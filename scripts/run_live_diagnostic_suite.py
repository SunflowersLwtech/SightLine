#!/usr/bin/env python3
"""Continuous diagnostic suite for SightLine live audio workflows.

Runs real websocket scenarios against server.py and writes a structured report:
- Audio ingest baseline
- Tool invocation (search + navigation)
- LOD/context injection (telemetry + gesture + image)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import websockets
from dotenv import load_dotenv

_WS_CONNECT_RETRIES = 4
_WS_CONNECT_BACKOFF_SEC = 2.5
_WS_OPEN_TIMEOUT_SEC = 35.0


@dataclass
class ScenarioResult:
    name: str
    passed: bool
    failures: list[str]
    counts: dict[str, int]
    tool_events: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    lod_updates: list[dict[str, Any]]
    search_results: int
    navigation_results: int
    transcripts: list[dict[str, str]]
    notes: list[str]


@dataclass
class LogEngineCheck:
    passed: bool
    failures: list[str]
    session_starts: dict[str, int]
    session_ends: dict[str, int]
    session_cleanups: dict[str, int]
    interactions: dict[str, int]


def _failed_scenario(name: str, exc: Exception) -> ScenarioResult:
    return ScenarioResult(
        name=name,
        passed=False,
        failures=[f"scenario_exception:{type(exc).__name__}:{exc}"],
        counts={},
        tool_events=[],
        tool_results=[],
        lod_updates=[],
        search_results=0,
        navigation_results=0,
        transcripts=[],
        notes=["exception_captured"],
    )


def _run_cmd(
    cmd: list[str],
    cwd: Path,
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _prepare_image_fixture(repo_root: Path, out_dir: Path) -> Path:
    src = repo_root / "SightLine/Assets.xcassets/AppIcon.appiconset/AppIcon.png"
    dst = out_dir / "test_frame.jpg"
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-q:v", "3",
        str(dst),
    ]
    result = _run_cmd(ffmpeg_cmd, repo_root)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg image conversion failed: {result.stderr[:240]}")
    return dst


def _prepare_video_frames_fixture(repo_root: Path, out_dir: Path) -> list[Path]:
    """Create a short MP4 clip and extract JPEG frames for websocket streaming."""
    src = repo_root / "SightLine/Assets.xcassets/AppIcon.appiconset/AppIcon.png"
    clip_path = out_dir / "test_clip.mp4"
    frames_dir = out_dir / "test_clip_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    clip_cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", str(src),
        "-vf", "scale=640:640,zoompan=z='min(zoom+0.0018,1.15)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=96:s=640x640,fps=12",
        "-t", "8",
        "-pix_fmt", "yuv420p",
        str(clip_path),
    ]
    result = _run_cmd(clip_cmd, repo_root)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg clip generation failed: {result.stderr[:240]}")

    for existing in frames_dir.glob("frame_*.jpg"):
        existing.unlink(missing_ok=True)

    frames_cmd = [
        "ffmpeg", "-y",
        "-i", str(clip_path),
        "-vf", "fps=3",
        "-q:v", "3",
        str(frames_dir / "frame_%03d.jpg"),
    ]
    result = _run_cmd(frames_cmd, repo_root)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {result.stderr[:240]}")

    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if len(frames) < 3:
        raise RuntimeError("insufficient video frames extracted for diagnostics")
    return frames


def _generate_local_tts_pcm(
    repo_root: Path,
    fixtures_dir: Path,
    *,
    turn_id: str,
    text: str,
) -> Path:
    """Generate fallback PCM fixture via macOS `say` + ffmpeg when Gemini TTS is rate-limited."""
    audio_dir = fixtures_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    aiff_path = fixtures_dir / f"{turn_id}.aiff"
    pcm_path = audio_dir / f"{turn_id}.pcm16000.raw"

    say_cmd = ["say", "-v", "Samantha", "-o", str(aiff_path), text]
    say_result = _run_cmd(say_cmd, repo_root)
    if say_result.returncode != 0:
        raise RuntimeError(f"local say failed for {turn_id}: {say_result.stderr[:200]}")

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", str(aiff_path),
        "-ac", "1",
        "-ar", "16000",
        "-f", "s16le",
        str(pcm_path),
    ]
    ff_result = _run_cmd(ffmpeg_cmd, repo_root)
    if ff_result.returncode != 0:
        raise RuntimeError(f"local ffmpeg pcm convert failed for {turn_id}: {ff_result.stderr[:200]}")
    return pcm_path


def _generate_gemini_tts_fixtures(repo_root: Path, out_dir: Path, voice: str, language: str) -> dict[str, str]:
    turns = [
        {"id": "baseline_ready", "text": "Hello, please confirm you are ready to guide me safely."},
        {"id": "search_weather", "text": "What's the weather in Kuala Lumpur today? Please answer in one sentence."},
        {
            "id": "nav_central_park",
            "text": (
                "Start walking navigation now: from Times Square to Central Park. "
                "Give one immediate next step."
            ),
        },
        {"id": "context_followup", "text": "Please summarize what you just perceived and what I should do next."},
        {"id": "memory_set", "text": "Please remember this destination: Central Park."},
        {
            "id": "memory_recall",
            "text": (
                "What destination did I just tell you? "
                "Reply with the exact two words only."
            ),
        },
        {
            "id": "interrupt_long",
            "text": (
                "Please describe the full scene in as much detail as possible, "
                "including people, obstacles, signs, and movement, using at least six sentences."
            ),
        },
        {
            "id": "interrupt_barge",
            "text": "Stop now. Give me only one immediate safety instruction in one short sentence.",
        },
        {
            "id": "active_followup",
            "text": "Now actively guide me with one concise next action.",
        },
    ]
    required_ids = [str(t["id"]) for t in turns]
    turn_text_by_id = {str(t["id"]): str(t["text"]) for t in turns}

    def _cached_pcm_map(cache_fixtures_dir: Path) -> dict[str, str] | None:
        audio_dir = cache_fixtures_dir / "audio"
        mapping: dict[str, str] = {}
        for turn_id in required_ids:
            pcm_path = (audio_dir / f"{turn_id}.pcm16000.raw").resolve()
            if not pcm_path.exists():
                return None
            mapping[turn_id] = str(pcm_path)
        return mapping

    turns_fingerprint = hashlib.sha1(
        json.dumps(turns, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    cache_root = Path(os.getenv("SIGHTLINE_TTS_CACHE_DIR", "/tmp/sightline_runs/tts_fixture_cache"))
    cache_key = f"{voice}_{language}_{turns_fingerprint}".replace("/", "_")
    cache_fixtures_dir = cache_root / cache_key / "tts_fixtures"
    cached_map = _cached_pcm_map(cache_fixtures_dir)
    if cached_map:
        return cached_map

    turns_path = out_dir / "suite_turns.json"
    turns_path.write_text(json.dumps(turns, ensure_ascii=False, indent=2), encoding="utf-8")

    cmd = [
        "python",
        "scripts/gemini_tts_multiturn_test.py",
        "--turns-file",
        str(turns_path),
        "--output-dir",
        str(out_dir / "tts_fixtures"),
        "--skip-run",
        "--voice",
        voice,
        "--language-code",
        language,
    ]
    tts_env = dict(os.environ)
    if tts_env.get("GOOGLE_CLOUD_PROJECT"):
        # Prefer Vertex credentials for TTS to avoid API-key-only rate buckets.
        tts_env["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"
        tts_env.pop("GEMINI_API_KEY", None)
        tts_env.pop("GOOGLE_API_KEY", None)
        tts_env.pop("_GOOGLE_AI_API_KEY", None)

    attempts = 3
    result: subprocess.CompletedProcess[str] | None = None
    transient_quota_seen = False
    for attempt in range(attempts):
        result = _run_cmd(cmd, repo_root, env=tts_env)
        if result.returncode == 0:
            break
        combined = (result.stdout or "") + "\n" + (result.stderr or "")
        transient_quota = "RESOURCE_EXHAUSTED" in combined or "429" in combined
        transient_quota_seen = transient_quota_seen or transient_quota
        cached_map = _cached_pcm_map(cache_fixtures_dir)
        if transient_quota and cached_map:
            print("[suite] TTS quota hit; falling back to cached fixtures")
            return cached_map
        if transient_quota and attempt + 1 < attempts:
            backoff_sec = 20 * (attempt + 1)
            print(f"[suite] TTS quota hit; retrying in {backoff_sec}s (attempt {attempt + 2}/{attempts})")
            time.sleep(backoff_sec)
            continue
        raise RuntimeError(
            "Gemini TTS fixture generation failed:\n"
            + combined
        )

    fixtures_dir = out_dir / "tts_fixtures"

    if result is None or result.returncode != 0:
        if transient_quota_seen:
            print("[suite] Gemini TTS quota exhausted; generating missing fixtures with local fallback.")
            for turn_id in required_ids:
                pcm_candidate = fixtures_dir / "audio" / f"{turn_id}.pcm16000.raw"
                if pcm_candidate.exists() and pcm_candidate.stat().st_size > 0:
                    continue
                _generate_local_tts_pcm(
                    repo_root,
                    fixtures_dir,
                    turn_id=turn_id,
                    text=turn_text_by_id.get(turn_id, turn_id),
                )
            local_map = _cached_pcm_map(fixtures_dir)
            if local_map:
                cache_fixtures_dir.parent.mkdir(parents=True, exist_ok=True)
                if cache_fixtures_dir.exists():
                    shutil.rmtree(cache_fixtures_dir)
                shutil.copytree(fixtures_dir, cache_fixtures_dir)
                return local_map
        raise RuntimeError("Gemini TTS fixture generation failed with no result.")

    report_path = fixtures_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    pcm_by_id: dict[str, str] = {}
    for turn in report.get("turns", []):
        turn_id = str(turn.get("id", "")).strip()
        pcm_path = str(turn.get("pcm_path", "")).strip()
        if turn_id and pcm_path:
            pcm_by_id[turn_id] = pcm_path

    if all(turn_id in pcm_by_id and Path(pcm_by_id[turn_id]).exists() for turn_id in required_ids):
        cache_fixtures_dir.parent.mkdir(parents=True, exist_ok=True)
        if cache_fixtures_dir.exists():
            shutil.rmtree(cache_fixtures_dir)
        shutil.copytree(fixtures_dir, cache_fixtures_dir)
    return pcm_by_id


async def _wait_for_type(ws: websockets.ClientConnection, msg_type: str, timeout_sec: float) -> dict:
    deadline = time.monotonic() + timeout_sec
    while True:
        remain = deadline - time.monotonic()
        if remain <= 0:
            raise TimeoutError(f"did not receive {msg_type}")
        raw = await asyncio.wait_for(ws.recv(), timeout=remain)
        if isinstance(raw, bytes):
            continue
        payload = json.loads(raw)
        if payload.get("type") == msg_type:
            return payload


async def _connect_ws_with_retry(ws_url: str) -> websockets.ClientConnection:
    last_exc: Exception | None = None
    for attempt in range(1, _WS_CONNECT_RETRIES + 1):
        try:
            return await websockets.connect(
                ws_url,
                max_size=None,
                ping_interval=None,
                ping_timeout=None,
                close_timeout=8.0,
                open_timeout=_WS_OPEN_TIMEOUT_SEC,
            )
        except Exception as exc:
            last_exc = exc
            if attempt >= _WS_CONNECT_RETRIES:
                break
            await asyncio.sleep(_WS_CONNECT_BACKOFF_SEC * attempt)
    raise RuntimeError(f"websocket_connect_failed:{last_exc}")


async def _drain_initial_output(
    ws: websockets.ClientConnection,
    *,
    timeout_sec: float = 20.0,
    quiet_sec: float = 1.5,
) -> None:
    """Drain startup greeting/events so turn collection starts from a clean boundary."""
    deadline = time.monotonic() + timeout_sec
    last_event_at: float | None = None
    while time.monotonic() < deadline:
        now = time.monotonic()
        if last_event_at is not None and (now - last_event_at) >= quiet_sec:
            return
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=0.6)
        except asyncio.TimeoutError:
            continue
        if isinstance(raw, bytes):
            last_event_at = now
            continue
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        msg_type = str(payload.get("type", ""))
        if msg_type in {
            "transcript",
            "tool_event",
            "tool_result",
            "lod_update",
            "debug_activity",
            "interrupted",
        }:
            last_event_at = now


async def _send_video_frames(
    ws: websockets.ClientConnection,
    frame_bytes_list: list[bytes],
    *,
    interval_sec: float,
) -> int:
    sent = 0
    for frame in frame_bytes_list:
        await ws.send(bytes([0x02]) + frame)
        sent += 1
        if interval_sec > 0:
            await asyncio.sleep(interval_sec)
    return sent


def _record_payload(
    payload: dict[str, Any],
    *,
    counts: dict[str, int],
    tool_events: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
    lod_updates: list[dict[str, Any]],
    transcripts: list[dict[str, str]],
) -> tuple[int, int]:
    search_results = 0
    navigation_results = 0

    t = str(payload.get("type", "unknown"))
    counts[t] = counts.get(t, 0) + 1
    if t == "interrupted":
        accepted = payload.get("accepted")
        if accepted is True:
            counts["interrupted_accepted"] = counts.get("interrupted_accepted", 0) + 1
        elif accepted is False:
            counts["interrupted_ignored"] = counts.get("interrupted_ignored", 0) + 1

    if t == "tool_event":
        tool_events.append(
            {
                "tool": payload.get("tool"),
                "status": payload.get("status"),
                "behavior": payload.get("behavior"),
            }
        )
    elif t == "tool_result":
        tool_results.append(
            {
                "tool": payload.get("tool"),
                "behavior": payload.get("behavior"),
            }
        )
    elif t == "lod_update":
        lod_updates.append({"lod": payload.get("lod"), "reason": payload.get("reason")})
    elif t == "search_result":
        search_results += 1
    elif t == "navigation_result":
        navigation_results += 1
    elif t == "transcript":
        transcripts.append(
            {
                "role": str(payload.get("role", "")),
                "text": str(payload.get("text", "")),
            }
        )

    return search_results, navigation_results


async def _run_probe(
    *,
    ws_url: str,
    pcm_path: str | None,
    wait_before_turn_sec: float = 10.0,
    telemetry_payload: dict[str, Any] | None = None,
    gesture: str | None = None,
    image_path: str | None = None,
    video_frame_paths: list[str] | None = None,
    video_frame_interval_sec: float = 0.35,
    barge_in_pcm_path: str | None = None,
    barge_in_after_audio_chunks: int = 3,
    collect_sec: float = 45.0,
    require_user_transcript: bool = True,
    require_agent_transcript: bool = True,
    require_agent_audio: bool = True,
    expect_interrupt: bool = False,
    expect_no_agent_response: bool = False,
) -> ScenarioResult:
    pcm = Path(pcm_path).read_bytes() if pcm_path else b""
    image_bytes = Path(image_path).read_bytes() if image_path else b""
    video_frames = [Path(p).read_bytes() for p in (video_frame_paths or []) if Path(p).exists()]
    barge_pcm = Path(barge_in_pcm_path).read_bytes() if barge_in_pcm_path else b""

    counts: dict[str, int] = {}
    tool_events: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    lod_updates: list[dict[str, Any]] = []
    transcripts: list[dict[str, str]] = []
    failures: list[str] = []
    notes: list[str] = []
    search_results = 0
    navigation_results = 0
    barge_sent = False
    saw_agent_transcript = False

    ws = await _connect_ws_with_retry(ws_url)
    try:
        await _wait_for_type(ws, "session_ready", timeout_sec=20.0)
        if wait_before_turn_sec > 0:
            await asyncio.sleep(wait_before_turn_sec)
        await _drain_initial_output(ws, timeout_sec=20.0, quiet_sec=1.2)

        if telemetry_payload:
            await ws.send(json.dumps({"type": "telemetry", "data": telemetry_payload}))
            notes.append("telemetry_sent")
        if gesture:
            await ws.send(json.dumps({"type": "gesture", "gesture": gesture}))
            notes.append(f"gesture_sent:{gesture}")
        if image_bytes:
            await ws.send(bytes([0x02]) + image_bytes)
            notes.append("image_frame_sent")
        if video_frames and not pcm:
            sent = await _send_video_frames(
                ws,
                video_frames,
                interval_sec=max(0.0, float(video_frame_interval_sec)),
            )
            notes.append(f"video_frames_sent:{sent}")

        if pcm:
            await ws.send(json.dumps({"type": "activity_start"}))
            frame_idx = 0
            next_frame_at = time.monotonic() + max(0.0, float(video_frame_interval_sec))
            for offset in range(0, len(pcm), 1280):
                chunk = pcm[offset : offset + 1280]
                await ws.send(bytes([0x01]) + chunk)
                now = time.monotonic()
                if frame_idx < len(video_frames) and now >= next_frame_at:
                    await ws.send(bytes([0x02]) + video_frames[frame_idx])
                    frame_idx += 1
                    next_frame_at = now + max(0.0, float(video_frame_interval_sec))
                await asyncio.sleep((len(chunk) / 2.0) / 16000.0)
            await ws.send(json.dumps({"type": "activity_end"}))
            if frame_idx:
                notes.append(f"video_frames_sent_during_audio:{frame_idx}")
        else:
            notes.append("audio_skipped")

        async def _try_send_barge() -> None:
            nonlocal barge_sent, deadline
            if not barge_pcm or barge_sent:
                return
            audio_ready = counts.get("audio_bytes", 0) >= max(1, int(barge_in_after_audio_chunks))
            transcript_ready = saw_agent_transcript
            if not (audio_ready or transcript_ready):
                return
            try:
                await ws.send(json.dumps({"type": "client_barge_in"}))
                await ws.send(json.dumps({"type": "activity_start"}))
                for offset in range(0, len(barge_pcm), 1280):
                    chunk = barge_pcm[offset : offset + 1280]
                    await ws.send(bytes([0x01]) + chunk)
                    await asyncio.sleep((len(chunk) / 2.0) / 16000.0)
                await ws.send(json.dumps({"type": "activity_end"}))
                barge_sent = True
                notes.append("barge_in_sent")
                # Give post-barge response enough time even for long model output.
                deadline = max(deadline, time.monotonic() + 18.0)
            except websockets.ConnectionClosed as exc:
                failures.append(f"connection_closed_during_barge_send:{exc}")

        deadline = time.monotonic() + collect_sec
        while time.monotonic() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except websockets.ConnectionClosed as exc:
                failures.append(f"connection_closed_during_collect:{exc}")
                break

            if isinstance(raw, bytes):
                counts["audio_bytes"] = counts.get("audio_bytes", 0) + 1
                await _try_send_barge()
                continue

            payload = json.loads(raw)
            inc_search, inc_nav = _record_payload(
                payload,
                counts=counts,
                tool_events=tool_events,
                tool_results=tool_results,
                lod_updates=lod_updates,
                transcripts=transcripts,
            )
            search_results += inc_search
            navigation_results += inc_nav
            if (
                str(payload.get("type")) == "transcript"
                and str(payload.get("role")) == "agent"
                and str(payload.get("text", "")).strip()
            ):
                saw_agent_transcript = True

            await _try_send_barge()
    finally:
        try:
            await ws.close()
        except Exception:
            pass

    # Generic baseline checks used by all scenarios.
    if require_user_transcript and not any(t.get("role") == "user" and t.get("text") for t in transcripts):
        failures.append("missing_user_transcript")
    if require_agent_transcript and not any(t.get("role") == "agent" and t.get("text") for t in transcripts):
        failures.append("missing_agent_transcript")
    if require_agent_audio and counts.get("audio_bytes", 0) <= 0:
        failures.append("missing_agent_audio_bytes")
    if expect_interrupt:
        accepted_interrupts = counts.get("interrupted_accepted", 0)
        legacy_interrupts = counts.get("interrupted", 0)
        if accepted_interrupts <= 0 and legacy_interrupts <= 0:
            failures.append("missing_interrupted_event")
        elif accepted_interrupts <= 0 and counts.get("interrupted_ignored", 0) > 0:
            notes.append("barge_in_ignored_by_server")
    if expect_no_agent_response:
        if any(t.get("role") == "agent" and t.get("text") for t in transcripts):
            failures.append("unexpected_agent_transcript")
        if counts.get("audio_bytes", 0) > 0:
            failures.append("unexpected_agent_audio_bytes")
    if barge_in_pcm_path and not barge_sent:
        failures.append("barge_in_not_sent")

    return ScenarioResult(
        name="probe",
        passed=not failures,
        failures=failures,
        counts=counts,
        tool_events=tool_events,
        tool_results=tool_results,
        lod_updates=lod_updates,
        search_results=search_results,
        navigation_results=navigation_results,
        transcripts=transcripts[:40],
        notes=notes,
    )


async def _run_passive_active_probe(
    *,
    ws_url: str,
    active_pcm_path: str,
    telemetry_payload: dict[str, Any] | None = None,
    video_frame_paths: list[str] | None = None,
    video_frame_interval_sec: float = 0.35,
    collect_passive_sec: float = 12.0,
    collect_active_sec: float = 35.0,
    wait_before_sec: float = 10.0,
) -> ScenarioResult:
    """Run passive-only context injection followed by active voice interaction on same session."""
    active_pcm = Path(active_pcm_path).read_bytes()
    video_frames = [Path(p).read_bytes() for p in (video_frame_paths or []) if Path(p).exists()]

    counts: dict[str, int] = {}
    tool_events: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    lod_updates: list[dict[str, Any]] = []
    transcripts: list[dict[str, str]] = []
    failures: list[str] = []
    notes: list[str] = []
    search_results = 0
    navigation_results = 0

    ws = await _connect_ws_with_retry(ws_url)
    try:
        await _wait_for_type(ws, "session_ready", timeout_sec=20.0)
        if wait_before_sec > 0:
            await asyncio.sleep(wait_before_sec)
        await _drain_initial_output(ws, timeout_sec=20.0, quiet_sec=1.2)

        if telemetry_payload:
            await ws.send(json.dumps({"type": "telemetry", "data": telemetry_payload}))
            notes.append("telemetry_sent")
        if video_frames:
            sent = await _send_video_frames(
                ws,
                video_frames,
                interval_sec=max(0.0, float(video_frame_interval_sec)),
            )
            notes.append(f"passive_video_frames_sent:{sent}")

        passive_agent_audio_before = counts.get("audio_bytes", 0)
        passive_agent_text_before = len(
            [t for t in transcripts if t.get("role") == "agent" and t.get("text")]
        )
        passive_tool_event_before = len(tool_events)
        passive_agent_texts: list[str] = []

        passive_deadline = time.monotonic() + max(1.0, float(collect_passive_sec))
        while time.monotonic() < passive_deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            if isinstance(raw, bytes):
                counts["audio_bytes"] = counts.get("audio_bytes", 0) + 1
                continue
            payload = json.loads(raw)
            if str(payload.get("type")) == "transcript" and str(payload.get("role")) == "agent":
                txt = str(payload.get("text", "")).strip()
                if txt:
                    passive_agent_texts.append(txt)
            inc_search, inc_nav = _record_payload(
                payload,
                counts=counts,
                tool_events=tool_events,
                tool_results=tool_results,
                lod_updates=lod_updates,
                transcripts=transcripts,
            )
            search_results += inc_search
            navigation_results += inc_nav

        passive_agent_audio_after = counts.get("audio_bytes", 0)
        passive_agent_text_after = len(
            [t for t in transcripts if t.get("role") == "agent" and t.get("text")]
        )
        passive_tool_event_after = len(tool_events)
        passive_audio_delta = passive_agent_audio_after - passive_agent_audio_before
        passive_text_delta = passive_agent_text_after - passive_agent_text_before

        def _is_startup_greeting(text: str) -> bool:
            t = text.lower()
            if "ready when you are" in t:
                return True
            # Model often streams greeting across short partial chunks:
            # "Hello. I'm" + "ready when you are."
            if len(t.strip()) <= 24 and ("hello" in t or "hi" in t or "ready" in t):
                return True
            return ("hello" in t or "hi" in t) and ("ready" in t or "help" in t)

        greeting_only = (
            passive_text_delta > 0
            and passive_agent_texts
            and all(_is_startup_greeting(t) for t in passive_agent_texts)
        )

        if greeting_only:
            notes.append("late_startup_greeting_ignored")
        else:
            if passive_audio_delta > 0:
                failures.append("passive_mode_unexpected_agent_audio")
            if passive_text_delta > 0:
                failures.append("passive_mode_unexpected_agent_transcript")

        a, b = await _send_turn_audio(
            ws=ws,
            pcm=active_pcm,
            counts=counts,
            tool_events=tool_events,
            tool_results=tool_results,
            lod_updates=lod_updates,
            transcripts=transcripts,
            collect_sec=max(5.0, float(collect_active_sec)),
        )
        search_results += a
        navigation_results += b
        notes.append("active_turn_sent")
    finally:
        try:
            await ws.close()
        except Exception:
            pass

    if not any(t.get("role") == "user" and t.get("text") for t in transcripts):
        failures.append("missing_user_transcript")
    if not any(t.get("role") == "agent" and t.get("text") for t in transcripts):
        failures.append("missing_agent_transcript")
    if counts.get("audio_bytes", 0) <= 0:
        failures.append("missing_agent_audio_bytes")

    return ScenarioResult(
        name="passive_active_cycle",
        passed=not failures,
        failures=failures,
        counts=counts,
        tool_events=tool_events,
        tool_results=tool_results,
        lod_updates=lod_updates,
        search_results=search_results,
        navigation_results=navigation_results,
        transcripts=transcripts[:60],
        notes=notes,
    )


async def _send_turn_audio(
    *,
    ws: websockets.ClientConnection,
    pcm: bytes,
    counts: dict[str, int],
    tool_events: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
    lod_updates: list[dict[str, Any]],
    transcripts: list[dict[str, str]],
    collect_sec: float,
) -> tuple[int, int]:
    search_results = 0
    navigation_results = 0
    await ws.send(json.dumps({"type": "activity_start"}))
    for offset in range(0, len(pcm), 1280):
        chunk = pcm[offset : offset + 1280]
        await ws.send(bytes([0x01]) + chunk)
        await asyncio.sleep((len(chunk) / 2.0) / 16000.0)
    await ws.send(json.dumps({"type": "activity_end"}))

    deadline = time.monotonic() + collect_sec
    while time.monotonic() < deadline:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
        except asyncio.TimeoutError:
            continue
        if isinstance(raw, bytes):
            counts["audio_bytes"] = counts.get("audio_bytes", 0) + 1
            continue
        payload = json.loads(raw)
        inc_search, inc_nav = _record_payload(
            payload,
            counts=counts,
            tool_events=tool_events,
            tool_results=tool_results,
            lod_updates=lod_updates,
            transcripts=transcripts,
        )
        search_results += inc_search
        navigation_results += inc_nav
    return search_results, navigation_results


async def _run_multiturn_memory_probe(
    *,
    ws_url: str,
    memory_set_pcm_path: str,
    memory_recall_pcm_path: str,
    wait_before_first_turn_sec: float = 10.0,
) -> ScenarioResult:
    counts: dict[str, int] = {}
    tool_events: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    lod_updates: list[dict[str, Any]] = []
    transcripts: list[dict[str, str]] = []
    failures: list[str] = []
    notes: list[str] = []
    search_results = 0
    navigation_results = 0

    set_pcm = Path(memory_set_pcm_path).read_bytes()
    recall_pcm = Path(memory_recall_pcm_path).read_bytes()

    ws = await _connect_ws_with_retry(ws_url)
    try:
        await _wait_for_type(ws, "session_ready", timeout_sec=20.0)
        if wait_before_first_turn_sec > 0:
            await asyncio.sleep(wait_before_first_turn_sec)

        a, b = await _send_turn_audio(
            ws=ws,
            pcm=set_pcm,
            counts=counts,
            tool_events=tool_events,
            tool_results=tool_results,
            lod_updates=lod_updates,
            transcripts=transcripts,
            collect_sec=20.0,
        )
        search_results += a
        navigation_results += b
        notes.append("turn_1_sent")

        a, b = await _send_turn_audio(
            ws=ws,
            pcm=recall_pcm,
            counts=counts,
            tool_events=tool_events,
            tool_results=tool_results,
            lod_updates=lod_updates,
            transcripts=transcripts,
            collect_sec=25.0,
        )
        search_results += a
        navigation_results += b
        notes.append("turn_2_sent")
    finally:
        try:
            await ws.close()
        except Exception:
            pass

    if not any(t.get("role") == "user" and t.get("text") for t in transcripts):
        failures.append("missing_user_transcript")
    if not any(t.get("role") == "agent" and t.get("text") for t in transcripts):
        failures.append("missing_agent_transcript")
    if counts.get("audio_bytes", 0) <= 0:
        failures.append("missing_agent_audio_bytes")

    agent_text = " ".join(
        t.get("text", "") for t in transcripts if t.get("role") == "agent"
    ).lower()
    if not any(token in agent_text for token in ("central park", "central", "park")):
        failures.append("memory_recall_missing_central_park")

    return ScenarioResult(
        name="multiturn_memory",
        passed=not failures,
        failures=failures,
        counts=counts,
        tool_events=tool_events,
        tool_results=tool_results,
        lod_updates=lod_updates,
        search_results=search_results,
        navigation_results=navigation_results,
        transcripts=transcripts[:60],
        notes=notes,
    )


def _log_marker(log_path: Path | None) -> int | None:
    if not log_path:
        return None
    if not log_path.exists():
        return 0
    return log_path.stat().st_size


def _analyze_server_log(log_path: Path, *, marker: int | None = None) -> dict[str, Any]:
    if not log_path.exists():
        return {"exists": False}
    if marker is not None:
        with log_path.open("rb") as f:
            if marker > 0:
                f.seek(marker)
            raw = f.read()
        text = raw.decode("utf-8", errors="ignore")
    else:
        text = log_path.read_text(encoding="utf-8", errors="ignore")

    session_starts: dict[str, int] = {}
    session_ends: dict[str, int] = {}
    session_cleanups: dict[str, int] = {}
    interactions: dict[str, int] = {}
    barge_accepted: dict[str, int] = {}
    barge_ignored: dict[str, int] = {}
    for line in text.splitlines():
        if "session_meta start written:" in line and "session=" in line:
            session = line.split("session=", 1)[1].strip()
            session_starts[session] = session_starts.get(session, 0) + 1
        elif "session_meta end written:" in line and "session=" in line:
            rest = line.split("session=", 1)[1]
            session = rest.split(" interactions=", 1)[0].strip()
            session_ends[session] = session_ends.get(session, 0) + 1
            if " interactions=" in rest:
                try:
                    interactions[session] = int(rest.split(" interactions=", 1)[1].strip())
                except ValueError:
                    interactions[session] = -1
        elif "Session cleaned up:" in line and "session=" in line:
            session = line.split("session=", 1)[1].strip()
            session_cleanups[session] = session_cleanups.get(session, 0) + 1
        elif "Client barge-in — suppressing audio forwarding" in line and "session=" in line:
            session = line.split("session=", 1)[1].strip()
            barge_accepted[session] = barge_accepted.get(session, 0) + 1
        elif "Client barge-in ignored — model not speaking" in line and "session=" in line:
            session = line.split("session=", 1)[1].strip()
            barge_ignored[session] = barge_ignored.get(session, 0) + 1

    return {
        "exists": True,
        "log_delta_bytes": len(text.encode("utf-8")),
        "function_call_count": text.count("Function call:"),
        "lod_update_injected_count": text.count("Injected [LOD UPDATE]"),
        "vision_injected_count": text.count("Injected [VISION ANALYSIS]"),
        "telemetry_queued_count": text.count("Queued [telemetry]"),
        "downstream_error_count": text.count("Error in downstream handler"),
        "model_503_count": text.count(" 503 ") + text.count("UNAVAILABLE"),
        "face_unavailable_count": text.count("InsightFace is unavailable in this environment"),
        "default_user_context_count": (
            text.count("for user default")
            + text.count("'user_id': 'default'")
            + text.count('"user_id": "default"')
        ),
        "session_starts": session_starts,
        "session_ends": session_ends,
        "session_cleanups": session_cleanups,
        "session_interactions": interactions,
        "barge_accepted_sessions": barge_accepted,
        "barge_ignored_sessions": barge_ignored,
        "traceback_count": text.count("Traceback (most recent call last):"),
        "context_flush_count": text.count("Flushed"),
    }


def _run_log_engine_checks(
    *,
    log_summary: dict[str, Any],
    expected_sessions: list[str],
) -> LogEngineCheck:
    starts = dict(log_summary.get("session_starts") or {})
    ends = dict(log_summary.get("session_ends") or {})
    cleanups = dict(log_summary.get("session_cleanups") or {})
    interactions = dict(log_summary.get("session_interactions") or {})
    failures: list[str] = []

    for sid in expected_sessions:
        if starts.get(sid, 0) < 1:
            failures.append(f"missing_session_meta_start:{sid}")
        if ends.get(sid, 0) < 1:
            failures.append(f"missing_session_meta_end:{sid}")
        if cleanups.get(sid, 0) < 1:
            failures.append(f"missing_session_cleanup:{sid}")
        if interactions.get(sid, 0) <= 0:
            failures.append(f"invalid_interactions_count:{sid}")

    return LogEngineCheck(
        passed=not failures,
        failures=failures,
        session_starts=starts,
        session_ends=ends,
        session_cleanups=cleanups,
        interactions=interactions,
    )


def _tool_names(result: ScenarioResult) -> set[str]:
    return {str(e.get("tool")) for e in result.tool_events if e.get("tool")}


def _check_reasonableness(scenarios: dict[str, ScenarioResult]) -> dict[str, list[str]]:
    issues: dict[str, list[str]] = {}
    search_tools = _tool_names(scenarios["tool_search"])
    if "google_search" not in search_tools:
        issues.setdefault("tool_search", []).append("missing_google_search")
    banned_for_search = {"get_walking_directions", "navigate_to"} & search_tools
    if banned_for_search:
        issues.setdefault("tool_search", []).append(
            f"unexpected_navigation_tools:{sorted(banned_for_search)}"
        )

    nav_tools = _tool_names(scenarios["tool_navigation"])
    if not ({"get_walking_directions", "navigate_to"} & nav_tools):
        issues.setdefault("tool_navigation", []).append("missing_navigation_tool")
    if "google_search" in nav_tools:
        issues.setdefault("tool_navigation", []).append("unexpected_google_search")

    context_tools = _tool_names(scenarios["context_lod_vision"])
    if "analyze_scene" not in context_tools:
        issues.setdefault("context_lod_vision", []).append("missing_analyze_scene")
    if {"navigate_to", "get_walking_directions"} & context_tools:
        issues.setdefault("context_lod_vision", []).append("unexpected_navigation_tool")

    memory_tools = _tool_names(scenarios["multiturn_memory"])
    if "google_search" in memory_tools:
        issues.setdefault("multiturn_memory", []).append("unexpected_google_search")

    if "interrupt_barge_in" in scenarios:
        interrupt = scenarios["interrupt_barge_in"]
        if any(f == "missing_interrupted_event" for f in interrupt.failures):
            issues.setdefault("interrupt_barge_in", []).append("missing_interrupted_event")

    if "passive_active_cycle" in scenarios:
        passive_active = scenarios["passive_active_cycle"]
        if any("passive_mode_unexpected" in f for f in passive_active.failures):
            issues.setdefault("passive_active_cycle", []).append("passive_mode_spoke_unexpectedly")

    return issues


def _build_issue_definitions(
    scenarios: dict[str, ScenarioResult],
    log_summary: dict[str, Any],
    log_engine_check: LogEngineCheck,
    reasonableness_issues: dict[str, list[str]],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    ingest = scenarios["audio_ingest"]
    if not ingest.passed:
        issues.append(
            {
                "id": "I-001",
                "title": "Live audio ingest regression",
                "definition": "Audio turn does not produce user transcript / agent response / audio output.",
                "evidence": ingest.failures,
                "severity": "critical",
            }
        )

    search = scenarios["tool_search"]
    if not any(e.get("tool") == "google_search" for e in search.tool_events):
        issues.append(
            {
                "id": "I-002",
                "title": "Search tool invocation missing",
                "definition": "Search intent did not invoke google_search tool.",
                "evidence": search.tool_events,
                "severity": "high",
            }
        )

    nav = scenarios["tool_navigation"]
    if not any(e.get("tool") in {"navigate_to", "get_walking_directions"} for e in nav.tool_events):
        issues.append(
            {
                "id": "I-003",
                "title": "Navigation tool invocation missing",
                "definition": "Navigation intent did not trigger navigation tool call.",
                "evidence": nav.tool_events,
                "severity": "high",
            }
        )

    context = scenarios["context_lod_vision"]
    if not context.lod_updates:
        issues.append(
            {
                "id": "I-004",
                "title": "LOD update missing under force gesture",
                "definition": "force_lod_3 gesture + telemetry failed to produce lod_update event.",
                "evidence": context.counts,
                "severity": "high",
            }
        )
    if not any(e.get("tool") == "analyze_scene" for e in context.tool_events):
        issues.append(
            {
                "id": "I-005",
                "title": "Vision queue trigger missing",
                "definition": "Image frame did not enqueue analyze_scene tool event.",
                "evidence": context.tool_events,
                "severity": "medium",
            }
        )

    memory = scenarios["multiturn_memory"]
    if not memory.passed:
        issues.append(
            {
                "id": "I-008",
                "title": "Multiturn context continuity regression",
                "definition": "Same-session second turn failed to correctly recall first-turn destination context.",
                "evidence": memory.failures,
                "severity": "high",
            }
        )

    interrupt = scenarios.get("interrupt_barge_in")
    if interrupt and not interrupt.passed:
        issues.append(
            {
                "id": "I-013",
                "title": "Barge-in interruption regression",
                "definition": "Model output was not reliably interrupted or resumed after client barge-in.",
                "evidence": interrupt.failures,
                "severity": "high",
            }
        )

    passive_active = scenarios.get("passive_active_cycle")
    if passive_active and not passive_active.passed:
        issues.append(
            {
                "id": "I-014",
                "title": "Passive/active interaction mode regression",
                "definition": "Passive mode emitted speech unexpectedly or failed to recover to active mode.",
                "evidence": passive_active.failures,
                "severity": "high",
            }
        )

    if reasonableness_issues:
        issues.append(
            {
                "id": "I-009",
                "title": "Input/output or tool-call reasonableness mismatch",
                "definition": "Observed tool call pattern does not match scenario intent contract.",
                "evidence": reasonableness_issues,
                "severity": "high",
            }
        )

    if bool(log_summary.get("exists")) and not log_engine_check.passed:
        issues.append(
            {
                "id": "I-010",
                "title": "Log Engine lifecycle contract mismatch",
                "definition": "session_meta start/end/cleanup contract is incomplete for one or more sessions.",
                "evidence": log_engine_check.failures,
                "severity": "high",
            }
        )

    if log_summary.get("face_unavailable_count", 0) > 0:
        issues.append(
            {
                "id": "I-006",
                "title": "Face pipeline unavailable",
                "definition": "Face recognition dependency is unavailable in runtime, resulting in SILENT tool errors.",
                "evidence": {"face_unavailable_count": log_summary.get("face_unavailable_count", 0)},
                "severity": "medium",
            }
        )

    if log_summary.get("traceback_count", 0) > 0:
        issues.append(
            {
                "id": "I-011",
                "title": "Runtime traceback observed",
                "definition": "Unhandled runtime traceback appeared in server log during suite execution.",
                "evidence": {"traceback_count": log_summary.get("traceback_count", 0)},
                "severity": "medium",
            }
        )

    if log_summary.get("default_user_context_count", 0) > 0:
        issues.append(
            {
                "id": "I-012",
                "title": "Cross-user context leakage risk (default user id)",
                "definition": "Tool calls or context writes used user_id=default during an authenticated test_user session.",
                "evidence": {"default_user_context_count": log_summary.get("default_user_context_count", 0)},
                "severity": "critical",
            }
        )

    if log_summary.get("model_503_count", 0) > 0:
        issues.append(
            {
                "id": "I-007",
                "title": "Upstream model capacity instability",
                "definition": "Gemini upstream returned UNAVAILABLE/503 during scenario execution.",
                "evidence": {"model_503_count": log_summary.get("model_503_count", 0)},
                "severity": "medium",
            }
        )

    return issues


async def _run_suite_once(
    *,
    repo_root: Path,
    ws_base_url: str,
    user_id: str,
    session_prefix: str,
    out_dir: Path,
    voice: str,
    language: str,
    server_log_path: Path | None,
    log_settle_sec: float,
    log_lifecycle_wait_sec: float,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_marker = _log_marker(server_log_path)
    pcm_by_id = _generate_gemini_tts_fixtures(repo_root, out_dir, voice=voice, language=language)
    image_path = _prepare_image_fixture(repo_root, out_dir)
    video_frames = _prepare_video_frames_fixture(repo_root, out_dir)
    video_frame_paths = [str(p) for p in video_frames]

    def ws_url(session: str) -> str:
        return f"{ws_base_url.rstrip('/')}/{user_id}/{session_prefix}{session}"

    scenarios: dict[str, ScenarioResult] = {}

    async def _safe_run(
        name: str,
        run_factory,
        *,
        attempts: int = 2,
        retry_delay_sec: float = 3.0,
    ) -> ScenarioResult:
        last_result: ScenarioResult | None = None
        total = max(1, int(attempts))
        for attempt in range(1, total + 1):
            try:
                result = await run_factory()
                result.name = name
            except Exception as exc:
                result = _failed_scenario(name, exc)
                result.name = name
            if result.passed:
                if attempt > 1:
                    result.notes.append(f"passed_on_retry:{attempt}/{total}")
                return result
            result.notes.append(f"attempt_failed:{attempt}/{total}")
            last_result = result
            if attempt < total:
                await asyncio.sleep(max(0.0, float(retry_delay_sec)) * attempt)
        return last_result or _failed_scenario(name, RuntimeError("unknown_scenario_failure"))

    ingest = await _safe_run(
        "audio_ingest",
        lambda: _run_probe(
            ws_url=ws_url("suite_audio_ingest"),
            pcm_path=pcm_by_id["baseline_ready"],
            wait_before_turn_sec=10.0,
            collect_sec=35.0,
            require_user_transcript=False,
        ),
    )
    ingest.name = "audio_ingest"
    if ingest.counts.get("debug_activity", 0) < 2:
        ingest.failures.append("missing_debug_activity_contract")
    ingest.passed = not ingest.failures
    scenarios["audio_ingest"] = ingest

    search = await _safe_run(
        "tool_search",
        lambda: _run_probe(
            ws_url=ws_url("suite_tool_search"),
            pcm_path=pcm_by_id["search_weather"],
            wait_before_turn_sec=10.0,
            collect_sec=45.0,
            require_user_transcript=False,
            require_agent_transcript=False,
            require_agent_audio=False,
        ),
    )
    search.name = "tool_search"
    if not (
        any(e.get("tool") == "google_search" for e in search.tool_events)
        or any(r.get("tool") == "google_search" for r in search.tool_results)
    ):
        search.failures.append("google_search_not_invoked")
    if search.search_results < 1:
        search.notes.append("search_result_not_emitted")
    search.passed = not search.failures
    scenarios["tool_search"] = search

    nav = await _safe_run(
        "tool_navigation",
        lambda: _run_probe(
            ws_url=ws_url("suite_tool_navigation"),
            pcm_path=pcm_by_id["nav_central_park"],
            wait_before_turn_sec=10.0,
            collect_sec=55.0,
            require_user_transcript=False,
            require_agent_transcript=False,
            require_agent_audio=False,
        ),
    )
    nav.name = "tool_navigation"
    if not (
        any(e.get("tool") in {"navigate_to", "get_walking_directions"} for e in nav.tool_events)
        or any(r.get("tool") in {"navigate_to", "get_walking_directions"} for r in nav.tool_results)
    ):
        nav.failures.append("navigation_tool_not_invoked")
    if nav.navigation_results < 1:
        nav.notes.append("navigation_result_not_emitted")
    nav.passed = not nav.failures
    scenarios["tool_navigation"] = nav

    context = await _safe_run(
        "context_lod_vision",
        lambda: _run_probe(
            ws_url=ws_url("suite_context_lod_vision"),
            pcm_path=pcm_by_id["context_followup"],
            wait_before_turn_sec=10.0,
            telemetry_payload={
                "motion_state": "running",
                "step_cadence": 140,
                "ambient_noise_db": 86,
                "heart_rate": 138,
                "heading": 95,
                "gps": {
                    "latitude": 40.7580,
                    "longitude": -73.9855,
                    "accuracy": 6.0,
                    "speed": 2.7,
                    "altitude": 12.0,
                },
                "time_context": "evening",
                "device_type": "phone_and_watch",
                "watch_stability_score": 0.42,
                "watch_noise_exposure": 89,
            },
            gesture="force_lod_3",
            image_path=str(image_path),
            video_frame_paths=video_frame_paths[:6],
            video_frame_interval_sec=0.35,
            collect_sec=45.0,
            require_user_transcript=False,
            require_agent_transcript=False,
            require_agent_audio=False,
        ),
    )
    context.name = "context_lod_vision"
    if not context.lod_updates:
        context.failures.append("lod_update_missing")
    if not any(e.get("tool") == "analyze_scene" for e in context.tool_events):
        context.failures.append("vision_tool_not_queued")
    context.passed = not context.failures
    scenarios["context_lod_vision"] = context

    memory = await _safe_run(
        "multiturn_memory",
        lambda: _run_multiturn_memory_probe(
            ws_url=ws_url("suite_multiturn_memory"),
            memory_set_pcm_path=pcm_by_id["memory_set"],
            memory_recall_pcm_path=pcm_by_id["memory_recall"],
            wait_before_first_turn_sec=10.0,
        ),
    )
    memory.name = "multiturn_memory"
    scenarios["multiturn_memory"] = memory

    interrupt = await _safe_run(
        "interrupt_barge_in",
        lambda: _run_probe(
            ws_url=ws_url("suite_interrupt_barge_in"),
            pcm_path=pcm_by_id["interrupt_long"],
            wait_before_turn_sec=10.0,
            image_path=None,
            video_frame_paths=None,
            barge_in_pcm_path=pcm_by_id["interrupt_barge"],
            barge_in_after_audio_chunks=1,
            collect_sec=30.0,
            require_user_transcript=False,
            require_agent_transcript=False,
            require_agent_audio=False,
            expect_interrupt=True,
        ),
    )
    interrupt.name = "interrupt_barge_in"
    scenarios["interrupt_barge_in"] = interrupt

    passive_active = await _safe_run(
        "passive_active_cycle",
        lambda: _run_passive_active_probe(
            ws_url=ws_url("suite_passive_active_cycle"),
            active_pcm_path=pcm_by_id["active_followup"],
            telemetry_payload={
                "motion_state": "stationary",
                "step_cadence": 0,
                "ambient_noise_db": 62,
                "heading": 135,
                "gps": {
                    "latitude": 40.7581,
                    "longitude": -73.9854,
                    "accuracy": 6.5,
                    "speed": 0.0,
                    "altitude": 11.2,
                },
                "time_context": "daytime",
                "device_type": "phone_and_watch",
                "watch_stability_score": 0.91,
                "watch_noise_exposure": 64,
            },
            video_frame_paths=video_frame_paths[:7],
            video_frame_interval_sec=0.3,
            collect_passive_sec=12.0,
            collect_active_sec=35.0,
            wait_before_sec=10.0,
        ),
    )
    passive_active.name = "passive_active_cycle"
    scenarios["passive_active_cycle"] = passive_active

    if log_settle_sec > 0:
        await asyncio.sleep(log_settle_sec)

    log_summary = (
        _analyze_server_log(server_log_path, marker=log_marker)
        if server_log_path
        else {"exists": False}
    )
    expected_sessions = [
        f"{session_prefix}suite_audio_ingest",
        f"{session_prefix}suite_tool_search",
        f"{session_prefix}suite_tool_navigation",
        f"{session_prefix}suite_context_lod_vision",
        f"{session_prefix}suite_multiturn_memory",
        f"{session_prefix}suite_interrupt_barge_in",
        f"{session_prefix}suite_passive_active_cycle",
    ]
    interrupt_session = f"{session_prefix}suite_interrupt_barge_in"
    if bool(log_summary.get("exists")):
        log_engine_check = _run_log_engine_checks(
            log_summary=log_summary,
            expected_sessions=expected_sessions,
        )
        observed_expected_entries = sum(
            log_engine_check.session_starts.get(sid, 0)
            + log_engine_check.session_ends.get(sid, 0)
            + log_engine_check.session_cleanups.get(sid, 0)
            for sid in expected_sessions
        )
        if observed_expected_entries == 0:
            # Wrong/rotated log path can produce complete false negatives.
            # Skip lifecycle contract check when none of the expected sessions
            # are present in the analyzed log segment.
            log_summary["lifecycle_check_skipped"] = "no_expected_sessions_in_log_segment"
            log_engine_check = LogEngineCheck(
                passed=True,
                failures=[],
                session_starts=log_engine_check.session_starts,
                session_ends=log_engine_check.session_ends,
                session_cleanups=log_engine_check.session_cleanups,
                interactions=log_engine_check.interactions,
            )
        if (
            server_log_path
            and log_engine_check.failures
            and log_lifecycle_wait_sec > 0
        ):
            wait_deadline = time.monotonic() + float(log_lifecycle_wait_sec)
            while time.monotonic() < wait_deadline and log_engine_check.failures:
                await asyncio.sleep(3.0)
                log_summary = _analyze_server_log(server_log_path, marker=log_marker)
                log_engine_check = _run_log_engine_checks(
                    log_summary=log_summary,
                    expected_sessions=expected_sessions,
                )
    else:
        log_engine_check = LogEngineCheck(
            passed=True,
            failures=[],
            session_starts={},
            session_ends={},
            session_cleanups={},
            interactions={},
        )

    # Observability fallback:
    # If interrupted event is missing but server log confirms client barge-in
    # suppression for this session, treat interruption as passed.
    interrupt = scenarios.get("interrupt_barge_in")
    if interrupt and any(f == "missing_interrupted_event" for f in interrupt.failures):
        accepted = int((log_summary.get("barge_accepted_sessions") or {}).get(interrupt_session, 0))
        if accepted > 0:
            interrupt.failures = [f for f in interrupt.failures if f != "missing_interrupted_event"]
            if interrupt.counts.get("interrupted", 0) <= 0:
                interrupt.notes.append("interrupted_event_missing_but_server_confirmed_barge")
            interrupt.passed = not interrupt.failures

    reasonableness_issues = _check_reasonableness(scenarios)
    for scenario_name, errs in reasonableness_issues.items():
        scenarios[scenario_name].failures.extend(errs)
        scenarios[scenario_name].passed = False

    issues = _build_issue_definitions(
        scenarios,
        log_summary,
        log_engine_check,
        reasonableness_issues,
    )

    all_passed = all(s.passed for s in scenarios.values()) and not issues
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ws_base_url": ws_base_url,
        "user_id": user_id,
        "session_prefix": session_prefix,
        "voice": voice,
        "language": language,
        "all_passed": all_passed,
        "scenarios": {k: vars(v) for k, v in scenarios.items()},
        "log_summary": log_summary,
        "log_engine_check": vars(log_engine_check),
        "reasonableness_issues": reasonableness_issues,
        "issues": issues,
    }
    return report


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run continuous SightLine live diagnostics.")
    p.add_argument("--ws-base-url", default="ws://127.0.0.1:8100/ws", help="WebSocket base URL without trailing user/session.")
    p.add_argument("--user-id", default="test_user", help="User id used in ws path /ws/{user_id}/{session_id}.")
    p.add_argument("--session-prefix", default="", help="Prefix added to every diagnostic session id for isolation.")
    p.add_argument("--output-root", type=Path, default=Path("artifacts/live_diagnostic_suite"))
    p.add_argument("--voice", default="Aoede")
    p.add_argument("--language", default="en-US")
    p.add_argument("--server-log-path", type=Path, default=Path("/private/tmp/sightline-server.log"))
    p.add_argument("--loops", type=int, default=1, help="How many full suite loops to run.")
    p.add_argument("--sleep-between-sec", type=float, default=60.0, help="Sleep between loops.")
    p.add_argument("--log-settle-sec", type=float, default=4.0, help="Wait before log analysis to avoid write-delay false negatives.")
    p.add_argument("--log-lifecycle-wait-sec", type=float, default=60.0, help="Extra wait window for delayed session_meta end/cleanup logs.")
    return p


async def _async_main(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = (args.output_root / stamp).resolve()
    root.mkdir(parents=True, exist_ok=True)
    session_prefix = str(args.session_prefix or f"diag_{stamp}_")
    if session_prefix and not session_prefix.endswith("_"):
        session_prefix = f"{session_prefix}_"

    last_report: dict[str, Any] | None = None
    for idx in range(args.loops):
        loop_dir = root / f"loop_{idx+1:02d}"
        loop_dir.mkdir(parents=True, exist_ok=True)
        report = await _run_suite_once(
            repo_root=repo_root,
            ws_base_url=args.ws_base_url,
            user_id=str(args.user_id).strip() or "test_user",
            session_prefix=session_prefix,
            out_dir=loop_dir,
            voice=args.voice,
            language=args.language,
            server_log_path=args.server_log_path,
            log_settle_sec=max(0.0, float(args.log_settle_sec)),
            log_lifecycle_wait_sec=max(0.0, float(args.log_lifecycle_wait_sec)),
        )
        report_path = loop_dir / "suite_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[loop {idx+1}] report={report_path}")
        print(f"[loop {idx+1}] all_passed={report['all_passed']} issues={len(report['issues'])}")
        last_report = report
        if idx + 1 < args.loops:
            await asyncio.sleep(max(0.0, float(args.sleep_between_sec)))

    if not last_report:
        return 1
    return 0 if bool(last_report.get("all_passed")) else 2


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    args = _build_parser().parse_args()
    try:
        return asyncio.run(_async_main(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
