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
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import websockets
from dotenv import load_dotenv


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


def _run_cmd(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)


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


def _generate_gemini_tts_fixtures(repo_root: Path, out_dir: Path, voice: str, language: str) -> dict[str, str]:
    turns = [
        {"id": "baseline_ready", "text": "Hello, please confirm you are ready to guide me safely."},
        {"id": "search_weather", "text": "Use the google_search tool and tell me today's weather in Kuala Lumpur in one sentence."},
        {"id": "nav_central_park", "text": "Use the get_walking_directions tool for Times Square to Central Park and give short walking guidance."},
        {"id": "context_followup", "text": "Please summarize what you just perceived and what I should do next."},
        {"id": "memory_set", "text": "Please remember this destination: Central Park."},
        {"id": "memory_recall", "text": "What destination did I just tell you? Answer briefly."},
    ]
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
    result = _run_cmd(cmd, repo_root)
    if result.returncode != 0:
        raise RuntimeError(
            "Gemini TTS fixture generation failed:\n"
            + (result.stdout or "")
            + "\n"
            + (result.stderr or "")
        )

    report_path = out_dir / "tts_fixtures" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    pcm_by_id: dict[str, str] = {}
    for turn in report.get("turns", []):
        turn_id = str(turn.get("id", "")).strip()
        pcm_path = str(turn.get("pcm_path", "")).strip()
        if turn_id and pcm_path:
            pcm_by_id[turn_id] = pcm_path
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
    pcm_path: str,
    wait_before_turn_sec: float = 10.0,
    telemetry_payload: dict[str, Any] | None = None,
    gesture: str | None = None,
    image_path: str | None = None,
    collect_sec: float = 45.0,
) -> ScenarioResult:
    pcm = Path(pcm_path).read_bytes()
    image_bytes = Path(image_path).read_bytes() if image_path else b""

    counts: dict[str, int] = {}
    tool_events: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    lod_updates: list[dict[str, Any]] = []
    transcripts: list[dict[str, str]] = []
    failures: list[str] = []
    notes: list[str] = []
    search_results = 0
    navigation_results = 0

    async with websockets.connect(ws_url, max_size=None) as ws:
        await _wait_for_type(ws, "session_ready", timeout_sec=20.0)
        if wait_before_turn_sec > 0:
            await asyncio.sleep(wait_before_turn_sec)

        if telemetry_payload:
            await ws.send(json.dumps({"type": "telemetry", "data": telemetry_payload}))
            notes.append("telemetry_sent")
        if gesture:
            await ws.send(json.dumps({"type": "gesture", "gesture": gesture}))
            notes.append(f"gesture_sent:{gesture}")
        if image_bytes:
            await ws.send(bytes([0x02]) + image_bytes)
            notes.append("image_frame_sent")

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

    # Generic baseline checks used by all scenarios.
    if not any(t.get("role") == "user" and t.get("text") for t in transcripts):
        failures.append("missing_user_transcript")
    if not any(t.get("role") == "agent" and t.get("text") for t in transcripts):
        failures.append("missing_agent_transcript")
    if counts.get("audio_bytes", 0) <= 0:
        failures.append("missing_agent_audio_bytes")

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

    async with websockets.connect(ws_url, max_size=None) as ws:
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

    if not any(t.get("role") == "user" and t.get("text") for t in transcripts):
        failures.append("missing_user_transcript")
    if not any(t.get("role") == "agent" and t.get("text") for t in transcripts):
        failures.append("missing_agent_transcript")
    if counts.get("audio_bytes", 0) <= 0:
        failures.append("missing_agent_audio_bytes")

    agent_text = " ".join(
        t.get("text", "") for t in transcripts if t.get("role") == "agent"
    ).lower()
    if "central park" not in agent_text:
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
    ingest_tools = _tool_names(scenarios["audio_ingest"])
    if ingest_tools:
        issues["audio_ingest"] = [f"unexpected_tools:{sorted(ingest_tools)}"]

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

    if not log_engine_check.passed:
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

    def ws_url(session: str) -> str:
        return f"{ws_base_url.rstrip('/')}/test_user/{session}"

    scenarios: dict[str, ScenarioResult] = {}

    ingest = await _run_probe(
        ws_url=ws_url("suite_audio_ingest"),
        pcm_path=pcm_by_id["baseline_ready"],
        wait_before_turn_sec=10.0,
        collect_sec=35.0,
    )
    ingest.name = "audio_ingest"
    if ingest.counts.get("debug_activity", 0) < 2:
        ingest.failures.append("missing_debug_activity_contract")
    ingest.passed = not ingest.failures
    scenarios["audio_ingest"] = ingest

    search = await _run_probe(
        ws_url=ws_url("suite_tool_search"),
        pcm_path=pcm_by_id["search_weather"],
        wait_before_turn_sec=10.0,
        collect_sec=45.0,
    )
    search.name = "tool_search"
    if not any(e.get("tool") == "google_search" for e in search.tool_events):
        search.failures.append("google_search_not_invoked")
    if search.search_results < 1:
        search.failures.append("search_result_not_emitted")
    search.passed = not search.failures
    scenarios["tool_search"] = search

    nav = await _run_probe(
        ws_url=ws_url("suite_tool_navigation"),
        pcm_path=pcm_by_id["nav_central_park"],
        wait_before_turn_sec=10.0,
        collect_sec=55.0,
    )
    nav.name = "tool_navigation"
    if not any(e.get("tool") in {"navigate_to", "get_walking_directions"} for e in nav.tool_events):
        nav.failures.append("navigation_tool_not_invoked")
    if nav.navigation_results < 1:
        nav.failures.append("navigation_result_not_emitted")
    nav.passed = not nav.failures
    scenarios["tool_navigation"] = nav

    context = await _run_probe(
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
        collect_sec=45.0,
    )
    context.name = "context_lod_vision"
    if not context.lod_updates:
        context.failures.append("lod_update_missing")
    if not any(e.get("tool") == "analyze_scene" for e in context.tool_events):
        context.failures.append("vision_tool_not_queued")
    context.passed = not context.failures
    scenarios["context_lod_vision"] = context

    memory = await _run_multiturn_memory_probe(
        ws_url=ws_url("suite_multiturn_memory"),
        memory_set_pcm_path=pcm_by_id["memory_set"],
        memory_recall_pcm_path=pcm_by_id["memory_recall"],
        wait_before_first_turn_sec=10.0,
    )
    memory.name = "multiturn_memory"
    scenarios["multiturn_memory"] = memory

    if log_settle_sec > 0:
        await asyncio.sleep(log_settle_sec)

    log_summary = (
        _analyze_server_log(server_log_path, marker=log_marker)
        if server_log_path
        else {"exists": False}
    )
    expected_sessions = [
        "suite_audio_ingest",
        "suite_tool_search",
        "suite_tool_navigation",
        "suite_context_lod_vision",
        "suite_multiturn_memory",
    ]
    log_engine_check = _run_log_engine_checks(
        log_summary=log_summary,
        expected_sessions=expected_sessions,
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
    p.add_argument("--output-root", type=Path, default=Path("artifacts/live_diagnostic_suite"))
    p.add_argument("--voice", default="Aoede")
    p.add_argument("--language", default="en-US")
    p.add_argument("--server-log-path", type=Path, default=Path("/tmp/sightline_runs/server_vertex_inmem_v2.log"))
    p.add_argument("--loops", type=int, default=1, help="How many full suite loops to run.")
    p.add_argument("--sleep-between-sec", type=float, default=60.0, help="Sleep between loops.")
    p.add_argument("--log-settle-sec", type=float, default=4.0, help="Wait before log analysis to avoid write-delay false negatives.")
    p.add_argument("--log-lifecycle-wait-sec", type=float, default=30.0, help="Extra wait window for delayed session_meta end/cleanup logs.")
    return p


async def _async_main(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = (args.output_root / stamp).resolve()
    root.mkdir(parents=True, exist_ok=True)

    last_report: dict[str, Any] | None = None
    for idx in range(args.loops):
        loop_dir = root / f"loop_{idx+1:02d}"
        loop_dir.mkdir(parents=True, exist_ok=True)
        report = await _run_suite_once(
            repo_root=repo_root,
            ws_base_url=args.ws_base_url,
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
