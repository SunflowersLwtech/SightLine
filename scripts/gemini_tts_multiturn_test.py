#!/usr/bin/env python3
"""Generate Gemini TTS fixtures and run continuous multi-turn WS tests.

This script is designed for real end-to-end speech loops:
1) Use Gemini TTS to synthesize high-quality utterances.
2) Convert output to 16k mono PCM expected by SightLine websocket upstream.
3) Replay turns continuously to /ws/{user_id}/{session_id}.
4) Collect transcripts/latency/audio stats and write a JSON report.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import re
import sys
import time
import wave
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import websockets
from dotenv import load_dotenv
from google import genai
from google.genai import types

_MAGIC_AUDIO = 0x01
_DEFAULT_TTS_MODEL = "gemini-2.5-flash-preview-tts"
_DEFAULT_TARGET_SAMPLE_RATE = 16000
_DEFAULT_OUTPUT_ROOT = Path("artifacts/multiturn_tts")


@dataclass
class TurnSpec:
    """Single test turn definition."""

    id: str
    text: str
    expect_contains: list[str] = field(default_factory=list)
    pcm_path: str | None = None


@dataclass
class GeneratedTurnAudio:
    """Generated audio artifacts for one turn."""

    id: str
    source_mime_type: str
    source_sample_rate: int
    target_sample_rate: int
    source_wav_path: str
    target_wav_path: str
    target_pcm_path: str
    source_duration_sec: float
    target_duration_sec: float


@dataclass
class TurnResult:
    """Runtime results for one replayed turn."""

    id: str
    text: str
    passed: bool
    failure_reasons: list[str]
    upload_duration_sec: float
    first_user_transcript_latency_sec: float | None
    first_agent_transcript_latency_sec: float | None
    user_transcripts: list[str]
    agent_transcripts: list[str]
    interrupted_events: int
    downstream_audio_bytes: int
    message_type_counts: dict[str, int]
    tool_events: list[dict[str, Any]]
    lod_updates: list[dict[str, Any]]
    activity_debug_events: list[dict[str, Any]]


@dataclass
class QueueEvent:
    """Single websocket incoming event with timestamp."""

    ts_mono: float
    kind: str
    payload: Any


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _default_turns() -> list[TurnSpec]:
    """Fallback turns when no JSON config is provided."""
    return [
        TurnSpec(
            id="turn_1",
            text="你好，我现在在室内，请先用一句话确认你已经准备好协助我。",
        ),
        TurnSpec(
            id="turn_2",
            text="请给我两条过马路前的安全提醒，语气简洁明确。",
        ),
        TurnSpec(
            id="turn_3",
            text="如果环境噪声很大，你会如何调整提示方式？请说两点。",
        ),
        TurnSpec(
            id="turn_4",
            text="请用三句话总结刚才我们的对话重点。",
        ),
    ]


def _load_turns(turns_file: Path | None) -> list[TurnSpec]:
    if turns_file is None:
        return _default_turns()

    raw = json.loads(turns_file.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("turns file must be a non-empty JSON array")

    turns: list[TurnSpec] = []
    for idx, item in enumerate(raw, start=1):
        if isinstance(item, str):
            turns.append(TurnSpec(id=f"turn_{idx}", text=item.strip()))
            continue
        if not isinstance(item, dict):
            raise ValueError(f"turn entry #{idx} must be string or object")

        text = str(item.get("text", "")).strip()
        if not text:
            raise ValueError(f"turn entry #{idx} missing non-empty 'text'")

        item_id = str(item.get("id") or f"turn_{idx}")
        expected = item.get("expect_contains", [])
        if isinstance(expected, str):
            expected = [expected]
        if not isinstance(expected, list):
            raise ValueError(f"turn entry #{idx} field 'expect_contains' must be a list")

        turns.append(
            TurnSpec(
                id=item_id,
                text=text,
                expect_contains=[str(v) for v in expected if str(v).strip()],
                pcm_path=(str(item.get("pcm_path")).strip() if item.get("pcm_path") else None),
            )
        )
    return turns


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")
    return slug[:48] or "turn"


def _parse_sample_rate_from_mime(mime_type: str, default_rate: int = 24000) -> int:
    match = re.search(r"rate=(\d+)", mime_type or "", flags=re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return default_rate
    return default_rate


def _write_wav(path: Path, pcm_mono_int16: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(np.asarray(pcm_mono_int16, dtype=np.int16).tobytes())


def _wav_bytes_to_mono_int16(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frame_count = wf.getnframes()
        raw_frames = wf.readframes(frame_count)

    if sample_width == 2:
        samples = np.frombuffer(raw_frames, dtype="<i2").astype(np.int16)
    elif sample_width == 1:
        # 8-bit PCM is unsigned in WAV.
        u8 = np.frombuffer(raw_frames, dtype=np.uint8).astype(np.int16)
        samples = ((u8 - 128) << 8).astype(np.int16)
    elif sample_width == 4:
        i32 = np.frombuffer(raw_frames, dtype="<i4")
        samples = (i32 >> 16).astype(np.int16)
    else:
        raise ValueError(f"unsupported WAV sample width: {sample_width} bytes")

    if channels > 1:
        samples = samples.reshape(-1, channels).astype(np.int32).mean(axis=1).astype(np.int16)
    return samples, sample_rate


def _resample_linear_int16(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return np.asarray(samples, dtype=np.int16)
    if len(samples) == 0:
        return np.asarray(samples, dtype=np.int16)

    src_len = len(samples)
    dst_len = max(1, int(round(src_len * (dst_rate / float(src_rate)))))
    src_x = np.linspace(0.0, 1.0, num=src_len, endpoint=False, dtype=np.float64)
    dst_x = np.linspace(0.0, 1.0, num=dst_len, endpoint=False, dtype=np.float64)
    dst = np.interp(dst_x, src_x, samples.astype(np.float64))
    return np.clip(np.round(dst), -32768, 32767).astype(np.int16)


def _signal_smoothness_score(samples: np.ndarray) -> float:
    """Lower is typically smoother (speech-like) for adjacent-sample deltas."""
    if samples.size < 2:
        return float("inf")
    diff = np.abs(np.diff(samples.astype(np.int32)))
    return float(diff.mean())


def _extract_inline_audio(response: types.GenerateContentResponse) -> tuple[bytes, str]:
    parts = response.parts or []
    for part in parts:
        if part.inline_data and part.inline_data.data:
            data = part.inline_data.data
            if isinstance(data, str):
                # SDK usually returns bytes, but some transports may return
                # Base64-encoded strings.
                try:
                    data = base64.b64decode(data, validate=True)
                except Exception:
                    data = data.encode("utf-8")
            mime_type = part.inline_data.mime_type or "application/octet-stream"
            return data, mime_type
    raise RuntimeError("TTS response contains no inline audio data")


def _create_genai_client() -> genai.Client:
    use_vertex_raw = (os.getenv("GOOGLE_GENAI_USE_VERTEXAI") or "").strip().lower()
    vertex_explicit_true = use_vertex_raw in {"1", "true", "yes", "on"}
    project = (os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip()
    location = (
        os.getenv("GOOGLE_CLOUD_LOCATION")
        or os.getenv("GOOGLE_CLOUD_REGION")
        or "us-central1"
    ).strip()
    auto_prefer_vertex = not use_vertex_raw and bool(project)
    use_vertex = vertex_explicit_true or auto_prefer_vertex

    if use_vertex:
        if not project:
            if vertex_explicit_true:
                raise RuntimeError(
                    "GOOGLE_GENAI_USE_VERTEXAI=true but GOOGLE_CLOUD_PROJECT is empty."
                )
        else:
            try:
                return genai.Client(vertexai=True, project=project, location=location)
            except Exception as exc:
                if vertex_explicit_true:
                    print(f"[warn] Vertex client init failed (explicit), falling back to API key: {exc}")
                else:
                    print(f"[info] Vertex auto-detect failed, falling back to API key: {exc}")

    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("_GOOGLE_AI_API_KEY")
        or ""
    ).strip()
    if not api_key:
        raise RuntimeError(
            "No API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY."
        )
    return genai.Client(api_key=api_key)


def _synthesize_turn_audio(
    *,
    client: genai.Client,
    model: str,
    voice: str,
    language_code: str | None,
    turn: TurnSpec,
    target_rate: int,
    output_dir: Path,
) -> GeneratedTurnAudio:
    # Keep prompt minimal to avoid instruction bleed being spoken in output.
    prompt = turn.text.strip()
    speech_config = types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
        ),
        language_code=language_code or None,
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=speech_config,
        ),
    )
    audio_data, mime_type = _extract_inline_audio(response)
    mime_lower = mime_type.lower()

    if "audio/wav" in mime_lower or "audio/x-wav" in mime_lower:
        src_samples, src_rate = _wav_bytes_to_mono_int16(audio_data)
        source_wav_bytes = audio_data
    elif "audio/pcm" in mime_lower or "audio/l16" in mime_lower:
        src_rate = _parse_sample_rate_from_mime(mime_type, default_rate=24000)
        usable = len(audio_data) - (len(audio_data) % 2)
        if "audio/l16" in mime_lower:
            # Some providers label L16 but still emit little-endian PCM.
            # Decode both endian variants and choose the smoother waveform.
            le = np.frombuffer(audio_data[:usable], dtype="<i2").astype(np.int16)
            be = np.frombuffer(audio_data[:usable], dtype=">i2").astype(np.int16)
            src_samples = le if _signal_smoothness_score(le) <= _signal_smoothness_score(be) else be
        else:
            src_samples = np.frombuffer(audio_data[:usable], dtype="<i2").astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(src_rate)
            wf.writeframes(src_samples.tobytes())
        source_wav_bytes = buf.getvalue()
    else:
        raise RuntimeError(f"Unsupported TTS output mime type: {mime_type}")

    dst_samples = _resample_linear_int16(src_samples, src_rate, target_rate)
    target_pcm = dst_samples.tobytes()

    turn_slug = _safe_slug(turn.id)
    source_wav_path = output_dir / "audio" / f"{turn_slug}.source.wav"
    target_wav_path = output_dir / "audio" / f"{turn_slug}.pcm{target_rate}.wav"
    target_pcm_path = output_dir / "audio" / f"{turn_slug}.pcm{target_rate}.raw"
    source_wav_path.parent.mkdir(parents=True, exist_ok=True)
    source_wav_path.write_bytes(source_wav_bytes)
    _write_wav(target_wav_path, dst_samples, target_rate)
    target_pcm_path.write_bytes(target_pcm)

    return GeneratedTurnAudio(
        id=turn.id,
        source_mime_type=mime_type,
        source_sample_rate=src_rate,
        target_sample_rate=target_rate,
        source_wav_path=str(source_wav_path),
        target_wav_path=str(target_wav_path),
        target_pcm_path=str(target_pcm_path),
        source_duration_sec=(len(src_samples) / float(src_rate)) if src_rate > 0 else 0.0,
        target_duration_sec=(len(dst_samples) / float(target_rate)),
    )


def _resolve_pcm_path(output_dir: Path, turn: TurnSpec, target_rate: int) -> Path:
    if turn.pcm_path:
        return Path(turn.pcm_path).expanduser().resolve()
    turn_slug = _safe_slug(turn.id)
    return (output_dir / "audio" / f"{turn_slug}.pcm{target_rate}.raw").resolve()


def _build_chunk_size_bytes(chunk_ms: int, sample_rate: int) -> int:
    samples = int(sample_rate * (chunk_ms / 1000.0))
    samples = max(samples, 1)
    return samples * 2  # int16 mono


async def _receiver(ws: websockets.ClientConnection, queue: asyncio.Queue[QueueEvent]) -> None:
    try:
        async for message in ws:
            now = time.monotonic()
            if isinstance(message, bytes):
                await queue.put(QueueEvent(ts_mono=now, kind="audio_bytes", payload=len(message)))
                continue
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                payload = {"type": "raw_text", "text": message}
            await queue.put(QueueEvent(ts_mono=now, kind="json", payload=payload))
    except websockets.ConnectionClosed:
        return


async def _wait_for_type(
    queue: asyncio.Queue[QueueEvent],
    message_type: str,
    timeout_sec: float,
) -> QueueEvent:
    deadline = time.monotonic() + timeout_sec
    while True:
        remain = deadline - time.monotonic()
        if remain <= 0:
            raise TimeoutError(f"timed out waiting for message type '{message_type}'")
        event = await asyncio.wait_for(queue.get(), timeout=remain)
        if event.kind != "json":
            continue
        payload = event.payload or {}
        if str(payload.get("type")) == message_type:
            return event


async def _await_initial_model_idle(
    queue: asyncio.Queue[QueueEvent],
    *,
    timeout_sec: float,
    quiet_sec: float,
) -> bool:
    """Wait for initial agent greeting/output to finish before replay turns."""
    deadline = time.monotonic() + timeout_sec
    saw_agent_output = False
    last_agent_event_ts: float | None = None

    while True:
        now = time.monotonic()
        if saw_agent_output and last_agent_event_ts is not None and (now - last_agent_event_ts) >= quiet_sec:
            return True
        if now >= deadline:
            return saw_agent_output

        wait_for = min(0.25, max(0.01, deadline - now))
        try:
            event = await asyncio.wait_for(queue.get(), timeout=wait_for)
        except asyncio.TimeoutError:
            continue

        if event.kind == "audio_bytes":
            saw_agent_output = True
            last_agent_event_ts = event.ts_mono
            continue

        if event.kind != "json":
            continue
        payload = event.payload if isinstance(event.payload, dict) else {}
        if str(payload.get("type")) == "transcript" and str(payload.get("role")) == "agent":
            saw_agent_output = True
            last_agent_event_ts = event.ts_mono


def _drain_queue(queue: asyncio.Queue[QueueEvent]) -> int:
    drained = 0
    while True:
        try:
            queue.get_nowait()
            drained += 1
        except asyncio.QueueEmpty:
            return drained


async def _stream_turn_audio(
    *,
    ws: websockets.ClientConnection,
    pcm_bytes: bytes,
    chunk_size_bytes: int,
    sample_rate: int,
    realtime_factor: float,
    post_turn_silence_ms: int,
) -> float:
    await ws.send(json.dumps({"type": "activity_start"}))
    started = time.monotonic()

    for offset in range(0, len(pcm_bytes), chunk_size_bytes):
        chunk = pcm_bytes[offset : offset + chunk_size_bytes]
        if not chunk:
            continue
        await ws.send(bytes([_MAGIC_AUDIO]) + chunk)

        if realtime_factor > 0:
            chunk_sec = (len(chunk) / 2.0) / float(sample_rate)
            await asyncio.sleep(chunk_sec / realtime_factor)

    await ws.send(json.dumps({"type": "activity_end"}))
    if post_turn_silence_ms > 0:
        await asyncio.sleep(post_turn_silence_ms / 1000.0)
    return time.monotonic() - started


async def _collect_turn_result(
    *,
    turn: TurnSpec,
    queue: asyncio.Queue[QueueEvent],
    turn_start_mono: float,
    timeout_sec: float,
    idle_after_agent_sec: float,
    upload_duration_sec: float,
) -> TurnResult:
    user_transcripts: list[str] = []
    agent_transcripts: list[str] = []
    failures: list[str] = []
    interrupted_events = 0
    downstream_audio_bytes = 0
    message_type_counts: dict[str, int] = {}
    tool_events: list[dict[str, Any]] = []
    lod_updates: list[dict[str, Any]] = []
    activity_debug_events: list[dict[str, Any]] = []

    first_user_latency: float | None = None
    first_agent_latency: float | None = None
    last_agent_ts: float | None = None

    deadline = time.monotonic() + timeout_sec
    while True:
        now = time.monotonic()
        if now >= deadline:
            break

        if last_agent_ts is not None and (now - last_agent_ts) >= idle_after_agent_sec:
            break

        wait_for = min(0.25, max(0.01, deadline - now))
        try:
            event = await asyncio.wait_for(queue.get(), timeout=wait_for)
        except asyncio.TimeoutError:
            continue

        if event.ts_mono < turn_start_mono:
            continue

        if event.kind == "audio_bytes":
            downstream_audio_bytes += int(event.payload)
            continue

        payload = event.payload if isinstance(event.payload, dict) else {}
        msg_type = str(payload.get("type"))
        message_type_counts[msg_type] = message_type_counts.get(msg_type, 0) + 1

        if msg_type == "transcript":
            role = str(payload.get("role", ""))
            text = str(payload.get("text", "")).strip()
            if not text:
                continue

            if role == "user":
                if first_user_latency is None:
                    first_user_latency = event.ts_mono - turn_start_mono
                user_transcripts.append(text)
            elif role == "agent":
                if first_agent_latency is None:
                    first_agent_latency = event.ts_mono - turn_start_mono
                agent_transcripts.append(text)
                last_agent_ts = event.ts_mono

        elif msg_type == "interrupted":
            interrupted_events += 1

        elif msg_type == "tool_event":
            tool_events.append({
                "tool": payload.get("tool"),
                "status": payload.get("status"),
                "behavior": payload.get("behavior"),
            })

        elif msg_type == "lod_update":
            lod_updates.append({
                "lod": payload.get("lod"),
                "reason": payload.get("reason"),
            })

        elif msg_type == "debug_activity":
            data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
            activity_debug_events.append({
                "event": data.get("event"),
                "state": data.get("state"),
                "queue_status": data.get("queue_status"),
            })

        elif msg_type == "error":
            err = payload.get("message") or payload.get("error") or "unknown error"
            failures.append(f"server_error:{err}")

    if not user_transcripts:
        failures.append("no_user_transcript")
    if not agent_transcripts:
        failures.append("no_agent_transcript")

    merged_agent = " ".join(agent_transcripts).lower()
    for expected in turn.expect_contains:
        needle = expected.strip().lower()
        if needle and needle not in merged_agent:
            failures.append(f"missing_expected_text:{expected}")

    return TurnResult(
        id=turn.id,
        text=turn.text,
        passed=not failures,
        failure_reasons=failures,
        upload_duration_sec=upload_duration_sec,
        first_user_transcript_latency_sec=first_user_latency,
        first_agent_transcript_latency_sec=first_agent_latency,
        user_transcripts=user_transcripts,
        agent_transcripts=agent_transcripts,
        interrupted_events=interrupted_events,
        downstream_audio_bytes=downstream_audio_bytes,
        message_type_counts=message_type_counts,
        tool_events=tool_events,
        lod_updates=lod_updates,
        activity_debug_events=activity_debug_events,
    )


async def _run_multiturn_replay(
    *,
    ws_url: str,
    turns: list[TurnSpec],
    output_dir: Path,
    sample_rate: int,
    chunk_ms: int,
    realtime_factor: float,
    warmup_sec: float,
    per_turn_timeout_sec: float,
    idle_after_agent_sec: float,
    post_turn_silence_ms: int,
    initial_idle_timeout_sec: float,
    initial_idle_quiet_sec: float,
) -> list[TurnResult]:
    event_queue: asyncio.Queue[QueueEvent] = asyncio.Queue()
    results: list[TurnResult] = []

    async with websockets.connect(ws_url, max_size=None) as ws:
        recv_task = asyncio.create_task(_receiver(ws, event_queue))
        try:
            await _wait_for_type(event_queue, "session_ready", timeout_sec=15.0)
            if warmup_sec > 0:
                await asyncio.sleep(warmup_sec)
            await _await_initial_model_idle(
                event_queue,
                timeout_sec=initial_idle_timeout_sec,
                quiet_sec=initial_idle_quiet_sec,
            )
            _ = _drain_queue(event_queue)

            chunk_size_bytes = _build_chunk_size_bytes(chunk_ms, sample_rate)

            for turn in turns:
                pcm_path = _resolve_pcm_path(output_dir, turn, sample_rate)
                if not pcm_path.exists():
                    raise FileNotFoundError(f"PCM file not found for turn '{turn.id}': {pcm_path}")

                pcm_bytes = pcm_path.read_bytes()
                if not pcm_bytes:
                    raise RuntimeError(f"PCM file is empty for turn '{turn.id}': {pcm_path}")

                _ = _drain_queue(event_queue)
                turn_start = time.monotonic()
                upload_sec = await _stream_turn_audio(
                    ws=ws,
                    pcm_bytes=pcm_bytes,
                    chunk_size_bytes=chunk_size_bytes,
                    sample_rate=sample_rate,
                    realtime_factor=realtime_factor,
                    post_turn_silence_ms=post_turn_silence_ms,
                )
                result = await _collect_turn_result(
                    turn=turn,
                    queue=event_queue,
                    turn_start_mono=turn_start,
                    timeout_sec=per_turn_timeout_sec,
                    idle_after_agent_sec=idle_after_agent_sec,
                    upload_duration_sec=upload_sec,
                )
                results.append(result)
        finally:
            recv_task.cancel()
            try:
                await recv_task
            except (asyncio.CancelledError, websockets.ConnectionClosed):
                pass

    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gemini TTS fixture generator + continuous multi-turn websocket tester."
    )
    parser.add_argument("--turns-file", type=Path, help="JSON array of turns. If omitted, built-in turns are used.")
    parser.add_argument("--output-dir", type=Path, help="Output directory. Default: artifacts/multiturn_tts/<timestamp>")
    parser.add_argument("--ws-url", type=str, help="Target websocket URL, e.g. ws://127.0.0.1:8000/ws/u1/s1")
    parser.add_argument("--skip-generate", action="store_true", help="Skip TTS generation and use existing pcm_path entries/files.")
    parser.add_argument("--skip-run", action="store_true", help="Generate fixtures only, do not run websocket replay.")
    parser.add_argument("--voice", type=str, default=os.getenv("GEMINI_TTS_VOICE", "Aoede"))
    parser.add_argument("--language-code", type=str, default=os.getenv("GEMINI_TTS_LANGUAGE", "zh-CN"))
    parser.add_argument("--tts-model", type=str, default=os.getenv("GEMINI_TTS_MODEL", _DEFAULT_TTS_MODEL))
    parser.add_argument("--target-sample-rate", type=int, default=_DEFAULT_TARGET_SAMPLE_RATE)
    parser.add_argument("--chunk-ms", type=int, default=40, help="Upstream audio chunk size in ms.")
    parser.add_argument("--realtime-factor", type=float, default=1.0, help="1.0=real-time, 2.0=2x faster upload.")
    parser.add_argument("--warmup-sec", type=float, default=2.5, help="Wait after session_ready before first turn.")
    parser.add_argument("--per-turn-timeout-sec", type=float, default=25.0, help="Max wait per turn for transcripts.")
    parser.add_argument("--idle-after-agent-sec", type=float, default=1.2, help="Turn closes after this long agent transcript silence.")
    parser.add_argument("--post-turn-silence-ms", type=int, default=250, help="Silence gap after each uploaded turn.")
    parser.add_argument("--initial-idle-timeout-sec", type=float, default=45.0, help="Max wait for initial greeting/output to settle.")
    parser.add_argument("--initial-idle-quiet-sec", type=float, default=1.5, help="Required quiet window after initial model output.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve config and print plan only.")
    return parser


def _print_config_summary(
    *,
    turns: list[TurnSpec],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    print("=== Gemini TTS Multi-Turn Test Config ===")
    print(f"turn_count: {len(turns)}")
    print(f"output_dir: {output_dir}")
    print(f"skip_generate: {args.skip_generate}")
    print(f"skip_run: {args.skip_run}")
    print(f"tts_model: {args.tts_model}")
    print(f"voice: {args.voice}")
    print(f"language_code: {args.language_code}")
    print(f"target_sample_rate: {args.target_sample_rate}")
    print(f"ws_url: {args.ws_url or '(none)'}")


async def _async_main(args: argparse.Namespace) -> int:
    turns = _load_turns(args.turns_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir or (_DEFAULT_OUTPUT_ROOT / timestamp)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _print_config_summary(turns=turns, output_dir=output_dir, args=args)
    if args.dry_run:
        return 0

    generation_records: list[GeneratedTurnAudio] = []
    if not args.skip_generate:
        client = _create_genai_client()
        for turn in turns:
            print(f"[generate] turn={turn.id}")
            generated = _synthesize_turn_audio(
                client=client,
                model=args.tts_model,
                voice=args.voice,
                language_code=args.language_code,
                turn=turn,
                target_rate=args.target_sample_rate,
                output_dir=output_dir,
            )
            generation_records.append(generated)
            turn.pcm_path = generated.target_pcm_path
            print(
                "[generate] saved "
                f"pcm={generated.target_pcm_path} "
                f"source_mime={generated.source_mime_type} "
                f"dur={generated.target_duration_sec:.2f}s"
            )
    else:
        for turn in turns:
            pcm_path = _resolve_pcm_path(output_dir, turn, args.target_sample_rate)
            if not pcm_path.exists():
                raise FileNotFoundError(
                    f"missing PCM file for turn '{turn.id}': {pcm_path} "
                    "(set pcm_path in turns file or disable --skip-generate)"
                )
            turn.pcm_path = str(pcm_path)

    run_results: list[TurnResult] = []
    if not args.skip_run:
        if not args.ws_url:
            raise ValueError("--ws-url is required unless --skip-run is set")
        run_results = await _run_multiturn_replay(
            ws_url=args.ws_url,
            turns=turns,
            output_dir=output_dir,
            sample_rate=args.target_sample_rate,
            chunk_ms=args.chunk_ms,
            realtime_factor=args.realtime_factor,
            warmup_sec=args.warmup_sec,
            per_turn_timeout_sec=args.per_turn_timeout_sec,
            idle_after_agent_sec=args.idle_after_agent_sec,
            post_turn_silence_ms=args.post_turn_silence_ms,
            initial_idle_timeout_sec=args.initial_idle_timeout_sec,
            initial_idle_quiet_sec=args.initial_idle_quiet_sec,
        )

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "ws_url": args.ws_url,
        "tts_model": args.tts_model,
        "voice": args.voice,
        "language_code": args.language_code,
        "target_sample_rate": args.target_sample_rate,
        "turns": [asdict(t) for t in turns],
        "generated_audio": [asdict(r) for r in generation_records],
        "results": [asdict(r) for r in run_results],
        "all_passed": all(r.passed for r in run_results) if run_results else None,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[report] {report_path}")
    for r in run_results:
        status = "PASS" if r.passed else "FAIL"
        print(
            f"[{status}] {r.id} "
            f"user_lat={r.first_user_transcript_latency_sec} "
            f"agent_lat={r.first_agent_transcript_latency_sec} "
            f"audio_bytes={r.downstream_audio_bytes} "
            f"reasons={','.join(r.failure_reasons) if r.failure_reasons else '-'}"
        )

    if not run_results:
        return 0
    return 0 if all(r.passed for r in run_results) else 2


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    parser = _build_parser()
    args = parser.parse_args()
    try:
        return asyncio.run(_async_main(args))
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
