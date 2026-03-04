#!/usr/bin/env python3
"""Multi-turn E2E test for SightLine.

Generates TTS audio + synthetic test images, then runs 12+ turn conversations
against the real server to validate full pipeline behavior.

Usage:
    python scripts/run_multiturn_e2e.py
    python scripts/run_multiturn_e2e.py --server ws://127.0.0.1:8100
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import re
import subprocess
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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("multiturn_e2e")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAGIC_AUDIO = 0x01
MAGIC_IMAGE = 0x02
TTS_MODEL = "gemini-2.5-flash-preview-tts"
TARGET_SAMPLE_RATE = 16000
IMAGEN_MODEL = "imagen-4.0-fast-generate-001"

# Scene prompts for Imagen image generation
SCENE_PROMPTS = {
    "street_scene": (
        "A busy city sidewalk during daytime with pedestrians walking, storefronts visible, "
        "crosswalk ahead, and a street sign. Realistic first-person pedestrian perspective photo."
    ),
    "text_sign": (
        "A close-up photo of a restaurant menu board on a wall with clearly readable text: "
        "'Today Special: Grilled Salmon $15, Caesar Salad $9, Coffee $4'. Realistic photo."
    ),
    "hazard_scene": (
        "A wet sidewalk with a large puddle, construction barrier with orange cones, "
        "and a yellow CAUTION WET FLOOR sign. First-person perspective photo."
    ),
    "park_scene": (
        "A sunny park with a paved walking path, green grass, two benches with people sitting, "
        "a small fountain, and trees. First-person perspective daytime photo."
    ),
    "building_scene": (
        "A modern building entrance with glass doors, a wheelchair accessible ramp, "
        "and a directory sign showing floor numbers. First-person perspective photo."
    ),
}

# Markers that should never appear in agent speech (context tag leak detection)
_LEAKED_MARKERS = [
    "[CONTEXT UPDATE", "[SILENT", "[DO NOT SPEAK", "[TELEMETRY",
    "[LOD UPDATE", "[VISION ANALYSIS", "<<<INTERNAL_CONTEXT>>>",
    "<<<SILENT_SENSOR_DATA>>>", "<<<END_",
]

# Mutex tool groups — at most one tool from each group per turn
_MUTEX_TOOL_GROUPS = [
    {"nearby_search", "maps_query"},
    {"navigate_to", "get_walking_directions"},
]

# ---------------------------------------------------------------------------
# Multi-turn conversation definitions
# ---------------------------------------------------------------------------

# Conversation A: Mixed scenario — greeting, vision, OCR, navigation, memory
CONVERSATION_A = {
    "id": "conv_mixed_full",
    "description": "Full mixed conversation: greeting → vision → OCR → nav → search → memory → follow-up",
    "turns": [
        {
            "id": "A01_greeting",
            "text": "Hello, I just stepped outside. Can you help me today?",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 20.0,
            "notes": "Initial greeting, expect warm response",
        },
        {
            "id": "A02_whats_around",
            "text": "What's around me right now?",
            "send_image": "street_scene",
            "send_telemetry": {
                "motion_state": "stationary",
                "step_cadence": 0,
                "ambient_noise_db": 55,
                "heading": 180,
                "gps": {"latitude": 40.7580, "longitude": -73.9855, "accuracy": 5.0},
            },
            "send_gesture": None,
            "send_video_frames": 3,
            "expect_agent_response": True,
            "expect_not_blocked": ["get_location_info", "nearby_search"],
            "collect_sec": 30.0,
            "notes": "Vision query with image + telemetry context",
        },
        {
            "id": "A03_read_sign",
            "text": "Can you read that sign for me?",
            "send_image": "text_sign",
            "send_telemetry": None,
            "send_gesture": None,
            "send_video_frames": 2,
            "expect_agent_response": True,
            "expect_tool": "extract_text_from_camera",
            "collect_sec": 30.0,
            "notes": "Explicit OCR request — should trigger extract_text_from_camera",
        },
        {
            "id": "A04_navigate",
            "text": "Navigate me to the nearest pharmacy please.",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_tool": "navigate_to",
            "collect_sec": 55.0,
            "notes": "Navigation request — should trigger navigate_to tool",
        },
        {
            "id": "A05_search",
            "text": "What's the weather like in New York today?",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_tool": "google_search",
            "collect_sec": 45.0,
            "notes": "Search request — should trigger google_search tool",
        },
        {
            "id": "A06_remember",
            "text": "Please remember that the pharmacy is called Walgreens on 5th Avenue.",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_tool": "remember_entity",
            "collect_sec": 25.0,
            "notes": "Memory store request",
        },
        {
            "id": "A07_whats_ahead",
            "text": "What's ahead of me now?",
            "send_image": "street_scene",
            "send_telemetry": {
                "motion_state": "walking",
                "step_cadence": 100,
                "ambient_noise_db": 65,
                "heading": 90,
                "gps": {"latitude": 40.7582, "longitude": -73.9850, "accuracy": 5.0},
            },
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 30.0,
            "notes": "Second vision query — different telemetry context, check context awareness",
        },
        {
            "id": "A08_danger_check",
            "text": "Is there any danger ahead?",
            "send_image": "hazard_scene",
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 30.0,
            "notes": "Safety-focused query with hazard image",
        },
        {
            "id": "A09_recall_memory",
            "text": "What pharmacy did I mention earlier?",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 25.0,
            "notes": "Memory recall — should remember Walgreens on 5th Avenue",
        },
        {
            "id": "A10_describe_detail",
            "text": "Describe everything you see in detail.",
            "send_image": "park_scene",
            "send_telemetry": None,
            "send_gesture": "force_lod_3",
            "expect_agent_response": True,
            "collect_sec": 35.0,
            "notes": "LOD3 detailed description request with gesture",
        },
        {
            "id": "A11_how_many_people",
            "text": "How many people can you see?",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 20.0,
            "notes": "Follow-up question about previous image — tests context retention",
        },
        {
            "id": "A12_goodbye",
            "text": "Thank you, that's all for now. Goodbye.",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 15.0,
            "notes": "Session closure — polite ending",
        },
    ],
}

# Conversation B: Rapid context switching stress test
CONVERSATION_B = {
    "id": "conv_rapid_context",
    "description": "Rapid context switching: alternating vision/OCR/nav/search to test model coherence",
    "turns": [
        {
            "id": "B01_greeting",
            "text": "Hi, I need your help navigating the city.",
            "send_image": None,
            "send_telemetry": {
                "motion_state": "walking",
                "step_cadence": 110,
                "ambient_noise_db": 70,
                "heading": 45,
                "gps": {"latitude": 37.7749, "longitude": -122.4194, "accuracy": 8.0},
            },
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 20.0,
            "notes": "Greeting with urban walking context",
        },
        {
            "id": "B02_read_menu",
            "text": "Read the menu for me please.",
            "send_image": "text_sign",
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_tool": "extract_text_from_camera",
            "collect_sec": 30.0,
            "notes": "OCR request",
        },
        {
            "id": "B03_search_restaurant",
            "text": "Search for the best Italian restaurant nearby.",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_tool": "google_search",
            "collect_sec": 45.0,
            "notes": "Immediate switch to search",
        },
        {
            "id": "B04_whats_this",
            "text": "What is this building?",
            "send_image": "building_scene",
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 30.0,
            "notes": "Vision query — context switch from search to vision",
        },
        {
            "id": "B05_navigate",
            "text": "Take me to Union Square.",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_tool": "navigate_to",
            "collect_sec": 55.0,
            "notes": "Navigation — context switch from vision to nav",
        },
        {
            "id": "B06_danger",
            "text": "Wait, is that area safe?",
            "send_image": "hazard_scene",
            "send_telemetry": {
                "motion_state": "running",
                "step_cadence": 150,
                "ambient_noise_db": 80,
                "heading": 270,
            },
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 25.0,
            "notes": "Safety concern mid-navigation with high-motion telemetry",
        },
        {
            "id": "B07_read_that",
            "text": "What does that say?",
            "send_image": "text_sign",
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_tool": "extract_text_from_camera",
            "collect_sec": 30.0,
            "notes": "Another OCR request — switching back to text reading",
        },
        {
            "id": "B08_where_am_i",
            "text": "Where am I right now?",
            "send_image": None,
            "send_telemetry": {
                "motion_state": "stationary",
                "step_cadence": 0,
                "ambient_noise_db": 50,
                "heading": 0,
                "gps": {"latitude": 37.7880, "longitude": -122.4075, "accuracy": 3.0},
            },
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 30.0,
            "notes": "Location query — tests GPS context integration",
        },
        {
            "id": "B09_remember",
            "text": "Remember that I like this intersection.",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_tool": "remember_entity",
            "collect_sec": 20.0,
            "notes": "Memory store",
        },
        {
            "id": "B10_lod_change",
            "text": "Give me more detail about my surroundings.",
            "send_image": "park_scene",
            "send_telemetry": None,
            "send_gesture": "force_lod_3",
            "expect_agent_response": True,
            "collect_sec": 35.0,
            "notes": "LOD3 detailed view with gesture",
        },
        {
            "id": "B11_followup",
            "text": "What else can you tell me about this place?",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 25.0,
            "notes": "Follow-up without new image — tests context retention",
        },
        {
            "id": "B12_recall",
            "text": "What intersection did I like?",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 25.0,
            "notes": "Memory recall test",
        },
    ],
}

# Conversation C: Defect regression suite (E2E-001 through E2E-007)
CONVERSATION_C = {
    "id": "conv_defect_regression",
    "description": "Defect regression: dedup, tag leak, nav blocking, token stability",
    "turns": [
        {
            "id": "C01_location_query",
            "text": "Where am I right now?",
            "send_image": "street_scene",
            "send_telemetry": {
                "motion_state": "stationary",
                "step_cadence": 0,
                "ambient_noise_db": 55,
                "heading": 180,
                "gps": {"latitude": 40.7580, "longitude": -73.9855, "accuracy": 5.0},
            },
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_not_blocked": ["get_location_info", "nearby_search"],
            "collect_sec": 25.0,
            "notes": "E2E-007: location query should NOT be blocked",
        },
        {
            "id": "C02_whats_nearby",
            "text": "What's nearby? Any restaurants?",
            "send_image": "street_scene",
            "send_telemetry": {
                "motion_state": "stationary",
                "step_cadence": 0,
                "ambient_noise_db": 55,
                "heading": 180,
                "gps": {"latitude": 40.7580, "longitude": -73.9855, "accuracy": 5.0},
            },
            "send_gesture": None,
            "send_video_frames": 3,
            "expect_agent_response": True,
            "expect_not_blocked": ["nearby_search", "maps_query"],
            "collect_sec": 30.0,
            "notes": "E2E-002/003: should call ONE of nearby_search/maps_query, not both",
        },
        {
            "id": "C03_read_menu",
            "text": "Read that menu for me.",
            "send_image": "text_sign",
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_tool": "extract_text_from_camera",
            "collect_sec": 30.0,
            "notes": "E2E-005: tool result should be spoken in this turn",
        },
        {
            "id": "C04_navigate_cafe",
            "text": "Take me to the closest cafe.",
            "send_image": None,
            "send_telemetry": {
                "motion_state": "stationary",
                "step_cadence": 0,
                "ambient_noise_db": 55,
                "heading": 180,
                "gps": {"latitude": 40.7580, "longitude": -73.9855, "accuracy": 5.0},
            },
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_tool": "navigate_to",
            "collect_sec": 35.0,
            "notes": "E2E-002: navigate_to called at most once",
        },
        {
            "id": "C05_whats_ahead_video",
            "text": "What's ahead of me?",
            "send_image": "hazard_scene",
            "send_telemetry": None,
            "send_gesture": None,
            "send_video_frames": 4,
            "expect_agent_response": True,
            "collect_sec": 30.0,
            "notes": "E2E-001/004: context tags must not leak in agent speech",
        },
        {
            "id": "C06_context_silence",
            "text": "Tell me more about this area.",
            "send_image": "park_scene",
            "send_telemetry": {
                "motion_state": "walking",
                "step_cadence": 90,
                "ambient_noise_db": 45,
                "heading": 90,
                "gps": {"latitude": 40.7585, "longitude": -73.9850, "accuracy": 5.0},
            },
            "send_gesture": None,
            "send_video_frames": 3,
            "expect_agent_response": True,
            "collect_sec": 30.0,
            "notes": "E2E-004: context injection should not be echoed",
        },
        {
            "id": "C07_search_then_nav",
            "text": "Find me a pharmacy nearby and then navigate there.",
            "send_image": None,
            "send_telemetry": {
                "motion_state": "stationary",
                "step_cadence": 0,
                "ambient_noise_db": 55,
                "heading": 180,
                "gps": {"latitude": 40.7580, "longitude": -73.9855, "accuracy": 5.0},
            },
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 40.0,
            "notes": "E2E-002/003: mutex should prevent nearby_search + maps_query both firing",
        },
        {
            "id": "C08_final_stability",
            "text": "Thanks, that's all for now. Goodbye.",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 15.0,
            "notes": "E2E-006: session should remain stable through turn 8",
        },
    ],
}

ALL_CONVERSATIONS = [CONVERSATION_A, CONVERSATION_B, CONVERSATION_C]


# ---------------------------------------------------------------------------
# Turn result
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    turn_id: str
    text: str
    passed: bool = False
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    user_transcripts: list[str] = field(default_factory=list)
    agent_transcripts: list[str] = field(default_factory=list)
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    lod_updates: list[dict[str, Any]] = field(default_factory=list)
    frame_acks: list[dict[str, Any]] = field(default_factory=list)
    message_counts: dict[str, int] = field(default_factory=dict)
    audio_bytes_received: int = 0
    first_user_transcript_latency: float | None = None
    first_agent_transcript_latency: float | None = None
    collect_duration_sec: float = 0.0
    notes: str = ""


@dataclass
class ConversationResult:
    conversation_id: str
    description: str
    total_turns: int = 0
    passed_turns: int = 0
    failed_turns: int = 0
    turns: list[TurnResult] = field(default_factory=list)
    issues: list[dict[str, Any]] = field(default_factory=list)
    total_duration_sec: float = 0.0


# ---------------------------------------------------------------------------
# Audio helpers (from gemini_tts_multiturn_test.py patterns)
# ---------------------------------------------------------------------------

def _create_client() -> genai.Client:
    """Create a Google AI API client (NOT Vertex AI) for TTS."""
    api_key = (
        os.getenv("_GOOGLE_AI_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or ""
    ).strip()
    if not api_key:
        raise RuntimeError("No API key. Set GEMINI_API_KEY or GOOGLE_API_KEY.")
    # Explicitly disable Vertex AI — TTS uses Google AI API with API key auth.
    # The env may have GOOGLE_GENAI_USE_VERTEXAI=TRUE for the Live API (orchestrator),
    # but TTS/Vision sub-agents go through Google AI API.
    return genai.Client(api_key=api_key, vertexai=False)


def _signal_smoothness(samples: np.ndarray) -> float:
    if samples.size < 2:
        return float("inf")
    return float(np.abs(np.diff(samples.astype(np.int32))).mean())


def _wav_to_mono_int16(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if sw == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.int16)
    elif sw == 4:
        samples = (np.frombuffer(raw, dtype="<i4") >> 16).astype(np.int16)
    else:
        raise ValueError(f"unsupported sample width {sw}")
    if ch > 1:
        samples = samples.reshape(-1, ch).astype(np.int32).mean(axis=1).astype(np.int16)
    return samples, sr


def _resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return samples.astype(np.int16)
    src_len = len(samples)
    dst_len = max(1, int(round(src_len * (dst_rate / float(src_rate)))))
    src_x = np.linspace(0, 1, src_len, endpoint=False)
    dst_x = np.linspace(0, 1, dst_len, endpoint=False)
    dst = np.interp(dst_x, src_x, samples.astype(np.float64))
    return np.clip(np.round(dst), -32768, 32767).astype(np.int16)


def _extract_audio(response: types.GenerateContentResponse) -> tuple[bytes, str]:
    for part in (response.parts or []):
        if part.inline_data and part.inline_data.data:
            data = part.inline_data.data
            if isinstance(data, str):
                data = base64.b64decode(data, validate=True)
            return data, part.inline_data.mime_type or "application/octet-stream"
    raise RuntimeError("No audio in TTS response")


def _parse_rate_from_mime(mime: str) -> int:
    m = re.search(r"rate=(\d+)", mime, re.IGNORECASE)
    return int(m.group(1)) if m else 24000


def synthesize_pcm(client: genai.Client, text: str, voice: str = "Aoede") -> bytes:
    """Synthesize text to 16kHz mono int16 PCM bytes."""
    response = client.models.generate_content(
        model=TTS_MODEL,
        contents=(
            "Read the following text naturally. Do not add extra words.\n\n"
            + text
        ),
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                ),
                language_code="en-US",
            ),
        ),
    )
    audio_data, mime = _extract_audio(response)
    mime_lower = mime.lower()

    if "audio/wav" in mime_lower or "audio/x-wav" in mime_lower:
        samples, src_rate = _wav_to_mono_int16(audio_data)
    elif "audio/pcm" in mime_lower or "audio/l16" in mime_lower:
        src_rate = _parse_rate_from_mime(mime)
        usable = len(audio_data) - (len(audio_data) % 2)
        if "audio/l16" in mime_lower:
            le = np.frombuffer(audio_data[:usable], dtype="<i2").astype(np.int16)
            be = np.frombuffer(audio_data[:usable], dtype=">i2").astype(np.int16)
            samples = le if _signal_smoothness(le) <= _signal_smoothness(be) else be
        else:
            samples = np.frombuffer(audio_data[:usable], dtype="<i2").astype(np.int16)
    else:
        raise RuntimeError(f"Unsupported mime: {mime}")

    resampled = _resample(samples, src_rate, TARGET_SAMPLE_RATE)
    return resampled.tobytes()


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

_IMG_MAX_DIM = 640
_IMG_JPEG_QUALITY = 80


def _compress_image(path: Path) -> None:
    """Resize + recompress an image to ≤640px and JPEG quality 80.

    Keeps images small enough for WebSocket streaming without overwhelming
    the server or Live API.  Uses opencv-python-headless (already a dep).
    """
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        return
    h, w = img.shape[:2]
    if max(h, w) > _IMG_MAX_DIM:
        scale = _IMG_MAX_DIM / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, _IMG_JPEG_QUALITY])


def _generate_synthetic_image(
    scene_id: str,
    output_dir: Path,
    *,
    client: genai.Client | None = None,
) -> Path:
    """Generate a test image: try Imagen first, fall back to ffmpeg.

    Images are cached — if the file already exists it is returned immediately.
    """
    out_path = output_dir / f"{scene_id}.jpg"

    if out_path.exists():
        return out_path

    # --- Try Imagen ---
    prompt = SCENE_PROMPTS.get(scene_id)
    if client and prompt:
        try:
            log.info("[imagen] generating %s ...", scene_id)
            response = client.models.generate_images(
                model=IMAGEN_MODEL,
                prompt=prompt,
                config={"number_of_images": 1},
            )
            if response.generated_images:
                image_bytes = response.generated_images[0].image.image_bytes
                if image_bytes:
                    out_path.write_bytes(image_bytes)
                    _compress_image(out_path)
                    log.info("[imagen] saved %s (%d bytes)", out_path.name, out_path.stat().st_size)
                    return out_path
            log.warning("[imagen] empty response for %s, falling back to ffmpeg", scene_id)
        except Exception as exc:
            log.warning("[imagen] failed for %s (%s), falling back to ffmpeg", scene_id, exc)

    # --- Fallback: ffmpeg synthetic image ---
    colors = {
        "street_scene": "0x4488BB",
        "text_sign": "0x228844",
        "hazard_scene": "0xCC4422",
        "park_scene": "0x44AA44",
        "building_scene": "0x666699",
    }
    color = colors.get(scene_id, "0x888888")

    cmd = [
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"color=c={color}:s=640x480:d=1",
        "-vf", f"drawtext=text='{scene_id}':fontsize=36:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2",
        "-vframes", "1", "-q:v", "3",
        str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        return out_path

    # Fallback: plain color (no text overlay)
    cmd2 = [
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"color=c={color}:s=640x480:d=1",
        "-vframes", "1", "-q:v", "3",
        str(out_path),
    ]
    r2 = subprocess.run(cmd2, capture_output=True, text=True)
    if r2.returncode == 0:
        return out_path

    raise RuntimeError(f"Failed to generate image for {scene_id}")


# ---------------------------------------------------------------------------
# WebSocket multi-turn runner
# ---------------------------------------------------------------------------

async def run_conversation(
    *,
    ws_base_url: str,
    conversation: dict[str, Any],
    pcm_cache: dict[str, bytes],
    image_dir: Path,
    timeout_mult: float = 1.0,
) -> ConversationResult:
    """Run a full multi-turn conversation against the server."""
    conv_id = conversation["id"]
    turns_def = conversation["turns"]
    log.info("=== Starting conversation: %s (%d turns) ===", conv_id, len(turns_def))

    # Vertex AI session service requires alphanumeric + hyphen IDs (no underscores)
    safe_session_id = conv_id.replace("_", "-")
    ws_url = f"{ws_base_url.rstrip('/')}/ws/e2e-user/{safe_session_id}"
    turn_results: list[TurnResult] = []
    t0_total = time.monotonic()

    try:
        async with websockets.connect(ws_url, max_size=None) as ws:
            # Wait for session_ready
            deadline = time.monotonic() + 20.0
            while True:
                remain = deadline - time.monotonic()
                if remain <= 0:
                    raise TimeoutError("session_ready timeout")
                raw = await asyncio.wait_for(ws.recv(), timeout=remain)
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                if msg.get("type") == "session_ready":
                    log.info("[%s] session_ready received", conv_id)
                    break

            # Wait for initial model greeting to settle
            await _drain_initial_greeting(ws, timeout_sec=15.0)

            # Run each turn
            for i, turn_def in enumerate(turns_def):
                turn_id = turn_def["id"]
                text = turn_def["text"]
                collect_sec = turn_def.get("collect_sec", 25.0) * timeout_mult
                log.info("[Turn %d/%d] %s: \"%s\"", i + 1, len(turns_def), turn_id, text)

                tr = await _run_single_turn(
                    ws=ws,
                    turn_def=turn_def,
                    pcm_cache=pcm_cache,
                    image_dir=image_dir,
                    collect_sec=collect_sec,
                )
                turn_results.append(tr)

                status = "PASS" if tr.passed else "FAIL"
                agent_text = " ".join(tr.agent_transcripts)[:120]
                log.info(
                    "[%s] %s | user_lat=%.2fs agent_lat=%.2fs audio=%d | agent: %s",
                    status, turn_id,
                    tr.first_user_transcript_latency or -1,
                    tr.first_agent_transcript_latency or -1,
                    tr.audio_bytes_received,
                    agent_text or "(no transcript)",
                )
                if tr.failures:
                    for f in tr.failures:
                        log.warning("  FAILURE: %s", f)

                # Brief pause between turns to let model settle
                await asyncio.sleep(1.5)

    except Exception as exc:
        log.error("Conversation %s failed: %s", conv_id, exc)
        # Add a failed turn for the connection error
        turn_results.append(TurnResult(
            turn_id="CONNECTION_ERROR",
            text=str(exc),
            failures=[f"connection_error: {exc}"],
        ))

    total_duration = time.monotonic() - t0_total
    passed = sum(1 for t in turn_results if t.passed)
    failed = sum(1 for t in turn_results if not t.passed)

    result = ConversationResult(
        conversation_id=conv_id,
        description=conversation["description"],
        total_turns=len(turn_results),
        passed_turns=passed,
        failed_turns=failed,
        turns=turn_results,
        total_duration_sec=total_duration,
    )

    # Build issues from failures
    result.issues = _build_issues(turn_results)
    return result


async def _drain_initial_greeting(
    ws: websockets.ClientConnection,
    timeout_sec: float = 15.0,
) -> None:
    """Drain any initial greeting audio/transcripts from the model."""
    deadline = time.monotonic() + timeout_sec
    last_event = time.monotonic()
    while True:
        now = time.monotonic()
        if now >= deadline:
            break
        # If 3s of silence after last event, model has settled
        if now - last_event >= 3.0:
            break
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            last_event = time.monotonic()
        except asyncio.TimeoutError:
            continue


async def _run_single_turn(
    *,
    ws: websockets.ClientConnection,
    turn_def: dict[str, Any],
    pcm_cache: dict[str, bytes],
    image_dir: Path,
    collect_sec: float,
) -> TurnResult:
    """Execute a single conversation turn and collect the response."""
    turn_id = turn_def["id"]
    text = turn_def["text"]
    pcm = pcm_cache.get(turn_id, b"")

    if not pcm:
        log.warning("No PCM audio for turn %s — skipping audio send", turn_id)
        return TurnResult(
            turn_id=turn_id,
            text=text,
            failures=["no_pcm_audio"],
            notes=turn_def.get("notes", ""),
        )

    # Containers
    counts: dict[str, int] = {}
    tool_events: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    lod_updates: list[dict[str, Any]] = []
    user_transcripts: list[str] = []
    agent_transcripts: list[str] = []
    frame_acks: list[dict[str, Any]] = []
    audio_bytes_count = 0

    t0 = time.monotonic()
    first_user_lat: float | None = None
    first_agent_lat: float | None = None

    # --- Send stimuli ---

    # 1. Telemetry (context setup)
    if turn_def.get("send_telemetry"):
        await ws.send(json.dumps({"type": "telemetry", "data": turn_def["send_telemetry"]}))

    # 2. Gesture
    if turn_def.get("send_gesture"):
        await ws.send(json.dumps({"type": "gesture", "gesture": turn_def["send_gesture"]}))

    # 3. Image (before audio so vision pipeline starts)
    img_bytes_for_frames: bytes | None = None
    if turn_def.get("send_image"):
        img_id = turn_def["send_image"]
        img_path = image_dir / f"{img_id}.jpg"
        if img_path.exists():
            img_bytes_for_frames = img_path.read_bytes()
            await ws.send(bytes([MAGIC_IMAGE]) + img_bytes_for_frames)
        else:
            log.warning("Image not found: %s", img_path)

    # 4. Audio + optional video frame streaming
    send_video_frames = turn_def.get("send_video_frames", 0)
    await ws.send(json.dumps({"type": "activity_start"}))

    audio_start = time.monotonic()
    audio_duration = len(pcm) / 2.0 / 16000.0
    if img_bytes_for_frames and send_video_frames:
        frame_interval = audio_duration / max(send_video_frames, 1)
    else:
        frame_interval = float("inf")
    frames_sent = 0

    for offset in range(0, len(pcm), 1280):
        chunk = pcm[offset: offset + 1280]
        await ws.send(bytes([MAGIC_AUDIO]) + chunk)
        await asyncio.sleep((len(chunk) / 2.0) / 16000.0)

        # Interleave video frames at ~1 FPS during audio streaming
        if img_bytes_for_frames and frames_sent < send_video_frames:
            elapsed = time.monotonic() - audio_start
            if elapsed >= frame_interval * (frames_sent + 1):
                await ws.send(bytes([MAGIC_IMAGE]) + img_bytes_for_frames)
                frames_sent += 1

    await ws.send(json.dumps({"type": "activity_end"}))

    # --- Collect responses ---
    deadline = time.monotonic() + collect_sec
    last_agent_ts: float | None = None

    while time.monotonic() < deadline:
        # Early exit: if we've received agent response and 4s of quiet
        if last_agent_ts and (time.monotonic() - last_agent_ts) >= 4.0:
            break

        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
        except asyncio.TimeoutError:
            continue

        now = time.monotonic()

        if isinstance(raw, bytes):
            audio_bytes_count += 1
            last_agent_ts = now
            continue

        payload = json.loads(raw)
        msg_type = str(payload.get("type", "unknown"))
        counts[msg_type] = counts.get(msg_type, 0) + 1

        if msg_type == "transcript":
            role = payload.get("role", "")
            txt = str(payload.get("text", "")).strip()
            if not txt:
                continue
            if role == "user":
                user_transcripts.append(txt)
                if first_user_lat is None:
                    first_user_lat = now - t0
            elif role == "agent":
                agent_transcripts.append(txt)
                last_agent_ts = now
                if first_agent_lat is None:
                    first_agent_lat = now - t0

        elif msg_type == "tool_event":
            tool_events.append({
                "tool": payload.get("tool"),
                "status": payload.get("status"),
                "behavior": payload.get("behavior"),
            })
        elif msg_type == "tool_result":
            tool_results.append({
                "tool": payload.get("tool"),
                "behavior": payload.get("behavior"),
                "data": payload.get("data"),
            })
        elif msg_type == "lod_update":
            lod_updates.append({
                "lod": payload.get("lod"),
                "reason": payload.get("reason"),
            })
        elif msg_type == "frame_ack":
            frame_acks.append({
                "queued_agents": payload.get("queued_agents"),
            })
        elif msg_type == "vision_result":
            tool_results.append({"_type": "vision_result", **payload})
        elif msg_type == "ocr_result":
            tool_results.append({"_type": "ocr_result", **payload})
        elif msg_type in ("search_result", "navigation_result"):
            tool_results.append({"_type": msg_type, **payload})

    duration = time.monotonic() - t0

    # --- Validate ---
    failures: list[str] = []
    warnings: list[str] = []

    if turn_def.get("expect_agent_response", True):
        if not agent_transcripts:
            failures.append("no_agent_transcript")
        if audio_bytes_count == 0:
            failures.append("no_agent_audio")

    if not user_transcripts:
        warnings.append("no_user_transcript_echoed")

    # Check expected tool
    expected_tool = turn_def.get("expect_tool")
    if expected_tool:
        tool_names = {e.get("tool") for e in tool_events}
        if expected_tool not in tool_names:
            failures.append(f"expected_tool_{expected_tool}_not_called")

    # Check for unwanted OCR on vision queries
    if "describe" in text.lower() or "what's around" in text.lower() or "what is this" in text.lower():
        ocr_tools = [e for e in tool_events if e.get("tool") == "extract_text_from_camera"]
        if ocr_tools and not turn_def.get("expect_tool") == "extract_text_from_camera":
            warnings.append("unwanted_ocr_triggered_on_vision_query")

    # --- Defect-specific assertions ---

    # E2E-001 / E2E-004: Context tag leak detection
    for marker in _LEAKED_MARKERS:
        if any(marker.lower() in t.lower() for t in agent_transcripts):
            failures.append(f"context_tag_leaked: {marker}")

    # E2E-002 / E2E-003: Duplicate tool call detection
    tool_call_counts: dict[str, int] = {}
    for e in tool_events:
        if e.get("status") == "invoked":
            name = e.get("tool", "")
            tool_call_counts[name] = tool_call_counts.get(name, 0) + 1
    for name, count in tool_call_counts.items():
        if count > 1:
            failures.append(f"duplicate_tool_call: {name} called {count}x")

    # E2E-002 / E2E-003: Mutex tool group violation detection
    invoked_tools = {e["tool"] for e in tool_events if e.get("status") == "invoked" and e.get("tool")}
    for group in _MUTEX_TOOL_GROUPS:
        overlap = invoked_tools & group
        if len(overlap) > 1:
            failures.append(f"mutex_violation: {sorted(overlap)} both called")

    # E2E-005: Tool result delivery — if tool expected and result returned, agent must speak
    if expected_tool and tool_results:
        if not agent_transcripts:
            failures.append("tool_result_not_spoken")

    # E2E-007: Navigation over-blocking detection
    expect_not_blocked = turn_def.get("expect_not_blocked")
    if expect_not_blocked:
        blocked_tools = [
            e for e in tool_events
            if e.get("status") == "blocked" and e.get("tool") in expect_not_blocked
        ]
        if blocked_tools:
            failures.append(f"over_blocked: {[e['tool'] for e in blocked_tools]}")

    return TurnResult(
        turn_id=turn_id,
        text=text,
        passed=not failures,
        failures=failures,
        warnings=warnings,
        user_transcripts=user_transcripts,
        agent_transcripts=agent_transcripts,
        tool_events=tool_events,
        tool_results=tool_results,
        lod_updates=lod_updates,
        frame_acks=frame_acks,
        message_counts=counts,
        audio_bytes_received=audio_bytes_count,
        first_user_transcript_latency=first_user_lat,
        first_agent_transcript_latency=first_agent_lat,
        collect_duration_sec=duration,
        notes=turn_def.get("notes", ""),
    )


def _build_issues(turns: list[TurnResult]) -> list[dict[str, Any]]:
    """Build issue definitions from turn results."""
    issues: list[dict[str, Any]] = []
    counter = 1

    for t in turns:
        # Failed turns
        for f in t.failures:
            issues.append({
                "id": f"MT-{counter:03d}",
                "turn": t.turn_id,
                "title": f,
                "severity": "high" if "no_agent" in f else "medium",
                "evidence": {
                    "user_transcripts": t.user_transcripts,
                    "agent_transcripts": t.agent_transcripts[:5],
                    "tool_events": t.tool_events,
                    "message_counts": t.message_counts,
                },
            })
            counter += 1

        # Warnings (observation issues)
        for w in t.warnings:
            issues.append({
                "id": f"MT-{counter:03d}",
                "turn": t.turn_id,
                "title": w,
                "severity": "observation",
                "evidence": {
                    "tool_events": t.tool_events,
                    "agent_transcripts": t.agent_transcripts[:3],
                },
            })
            counter += 1

    return issues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir).resolve()
    image_dir = out_dir / "images"
    audio_dir = out_dir / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(exist_ok=True)
    audio_dir.mkdir(exist_ok=True)

    # Create API client (used for Imagen + TTS)
    client = _create_client()

    # Step 1: Generate test images (Imagen with ffmpeg fallback)
    log.info("--- Generating test images ---")
    needed_images = set()
    for conv in ALL_CONVERSATIONS:
        for turn in conv["turns"]:
            img = turn.get("send_image")
            if img:
                needed_images.add(img)

    for img_id in sorted(needed_images):
        path = _generate_synthetic_image(img_id, image_dir, client=client)
        log.info("Image ready: %s (%d bytes)", path.name, path.stat().st_size)

    # Step 2: Generate TTS audio for all turns
    log.info("--- Generating TTS audio for all turns ---")
    pcm_cache: dict[str, bytes] = {}

    all_turns = []
    for conv in ALL_CONVERSATIONS:
        all_turns.extend(conv["turns"])

    for i, turn in enumerate(all_turns):
        turn_id = turn["id"]
        text = turn["text"]
        cache_path = audio_dir / f"{turn_id}.pcm16k.raw"

        if cache_path.exists() and cache_path.stat().st_size > 0:
            pcm_cache[turn_id] = cache_path.read_bytes()
            log.info("[%d/%d] Loaded cached: %s", i + 1, len(all_turns), turn_id)
            continue

        log.info("[%d/%d] Synthesizing: %s → \"%s\"", i + 1, len(all_turns), turn_id, text)
        try:
            pcm = synthesize_pcm(client, text, voice=args.voice)
            cache_path.write_bytes(pcm)
            pcm_cache[turn_id] = pcm
            dur = len(pcm) / 2.0 / TARGET_SAMPLE_RATE
            log.info("  → %d bytes, %.1fs", len(pcm), dur)
            # TTS rate limit: 10 RPM. Sleep 7s between requests to stay safe.
            if i + 1 < len(all_turns):
                await asyncio.sleep(7.0)
        except Exception as exc:
            log.error("  → TTS FAILED: %s", exc)
            pcm_cache[turn_id] = b""
            # On rate limit, wait longer before next attempt
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                log.info("  → Rate limited, waiting 30s...")
                await asyncio.sleep(30.0)

    # Step 3: Run conversations
    log.info("--- Running multi-turn conversations ---")
    results: list[ConversationResult] = []

    for conv in ALL_CONVERSATIONS:
        result = await run_conversation(
            ws_base_url=args.server,
            conversation=conv,
            pcm_cache=pcm_cache,
            image_dir=image_dir,
            timeout_mult=args.timeout,
        )
        results.append(result)
        log.info(
            "Conversation %s: %d/%d passed, %d issues, %.1fs total",
            result.conversation_id,
            result.passed_turns,
            result.total_turns,
            len(result.issues),
            result.total_duration_sec,
        )
        # Pause between conversations
        await asyncio.sleep(3.0)

    # Step 4: Write report
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "suite_id": f"multiturn_e2e_{stamp}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "server_url": args.server,
        "conversations": [],
    }

    total_turns = 0
    total_passed = 0
    total_failed = 0
    all_issues: list[dict[str, Any]] = []

    for r in results:
        total_turns += r.total_turns
        total_passed += r.passed_turns
        total_failed += r.failed_turns
        all_issues.extend(r.issues)

        conv_data = {
            "id": r.conversation_id,
            "description": r.description,
            "total_turns": r.total_turns,
            "passed": r.passed_turns,
            "failed": r.failed_turns,
            "duration_sec": round(r.total_duration_sec, 1),
            "turns": [asdict(t) for t in r.turns],
            "issues": r.issues,
        }
        report["conversations"].append(conv_data)

    report["summary"] = {
        "total_turns": total_turns,
        "passed": total_passed,
        "failed": total_failed,
        "pass_rate": f"{total_passed / total_turns * 100:.1f}%" if total_turns > 0 else "N/A",
        "total_issues": len(all_issues),
    }
    report["all_issues"] = all_issues

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Report: %s", report_path)

    # Print summary
    print("\n" + "=" * 70)
    print("  SightLine Multi-Turn E2E Test Results")
    print("=" * 70)
    print(f"  Total turns: {total_turns}")
    print(f"  Passed:      {total_passed}")
    print(f"  Failed:      {total_failed}")
    print(f"  Pass rate:   {report['summary']['pass_rate']}")
    print()

    for r in results:
        status = "OK" if r.failed_turns == 0 else "ISSUES"
        print(f"  [{status}] {r.conversation_id}: {r.passed_turns}/{r.total_turns} passed ({r.total_duration_sec:.0f}s)")
        for t in r.turns:
            mark = "+" if t.passed else "X"
            agent_preview = " ".join(t.agent_transcripts)[:80]
            tools = ", ".join(e.get("tool", "?") for e in t.tool_events) or "-"
            print(f"    [{mark}] {t.turn_id}: tools=[{tools}] agent=\"{agent_preview}\"")
            for f in t.failures:
                print(f"        FAIL: {f}")
            for w in t.warnings:
                print(f"        WARN: {w}")

    if all_issues:
        print(f"\n  Issues ({len(all_issues)}):")
        for iss in all_issues:
            print(f"    {iss['id']} [{iss['severity']}] {iss['turn']}: {iss['title']}")

    print("=" * 70 + "\n")

    return 0 if total_failed == 0 else 2


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    p = argparse.ArgumentParser(description="Multi-turn E2E test for SightLine.")
    p.add_argument("--server", default="ws://127.0.0.1:8100", help="Server WebSocket URL")
    p.add_argument("--output-dir", default="artifacts/e2e_multiturn", help="Output directory")
    p.add_argument("--voice", default="Aoede", help="TTS voice")
    p.add_argument("--timeout", type=float, default=1.0, help="Timeout multiplier")
    args = p.parse_args()
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
