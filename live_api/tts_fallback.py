"""Server-side TTS fallback for tool results.

Generates PCM 24kHz mono 16-bit audio so the iOS client can play fallback
speech when Gemini Live fails to verbalize a tool result in time.
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import re
import subprocess
import uuid
import wave
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types

TTS_MODEL = "gemini-2.5-flash-preview-tts"
TARGET_SAMPLE_RATE = 24000
DEFAULT_VOICE = os.getenv("GEMINI_TTS_VOICE", "Aoede")

_client: genai.Client | None = None


def _create_client() -> genai.Client:
    global _client
    if _client is not None:
        return _client

    use_vertex = (os.getenv("GOOGLE_GENAI_USE_VERTEXAI") or "").strip().lower() in {
        "1", "true", "yes", "on"
    }
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    location = (
        os.getenv("GOOGLE_CLOUD_REGION", "").strip()
        or os.getenv("GOOGLE_CLOUD_LOCATION", "").strip()
        or "us-central1"
    )

    if use_vertex and project:
        _client = genai.Client(vertexai=True, project=project, location=location)
        return _client

    api_key = (
        os.getenv("_GOOGLE_AI_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or ""
    ).strip()
    if not api_key:
        raise RuntimeError("No Gemini API credentials available for TTS fallback")
    _client = genai.Client(api_key=api_key, vertexai=False)
    return _client


def _parse_rate_from_mime(mime: str) -> int:
    match = re.search(r"rate=(\d+)", mime or "", re.IGNORECASE)
    return int(match.group(1)) if match else TARGET_SAMPLE_RATE


def _wav_to_mono_int16(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        raw_frames = wf.readframes(wf.getnframes())

    if sample_width == 2:
        samples = np.frombuffer(raw_frames, dtype="<i2").astype(np.int16)
    elif sample_width == 4:
        samples = (np.frombuffer(raw_frames, dtype="<i4") >> 16).astype(np.int16)
    else:
        raise ValueError(f"unsupported WAV sample width {sample_width}")

    if channels > 1:
        samples = samples.reshape(-1, channels).astype(np.int32).mean(axis=1).astype(np.int16)
    return samples, sample_rate


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
    raise RuntimeError("No audio payload in TTS response")


def _silent_pcm(duration_sec: float = 1.0) -> bytes:
    """Generate silent PCM audio as a last-resort fallback."""
    num_samples = int(TARGET_SAMPLE_RATE * duration_sec)
    return np.zeros(num_samples, dtype=np.int16).tobytes()


def _local_fallback_pcm(text: str) -> bytes:
    import platform
    import shutil

    tmp_root = Path("/tmp/sightline_runs/server_tts_fallback")
    tmp_root.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:10]
    pcm_path = tmp_root / f"{token}.pcm24k.raw"

    # macOS: use `say` + ffmpeg
    if platform.system() == "Darwin" and shutil.which("say"):
        aiff_path = tmp_root / f"{token}.aiff"
        say_result = subprocess.run(
            ["say", "-v", "Samantha", "-o", str(aiff_path), text],
            capture_output=True, text=True,
        )
        if say_result.returncode == 0 and shutil.which("ffmpeg"):
            ffmpeg_result = subprocess.run(
                ["ffmpeg", "-y", "-i", str(aiff_path),
                 "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE), "-f", "s16le",
                 str(pcm_path)],
                capture_output=True, text=True,
            )
            if ffmpeg_result.returncode == 0:
                return pcm_path.read_bytes()

    # Linux: use espeak-ng if available
    if shutil.which("espeak-ng"):
        wav_path = tmp_root / f"{token}.wav"
        espeak_result = subprocess.run(
            ["espeak-ng", "-v", "en", "-w", str(wav_path), text],
            capture_output=True, text=True,
        )
        if espeak_result.returncode == 0 and wav_path.exists():
            try:
                samples, src_rate = _wav_to_mono_int16(wav_path.read_bytes())
                return _resample(samples, src_rate, TARGET_SAMPLE_RATE).tobytes()
            except Exception:
                pass

    # Last resort: return silent audio (text transcript still delivered)
    return _silent_pcm(0.5)


async def synthesize_pcm(text: str, voice: str = DEFAULT_VOICE) -> bytes:
    client = _create_client()
    try:
        response = await client.aio.models.generate_content(
            model=TTS_MODEL,
            contents="Read the following text naturally. Do not add extra words.\n\n" + text,
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
        else:
            src_rate = _parse_rate_from_mime(mime)
            usable = len(audio_data) - (len(audio_data) % 2)
            samples = np.frombuffer(audio_data[:usable], dtype="<i2").astype(np.int16)
        return _resample(samples, src_rate, TARGET_SAMPLE_RATE).tobytes()
    except Exception as exc:
        exc_text = str(exc)
        if "429" in exc_text or "RESOURCE_EXHAUSTED" in exc_text:
            return await asyncio.to_thread(_local_fallback_pcm, text)
        raise
