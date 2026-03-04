#!/usr/bin/env python3
"""Generate test assets for SightLine E2E test suite.

Produces 8 test images (via Gemini Imagen API or synthetic ffmpeg fallback)
and 13 voice command audio files (via Gemini TTS), then writes a manifest.json
describing every generated artifact.

Usage:
    python scripts/generate_e2e_assets.py
    python scripts/generate_e2e_assets.py --force-synthetic --skip-audio
    python scripts/generate_e2e_assets.py --output-dir /tmp/e2e --voice Kore
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import re
import shutil
import subprocess
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger("generate_e2e_assets")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_OUTPUT_DIR = Path("artifacts/e2e_assets")
_DEFAULT_TTS_MODEL = "gemini-2.5-flash-preview-tts"
_DEFAULT_IMAGEN_MODEL = "imagen-3.0-generate-002"
_TARGET_SAMPLE_RATE = 16000

IMAGE_SCENES: list[dict[str, Any]] = [
    {
        "id": "street_crossing",
        "prompt": (
            "A city crosswalk at a busy intersection, with a pedestrian Don't Walk "
            "hand signal visible, and a car approaching from the left. Realistic photo "
            "taken from a pedestrian's eye level on the sidewalk."
        ),
        "tests": ["safety_detection", "ocr_safety"],
    },
    {
        "id": "restaurant_menu",
        "prompt": (
            "A restaurant menu board mounted on a wall with clearly readable food items "
            "and prices in English. Items include Burger $12, Salad $8, Coffee $4. "
            "Realistic photo."
        ),
        "tests": ["ocr_accuracy", "user_intent_ocr"],
    },
    {
        "id": "staircase_hazard",
        "prompt": (
            "A wet concrete staircase going down with no handrail, puddles of water "
            "visible on the steps. Warning: slippery surface. Realistic photo from top "
            "of stairs looking down."
        ),
        "tests": ["safety_all_lods"],
    },
    {
        "id": "indoor_lobby",
        "prompt": (
            "A modern office building lobby with an elevator on the left, a reception "
            "desk straight ahead, and a hallway to the right with room numbers visible. "
            "Realistic photo."
        ),
        "tests": ["navigation_landmarks"],
    },
    {
        "id": "construction_zone",
        "prompt": (
            "A sidewalk construction zone with orange barriers, a yellow DANGER sign, "
            "and exposed ground. Workers wearing hard hats in background. Realistic photo."
        ),
        "tests": ["multimodal_safety"],
    },
    {
        "id": "park_scene",
        "prompt": (
            "A sunny park with green grass, three people sitting on a bench, a dog on "
            "a leash, and a paved walking path. Trees and a small pond visible. "
            "Realistic photo."
        ),
        "tests": ["lod3_narrative"],
    },
    {
        "id": "blank_wall",
        "prompt": (
            "A plain white wall with no features, no text, no objects. Just a flat "
            "white painted wall. Realistic photo."
        ),
        "tests": ["negative_test"],
    },
    {
        "id": "street_signs",
        "prompt": (
            "A street corner with clearly readable road signs showing 5th Avenue and "
            "Main Street, with a green traffic light visible. Realistic photo from "
            "pedestrian perspective."
        ),
        "tests": ["navigation_ocr"],
    },
]

VOICE_COMMANDS: list[dict[str, str | list[str]]] = [
    {"id": "greeting", "text": "Hello, can you help me navigate?"},
    {"id": "whats_ahead", "text": "What's ahead of me right now?"},
    {"id": "read_sign", "text": "Can you read that sign for me?"},
    {"id": "navigate_pharmacy", "text": "Navigate me to the nearest pharmacy."},
    {"id": "search_weather", "text": "What's the weather like today?"},
    {"id": "describe_scene", "text": "Describe what you see around me."},
    {"id": "whats_that_say", "text": "What does that say?"},
    {"id": "any_danger", "text": "Is there any danger ahead?"},
    {"id": "how_many_people", "text": "How many people are around me?"},
    {"id": "read_menu", "text": "Read the menu for me please."},
    {"id": "where_am_i", "text": "Where am I right now?"},
    {"id": "remember_place", "text": "Remember this place as my favorite cafe."},
    {"id": "repeat_last", "text": "Can you repeat what you just said?"},
]

# Distinct colors for synthetic fallback images (one per scene).
_SYNTH_COLORS = [
    "0x3366CC",  # street_crossing  - blue
    "0xCC6633",  # restaurant_menu  - orange
    "0x999933",  # staircase_hazard - olive
    "0x339966",  # indoor_lobby     - green
    "0xCC3333",  # construction_zone- red
    "0x33CC66",  # park_scene       - lime
    "0xCCCCCC",  # blank_wall       - light grey
    "0x6633CC",  # street_signs     - purple
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_api_key() -> str:
    """Return the first non-empty Gemini/Google API key from env."""
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "_GOOGLE_AI_API_KEY"):
        val = (os.getenv(var) or "").strip()
        if val:
            return val
    raise RuntimeError(
        "No API key found. Set GEMINI_API_KEY, GOOGLE_API_KEY, or _GOOGLE_AI_API_KEY."
    )


def _create_genai_client() -> genai.Client:
    """Create a google-genai client, preferring Vertex when configured/available."""
    use_vertex_raw = (os.getenv("GOOGLE_GENAI_USE_VERTEXAI") or "").strip().lower()
    vertex_explicit_true = use_vertex_raw in {"1", "true", "yes", "on"}
    vertex_explicit_false = use_vertex_raw in {"0", "false", "no", "off"}
    project = (os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip()
    location = (
        os.getenv("GOOGLE_CLOUD_LOCATION")
        or os.getenv("GOOGLE_CLOUD_REGION")
        or "us-central1"
    ).strip()
    auto_prefer_vertex = not use_vertex_raw and bool(project)
    use_vertex = vertex_explicit_true or auto_prefer_vertex

    if use_vertex and project:
        try:
            return genai.Client(vertexai=True, project=project, location=location)
        except Exception as exc:
            if vertex_explicit_true:
                log.warning("Vertex client init failed (explicit), falling back to API key: %s", exc)
            else:
                log.info("Vertex auto-detect failed, falling back to API key: %s", exc)
    elif vertex_explicit_true and not project:
        log.warning("GOOGLE_GENAI_USE_VERTEXAI=true but GOOGLE_CLOUD_PROJECT is empty; using API key.")
    elif vertex_explicit_false:
        log.info("GOOGLE_GENAI_USE_VERTEXAI explicitly disabled; using API key.")

    return genai.Client(api_key=_resolve_api_key())


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
        raw_frames = wf.readframes(wf.getnframes())

    if sample_width == 2:
        samples = np.frombuffer(raw_frames, dtype="<i2").astype(np.int16)
    elif sample_width == 1:
        u8 = np.frombuffer(raw_frames, dtype=np.uint8).astype(np.int16)
        samples = ((u8 - 128) << 8).astype(np.int16)
    elif sample_width == 4:
        i32 = np.frombuffer(raw_frames, dtype="<i4")
        samples = (i32 >> 16).astype(np.int16)
    else:
        raise ValueError(f"unsupported WAV sample width: {sample_width} bytes")

    if channels > 1:
        samples = (
            samples.reshape(-1, channels).astype(np.int32).mean(axis=1).astype(np.int16)
        )
    return samples, sample_rate


def _signal_smoothness_score(samples: np.ndarray) -> float:
    if samples.size < 2:
        return float("inf")
    diff = np.abs(np.diff(samples.astype(np.int32)))
    return float(diff.mean())


def _resample_linear_int16(
    samples: np.ndarray, src_rate: int, dst_rate: int
) -> np.ndarray:
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


def _extract_inline_audio(
    response: types.GenerateContentResponse,
) -> tuple[bytes, str]:
    parts = response.parts or []
    for part in parts:
        if part.inline_data and part.inline_data.data:
            data = part.inline_data.data
            if isinstance(data, str):
                try:
                    data = base64.b64decode(data, validate=True)
                except Exception:
                    data = data.encode("utf-8")
            mime_type = part.inline_data.mime_type or "application/octet-stream"
            return data, mime_type
    raise RuntimeError("TTS response contains no inline audio data")


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


# ---------------------------------------------------------------------------
# Image Generation
# ---------------------------------------------------------------------------


def _generate_imagen_image(
    client: genai.Client,
    scene_def: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Generate a single scene image using the Imagen API.

    Returns a manifest entry dict on success; raises on failure.
    """
    scene_id = scene_def["id"]
    prompt = scene_def["prompt"]
    dest = output_dir / "images" / f"{scene_id}.jpg"
    dest.parent.mkdir(parents=True, exist_ok=True)

    log.info("  [imagen] generating %s ...", scene_id)
    response = client.models.generate_images(
        model=_DEFAULT_IMAGEN_MODEL,
        prompt=prompt,
        config={"number_of_images": 1},
    )

    if not response.generated_images:
        raise RuntimeError(f"Imagen returned no images for scene '{scene_id}'")

    image_bytes = response.generated_images[0].image.image_bytes
    if not image_bytes:
        raise RuntimeError(f"Imagen returned empty image data for scene '{scene_id}'")

    dest.write_bytes(image_bytes)
    log.info("  [imagen] saved %s (%d bytes)", dest, len(image_bytes))

    return {
        "id": scene_id,
        "path": f"images/{scene_id}.jpg",
        "source": "imagen",
        "tests": scene_def.get("tests", []),
    }


def _generate_synthetic_image(
    scene_def: dict[str, Any],
    output_dir: Path,
    color: str,
) -> dict[str, Any]:
    """Generate a synthetic placeholder JPEG using ffmpeg.

    Falls back to a minimal numpy-generated image if ffmpeg is unavailable.
    """
    scene_id = scene_def["id"]
    dest = output_dir / "images" / f"{scene_id}.jpg"
    dest.parent.mkdir(parents=True, exist_ok=True)

    if _ffmpeg_available():
        # Try with drawtext overlay first.
        cmd_with_text = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c={color}:s=640x480:d=1",
            "-vf", (
                f"drawtext=text='{scene_id}':fontsize=36:"
                "fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2"
            ),
            "-vframes", "1",
            "-q:v", "3",
            str(dest),
        ]
        result = subprocess.run(
            cmd_with_text,
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            # drawtext may fail if fontconfig/freetype is missing; retry without.
            cmd_plain = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"color=c={color}:s=640x480:d=1",
                "-vframes", "1",
                "-q:v", "3",
                str(dest),
            ]
            result = subprocess.run(
                cmd_plain,
                capture_output=True,
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg failed for scene '{scene_id}': "
                    f"{result.stderr.decode(errors='replace')}"
                )

        log.info("  [synthetic/ffmpeg] saved %s", dest)
    else:
        # No ffmpeg -- create a minimal valid JPEG via raw bytes.
        # We write a small BMP-in-memory and convert to JPEG concept.
        # Simpler: create a solid-color image using struct + JFIF header.
        _write_solid_jpeg(dest, color, 640, 480)
        log.info("  [synthetic/numpy] saved %s", dest)

    return {
        "id": scene_id,
        "path": f"images/{scene_id}.jpg",
        "source": "synthetic",
        "tests": scene_def.get("tests", []),
    }


def _write_solid_jpeg(path: Path, hex_color: str, width: int, height: int) -> None:
    """Write a solid-color image as an uncompressed BMP (universally readable).

    Since we want a .jpg but don't have PIL/ffmpeg, we write a valid BMP
    that most image loaders accept.  If cv2 is available we use it for
    proper JPEG encoding; otherwise we fall back to BMP renamed as .jpg
    which is still loadable by most test harnesses.
    """
    # Parse 0xRRGGBB
    hex_color = hex_color.replace("0x", "").replace("#", "")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    try:
        import cv2  # noqa: F811

        img = np.full((height, width, 3), [b, g, r], dtype=np.uint8)
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return
    except ImportError:
        pass

    # Fallback: raw PPM (universally parseable, renamed to .jpg for the manifest).
    # Switch extension to .ppm so the file is genuinely valid.
    ppm_path = path.with_suffix(".ppm")
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    pixel = bytes([r, g, b])
    ppm_path.write_bytes(header + pixel * (width * height))
    # Rename to .jpg -- downstream test code loads via path from manifest.
    if path.exists():
        path.unlink()
    ppm_path.rename(path)


def _generate_images(
    client: genai.Client | None,
    output_dir: Path,
    *,
    force_synthetic: bool = False,
) -> list[dict[str, Any]]:
    """Generate all 8 scene images.  Returns list of manifest entries."""
    results: list[dict[str, Any]] = []
    imagen_failures = 0

    for idx, scene in enumerate(IMAGE_SCENES):
        scene_id = scene["id"]

        if force_synthetic:
            entry = _generate_synthetic_image(
                scene, output_dir, _SYNTH_COLORS[idx]
            )
            results.append(entry)
            continue

        # Try Imagen first, fall back to synthetic on any error.
        try:
            if client is None:
                raise RuntimeError("no genai client")
            entry = _generate_imagen_image(client, scene, output_dir)
            results.append(entry)
        except Exception as exc:
            imagen_failures += 1
            log.warning(
                "  [imagen] failed for %s (%s), falling back to synthetic",
                scene_id,
                exc,
            )
            entry = _generate_synthetic_image(
                scene, output_dir, _SYNTH_COLORS[idx]
            )
            results.append(entry)

    if imagen_failures > 0:
        log.warning(
            "[images] %d/%d scenes fell back to synthetic generation",
            imagen_failures,
            len(IMAGE_SCENES),
        )

    return results


# ---------------------------------------------------------------------------
# Audio / TTS Generation
# ---------------------------------------------------------------------------


def _synthesize_voice_command(
    client: genai.Client,
    command_def: dict[str, Any],
    output_dir: Path,
    voice: str,
    model: str,
) -> dict[str, Any]:
    """Synthesize a single voice command via Gemini TTS.

    Saves both a .wav and a .raw (PCM int16) file at 16 kHz mono.
    Returns a manifest entry dict.
    """
    cmd_id = str(command_def["id"])
    text = str(command_def["text"])
    target_rate = _TARGET_SAMPLE_RATE

    log.info("  [tts] synthesizing %s: %r", cmd_id, text)

    prompt = (
        "Read the following text naturally. "
        "Do not add any extra words, headers, or explanations.\n\n"
        f"{text}"
    )
    speech_config = types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
        ),
        language_code="en-US",
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

    # Decode to mono int16
    if "audio/wav" in mime_lower or "audio/x-wav" in mime_lower:
        src_samples, src_rate = _wav_bytes_to_mono_int16(audio_data)
    elif "audio/pcm" in mime_lower or "audio/l16" in mime_lower:
        src_rate = _parse_sample_rate_from_mime(mime_type, default_rate=24000)
        usable = len(audio_data) - (len(audio_data) % 2)
        if "audio/l16" in mime_lower:
            le = np.frombuffer(audio_data[:usable], dtype="<i2").astype(np.int16)
            be = np.frombuffer(audio_data[:usable], dtype=">i2").astype(np.int16)
            src_samples = (
                le
                if _signal_smoothness_score(le) <= _signal_smoothness_score(be)
                else be
            )
        else:
            src_samples = np.frombuffer(
                audio_data[:usable], dtype="<i2"
            ).astype(np.int16)
    else:
        raise RuntimeError(f"Unsupported TTS output mime type: {mime_type}")

    # Resample to target rate
    dst_samples = _resample_linear_int16(src_samples, src_rate, target_rate)

    # Write artifacts
    wav_path = output_dir / "audio" / f"{cmd_id}.pcm{target_rate}.wav"
    raw_path = output_dir / "audio" / f"{cmd_id}.pcm{target_rate}.raw"
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    _write_wav(wav_path, dst_samples, target_rate)
    raw_path.write_bytes(dst_samples.tobytes())

    dur = len(dst_samples) / float(target_rate)
    log.info(
        "  [tts] saved %s  mime=%s  src_rate=%d  dur=%.2fs",
        cmd_id,
        mime_type,
        src_rate,
        dur,
    )

    return {
        "id": cmd_id,
        "text": text,
        "pcm_path": f"audio/{cmd_id}.pcm{target_rate}.raw",
        "wav_path": f"audio/{cmd_id}.pcm{target_rate}.wav",
    }


def _generate_audio(
    client: genai.Client,
    output_dir: Path,
    voice: str = "Aoede",
) -> list[dict[str, Any]]:
    """Generate TTS audio for all 13 voice commands.  Returns manifest entries."""
    results: list[dict[str, Any]] = []

    for cmd in VOICE_COMMANDS:
        try:
            entry = _synthesize_voice_command(
                client,
                cmd,
                output_dir,
                voice=voice,
                model=_DEFAULT_TTS_MODEL,
            )
            results.append(entry)
        except Exception:
            log.exception("  [tts] FAILED for %s", cmd["id"])
            raise

    return results


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _write_manifest(
    output_dir: Path,
    images: list[dict[str, Any]],
    audio: list[dict[str, Any]],
) -> Path:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "images": images,
        "audio": audio,
        "image_count": len(images),
        "audio_count": len(audio),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    log.info("[manifest] written to %s", manifest_path)
    return manifest_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate test assets (images + voice commands) for SightLine E2E test suite.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        metavar="PATH",
        help=f"Output directory (default: {_DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="Aoede",
        help="TTS voice name (default: Aoede)",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip image generation (keep existing images)",
    )
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Skip audio generation (keep existing audio)",
    )
    parser.add_argument(
        "--force-synthetic",
        action="store_true",
        help="Force synthetic images via ffmpeg instead of Imagen API",
    )
    return parser


def main() -> int:
    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = _build_parser()
    args = parser.parse_args()

    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("output_dir = %s", output_dir)
    log.info("skip_images = %s", args.skip_images)
    log.info("skip_audio  = %s", args.skip_audio)
    log.info("force_synth = %s", args.force_synthetic)
    log.info("voice       = %s", args.voice)

    # ------------------------------------------------------------------
    # Create genai client (only when needed)
    # ------------------------------------------------------------------
    need_api = (not args.skip_images and not args.force_synthetic) or (
        not args.skip_audio
    )
    client: genai.Client | None = None
    if need_api:
        try:
            client = _create_genai_client()
        except RuntimeError as exc:
            if not args.skip_audio:
                # Audio absolutely requires API.  Abort.
                log.error("Cannot create genai client: %s", exc)
                return 1
            # Images can still fall back to synthetic.
            log.warning("No API key; images will be synthetic: %s", exc)

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------
    images: list[dict[str, Any]] = []
    if not args.skip_images:
        log.info("[images] generating %d scenes ...", len(IMAGE_SCENES))
        images = _generate_images(
            client,
            output_dir,
            force_synthetic=args.force_synthetic,
        )
        log.info(
            "[images] done: %d generated (%d imagen, %d synthetic)",
            len(images),
            sum(1 for i in images if i["source"] == "imagen"),
            sum(1 for i in images if i["source"] == "synthetic"),
        )
    else:
        # Collect existing image entries for manifest.
        for scene in IMAGE_SCENES:
            img_path = output_dir / "images" / f"{scene['id']}.jpg"
            if img_path.exists():
                images.append({
                    "id": scene["id"],
                    "path": f"images/{scene['id']}.jpg",
                    "source": "existing",
                    "tests": scene.get("tests", []),
                })
            else:
                log.warning(
                    "  [images] skipped but %s not found; omitting from manifest",
                    img_path,
                )

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------
    audio: list[dict[str, Any]] = []
    if not args.skip_audio:
        if client is None:
            log.error("Cannot generate audio without a genai client.")
            return 1
        log.info("[audio] generating %d voice commands ...", len(VOICE_COMMANDS))
        audio = _generate_audio(client, output_dir, voice=args.voice)
        log.info("[audio] done: %d generated", len(audio))
    else:
        # Collect existing audio entries for manifest.
        for cmd in VOICE_COMMANDS:
            raw_path = (
                output_dir / "audio" / f"{cmd['id']}.pcm{_TARGET_SAMPLE_RATE}.raw"
            )
            wav_path = (
                output_dir / "audio" / f"{cmd['id']}.pcm{_TARGET_SAMPLE_RATE}.wav"
            )
            if raw_path.exists() and wav_path.exists():
                audio.append({
                    "id": cmd["id"],
                    "text": cmd["text"],
                    "pcm_path": f"audio/{cmd['id']}.pcm{_TARGET_SAMPLE_RATE}.raw",
                    "wav_path": f"audio/{cmd['id']}.pcm{_TARGET_SAMPLE_RATE}.wav",
                })
            else:
                log.warning(
                    "  [audio] skipped but %s not found; omitting from manifest",
                    raw_path,
                )

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------
    manifest_path = _write_manifest(output_dir, images, audio)

    log.info("=== Summary ===")
    log.info("  images: %d", len(images))
    log.info("  audio:  %d", len(audio))
    log.info("  manifest: %s", manifest_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
