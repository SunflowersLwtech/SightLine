"""SightLine Vision Sub-Agent.

Asynchronous scene analysis using Gemini Vision (gemini-3.1-pro-preview)
with LOD-adaptive prompting. Each LOD level adjusts both the media resolution
and the system prompt to extract information appropriate for the user's
current context.

LOD Levels:
    1 — Safety threats only (low resolution, ~70 tokens/frame)
    2 — Spatial navigation (medium resolution, ~560 tokens/frame)
    3 — Full narrative (high resolution, ~1120 tokens/frame)
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any

from google import genai
from google.genai import types

logger = logging.getLogger("sightline.vision_agent")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-3.1-pro-preview")

_MEDIA_RESOLUTION_BY_LOD: dict[int, types.MediaResolution] = {
    1: types.MediaResolution.MEDIA_RESOLUTION_LOW,
    2: types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
    3: types.MediaResolution.MEDIA_RESOLUTION_HIGH,
}

# ---------------------------------------------------------------------------
# LOD-adaptive system prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_LOD1 = """\
You are a safety analysis system for a blind user navigating in real time.

ONLY report immediate physical hazards visible in this image:
- Stairs, steps, drop-offs, curbs
- Approaching vehicles or cyclists
- Obstacles in the walking path (poles, furniture, construction)
- Wet/slippery surfaces, uneven ground
- Low-hanging objects at head height

Use clock positions (e.g. "obstacle at 12 o'clock, 2 meters").
If no hazards are visible, return an empty safety_warnings list.
Do NOT describe anything else — no people, colors, atmosphere, or text.
"""

_SYSTEM_PROMPT_LOD2 = """\
You are a spatial navigation assistant for a blind user.

Analyze this image for navigation-relevant information:
1. Spatial layout: entrances, exits, paths, corridors, intersections.
2. Signage and wayfinding: readable signs, door numbers, directions.
3. People: approximate count and proximity (not descriptions).
4. Key landmarks: counters, elevators, escalators, seating areas.

Use clock positions for spatial references.
Be concise — focus on what helps the user navigate, not decorative details.
"""

_SYSTEM_PROMPT_LOD3 = """\
You are a detailed scene narrator for a blind user who wants a rich \
understanding of their surroundings.

Provide a comprehensive description:
1. SAFETY: Any hazards (always first priority).
2. Spatial layout: full description of the space and its organization.
3. People: count, approximate positions, expressions, activities.
4. Text: all readable text (signs, menus, labels, screens).
5. Objects: notable items, their positions and colors.
6. Atmosphere: lighting, weather, mood, sounds you might infer.

Use clock positions for spatial references.
Be thorough but organized — the user relies on you as their eyes.
"""

_SYSTEM_PROMPTS: dict[int, str] = {
    1: _SYSTEM_PROMPT_LOD1,
    2: _SYSTEM_PROMPT_LOD2,
    3: _SYSTEM_PROMPT_LOD3,
}

# ---------------------------------------------------------------------------
# Response schema for structured JSON output
# ---------------------------------------------------------------------------

_RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "safety_warnings": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING),
            description="List of immediate safety hazards using clock positions.",
        ),
        "navigation_info": types.Schema(
            type=types.Type.OBJECT,
            properties={
                "entrances": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING),
                ),
                "paths": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING),
                ),
                "landmarks": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING),
                ),
            },
            description="Spatial navigation information.",
        ),
        "scene_description": types.Schema(
            type=types.Type.STRING,
            description="Natural language scene description appropriate to LOD.",
        ),
        "detected_text": types.Schema(
            type=types.Type.STRING,
            nullable=True,
            description="Any readable text found in the scene.",
        ),
        "people_count": types.Schema(
            type=types.Type.INTEGER,
            description="Number of people visible in the scene.",
        ),
        "confidence": types.Schema(
            type=types.Type.NUMBER,
            description="Confidence score from 0.0 to 1.0.",
        ),
    },
    required=[
        "safety_warnings",
        "navigation_info",
        "scene_description",
        "people_count",
        "confidence",
    ],
)

# ---------------------------------------------------------------------------
# Empty / fallback result
# ---------------------------------------------------------------------------

_EMPTY_RESULT: dict[str, Any] = {
    "safety_warnings": [],
    "navigation_info": {"entrances": [], "paths": [], "landmarks": []},
    "scene_description": "",
    "detected_text": None,
    "people_count": 0,
    "confidence": 0.0,
}


def _build_context_user_message(lod: int, session_context: dict) -> str:
    """Build a context-aware user message that includes session info."""
    parts = [f"Analyze this image at LOD level {lod}."]

    space = session_context.get("space_type")
    if space and space != "unknown":
        parts.append(f"The user is currently in a {space} environment.")

    trip = session_context.get("trip_purpose")
    if trip:
        parts.append(f"Trip purpose: {trip}.")

    task = session_context.get("active_task")
    if task:
        parts.append(f"Currently engaged in: {task}.")

    motion = session_context.get("motion_state")
    if motion:
        parts.append(f"Motion state: {motion}.")

    # Depth data (from CoreML depth estimation)
    depth_center = session_context.get("depth_center")
    depth_min = session_context.get("depth_min")
    depth_min_region = session_context.get("depth_min_region")
    if depth_center is not None and depth_center > 0:
        depth_info = f"Depth data: center distance {depth_center:.1f}m"
        if depth_min is not None and depth_min > 0 and depth_min_region:
            depth_info += f", closest object at {depth_min:.1f}m ({depth_min_region})"
        depth_info += "."
        parts.append(depth_info)

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Lazily initialize the Gemini client."""
    global _client
    if _client is None:
        api_key = os.environ.get("_GOOGLE_AI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
        _client = genai.Client(api_key=api_key, vertexai=False)
    return _client


async def analyze_scene(
    image_base64: str,
    lod: int,
    session_context: dict | None = None,
) -> dict[str, Any]:
    """Analyze a scene image with LOD-adaptive prompting.

    Args:
        image_base64: Base64-encoded JPEG image data.
        lod: Level of detail (1, 2, or 3).
        session_context: Optional dict with keys like space_type,
            trip_purpose, active_task, motion_state.

    Returns:
        Structured dict with safety_warnings, navigation_info,
        scene_description, detected_text, people_count, confidence.
        Returns empty result on failure (never raises).
    """
    if session_context is None:
        session_context = {}

    # Clamp LOD to valid range
    lod = max(1, min(3, lod))

    try:
        image_bytes = base64.b64decode(image_base64)
    except Exception:
        logger.error("Failed to decode base64 image data")
        return dict(_EMPTY_RESULT)

    system_prompt = _SYSTEM_PROMPTS[lod]
    media_resolution = _MEDIA_RESOLUTION_BY_LOD[lod]
    user_message = _build_context_user_message(lod, session_context)

    response = None
    try:
        client = _get_client()
        response = await client.aio.models.generate_content(
            model=VISION_MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type="image/jpeg",
                        ),
                        types.Part.from_text(text=user_message),
                    ],
                ),
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                media_resolution=media_resolution,
                response_mime_type="application/json",
                response_schema=_RESPONSE_SCHEMA,
                temperature=0.2,
            ),
        )

        if not response.text:
            logger.warning("Vision model returned empty response")
            return dict(_EMPTY_RESULT)

        result = json.loads(response.text)

        # Ensure all expected keys exist
        for key, default_val in _EMPTY_RESULT.items():
            if key not in result:
                result[key] = default_val

        return result

    except json.JSONDecodeError:
        raw = response.text if response else "<no response>"
        logger.error("Failed to parse vision model JSON response: %s", raw)
        return dict(_EMPTY_RESULT)
    except Exception:
        logger.exception("Vision analysis failed")
        return dict(_EMPTY_RESULT)
