"""SightLine OCR Sub-Agent.

Asynchronous text extraction using Gemini Flash (gemini-3-flash-preview)
optimized for menus, signage, documents, and labels. Uses the free-tier
Gemini Developer API.

The context_hint parameter helps the model focus on the most relevant
text type (e.g., "user is at a restaurant" prioritizes menu items).
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any

from google import genai
from google.genai import types

logger = logging.getLogger("sightline.ocr_agent")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OCR_MODEL = os.getenv("GEMINI_FLASH_MODEL", "gemini-3-flash-preview")

_SYSTEM_PROMPT = """\
You are a text extraction system for a blind user. Your job is to read ALL \
visible text in the image accurately.

Rules:
1. Extract every piece of readable text — signs, menus, labels, documents, \
   screens, handwriting.
2. Classify the text type: "menu", "sign", "document", "label", or "unknown".
3. For menus: parse into individual items with prices when visible. Format \
   each item as "Item Name - $Price" or just "Item Name" if no price.
4. For signs: preserve the exact wording.
5. For documents: maintain reading order (top to bottom, left to right).
6. Report confidence based on text clarity (0.0 = unreadable, 1.0 = crystal clear).
7. If no text is visible, return empty results with confidence 0.0.

Priority: accuracy over speed. A blind user depends on correct text reading.
"""

_SAFETY_SYSTEM_PROMPT = """\
You are a safety text detector for a blind user. Focus ONLY on identifying \
warning signs, traffic signs, danger labels, hazard notices, and any \
safety-critical text visible in the image.

Rules:
1. Only extract text related to warnings, traffic, danger, or hazards.
2. Ignore menus, decorative text, brand names, and non-safety content.
3. If no safety-related text is visible, return empty results with confidence 0.0.
"""

# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

_RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "text": types.Schema(
            type=types.Type.STRING,
            description="All extracted text as a single string.",
        ),
        "text_type": types.Schema(
            type=types.Type.STRING,
            enum=["menu", "sign", "document", "label", "unknown"],
            description="Classification of the dominant text type.",
        ),
        "items": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=types.Type.STRING),
            description="Parsed items (menu items with prices, sign lines, etc).",
        ),
        "confidence": types.Schema(
            type=types.Type.NUMBER,
            description="Confidence score from 0.0 to 1.0.",
        ),
    },
    required=["text", "text_type", "items", "confidence"],
)

# ---------------------------------------------------------------------------
# Empty / fallback result
# ---------------------------------------------------------------------------

_EMPTY_RESULT: dict[str, Any] = {
    "text": "",
    "text_type": "unknown",
    "items": [],
    "confidence": 0.0,
}

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


async def extract_text(
    image_base64: str,
    context_hint: str = "",
    safety_only: bool = False,
) -> dict[str, Any]:
    """Extract text from an image using Gemini Flash OCR.

    Args:
        image_base64: Base64-encoded image data (JPEG or PNG).
        context_hint: Optional context about the user's situation
            (e.g., "user is at a restaurant") to help focus extraction.
        safety_only: When True, only detect safety-critical text
            (warnings, traffic signs, danger labels) using low resolution.

    Returns:
        Structured dict with text, text_type, items, confidence.
        Returns empty result on failure (never raises).
    """
    try:
        image_bytes = base64.b64decode(image_base64)
    except Exception:
        logger.error("Failed to decode base64 image data")
        return dict(_EMPTY_RESULT)

    if safety_only:
        system_prompt = _SAFETY_SYSTEM_PROMPT
        user_message = (
            "Identify any warning signs, traffic signs, danger labels, "
            "or safety-critical text visible in this image. "
            "If none, respond with empty string."
        )
        media_res = types.MediaResolution.MEDIA_RESOLUTION_LOW
    else:
        system_prompt = _SYSTEM_PROMPT
        user_message = "Extract all visible text from this image."
        if context_hint:
            user_message += f" Context: {context_hint}"
        media_res = types.MediaResolution.MEDIA_RESOLUTION_MEDIUM

    response = None
    try:
        client = _get_client()
        response = await client.aio.models.generate_content(
            model=OCR_MODEL,
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
                media_resolution=media_res,
                response_mime_type="application/json",
                response_schema=_RESPONSE_SCHEMA,
                temperature=0.1,
            ),
        )

        if not response.text:
            logger.warning("OCR model returned empty response")
            return dict(_EMPTY_RESULT)

        result = json.loads(response.text)

        # Ensure all expected keys exist
        for key, default_val in _EMPTY_RESULT.items():
            if key not in result:
                result[key] = default_val

        return result

    except json.JSONDecodeError:
        raw = response.text if response else "<no response>"
        logger.error("Failed to parse OCR model JSON response: %s", raw)
        return dict(_EMPTY_RESULT)
    except Exception:
        logger.exception("OCR extraction failed")
        return dict(_EMPTY_RESULT)
