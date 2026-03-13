"""SightLine OCR Sub-Agent.

Asynchronous text extraction using Gemini Flash (gemini-3-flash-preview)
optimized for menus, signage, documents, and labels. Uses the free-tier
Gemini Developer API.

The context_hint parameter helps the model focus on the most relevant
text type (e.g., "user is at a restaurant" prioritizes menu items).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from typing import Any

from google import genai
from google.genai import types

from gemini_client import get_gemini_api_client

logger = logging.getLogger("sightline.ocr_agent")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OCR_MODEL = os.getenv("GEMINI_FLASH_MODEL", "gemini-3-flash-preview")
_REQUEST_TIMEOUT_SEC = float(os.getenv("OCR_REQUEST_TIMEOUT_SEC", "8"))

_SYSTEM_PROMPT = """\
You are a text extraction system for a blind user. Your job is to read ALL \
visible text in the image accurately.

Rules:
1. Extract every piece of readable text — signs, menus, labels, documents, \
   screens, handwriting.
2. Classify the text type: "menu", "sign", "document", "label", or "unknown".
3. For menus: parse into individual items with prices when visible. Format \
   each item as "Item Name - $Price" or just "Item Name" if no price. \
   Group items by category when clear (e.g. appetizers, mains, drinks, desserts).
4. For signs: preserve the exact wording.
5. For documents: maintain reading order (top to bottom, left to right).
6. Report confidence based on text clarity (0.0 = unreadable, 1.0 = crystal clear).
7. If no text is visible, return empty results with confidence 0.0.

Text priority (extract all, but rank by importance):
1. Safety-critical: warnings, caution signs, traffic signals, hazard labels.
2. Actionable: prices, opening hours, directions, instructions, dosage info.
3. Informational: names, titles, descriptions, news headlines.
4. Decorative: brand slogans, decorative quotes, background text.

Multi-language handling:
- Always extract text in its original language first.
- If the text is not in the user's language, add a brief translation \
  in parentheses: "注意安全 (Caution: be safe)".
- For mixed-language text, preserve both languages.

Unclear text:
- Mark partially obscured or blurry characters with [?]: "Pha[?]macy".
- If a word is completely unreadable, note "[unreadable]" in its position.

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

def _get_client() -> genai.Client:
    """Return the shared Gemini client with OCR-specific error text."""
    try:
        return get_gemini_api_client()
    except RuntimeError as exc:
        raise RuntimeError(
            "OCR agent requires a Gemini API key. "
            "Set _GOOGLE_AI_API_KEY or GOOGLE_API_KEY environment variable."
        ) from exc


def _image_part(image_bytes: bytes) -> Any:
    """Build a bytes part, tolerating lightweight test doubles."""
    if hasattr(types.Part, "from_bytes"):
        return types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
    return types.Part(text="[image bytes omitted in test stub]")


def _text_part(text: str) -> Any:
    if hasattr(types.Part, "from_text"):
        return types.Part.from_text(text=text)
    return types.Part(text=text)


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
            user_message += f" Context: {context_hint}. Prioritize text most relevant to this context."
        media_res = types.MediaResolution.MEDIA_RESOLUTION_MEDIUM

    response = None
    try:
        client = _get_client()
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=OCR_MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            _image_part(image_bytes),
                            _text_part(user_message),
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
            ),
            timeout=_REQUEST_TIMEOUT_SEC,
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
    except TimeoutError:
        logger.warning("OCR extraction timed out after %.1fs", _REQUEST_TIMEOUT_SEC)
        return dict(_EMPTY_RESULT)
    except Exception:
        logger.exception("OCR extraction failed")
        return dict(_EMPTY_RESULT)
