"""SightLine Plus Codes tool — offline replacement for What3Words.

Uses Google Open Location Code (Plus Codes) for precise micro-location
encoding/decoding.  Fully offline — no API calls needed.

Code length 11 gives ~3m precision, matching W3W's 3m×3m grid.
"""

from __future__ import annotations

import logging
from typing import Any

from openlocationcode import openlocationcode as olc

logger = logging.getLogger("sightline.tools.plus_codes")

# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------


def resolve_plus_code(code: str) -> dict[str, Any]:
    """Decode a Plus Code to GPS coordinates.

    Args:
        code: A full or short Plus Code (e.g. "849VQJQ5+JQ").

    Returns:
        Dict with ``success``, ``latitude``, ``longitude``, ``code``.
    """
    try:
        clean = code.strip()
        if not olc.isValid(clean):
            return {
                "success": False,
                "error": f"Invalid Plus Code: '{clean}'",
                "code": clean,
            }

        if not olc.isFull(clean):
            return {
                "success": False,
                "error": (
                    f"Short Plus Code '{clean}' requires a reference location. "
                    "Please provide a full Plus Code (e.g. '849VQJQ5+JQ')."
                ),
                "code": clean,
            }

        area = olc.decode(clean)
        lat = (area.latitudeLo + area.latitudeHi) / 2
        lng = (area.longitudeLo + area.longitudeHi) / 2

        return {
            "success": True,
            "latitude": round(lat, 7),
            "longitude": round(lng, 7),
            "code": clean,
        }

    except Exception as e:
        logger.exception("resolve_plus_code failed: %s", e)
        return {
            "success": False,
            "error": f"Failed to resolve Plus Code: {e}",
            "code": code,
        }


def convert_to_plus_code(lat: float, lng: float) -> dict[str, Any]:
    """Encode GPS coordinates to a Plus Code.

    Args:
        lat: Latitude in decimal degrees.
        lng: Longitude in decimal degrees.

    Returns:
        Dict with ``success``, ``code``, ``latitude``, ``longitude``.
    """
    try:
        code = olc.encode(lat, lng, codeLength=11)
        return {
            "success": True,
            "code": code,
            "latitude": lat,
            "longitude": lng,
        }

    except Exception as e:
        logger.exception("convert_to_plus_code failed: %s", e)
        return {
            "success": False,
            "error": f"Failed to encode Plus Code: {e}",
            "latitude": lat,
            "longitude": lng,
        }


# ---------------------------------------------------------------------------
# ADK FunctionDeclaration for Gemini Live API
# ---------------------------------------------------------------------------

PLUS_CODES_TOOL_DECLARATIONS = [
    {
        "name": "resolve_plus_code",
        "description": (
            "Decode a Google Plus Code (e.g. '849VQJQ5+JQ') to GPS coordinates. "
            "Plus Codes are short alphanumeric codes that represent a location. "
            "Use when the user provides a Plus Code. Behavior: WHEN_IDLE."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Plus Code to decode (e.g. '849VQJQ5+JQ')",
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "convert_to_plus_code",
        "description": (
            "Convert the user's current GPS coordinates to a Google Plus Code. "
            "Use when the user asks for a location code or wants to share their "
            "precise position. Behavior: WHEN_IDLE."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "lat": {
                    "type": "number",
                    "description": "Latitude (auto-injected from GPS)",
                },
                "lng": {
                    "type": "number",
                    "description": "Longitude (auto-injected from GPS)",
                },
            },
            "required": ["lat", "lng"],
        },
    },
]

PLUS_CODES_FUNCTIONS = {
    "resolve_plus_code": resolve_plus_code,
    "convert_to_plus_code": convert_to_plus_code,
}
