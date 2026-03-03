"""SightLine What3Words tool.

Provides resolution of what3words three-word addresses to GPS coordinates
and conversion of GPS coordinates to three-word addresses.  Useful for
precise micro-location sharing with blind/low-vision users.

Behavior mode: WHEN_IDLE — results delivered after speech finishes.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import what3words

logger = logging.getLogger("sightline.tools.what3words")

# ---------------------------------------------------------------------------
# Client singleton
# ---------------------------------------------------------------------------

_client: what3words.Geocoder | None = None


def _get_client() -> what3words.Geocoder:
    """Return a lazily-initialised What3Words geocoder."""
    global _client
    if _client is None:
        api_key = os.environ.get("WHAT3WORDS_API_KEY")
        if not api_key:
            raise RuntimeError("WHAT3WORDS_API_KEY environment variable not set")
        _client = what3words.Geocoder(api_key)
    return _client


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------


def resolve_what3words(words: str) -> dict[str, Any]:
    """Resolve a what3words three-word address to GPS coordinates.

    Args:
        words: A what3words address (e.g. "filled.count.soap").
              The leading ``///`` prefix is stripped automatically.

    Returns:
        Dict with ``latitude``, ``longitude``, ``nearest_place``,
        ``country``, and the original ``words``.
    """
    try:
        # Strip common prefixes
        clean = words.strip().lstrip("/").strip()

        client = _get_client()
        result = client.convert_to_coordinates(clean)

        if "error" in result:
            code = result["error"].get("code", "")
            message = result["error"].get("message", "Unknown error")
            logger.warning("What3Words error %s: %s", code, message)
            return {
                "success": False,
                "error": message,
                "words": clean,
            }

        coords = result.get("coordinates", {})
        return {
            "success": True,
            "latitude": coords.get("lat", 0.0),
            "longitude": coords.get("lng", 0.0),
            "nearest_place": result.get("nearestPlace", ""),
            "country": result.get("country", ""),
            "words": result.get("words", clean),
        }

    except Exception as e:
        logger.exception("resolve_what3words failed: %s", e)
        return {
            "success": False,
            "error": f"Failed to resolve what3words address: {e}",
            "words": words,
        }


def convert_to_what3words(lat: float, lng: float) -> dict[str, Any]:
    """Convert GPS coordinates to a what3words three-word address.

    Args:
        lat: Latitude in decimal degrees.
        lng: Longitude in decimal degrees.

    Returns:
        Dict with ``words``, ``nearest_place``, ``country``,
        and the original coordinates.
    """
    try:
        client = _get_client()
        result = client.convert_to_3wa(what3words.Coordinates(lat, lng))

        if "error" in result:
            code = result["error"].get("code", "")
            message = result["error"].get("message", "Unknown error")
            logger.warning("What3Words error %s: %s", code, message)
            return {
                "success": False,
                "error": message,
                "latitude": lat,
                "longitude": lng,
            }

        return {
            "success": True,
            "words": result.get("words", ""),
            "nearest_place": result.get("nearestPlace", ""),
            "country": result.get("country", ""),
            "latitude": lat,
            "longitude": lng,
        }

    except Exception as e:
        logger.exception("convert_to_what3words failed: %s", e)
        return {
            "success": False,
            "error": f"Failed to convert coordinates: {e}",
            "latitude": lat,
            "longitude": lng,
        }


# ---------------------------------------------------------------------------
# ADK FunctionDeclaration for Gemini Live API
# ---------------------------------------------------------------------------

WHAT3WORDS_TOOL_DECLARATIONS = [
    {
        "name": "resolve_what3words",
        "description": (
            "Resolve a what3words three-word address (e.g. 'filled.count.soap') "
            "to GPS coordinates. Use when the user provides a what3words address "
            "or mentions '///' prefix. Returns latitude, longitude, and nearest place. "
            "Behavior: WHEN_IDLE."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "words": {
                    "type": "string",
                    "description": (
                        "The what3words address to resolve "
                        "(e.g. 'filled.count.soap' or '///filled.count.soap')"
                    ),
                },
            },
            "required": ["words"],
        },
    },
    {
        "name": "convert_to_what3words",
        "description": (
            "Convert the user's current GPS coordinates to a what3words address. "
            "Use when the user asks 'what is my what3words address?' or wants to "
            "share their precise location. Behavior: WHEN_IDLE."
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

WHAT3WORDS_FUNCTIONS = {
    "resolve_what3words": resolve_what3words,
    "convert_to_what3words": convert_to_what3words,
}
