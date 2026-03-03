"""SightLine OpenStreetMap accessibility data tool.

Queries the Overpass API for accessibility-related features near the user's
location: tactile paving, wheelchair ramps, audio traffic signals, sidewalk
surfaces, crossings, and more.

Behavior mode: WHEN_IDLE — results delivered after speech finishes.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger("sightline.tools.accessibility")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"
_REQUEST_TIMEOUT = 10.0  # seconds
_CACHE_TTL = 300  # 5 minutes

# ---------------------------------------------------------------------------
# Location cache (avoid repeated queries for the same area)
# ---------------------------------------------------------------------------

_cache: dict[str, tuple[float, dict]] = {}  # key → (timestamp, result)


def _cache_key(lat: float, lng: float, radius: int) -> str:
    """Round coordinates to ~100m grid for cache grouping."""
    return f"{lat:.3f},{lng:.3f},{radius}"


# ---------------------------------------------------------------------------
# Overpass query
# ---------------------------------------------------------------------------

_OVERPASS_QUERY_TEMPLATE = """
[out:json][timeout:10];
(
  // Tactile paving
  node["tactile_paving"](around:{radius},{lat},{lng});
  way["tactile_paving"](around:{radius},{lat},{lng});

  // Wheelchair accessibility
  node["wheelchair"](around:{radius},{lat},{lng});
  way["wheelchair"](around:{radius},{lat},{lng});

  // Sidewalk surface quality
  way["sidewalk:surface"](around:{radius},{lat},{lng});
  way["surface"]["highway"="footway"](around:{radius},{lat},{lng});

  // Audio traffic signals
  node["traffic_signals:sound"](around:{radius},{lat},{lng});
  node["crossing"]["highway"="traffic_signals"](around:{radius},{lat},{lng});

  // Pedestrian crossings
  node["crossing"]["highway"="crossing"](around:{radius},{lat},{lng});

  // Steps and ramps
  way["highway"="steps"](around:{radius},{lat},{lng});
  node["ramp"](around:{radius},{lat},{lng});
  way["ramp"](around:{radius},{lat},{lng});

  // Handrails
  way["handrail"](around:{radius},{lat},{lng});
);
out body;
"""


def _classify_feature(element: dict) -> dict[str, Any] | None:
    """Classify an OSM element into an accessibility feature."""
    tags = element.get("tags", {})
    if not tags:
        return None

    feature: dict[str, Any] = {
        "osm_id": element.get("id"),
        "type": element.get("type"),
    }

    # Add coordinates if available
    if "lat" in element and "lon" in element:
        feature["lat"] = element["lat"]
        feature["lng"] = element["lon"]

    # Classify by tag priority
    if "tactile_paving" in tags:
        feature["category"] = "tactile_paving"
        feature["value"] = tags["tactile_paving"]  # yes/no/incorrect
        feature["description"] = (
            "Tactile paving present" if tags["tactile_paving"] == "yes"
            else f"Tactile paving: {tags['tactile_paving']}"
        )

    elif "traffic_signals:sound" in tags:
        feature["category"] = "audio_signal"
        feature["value"] = tags["traffic_signals:sound"]
        feature["description"] = "Audio traffic signal available"

    elif tags.get("highway") == "steps":
        feature["category"] = "stairs"
        step_count = tags.get("step_count", "unknown")
        handrail = tags.get("handrail", "unknown")
        feature["description"] = f"Stairs (steps: {step_count}, handrail: {handrail})"

    elif "ramp" in tags:
        feature["category"] = "ramp"
        feature["value"] = tags["ramp"]
        feature["description"] = (
            "Ramp available" if tags["ramp"] == "yes"
            else f"Ramp: {tags['ramp']}"
        )

    elif "wheelchair" in tags:
        feature["category"] = "wheelchair_access"
        feature["value"] = tags["wheelchair"]
        name = tags.get("name", "")
        feature["description"] = (
            f"{'(' + name + ') ' if name else ''}"
            f"Wheelchair: {tags['wheelchair']}"
        )

    elif tags.get("highway") == "crossing" or "crossing" in tags:
        feature["category"] = "crossing"
        crossing_type = tags.get("crossing", "unmarked")
        feature["description"] = f"Pedestrian crossing ({crossing_type})"

    elif "handrail" in tags:
        feature["category"] = "handrail"
        feature["value"] = tags["handrail"]
        feature["description"] = (
            "Handrail present" if tags["handrail"] == "yes"
            else f"Handrail: {tags['handrail']}"
        )

    elif "sidewalk:surface" in tags or (
        "surface" in tags and tags.get("highway") == "footway"
    ):
        surface = tags.get("sidewalk:surface") or tags.get("surface", "unknown")
        feature["category"] = "surface"
        feature["value"] = surface
        feature["description"] = f"Walking surface: {surface}"

    else:
        return None

    return feature


def _build_summary(features: list[dict]) -> str:
    """Build a human-readable summary of accessibility features."""
    if not features:
        return "No specific accessibility features found in this area."

    counts: dict[str, int] = {}
    for f in features:
        cat = f.get("category", "other")
        counts[cat] = counts.get(cat, 0) + 1

    parts = []
    if counts.get("tactile_paving"):
        parts.append(f"{counts['tactile_paving']} tactile paving locations")
    if counts.get("audio_signal"):
        parts.append(f"{counts['audio_signal']} audio traffic signals")
    if counts.get("crossing"):
        parts.append(f"{counts['crossing']} pedestrian crossings")
    if counts.get("stairs"):
        parts.append(f"{counts['stairs']} stairways")
    if counts.get("ramp"):
        parts.append(f"{counts['ramp']} ramps")
    if counts.get("wheelchair_access"):
        parts.append(f"{counts['wheelchair_access']} wheelchair-accessible locations")
    if counts.get("handrail"):
        parts.append(f"{counts['handrail']} handrails")
    if counts.get("surface"):
        parts.append(f"{counts['surface']} surface-tagged paths")

    return "Found: " + ", ".join(parts) + "."


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------


def get_accessibility_info(
    lat: float,
    lng: float,
    radius: int = 200,
) -> dict[str, Any]:
    """Query OpenStreetMap for accessibility features near a location.

    Args:
        lat: Latitude in decimal degrees.
        lng: Longitude in decimal degrees.
        radius: Search radius in meters (default 200, max 500).

    Returns:
        Dict with ``features`` list, ``summary`` text, ``count``,
        and search ``area`` description.
    """
    radius = max(50, min(500, radius))

    # Check cache
    key = _cache_key(lat, lng, radius)
    if key in _cache:
        ts, cached = _cache[key]
        if time.time() - ts < _CACHE_TTL:
            logger.debug("Returning cached accessibility data for %s", key)
            return cached

    try:
        query = _OVERPASS_QUERY_TEMPLATE.format(
            lat=lat, lng=lng, radius=radius,
        )

        with httpx.Client(timeout=_REQUEST_TIMEOUT) as client:
            resp = client.post(OVERPASS_API_URL, data={"data": query})
            resp.raise_for_status()

        data = resp.json()
        elements = data.get("elements", [])

        # Classify each element
        features = []
        seen_ids = set()
        for el in elements:
            osm_id = el.get("id")
            if osm_id in seen_ids:
                continue
            seen_ids.add(osm_id)

            feature = _classify_feature(el)
            if feature:
                features.append(feature)

        summary = _build_summary(features)

        result: dict[str, Any] = {
            "success": True,
            "features": features[:50],  # Cap to avoid huge payloads
            "summary": summary,
            "count": len(features),
            "area": f"{radius}m radius around ({lat:.4f}, {lng:.4f})",
        }

        # Update cache
        _cache[key] = (time.time(), result)

        return result

    except httpx.TimeoutException:
        logger.warning("Overpass API timeout for (%s, %s)", lat, lng)
        return {
            "success": False,
            "error": "Accessibility data request timed out. Try again shortly.",
            "features": [],
            "summary": "Data temporarily unavailable.",
            "count": 0,
        }

    except Exception as e:
        logger.exception("get_accessibility_info failed: %s", e)
        return {
            "success": False,
            "error": f"Failed to query accessibility data: {e}",
            "features": [],
            "summary": "Data unavailable.",
            "count": 0,
        }


# ---------------------------------------------------------------------------
# ADK FunctionDeclaration for Gemini Live API
# ---------------------------------------------------------------------------

ACCESSIBILITY_TOOL_DECLARATIONS = [
    {
        "name": "get_accessibility_info",
        "description": (
            "Query nearby accessibility features from OpenStreetMap: tactile paving, "
            "wheelchair ramps, audio traffic signals, pedestrian crossings, stairs, "
            "handrails, and sidewalk surface quality. Use when the user asks about "
            "accessibility of their surroundings or when navigating in unfamiliar areas. "
            "Behavior: WHEN_IDLE."
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
                "radius": {
                    "type": "integer",
                    "description": "Search radius in meters (default 200, max 500)",
                },
            },
            "required": ["lat", "lng"],
        },
    },
]

ACCESSIBILITY_FUNCTIONS = {
    "get_accessibility_info": get_accessibility_info,
}
