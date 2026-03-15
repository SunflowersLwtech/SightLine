"""SightLine emergency help tool.

Provides emergency assistance for blind users: finds nearest emergency
services (hospital, police, fire station), provides country-specific
emergency numbers, and generates a shareable Plus Code location.

Behavior mode: INTERRUPT — always delivered immediately (safety-critical).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import googlemaps

logger = logging.getLogger("sightline.tools.emergency")

# ---------------------------------------------------------------------------
# Client singleton (reuses pattern from navigation.py)
# ---------------------------------------------------------------------------

_client: googlemaps.Client | None = None


def _get_client() -> googlemaps.Client:
    """Return a lazily-initialised Google Maps client."""
    global _client
    if _client is None:
        api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_MAPS_API_KEY environment variable not set")
        _client = googlemaps.Client(
            key=api_key,
            connect_timeout=5,
            read_timeout=5,
        )
    return _client


# ---------------------------------------------------------------------------
# Country → emergency number mapping
# ---------------------------------------------------------------------------

_EMERGENCY_NUMBERS: dict[str, dict[str, str]] = {
    "US": {"general": "911", "police": "911", "fire": "911", "ambulance": "911"},
    "CA": {"general": "911", "police": "911", "fire": "911", "ambulance": "911"},
    "GB": {"general": "999", "police": "999", "fire": "999", "ambulance": "999"},
    "AU": {"general": "000", "police": "000", "fire": "000", "ambulance": "000"},
    "EU": {"general": "112", "police": "112", "fire": "112", "ambulance": "112"},
    "DE": {"general": "112", "police": "110", "fire": "112", "ambulance": "112"},
    "FR": {"general": "112", "police": "17", "fire": "18", "ambulance": "15"},
    "JP": {"general": "110", "police": "110", "fire": "119", "ambulance": "119"},
    "CN": {"general": "110", "police": "110", "fire": "119", "ambulance": "120"},
    "KR": {"general": "112", "police": "112", "fire": "119", "ambulance": "119"},
    "IN": {"general": "112", "police": "100", "fire": "101", "ambulance": "102"},
    "BR": {"general": "190", "police": "190", "fire": "193", "ambulance": "192"},
    "MX": {"general": "911", "police": "911", "fire": "911", "ambulance": "911"},
    "NZ": {"general": "111", "police": "111", "fire": "111", "ambulance": "111"},
    "SG": {"general": "999", "police": "999", "fire": "995", "ambulance": "995"},
    "HK": {"general": "999", "police": "999", "fire": "999", "ambulance": "999"},
    "TW": {"general": "110", "police": "110", "fire": "119", "ambulance": "119"},
}

# Default for unknown countries — 112 is the international emergency number
_DEFAULT_EMERGENCY = {"general": "112", "police": "112", "fire": "112", "ambulance": "112"}


def _get_emergency_numbers(country_code: str) -> dict[str, str]:
    """Get emergency numbers for a country code."""
    code = country_code.upper().strip()
    # Check EU members that use 112 as primary
    eu_countries = {"AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI",
                    "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
                    "PL", "PT", "RO", "SK", "SI", "ES", "SE"}
    if code in _EMERGENCY_NUMBERS:
        return _EMERGENCY_NUMBERS[code]
    if code in eu_countries:
        return _EMERGENCY_NUMBERS["EU"]
    return _DEFAULT_EMERGENCY


# ---------------------------------------------------------------------------
# Plus Code generation
# ---------------------------------------------------------------------------

def _to_plus_code(lat: float, lng: float) -> str:
    """Convert lat/lng to an Open Location Code (Plus Code).

    Uses a simple implementation that generates a 10-character Plus Code.
    """
    try:
        # Use the openlocationcode library if available
        from openlocationcode import openlocationcode as olc
        return olc.encode(lat, lng, codeLength=10)
    except ImportError:
        # Fallback: return coordinates formatted for sharing
        return f"{lat:.6f},{lng:.6f}"


# ---------------------------------------------------------------------------
# Emergency type → Places API query mapping
# ---------------------------------------------------------------------------

_EMERGENCY_PLACE_QUERIES: dict[str, str] = {
    "medical": "hospital emergency room",
    "police": "police station",
    "fire": "fire station",
    "general": "hospital emergency room",
}


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------


def get_emergency_help(
    emergency_type: str = "general",
    lat: float = 0.0,
    lng: float = 0.0,
) -> dict[str, Any]:
    """Find nearest emergency services and provide emergency contact info.

    Args:
        emergency_type: Type of emergency — "medical", "police", "fire", or "general".
        lat: User's current latitude (auto-injected from GPS).
        lng: User's current longitude (auto-injected from GPS).

    Returns:
        Dict with nearest_services, emergency_numbers, user_location_code,
        and a spoken summary.
    """
    emergency_type = (emergency_type or "general").strip().lower()
    if emergency_type not in _EMERGENCY_PLACE_QUERIES:
        emergency_type = "general"

    # Step 1: Reverse geocode to get country code
    country_code = "US"  # Default fallback
    try:
        client = _get_client()
        geocode_results = client.reverse_geocode((lat, lng))
        if geocode_results:
            for component in geocode_results[0].get("address_components", []):
                if "country" in component.get("types", []):
                    country_code = component.get("short_name", "US")
                    break
    except Exception:
        logger.warning("Failed to reverse geocode for emergency, using default country")

    # Step 2: Get emergency numbers
    emergency_numbers = _get_emergency_numbers(country_code)

    # Step 3: Find nearest emergency services using Places Nearby Search
    nearest_services: list[dict[str, Any]] = []
    place_query = _EMERGENCY_PLACE_QUERIES[emergency_type]
    try:
        client = _get_client()
        places_result = client.places_nearby(
            location=(lat, lng),
            keyword=place_query,
            rank_by="distance",
        )
        for place in places_result.get("results", [])[:3]:
            place_loc = place.get("geometry", {}).get("location", {})
            place_lat = place_loc.get("lat", 0)
            place_lng = place_loc.get("lng", 0)

            # Calculate approximate walking distance (rough estimate)
            import math
            dlat = math.radians(place_lat - lat)
            dlng = math.radians(place_lng - lng)
            a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat)) * math.cos(math.radians(place_lat)) * math.sin(dlng / 2) ** 2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            distance_m = 6371000 * c  # Earth radius in meters
            walk_min = max(1, int(distance_m / 80))  # ~80m/min walking

            nearest_services.append({
                "name": place.get("name", "Unknown"),
                "address": place.get("vicinity", ""),
                "distance_meters": int(distance_m),
                "walking_minutes": walk_min,
                "open_now": place.get("opening_hours", {}).get("open_now"),
            })
    except Exception:
        logger.warning("Failed to search for nearby emergency services")

    # Step 4: Generate Plus Code for location sharing
    location_code = _to_plus_code(lat, lng)

    # Step 5: Build spoken summary
    primary_number = emergency_numbers.get(emergency_type, emergency_numbers["general"])
    summary_parts = [f"Emergency number: {primary_number}."]
    if nearest_services:
        svc = nearest_services[0]
        summary_parts.append(
            f"Nearest: {svc['name']}, {svc['distance_meters']} meters away, "
            f"about {svc['walking_minutes']} minute walk."
        )
    summary_parts.append(f"Your location code: {location_code}")

    return {
        "success": True,
        "emergency_type": emergency_type,
        "country_code": country_code,
        "emergency_numbers": emergency_numbers,
        "nearest_services": nearest_services,
        "user_location_code": location_code,
        "user_coordinates": {"lat": lat, "lng": lng},
        "summary": " ".join(summary_parts),
    }


# ---------------------------------------------------------------------------
# ADK FunctionDeclaration for Gemini Live API
# ---------------------------------------------------------------------------

EMERGENCY_TOOL_DECLARATIONS = [
    {
        "name": "get_emergency_help",
        "description": (
            "Find nearest emergency services (hospital, police, fire station) and "
            "provide country-specific emergency phone numbers. Also generates a "
            "shareable location code. Use when the user says 'help', 'emergency', "
            "'I need help', 'call 911', or expresses urgent distress. "
            "Behavior: INTERRUPT — always delivered immediately."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "emergency_type": {
                    "type": "string",
                    "enum": ["medical", "police", "fire", "general"],
                    "description": "Type of emergency. Default 'general' if unclear.",
                },
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

EMERGENCY_FUNCTIONS = {
    "get_emergency_help": get_emergency_help,
}
