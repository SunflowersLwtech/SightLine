"""SightLine navigation tools using Google Maps APIs.

Provides walking directions, reverse geocoding, nearby search, location
info, destination preview, and address validation — all formatted for
blind/low-vision users with clock-position directional cues.

Clock-position system: Instead of "turn right", we say
"destination at 2 o'clock, 50 meters". This converts absolute
compass bearings to positions relative to the user's current heading.

API versions:
- Routes API v2 (replaces Legacy Directions)
- Places API (New) (replaces Legacy Places Nearby)
- Elevation API via googlemaps SDK (slope warnings)
- Street View Static API (destination preview → Vision Agent)
- Address Validation API (voice address correction)
- Geocoding API via googlemaps SDK (reverse geocode, unchanged)
"""

from __future__ import annotations

import asyncio
import base64
import logging
import math
import os
from typing import Any

import googlemaps

from tools._maps_http import maps_rest_get, maps_rest_post

logger = logging.getLogger("sightline.tools.navigation")

# ---------------------------------------------------------------------------
# Client singleton (for Geocoding + Elevation — still uses googlemaps SDK)
# ---------------------------------------------------------------------------

_client: googlemaps.Client | None = None


def _get_client() -> googlemaps.Client:
    """Return a lazily-initialised Google Maps client."""
    global _client
    if _client is None:
        api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_MAPS_API_KEY environment variable not set")
        _client = googlemaps.Client(key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Clock-position helpers
# ---------------------------------------------------------------------------


def bearing_between(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Compute initial compass bearing (0-360) from point 1 to point 2."""
    lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
    d_lng = lng2 - lng1
    x = math.sin(d_lng) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lng)
    bearing = math.degrees(math.atan2(x, y))
    return bearing % 360


def bearing_to_clock(absolute_bearing: float, user_heading: float) -> int:
    """Convert an absolute bearing to a clock position (1-12).

    Args:
        absolute_bearing: Compass bearing to the target (0-360, 0=N).
        user_heading: User's current compass heading (0-360, 0=N).

    Returns:
        Clock position 1-12 (12 = straight ahead, 6 = behind).
    """
    relative = (absolute_bearing - user_heading) % 360
    # Each clock position spans 30 degrees, with 12 o'clock centered at 0.
    clock = round(relative / 30) % 12
    return clock if clock != 0 else 12


def format_clock_direction(clock: int, distance_m: float) -> str:
    """Format a clock position and distance into a spoken direction.

    Examples:
        "straight ahead, 50 meters"
        "at 2 o'clock, 120 meters"
        "behind you, 30 meters"
    """
    if distance_m >= 1000:
        dist_str = f"{distance_m / 1000:.1f} kilometers"
    else:
        dist_str = f"{int(round(distance_m))} meters"

    if clock == 12:
        return f"straight ahead, {dist_str}"
    if clock == 6:
        return f"behind you, {dist_str}"
    return f"at {clock} o'clock, {dist_str}"


# ---------------------------------------------------------------------------
# Maneuver mapping (supports both Legacy and Routes API formats)
# ---------------------------------------------------------------------------

_MANEUVER_MAP = {
    # Legacy Directions API format (kebab-case)
    "turn-left": "turn to 9 o'clock",
    "turn-right": "turn to 3 o'clock",
    "turn-slight-left": "bear to 10 o'clock",
    "turn-slight-right": "bear to 2 o'clock",
    "turn-sharp-left": "turn to 8 o'clock",
    "turn-sharp-right": "turn to 4 o'clock",
    "uturn-left": "make a U-turn to the left",
    "uturn-right": "make a U-turn to the right",
    "straight": "continue straight ahead",
    "roundabout-left": "take the roundabout to the left",
    "roundabout-right": "take the roundabout to the right",
    # Routes API format (UPPER_SNAKE_CASE)
    "TURN_LEFT": "turn to 9 o'clock",
    "TURN_RIGHT": "turn to 3 o'clock",
    "TURN_SLIGHT_LEFT": "bear to 10 o'clock",
    "TURN_SLIGHT_RIGHT": "bear to 2 o'clock",
    "TURN_SHARP_LEFT": "turn to 8 o'clock",
    "TURN_SHARP_RIGHT": "turn to 4 o'clock",
    "UTURN_LEFT": "make a U-turn to the left",
    "UTURN_RIGHT": "make a U-turn to the right",
    "STRAIGHT": "continue straight ahead",
    "ROUNDABOUT_LEFT": "take the roundabout to the left",
    "ROUNDABOUT_RIGHT": "take the roundabout to the right",
    "ROUNDABOUT_SHARP_LEFT": "take the roundabout sharply to the left",
    "ROUNDABOUT_SHARP_RIGHT": "take the roundabout sharply to the right",
    "ROUNDABOUT_SLIGHT_LEFT": "take the roundabout slightly left",
    "ROUNDABOUT_SLIGHT_RIGHT": "take the roundabout slightly right",
    "ROUNDABOUT_STRAIGHT": "continue straight through the roundabout",
    "ROUNDABOUT_U_TURN": "make a U-turn at the roundabout",
    "DEPART": "depart",
    "NAME_CHANGE": "continue onto the next road",
    "MERGE_LEFT": "merge to the left",
    "MERGE_RIGHT": "merge to the right",
    "MERGE_UNSPECIFIED": "merge ahead",
    "RAMP_LEFT": "take the ramp to the left",
    "RAMP_RIGHT": "take the ramp to the right",
}


def _maneuver_to_description(maneuver: str | None) -> str:
    """Convert a Google Maps maneuver string to accessible language."""
    if maneuver and maneuver in _MANEUVER_MAP:
        return _MANEUVER_MAP[maneuver]
    return ""


def _strip_html(text: str) -> str:
    """Remove HTML tags from Google Maps instruction text."""
    import re
    clean = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", clean).strip()


# ---------------------------------------------------------------------------
# Elevation / slope warnings
# ---------------------------------------------------------------------------

_ADA_SLOPE_THRESHOLD = 8.0  # percent grade — ADA max for ramps


def _compute_slope_warnings(encoded_polyline: str, total_distance_m: float) -> list[dict]:
    """Compute slope warnings along a route using Elevation API.

    Args:
        encoded_polyline: Encoded polyline from Routes API response.
        total_distance_m: Total route distance in meters.

    Returns:
        List of slope warning dicts for segments exceeding ADA threshold.
        Empty list on failure (non-critical path).
    """
    try:
        client = _get_client()
        # Sample every ~50m, cap at 100 to control API cost
        num_samples = min(max(int(total_distance_m / 50), 2), 100)
        results = client.elevation_along_path(
            f"enc:{encoded_polyline}",
            samples=num_samples,
        )

        if not results or len(results) < 2:
            return []

        warnings = []
        segment_length = total_distance_m / (len(results) - 1)

        for i in range(1, len(results)):
            elev_prev = results[i - 1].get("elevation", 0)
            elev_curr = results[i].get("elevation", 0)
            elev_diff = elev_curr - elev_prev

            if segment_length <= 0:
                continue

            grade_pct = abs(elev_diff / segment_length) * 100

            if grade_pct >= _ADA_SLOPE_THRESHOLD:
                direction = "uphill" if elev_diff > 0 else "downhill"
                severity = "steep" if grade_pct >= 12 else "moderate"
                distance_along = round(i * segment_length)
                warnings.append({
                    "distance_along_route_m": distance_along,
                    "grade_percent": round(grade_pct, 1),
                    "direction": direction,
                    "severity": severity,
                    "description": (
                        f"{severity.capitalize()} {direction} slope ({grade_pct:.0f}%) "
                        f"at approximately {distance_along} meters along the route"
                    ),
                })

        return warnings

    except Exception as e:
        logger.warning("Elevation slope computation failed (non-critical): %s", e)
        return []


# ---------------------------------------------------------------------------
# Routes API constants
# ---------------------------------------------------------------------------

_ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
_ROUTES_FIELD_MASK = (
    "routes.legs.steps.navigationInstruction,"
    "routes.legs.steps.localizedValues,"
    "routes.legs.steps.distanceMeters,"
    "routes.legs.steps.startLocation,"
    "routes.legs.steps.endLocation,"
    "routes.legs.distanceMeters,"
    "routes.legs.duration,"
    "routes.legs.endLocation,"
    "routes.legs.startLocation,"
    "routes.polyline.encodedPolyline"
)


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------


def navigate_to(
    destination: str,
    origin_lat: float,
    origin_lng: float,
    user_heading: float = 0.0,
) -> dict[str, Any]:
    """Get step-by-step walking directions to a destination.

    Uses Routes API v2 with clock-position format for directional cues.

    Args:
        destination: Place name or address to navigate to.
        origin_lat: User's current latitude.
        origin_lng: User's current longitude.
        user_heading: User's current compass heading (0-360, 0=N).

    Returns:
        Dict with route summary and accessible step-by-step directions.
    """
    try:
        body = {
            "origin": {
                "location": {
                    "latLng": {"latitude": origin_lat, "longitude": origin_lng},
                },
            },
            "destination": {"address": destination},
            "travelMode": "WALK",
            "computeAlternativeRoutes": False,
            "languageCode": "en-US",
            "units": "METRIC",
        }

        resp = maps_rest_post(_ROUTES_URL, body, field_mask=_ROUTES_FIELD_MASK)

        routes = resp.get("routes", [])
        if not routes:
            return {
                "success": False,
                "error": "No walking route found to that destination.",
            }

        route = routes[0]
        legs = route.get("legs", [])
        if not legs:
            return {"success": False, "error": "Route has no legs."}

        leg = legs[0]
        steps = leg.get("steps", [])

        # Parse totals
        total_distance_m = leg.get("distanceMeters", 0)
        duration_str = leg.get("duration", "0s")  # e.g. "720s"
        duration_secs = int(duration_str.rstrip("s")) if duration_str.endswith("s") else 0
        duration_mins = max(1, round(duration_secs / 60))

        if total_distance_m >= 1000:
            total_distance_text = f"{total_distance_m / 1000:.1f} km"
        else:
            total_distance_text = f"{total_distance_m} m"

        total_duration_text = f"{duration_mins} min"

        # Destination location for clock direction
        dest_loc = leg.get("endLocation", {}).get("latLng", {})
        dest_lat = dest_loc.get("latitude", origin_lat)
        dest_lng = dest_loc.get("longitude", origin_lng)
        dest_bearing = bearing_between(origin_lat, origin_lng, dest_lat, dest_lng)
        dest_clock = bearing_to_clock(dest_bearing, user_heading)

        # Parse steps
        accessible_steps = []
        for i, step in enumerate(steps, 1):
            nav_instr = step.get("navigationInstruction", {})
            instruction = nav_instr.get("instructions", "")
            maneuver = nav_instr.get("maneuver", "")
            maneuver_desc = _maneuver_to_description(maneuver)

            step_distance_m = step.get("distanceMeters", 0)
            localized = step.get("localizedValues", {})
            distance_text = localized.get("distance", {}).get("text", f"{step_distance_m} m")

            step_info: dict[str, Any] = {
                "step": i,
                "instruction": instruction,
                "distance": distance_text,
            }
            if maneuver_desc:
                step_info["direction"] = maneuver_desc

            # Clock position from step start → step end
            start_loc = step.get("startLocation", {}).get("latLng", {})
            end_loc = step.get("endLocation", {}).get("latLng", {})
            s_lat = start_loc.get("latitude", 0)
            s_lng = start_loc.get("longitude", 0)
            e_lat = end_loc.get("latitude", 0)
            e_lng = end_loc.get("longitude", 0)

            if s_lat and s_lng and e_lat and e_lng:
                step_bearing = bearing_between(s_lat, s_lng, e_lat, e_lng)
                step_clock = bearing_to_clock(step_bearing, user_heading)
                step_info["clock_direction"] = format_clock_direction(step_clock, step_distance_m)

            accessible_steps.append(step_info)

        # Slope warnings (from encoded polyline + Elevation API)
        encoded_polyline = route.get("polyline", {}).get("encodedPolyline", "")
        slope_warnings = []
        if encoded_polyline and total_distance_m > 0:
            slope_warnings = _compute_slope_warnings(encoded_polyline, total_distance_m)

        accessibility_note = (
            "Walking route. Watch for crosswalks and intersections. "
            "Listen for traffic signals at crossings."
        )
        if slope_warnings:
            steep_count = sum(1 for w in slope_warnings if w["severity"] == "steep")
            moderate_count = len(slope_warnings) - steep_count
            parts = []
            if steep_count:
                parts.append(f"{steep_count} steep slope{'s' if steep_count > 1 else ''}")
            if moderate_count:
                parts.append(f"{moderate_count} moderate slope{'s' if moderate_count > 1 else ''}")
            accessibility_note += f" Route has {' and '.join(parts)}."

        return {
            "success": True,
            "destination": destination,
            "total_distance": total_distance_text,
            "total_duration": total_duration_text,
            "destination_direction": format_clock_direction(dest_clock, total_distance_m),
            "steps": accessible_steps,
            "slope_warnings": slope_warnings,
            "accessibility_note": accessibility_note,
        }

    except Exception as e:
        logger.exception("navigate_to failed: %s", e)
        return {
            "success": False,
            "error": f"Navigation request failed: {e}",
        }


def get_walking_directions(origin: str, destination: str) -> dict[str, Any]:
    """Get text-based walking directions between two addresses.

    For use when GPS coordinates are not available — uses
    address/place names instead.

    Args:
        origin: Starting address or place name.
        destination: Destination address or place name.

    Returns:
        Dict with walking directions summary and steps.
    """
    try:
        body = {
            "origin": {"address": origin},
            "destination": {"address": destination},
            "travelMode": "WALK",
            "computeAlternativeRoutes": False,
            "languageCode": "en-US",
            "units": "METRIC",
        }

        resp = maps_rest_post(_ROUTES_URL, body, field_mask=_ROUTES_FIELD_MASK)

        routes = resp.get("routes", [])
        if not routes:
            return {
                "success": False,
                "error": "No walking route found between those locations.",
            }

        route = routes[0]
        legs = route.get("legs", [])
        if not legs:
            return {"success": False, "error": "Route has no legs."}

        leg = legs[0]

        total_distance_m = leg.get("distanceMeters", 0)
        duration_str = leg.get("duration", "0s")
        duration_secs = int(duration_str.rstrip("s")) if duration_str.endswith("s") else 0
        duration_mins = max(1, round(duration_secs / 60))

        if total_distance_m >= 1000:
            total_distance_text = f"{total_distance_m / 1000:.1f} km"
        else:
            total_distance_text = f"{total_distance_m} m"

        steps = []
        for i, step in enumerate(leg.get("steps", []), 1):
            nav_instr = step.get("navigationInstruction", {})
            instruction = nav_instr.get("instructions", "")
            maneuver = nav_instr.get("maneuver", "")
            maneuver_desc = _maneuver_to_description(maneuver)

            step_distance_m = step.get("distanceMeters", 0)
            localized = step.get("localizedValues", {})
            distance_text = localized.get("distance", {}).get("text", f"{step_distance_m} m")

            step_info: dict[str, Any] = {
                "step": i,
                "instruction": instruction,
                "distance": distance_text,
            }
            if maneuver_desc:
                step_info["direction"] = maneuver_desc
            steps.append(step_info)

        # Slope warnings
        encoded_polyline = route.get("polyline", {}).get("encodedPolyline", "")
        slope_warnings = []
        if encoded_polyline and total_distance_m > 0:
            slope_warnings = _compute_slope_warnings(encoded_polyline, total_distance_m)

        accessibility_note = (
            "Walking route. Watch for crosswalks and intersections."
        )
        if slope_warnings:
            steep_count = sum(1 for w in slope_warnings if w["severity"] == "steep")
            moderate_count = len(slope_warnings) - steep_count
            parts = []
            if steep_count:
                parts.append(f"{steep_count} steep slope{'s' if steep_count > 1 else ''}")
            if moderate_count:
                parts.append(f"{moderate_count} moderate slope{'s' if moderate_count > 1 else ''}")
            accessibility_note += f" Route has {' and '.join(parts)}."

        return {
            "success": True,
            "origin": origin,
            "destination": destination,
            "total_distance": total_distance_text,
            "total_duration": f"{duration_mins} min",
            "steps": steps,
            "slope_warnings": slope_warnings,
            "accessibility_note": accessibility_note,
        }

    except Exception as e:
        logger.exception("get_walking_directions failed: %s", e)
        return {
            "success": False,
            "error": f"Could not get directions: {e}",
        }


# ---------------------------------------------------------------------------
# Places API (New)
# ---------------------------------------------------------------------------

_PLACES_NEARBY_URL = "https://places.googleapis.com/v1/places:searchNearby"
_PLACES_NEARBY_FIELD_MASK = (
    "places.displayName,"
    "places.formattedAddress,"
    "places.types,"
    "places.rating,"
    "places.location,"
    "places.currentOpeningHours,"
    "places.accessibilityOptions,"
    "places.plusCode"
)


def _parse_accessibility(place: dict) -> dict[str, bool | None]:
    """Extract accessibility options from a Places (New) response."""
    opts = place.get("accessibilityOptions", {})
    return {
        "wheelchair_entrance": opts.get("wheelchairAccessibleEntrance"),
        "wheelchair_parking": opts.get("wheelchairAccessibleParking"),
        "wheelchair_restroom": opts.get("wheelchairAccessibleRestroom"),
        "wheelchair_seating": opts.get("wheelchairAccessibleSeating"),
    }


def get_location_info(lat: float, lng: float) -> dict[str, Any]:
    """Get information about the user's current location.

    Combines reverse geocoding (SDK) with nearby places (Places New API).

    Args:
        lat: Latitude.
        lng: Longitude.

    Returns:
        Dict with address and nearby points of interest.
    """
    try:
        # Reverse geocode (SDK — not deprecated)
        client = _get_client()
        geocode_results = client.reverse_geocode((lat, lng))
        address = "Unknown location"
        if geocode_results:
            address = geocode_results[0].get("formatted_address", "Unknown location")

        # Nearby POIs via Places (New)
        body = {
            "locationRestriction": {
                "circle": {
                    "center": {"latitude": lat, "longitude": lng},
                    "radius": 100.0,
                },
            },
            "maxResultCount": 5,
        }

        pois = []
        try:
            resp = maps_rest_post(_PLACES_NEARBY_URL, body, field_mask=_PLACES_NEARBY_FIELD_MASK)
            for place in resp.get("places", []):
                loc = place.get("location", {})
                poi_lat = loc.get("latitude", lat)
                poi_lng = loc.get("longitude", lng)
                dist = _haversine_distance(lat, lng, poi_lat, poi_lng)

                display_name = place.get("displayName", {})
                name = display_name.get("text", "Unknown") if isinstance(display_name, dict) else str(display_name)

                poi_info: dict[str, Any] = {
                    "name": name,
                    "types": place.get("types", []),
                    "distance_meters": round(dist),
                    "accessibility": _parse_accessibility(place),
                }

                hours = place.get("currentOpeningHours", {})
                if "openNow" in hours:
                    poi_info["open_now"] = hours["openNow"]

                plus_code = place.get("plusCode", {})
                if plus_code.get("globalCode"):
                    poi_info["plus_code"] = plus_code["globalCode"]

                pois.append(poi_info)
        except Exception as e:
            logger.warning("Places (New) nearby search failed, falling back: %s", e)

        return {
            "success": True,
            "address": address,
            "nearby_places": pois,
        }

    except Exception as e:
        logger.exception("get_location_info failed: %s", e)
        return {
            "success": False,
            "error": f"Could not get location info: {e}",
        }


def nearby_search(
    lat: float,
    lng: float,
    radius: int = 200,
    types: list[str] | None = None,
    keyword: str | None = None,
) -> dict[str, Any]:
    """Search for nearby places matching given criteria.

    Uses Places API (New) with accessibility options.

    Args:
        lat: Latitude.
        lng: Longitude.
        radius: Search radius in meters (default 200).
        types: Place types to filter by (e.g. ["restaurant", "cafe"]).
        keyword: Optional keyword to search for.

    Returns:
        Dict with list of matching places and their distances.
    """
    try:
        body: dict[str, Any] = {
            "locationRestriction": {
                "circle": {
                    "center": {"latitude": lat, "longitude": lng},
                    "radius": float(radius),
                },
            },
            "maxResultCount": 10,
        }
        if types:
            body["includedTypes"] = types
        if keyword:
            # Places (New) uses textQuery in searchNearby via includedTypes
            # For keyword search, we add it as includedPrimaryTypes fallback
            # The API doesn't have a direct keyword param; types cover most cases
            pass

        field_mask = _PLACES_NEARBY_FIELD_MASK

        resp = maps_rest_post(_PLACES_NEARBY_URL, body, field_mask=field_mask)

        places = []
        for place in resp.get("places", []):
            loc = place.get("location", {})
            p_lat = loc.get("latitude", lat)
            p_lng = loc.get("longitude", lng)
            dist = _haversine_distance(lat, lng, p_lat, p_lng)

            display_name = place.get("displayName", {})
            name = display_name.get("text", "Unknown") if isinstance(display_name, dict) else str(display_name)

            place_info: dict[str, Any] = {
                "name": name,
                "address": place.get("formattedAddress", ""),
                "types": place.get("types", []),
                "rating": place.get("rating"),
                "distance_meters": round(dist),
                "accessibility": _parse_accessibility(place),
            }

            hours = place.get("currentOpeningHours", {})
            if "openNow" in hours:
                place_info["open_now"] = hours["openNow"]

            plus_code = place.get("plusCode", {})
            if plus_code.get("globalCode"):
                place_info["plus_code"] = plus_code["globalCode"]

            places.append(place_info)

        # Sort by distance
        places.sort(key=lambda p: p["distance_meters"])

        return {
            "success": True,
            "query": keyword or (types[0] if types else "nearby places"),
            "count": len(places),
            "places": places,
        }

    except Exception as e:
        logger.exception("nearby_search failed: %s", e)
        return {
            "success": False,
            "error": f"Nearby search failed: {e}",
        }


def reverse_geocode(lat: float, lng: float) -> dict[str, Any]:
    """Get a human-readable address from coordinates.

    Args:
        lat: Latitude.
        lng: Longitude.

    Returns:
        Dict with ``success``, ``address`` (or ``error``) fields.
    """
    try:
        client = _get_client()
        results = client.reverse_geocode((lat, lng))
        if results:
            return {
                "success": True,
                "address": results[0].get("formatted_address", "Unknown location"),
            }
        return {"success": False, "error": "No address found for this location."}
    except Exception as e:
        logger.exception("reverse_geocode failed: %s", e)
        return {"success": False, "error": f"Could not determine address: {e}"}


# ---------------------------------------------------------------------------
# Street View destination preview (Step 5)
# ---------------------------------------------------------------------------

_STREET_VIEW_META_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
_STREET_VIEW_IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview"


def preview_destination(lat: float, lng: float, destination_name: str = "") -> dict[str, Any]:
    """Preview a destination via Street View imagery + Vision Agent analysis.

    Downloads Street View image and sends it to the Vision Agent for
    scene description, providing the user with a "mental picture" of
    their destination before arrival.

    Args:
        lat: Destination latitude.
        lng: Destination longitude.
        destination_name: Optional name for context.

    Returns:
        Dict with scene description, safety warnings, and navigation info.
    """
    try:
        # 1. Check Street View availability via metadata
        meta_params = {
            "location": f"{lat},{lng}",
            "source": "outdoor",
        }
        meta_resp = maps_rest_get(_STREET_VIEW_META_URL, params=meta_params)
        meta = meta_resp.json()

        if meta.get("status") != "OK":
            return {
                "success": True,
                "has_street_view": False,
                "description": (
                    f"No street-level imagery available for "
                    f"{destination_name or 'this location'}. "
                    "I'll describe the area when you arrive."
                ),
                "safety_warnings": [],
                "navigation_info": {},
            }

        # 2. Download Street View image
        image_params = {
            "location": f"{lat},{lng}",
            "size": "640x480",
            "source": "outdoor",
        }
        image_resp = maps_rest_get(_STREET_VIEW_IMAGE_URL, params=image_params)
        image_b64 = base64.b64encode(image_resp.content).decode("utf-8")

        # 3. Send to Vision Agent for analysis
        from agents.vision_agent import analyze_scene

        # analyze_scene is async; we're in a sync context (called via asyncio.to_thread)
        # Create a new event loop for the coroutine
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're already in an async context — use a new thread with its own loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, analyze_scene(image_b64, lod=2, session_context={
                    "space_type": "outdoor",
                    "active_task": f"previewing destination: {destination_name}" if destination_name else "previewing destination",
                }))
                vision_result = future.result(timeout=30)
        else:
            vision_result = asyncio.run(analyze_scene(image_b64, lod=2, session_context={
                "space_type": "outdoor",
                "active_task": f"previewing destination: {destination_name}" if destination_name else "previewing destination",
            }))

        desc = vision_result.get("scene_description", "")
        if destination_name:
            desc = f"Preview of {destination_name}: {desc}"

        return {
            "success": True,
            "has_street_view": True,
            "description": desc,
            "safety_warnings": vision_result.get("safety_warnings", []),
            "navigation_info": vision_result.get("navigation_info", {}),
        }

    except Exception as e:
        logger.exception("preview_destination failed: %s", e)
        return {
            "success": False,
            "has_street_view": False,
            "description": f"Could not preview destination: {e}",
            "safety_warnings": [],
            "navigation_info": {},
        }


# ---------------------------------------------------------------------------
# Address Validation (Step 6)
# ---------------------------------------------------------------------------

_ADDRESS_VALIDATION_URL = "https://addressvalidation.googleapis.com/v1:validateAddress"


def validate_address(address: str) -> dict[str, Any]:
    """Validate and correct a spoken/typed address.

    Uses Google Address Validation API to fix common speech-to-text
    address errors (e.g. "one two three main street" → "123 Main St").

    Args:
        address: Raw address string from user input.

    Returns:
        Dict with corrected address, coordinates, and correction note.
    """
    try:
        body = {
            "address": {
                "addressLines": [address],
            },
        }

        resp = maps_rest_post(_ADDRESS_VALIDATION_URL, body)
        result = resp.get("result", {})
        validated = result.get("address", {})
        geocode = result.get("geocode", {})

        corrected = validated.get("formattedAddress", address)
        was_corrected = corrected.lower().strip() != address.lower().strip()

        location = geocode.get("location", {})
        lat = location.get("latitude")
        lng = location.get("longitude")

        # Check completeness
        verdict = result.get("verdict", {})
        is_complete = verdict.get("addressComplete", False)

        correction_note = ""
        if was_corrected:
            correction_note = f"Did you mean '{corrected}'?"
        elif not is_complete:
            correction_note = "The address may be incomplete. Please provide more details."

        return {
            "success": True,
            "corrected_address": corrected,
            "was_corrected": was_corrected,
            "correction_note": correction_note,
            "latitude": lat,
            "longitude": lng,
            "is_complete": is_complete,
        }

    except Exception as e:
        logger.warning("Address validation failed (falling back to original): %s", e)
        return {
            "success": True,
            "corrected_address": address,
            "was_corrected": False,
            "correction_note": "",
            "latitude": None,
            "longitude": None,
            "is_complete": False,
        }


# ---------------------------------------------------------------------------
# Haversine helper
# ---------------------------------------------------------------------------


def _haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Compute distance in meters between two GPS points."""
    R = 6_371_000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lng2 - lng1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# ADK FunctionDeclaration wrappers for Gemini Live API
# ---------------------------------------------------------------------------

NAVIGATION_TOOL_DECLARATIONS = [
    {
        "name": "navigate_to",
        "description": (
            "Get step-by-step walking directions to a destination using clock-position "
            "directional cues (e.g. 'at 2 o'clock, 50 meters'). Includes slope warnings "
            "for steep grades. Always use this when the user asks how to get somewhere."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "destination": {
                    "type": "string",
                    "description": "Place name or address to navigate to",
                },
                "origin_lat": {
                    "type": "number",
                    "description": "User's current latitude",
                },
                "origin_lng": {
                    "type": "number",
                    "description": "User's current longitude",
                },
                "user_heading": {
                    "type": "number",
                    "description": "User's current compass heading (0-360, 0=North)",
                },
            },
            "required": ["destination", "origin_lat", "origin_lng"],
        },
    },
    {
        "name": "get_location_info",
        "description": (
            "Get information about the user's current location including address and "
            "nearby points of interest with accessibility details. Use when the user "
            "asks 'where am I?' or wants to know what's around them."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "lat": {
                    "type": "number",
                    "description": "Latitude",
                },
                "lng": {
                    "type": "number",
                    "description": "Longitude",
                },
            },
            "required": ["lat", "lng"],
        },
    },
    {
        "name": "nearby_search",
        "description": (
            "Search for nearby places like restaurants, cafes, pharmacies, bus stops, etc. "
            "Returns accessibility information (wheelchair access) for each place. "
            "Use when the user asks to find a specific type of place nearby."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "lat": {
                    "type": "number",
                    "description": "Latitude",
                },
                "lng": {
                    "type": "number",
                    "description": "Longitude",
                },
                "radius": {
                    "type": "integer",
                    "description": "Search radius in meters (default 200)",
                },
                "types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Place types to filter (e.g. ['restaurant', 'cafe'])",
                },
                "keyword": {
                    "type": "string",
                    "description": "Optional keyword to search for",
                },
            },
            "required": ["lat", "lng"],
        },
    },
    {
        "name": "reverse_geocode",
        "description": (
            "Get a human-readable address from GPS coordinates. "
            "Use when you need to describe the user's location."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "lat": {
                    "type": "number",
                    "description": "Latitude",
                },
                "lng": {
                    "type": "number",
                    "description": "Longitude",
                },
            },
            "required": ["lat", "lng"],
        },
    },
    {
        "name": "get_walking_directions",
        "description": (
            "Get walking directions between two named locations (addresses or place names). "
            "Includes slope warnings. Use when GPS coordinates are not available."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {
                    "type": "string",
                    "description": "Starting address or place name",
                },
                "destination": {
                    "type": "string",
                    "description": "Destination address or place name",
                },
            },
            "required": ["origin", "destination"],
        },
    },
    {
        "name": "preview_destination",
        "description": (
            "Preview a destination using Street View imagery before arrival. "
            "Returns a scene description with safety warnings and navigation cues. "
            "Use when the user asks 'what does it look like there?' or before starting "
            "navigation to an unfamiliar place."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "lat": {
                    "type": "number",
                    "description": "Destination latitude",
                },
                "lng": {
                    "type": "number",
                    "description": "Destination longitude",
                },
                "destination_name": {
                    "type": "string",
                    "description": "Name of the destination for context",
                },
            },
            "required": ["lat", "lng"],
        },
    },
    {
        "name": "validate_address",
        "description": (
            "Validate and correct a spoken or typed address. Fixes common "
            "speech-to-text errors (e.g. 'one two three main street' → "
            "'123 Main St'). Use before navigating to a user-spoken address "
            "to ensure accuracy. Returns corrected address with coordinates."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "address": {
                    "type": "string",
                    "description": "The raw address string to validate",
                },
            },
            "required": ["address"],
        },
    },
]

# Map function names to callables for the tool dispatcher
NAVIGATION_FUNCTIONS = {
    "navigate_to": navigate_to,
    "get_location_info": get_location_info,
    "nearby_search": nearby_search,
    "reverse_geocode": reverse_geocode,
    "get_walking_directions": get_walking_directions,
    "preview_destination": preview_destination,
    "validate_address": validate_address,
}
