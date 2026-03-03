"""Telemetry data parser for SightLine.

Converts raw sensor JSON from the iOS client into semantic text
suitable for injection into the Gemini Live context.

Phase 2 additions:
- ``parse_telemetry_to_ephemeral()`` — converts raw JSON into
  ``EphemeralContext`` for the LOD decision engine.
"""

import json
import logging

from typing import Optional

from lod.models import EphemeralContext, GPSData

logger = logging.getLogger(__name__)


def _to_float(value, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _to_optional_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _noise_bucket(noise_db: float) -> str:
    """Classify ambient noise level into a bucket label."""
    if noise_db < 40:
        return "quiet"
    elif noise_db < 65:
        return "moderate"
    elif noise_db < 80:
        return "loud"
    else:
        return "very_loud"


def _hr_bucket(hr: float) -> str:
    """Classify heart rate into a bucket label."""
    if hr > 100:
        return "elevated"
    return "normal"


def parse_telemetry(data: dict) -> str:
    """Convert raw telemetry JSON to structured KV text for Gemini context injection.

    Args:
        data: Raw telemetry dict from iOS client containing fields like
              motion_state, step_cadence, ambient_noise_db, heart_rate, gps.

    Returns:
        Structured key=value string with ``[TELEMETRY UPDATE]`` prefix.
    """
    pairs: list[str] = []

    # Motion state
    motion_state = data.get("motion_state")
    if motion_state:
        pairs.append(f"motion={motion_state}")

    # Step cadence
    step_cadence = data.get("step_cadence")
    if step_cadence is not None:
        try:
            cadence = float(step_cadence)
            if cadence > 0:
                pairs.append(f"cadence={cadence:.0f}spm")
        except (ValueError, TypeError):
            pass

    # Ambient noise
    ambient_noise_db = data.get("ambient_noise_db")
    if ambient_noise_db is not None:
        try:
            noise = float(ambient_noise_db)
            bucket = _noise_bucket(noise)
            pairs.append(f"noise={noise:.0f}dB/{bucket}")
        except (ValueError, TypeError):
            pass

    # Heart rate
    heart_rate = data.get("heart_rate")
    if heart_rate is not None:
        try:
            hr = float(heart_rate)
            if hr > 0:
                bucket = _hr_bucket(hr)
                pairs.append(f"hr={hr:.0f}/{bucket}")
        except (ValueError, TypeError):
            pass

    # GPS
    gps = data.get("gps")
    if gps and isinstance(gps, dict):
        lat = gps.get("latitude")
        lon = gps.get("longitude")
        accuracy = gps.get("accuracy")
        speed = gps.get("speed")

        if lat is not None and lon is not None:
            loc = f"loc={lat:.6f},{lon:.6f}"
            if accuracy is not None:
                loc += f"/acc={accuracy:.0f}m"
            pairs.append(loc)

        if speed is not None:
            try:
                spd = float(speed)
                if spd > 0:
                    pairs.append(f"speed={spd:.1f}m/s")
            except (ValueError, TypeError):
                pass

    # Heading
    heading = data.get("heading")
    if heading is not None:
        try:
            h = float(heading)
            cardinal = _degrees_to_cardinal(h)
            pairs.append(f"heading={cardinal}/{h:.0f}deg")
        except (ValueError, TypeError):
            pass

    # Weather
    weather = data.get("weather")
    if weather and isinstance(weather, dict):
        condition = weather.get("condition", "")
        if condition:
            parts = [f"weather={condition.lower()}"]
            precip_chance = weather.get("precipitationChance")
            if precip_chance is not None and precip_chance > 0.1:
                parts.append(f"precipitation_chance={precip_chance:.0%}")
            visibility = weather.get("visibility")
            if visibility is not None and visibility < 5000:
                parts.append(f"visibility={visibility:.0f}m")
            wind = weather.get("windSpeed")
            if wind is not None and wind > 5:
                parts.append(f"wind={wind:.0f}m/s")
            pairs.append(" ".join(parts))

    # Depth
    depth = data.get("depth")
    if depth and isinstance(depth, dict):
        center = depth.get("center_distance")
        min_d = depth.get("min_distance")
        min_region = depth.get("min_distance_region", "")
        if center is not None and center > 0:
            depth_parts = [f"depth_center={center:.1f}m"]
            if min_d is not None and min_d > 0:
                depth_parts.append(f"depth_closest={min_d:.1f}m/{min_region}")
            pairs.append(" ".join(depth_parts))

    # Watch extended context
    stability = data.get("watch_stability_score")
    if stability is not None:
        pairs.append(f"stability={float(stability):.2f}")

    w_heading = data.get("watch_heading")
    if w_heading is not None:
        cardinal = _degrees_to_cardinal(float(w_heading))
        pairs.append(f"watch_heading={cardinal}({float(w_heading):.0f}°)")

    sp_o2 = data.get("sp_o2")
    if sp_o2 is not None:
        pairs.append(f"spO2={float(sp_o2):.0f}%")

    w_noise = data.get("watch_noise_exposure")
    if w_noise is not None:
        pairs.append(f"watch_noise={float(w_noise):.0f}dB")

    if not pairs:
        logger.debug("Telemetry data had no parseable fields: %s", json.dumps(data))
        return "[TELEMETRY UPDATE] No sensor data available."

    return "[TELEMETRY UPDATE] " + " ".join(pairs)


def _degrees_to_cardinal(degrees: float) -> str:
    """Convert compass degrees to cardinal direction."""
    directions = [
        "North", "North-Northeast", "Northeast", "East-Northeast",
        "East", "East-Southeast", "Southeast", "South-Southeast",
        "South", "South-Southwest", "Southwest", "West-Southwest",
        "West", "West-Northwest", "Northwest", "North-Northwest",
    ]
    idx = round(degrees / 22.5) % 16
    return directions[idx]


# ---------------------------------------------------------------------------
# Phase 2: Raw telemetry → EphemeralContext (for LOD engine)
# ---------------------------------------------------------------------------

# iOS CMMotionActivity types → normalised motion states
_MOTION_STATE_MAP: dict[str, str] = {
    "stationary": "stationary",
    "walking": "walking",
    "running": "running",
    "automotive": "in_vehicle",
    "cycling": "cycling",
    "unknown": "stationary",
}


def parse_telemetry_to_ephemeral(data: dict) -> EphemeralContext:
    """Convert raw telemetry JSON from iOS into an ``EphemeralContext``.

    This is the LOD engine's input — a structured snapshot of the user's
    physical state, used alongside ``SessionContext`` and ``UserProfile``
    by ``decide_lod()``.

    Args:
        data: Raw telemetry dict from iOS TelemetryData.toJSON().

    Returns:
        Populated EphemeralContext dataclass.
    """
    ctx = EphemeralContext()

    # Motion state
    raw_motion = data.get("motion_state", "stationary")
    ctx.motion_state = _MOTION_STATE_MAP.get(raw_motion, "stationary")

    # Step cadence
    try:
        ctx.step_cadence = float(data.get("step_cadence", 0.0))
    except (ValueError, TypeError):
        ctx.step_cadence = 0.0

    # Ambient noise
    try:
        ctx.ambient_noise_db = float(data.get("ambient_noise_db", 50.0))
    except (ValueError, TypeError):
        ctx.ambient_noise_db = 50.0

    # GPS
    gps_raw = data.get("gps")
    if gps_raw and isinstance(gps_raw, dict):
        try:
            ctx.gps = GPSData(
                lat=float(gps_raw.get("latitude", 0.0)),
                lng=float(gps_raw.get("longitude", 0.0)),
                accuracy=float(gps_raw.get("accuracy", 0.0)),
                speed=float(gps_raw.get("speed", 0.0)),
                altitude=float(gps_raw.get("altitude", 0.0)),
            )
        except (ValueError, TypeError):
            ctx.gps = None

    # Heading
    try:
        ctx.heading = float(data.get("heading", 0.0))
    except (ValueError, TypeError):
        ctx.heading = 0.0

    # Time context
    ctx.time_context = data.get("time_context", "unknown")

    # Heart rate (None when watch not connected)
    hr = data.get("heart_rate")
    if hr is not None:
        try:
            ctx.heart_rate = float(hr)
        except (ValueError, TypeError):
            ctx.heart_rate = None

    # User gesture (lod_up, lod_down, tap, shake)
    ctx.user_gesture = data.get("user_gesture")

    # Device type
    ctx.device_type = data.get("device_type", "phone_only")

    # Weather
    weather = data.get("weather")
    if weather and isinstance(weather, dict):
        ctx.weather_condition = weather.get("condition", "unknown").lower()
        ctx.weather_visibility = float(weather.get("visibility", 10000.0))
        ctx.weather_precipitation = weather.get("precipitation", "none")

    # Depth
    depth = data.get("depth")
    if depth and isinstance(depth, dict):
        try:
            center = depth.get("center_distance")
            if center is not None:
                ctx.depth_center = float(center)
            min_d = depth.get("min_distance")
            if min_d is not None:
                ctx.depth_min = float(min_d)
            ctx.depth_min_region = depth.get("min_distance_region")
        except (ValueError, TypeError):
            pass

    # Watch extended context
    ctx.watch_pitch = _to_float(data.get("watch_pitch"), 0.0)
    ctx.watch_roll = _to_float(data.get("watch_roll"), 0.0)
    ctx.watch_yaw = _to_float(data.get("watch_yaw"), 0.0)
    ctx.watch_stability_score = _to_float(data.get("watch_stability_score"), 1.0)
    ctx.watch_heading = _to_optional_float(data.get("watch_heading"))
    ctx.watch_heading_accuracy = _to_optional_float(data.get("watch_heading_accuracy"))
    ctx.sp_o2 = _to_optional_float(data.get("sp_o2"))
    ctx.watch_noise_exposure = _to_optional_float(data.get("watch_noise_exposure"))

    return ctx
