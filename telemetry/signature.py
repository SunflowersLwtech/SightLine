"""Telemetry signature helpers extracted from ``server.py``."""

from __future__ import annotations

TELEMETRY_FORCE_REFRESH_SEC = 60.0

_MEANINGFUL_TELEMETRY_FIELDS = {
    "motion_state",
    "hr_bucket",
    "noise_bucket",
    "cadence_bucket",
    "heading_bucket",
    "gps_bucket",
    "time_context",
    "device_type",
}


def _heart_rate_bucket(heart_rate: float | None) -> str:
    if heart_rate is None or heart_rate <= 0:
        return "unknown"
    if heart_rate > 100:
        return "elevated"
    return "normal"


def _noise_bucket(noise_db: float) -> str:
    if noise_db < 40:
        return "quiet"
    if noise_db < 65:
        return "moderate"
    if noise_db < 80:
        return "loud"
    return "very_loud"


def _cadence_bucket(step_cadence: float) -> str:
    if step_cadence <= 0:
        return "still"
    if step_cadence < 60:
        return "slow"
    if step_cadence < 120:
        return "walk"
    return "fast"


def _heading_bucket(heading: float | None) -> int | None:
    if heading is None:
        return None
    return int((heading % 360) // 30)


def _gps_bucket(gps) -> tuple[float, float] | None:
    if gps is None:
        return None
    try:
        return (round(float(gps.lat), 3), round(float(gps.lng), 3))
    except (TypeError, ValueError, AttributeError):
        return None


def _build_telemetry_signature(ephemeral_ctx) -> dict[str, object]:
    """Build coarse signature to detect meaningful telemetry changes."""
    heading_value = getattr(ephemeral_ctx, "heading", None)
    heading_bucket = _heading_bucket(heading_value if heading_value not in (None, 0.0) else None)
    return {
        "motion_state": getattr(ephemeral_ctx, "motion_state", "unknown"),
        "hr_bucket": _heart_rate_bucket(getattr(ephemeral_ctx, "heart_rate", None)),
        "noise_bucket": _noise_bucket(float(getattr(ephemeral_ctx, "ambient_noise_db", 50.0) or 50.0)),
        "cadence_bucket": _cadence_bucket(float(getattr(ephemeral_ctx, "step_cadence", 0.0) or 0.0)),
        "heading_bucket": heading_bucket,
        "gps_bucket": _gps_bucket(getattr(ephemeral_ctx, "gps", None)),
        "time_context": getattr(ephemeral_ctx, "time_context", "unknown"),
        "device_type": getattr(ephemeral_ctx, "device_type", "phone_only"),
    }


def _changed_signature_fields(
    previous_signature: dict[str, object] | None,
    current_signature: dict[str, object],
) -> list[str]:
    if previous_signature is None:
        return ["initial"]
    changed: list[str] = []
    for key, value in current_signature.items():
        if previous_signature.get(key) != value:
            changed.append(key)
    return changed


def _should_inject_telemetry_context(
    *,
    previous_signature: dict[str, object] | None,
    current_signature: dict[str, object],
    last_injected_ts: float,
    now_ts: float,
    force_refresh_sec: float = TELEMETRY_FORCE_REFRESH_SEC,
) -> tuple[bool, list[str]]:
    """Decide if telemetry context should be injected into the model."""
    changed = _changed_signature_fields(previous_signature, current_signature)
    if previous_signature is None:
        return True, changed

    meaningful_change = [field for field in changed if field in _MEANINGFUL_TELEMETRY_FIELDS]
    if meaningful_change:
        return True, meaningful_change

    if now_ts - last_injected_ts >= force_refresh_sec:
        return True, ["periodic_refresh"]

    return False, changed

