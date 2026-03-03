"""Tests for SightLine telemetry parser."""

import telemetry.telemetry_parser as _tp_mod


def parse_telemetry(*args, **kwargs):
    """Late-bind to dodge import-time MagicMock pollution."""
    return _tp_mod.parse_telemetry(*args, **kwargs)


def parse_telemetry_to_ephemeral(*args, **kwargs):
    """Late-bind to dodge import-time MagicMock pollution."""
    return _tp_mod.parse_telemetry_to_ephemeral(*args, **kwargs)


# =====================================================================
# parse_telemetry() — semantic text output
# =====================================================================


def test_walking_state_kv():
    result = parse_telemetry({"motion_state": "walking"})
    assert "motion=walking" in result


def test_high_noise_kv():
    result = parse_telemetry({"ambient_noise_db": 85})
    assert "noise=85dB/very_loud" in result


def test_moderate_noise_kv():
    result = parse_telemetry({"ambient_noise_db": 55})
    assert "noise=55dB/moderate" in result


def test_elevated_heart_rate_kv():
    result = parse_telemetry({"heart_rate": 125})
    assert "hr=125/elevated" in result


def test_normal_heart_rate_kv():
    result = parse_telemetry({"heart_rate": 72})
    assert "hr=72/normal" in result


def test_gps_kv_format():
    result = parse_telemetry({"gps": {"latitude": 37.7749, "longitude": -122.4194}})
    assert "loc=37.774900,-122.419400" in result


def test_heading_cardinal_kv():
    result = parse_telemetry({"heading": 90})
    assert "heading=East/90deg" in result


def test_empty_data_fallback():
    result = parse_telemetry({})
    assert "No sensor data available" in result


def test_cadence_kv():
    result = parse_telemetry({"step_cadence": 80})
    assert "cadence=80spm" in result


def test_full_telemetry_kv():
    """Verify all fields combine as KV pairs with prefix."""
    result = parse_telemetry({
        "motion_state": "walking",
        "step_cadence": 80,
        "ambient_noise_db": 55,
        "heart_rate": 72,
    })
    assert result.startswith("[TELEMETRY UPDATE]")
    assert "motion=walking" in result
    assert "cadence=80spm" in result
    assert "noise=55dB/moderate" in result
    assert "hr=72/normal" in result


# =====================================================================
# parse_telemetry_to_ephemeral() — structured context
# =====================================================================


def test_ephemeral_motion_state_mapping():
    ctx = parse_telemetry_to_ephemeral({"motion_state": "automotive"})
    assert ctx.motion_state == "in_vehicle"


def test_ephemeral_unknown_motion_default():
    ctx = parse_telemetry_to_ephemeral({"motion_state": "unknown"})
    assert ctx.motion_state == "stationary"


def test_ephemeral_gps_parsing():
    ctx = parse_telemetry_to_ephemeral({
        "gps": {"latitude": 37.77, "longitude": -122.42, "accuracy": 5.0, "speed": 1.2}
    })
    assert ctx.gps is not None
    assert abs(ctx.gps.lat - 37.77) < 0.01
    assert abs(ctx.gps.lng - (-122.42)) < 0.01
    assert ctx.gps.accuracy == 5.0
    assert ctx.gps.speed == 1.2


def test_ephemeral_heart_rate_none():
    ctx = parse_telemetry_to_ephemeral({})
    assert ctx.heart_rate is None


def test_ephemeral_watch_stability_score():
    ctx = parse_telemetry_to_ephemeral({"watch_stability_score": 0.75})
    assert abs(ctx.watch_stability_score - 0.75) < 0.01


def test_ephemeral_invalid_values():
    ctx = parse_telemetry_to_ephemeral({"step_cadence": "abc", "ambient_noise_db": "xyz"})
    assert ctx.step_cadence == 0.0
    assert ctx.ambient_noise_db == 50.0


def test_ephemeral_device_type_passthrough():
    ctx = parse_telemetry_to_ephemeral({"device_type": "phone_and_watch"})
    assert ctx.device_type == "phone_and_watch"
