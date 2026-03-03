"""Integration smoke tests for the SightLine server.

These tests verify:
- FastAPI app startup and health endpoint
- WebSocket connection acceptance
- Upstream message type handling (structural only — no live Gemini connection)

Note: These tests do NOT require a real Gemini API connection.
They verify the server's structural correctness: routing, message parsing,
LOD state initialisation, and protocol compliance.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixture: FastAPI TestClient
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """Create a TestClient for the SightLine FastAPI app."""
    from fastapi.testclient import TestClient
    from server import app
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Verify the /health endpoint for Cloud Run probes."""

    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model" in data

    def test_health_reports_phase(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["phase"] == 6


# ---------------------------------------------------------------------------
# Session Manager
# ---------------------------------------------------------------------------


class TestSessionManager:
    """Verify SessionManager initialisation and state management."""

    def test_session_context_creation(self):
        from live_api.session_manager import SessionManager
        mgr = SessionManager()
        ctx = mgr.get_session_context("test_session")
        assert ctx.current_lod == 2  # default LOD
        assert ctx.space_type == "unknown"

    def test_user_profile_defaults(self):
        from live_api.session_manager import SessionManager
        mgr = SessionManager()
        profile = mgr.get_user_profile("test_user")
        assert profile.user_id == "test_user"
        assert profile.vision_status == "totally_blind"
        assert profile.verbosity_preference == "concise"

    def test_ephemeral_context_creation(self):
        from live_api.session_manager import SessionManager
        mgr = SessionManager()
        ctx = mgr.get_ephemeral_context("test_session")
        assert ctx.motion_state == "stationary"

    def test_session_handle_cache(self):
        from live_api.session_manager import SessionManager
        mgr = SessionManager()
        assert mgr.get_handle("s1") is None
        mgr.update_handle("s1", "handle_abc")
        assert mgr.get_handle("s1") == "handle_abc"

    def test_session_cleanup(self):
        from live_api.session_manager import SessionManager
        mgr = SessionManager()
        mgr.get_session_context("s1")
        mgr.update_handle("s1", "h1")
        mgr.remove_session("s1")
        assert mgr.get_handle("s1") is None

    def test_run_config_has_vad_params(self):
        """SL-36: RunConfig should include LOD-specific VAD settings."""
        from live_api.session_manager import SessionManager
        mgr = SessionManager()
        config = mgr.get_run_config("test", lod=1)
        assert config.streaming_mode is not None
        assert config.realtime_input_config is not None

    def test_run_config_lod3_longer_silence(self):
        """LOD 3 should have longer silence duration than LOD 1."""
        from live_api.session_manager import SessionManager, LOD_VAD_PRESETS
        assert LOD_VAD_PRESETS[3]["silence_duration_ms"] > LOD_VAD_PRESETS[1]["silence_duration_ms"]

    def test_context_window_compression_guardrail(self):
        """High-risk guardrail: keep context compression enabled for long AV sessions."""
        from live_api.session_manager import SessionManager
        mgr = SessionManager()
        config = mgr.get_run_config("test", lod=2)
        compression = config.context_window_compression
        assert compression is not None
        assert compression.trigger_tokens == 100000
        assert compression.sliding_window is not None
        assert compression.sliding_window.target_tokens == 80000

    def test_runtime_vad_update_payload_contract(self):
        """Runtime VAD updates should expose explicit payload+capability metadata."""
        from live_api.session_manager import (
            build_vad_runtime_update_message,
            build_vad_runtime_update_payload,
            supports_runtime_vad_reconfiguration,
        )

        supported, reason = supports_runtime_vad_reconfiguration()
        assert supported is False
        assert reason

        payload = build_vad_runtime_update_payload(1)
        assert payload["lod"] == 1
        assert 500 <= payload["silence_duration_ms"] <= 1000
        assert payload["prefix_padding_ms"] >= 0

        message = build_vad_runtime_update_message(1)
        assert "[VAD UPDATE]" in message
        assert "silence_duration_ms" in message


# ---------------------------------------------------------------------------
# Telemetry → LOD integration
# ---------------------------------------------------------------------------


class TestTelemetryLODIntegration:
    """Verify the telemetry → LOD engine integration path."""

    def test_telemetry_to_ephemeral_to_lod(self):
        """Full pipeline: raw telemetry JSON → EphemeralContext → LOD decision."""
        from telemetry.telemetry_parser import parse_telemetry_to_ephemeral
        from lod import decide_lod
        from lod.models import SessionContext, UserProfile

        raw = {
            "motion_state": "walking",
            "step_cadence": 80,
            "ambient_noise_db": 55,
            "heart_rate": 75,
        }
        ephemeral = parse_telemetry_to_ephemeral(raw)
        session = SessionContext()
        profile = UserProfile.default()

        lod, log = decide_lod(ephemeral, session, profile)

        # Walking at 80 spm → LOD 2 (experience-driven: walking = standard)
        assert lod == 2
        assert log.motion_state == "walking"
        assert len(log.triggered_rules) > 0

    def test_stationary_gives_lod2_with_default_concise(self):
        """Stationary user with default concise pref should get LOD 2 (Rule 5 reduces 3→2)."""
        from telemetry.telemetry_parser import parse_telemetry_to_ephemeral
        from lod import decide_lod
        from lod.models import SessionContext, UserProfile

        raw = {
            "motion_state": "stationary",
            "step_cadence": 0,
            "ambient_noise_db": 35,
        }
        ephemeral = parse_telemetry_to_ephemeral(raw)
        lod, _ = decide_lod(ephemeral, SessionContext(), UserProfile.default())
        assert lod == 2


# ---------------------------------------------------------------------------
# TelemetryAggregator wiring
# ---------------------------------------------------------------------------


class TestTelemetryAggregator:
    """Verify TelemetryAggregator LOD-aware throttling."""

    def test_immediate_first_send(self):
        from lod.telemetry_aggregator import TelemetryAggregator
        agg = TelemetryAggregator()
        assert agg.should_send(0.0) is True

    def test_throttle_within_interval(self):
        from lod.telemetry_aggregator import TelemetryAggregator
        agg = TelemetryAggregator(current_lod=2)
        agg.mark_sent(0.0)
        # LOD 2 midpoint interval = 2.5s; at 1s should NOT send
        assert agg.should_send(1.0) is False

    def test_send_after_interval(self):
        from lod.telemetry_aggregator import TelemetryAggregator
        agg = TelemetryAggregator(current_lod=2)
        agg.mark_sent(0.0)
        # LOD 2 midpoint interval = 2.5s; at 3s should send
        assert agg.should_send(3.0) is True

    def test_lod_change_updates_interval(self):
        from lod.telemetry_aggregator import TelemetryAggregator
        agg = TelemetryAggregator(current_lod=1)
        interval_lod1 = agg.send_interval
        agg.update_lod(3)
        interval_lod3 = agg.send_interval
        assert interval_lod3 > interval_lod1  # LOD 3 is slower


# ---------------------------------------------------------------------------
# Repeat suppression guards
# ---------------------------------------------------------------------------


class TestRepeatSuppressionGuards:
    """Verify anti-repeat helpers used by server-side speech throttling."""

    def test_repeated_text_detected_within_cooldown(self):
        from server import _is_repeated_text

        repeated = _is_repeated_text(
            "Facing north. Hallway ahead.",
            previous_text="Facing north. Hallway ahead!",
            now_ts=12.0,
            previous_ts=4.0,
            cooldown_sec=10.0,
        )
        assert repeated is True

    def test_short_text_not_suppressed(self):
        from server import _is_repeated_text

        repeated = _is_repeated_text(
            "OK",
            previous_text="OK",
            now_ts=5.0,
            previous_ts=1.0,
            cooldown_sec=10.0,
        )
        assert repeated is False

    def test_telemetry_injection_requires_meaningful_change(self):
        from lod.models import EphemeralContext
        from server import _build_telemetry_signature, _should_inject_telemetry_context

        previous_ctx = EphemeralContext(
            motion_state="walking",
            step_cadence=80,
            ambient_noise_db=55,
            heading=80,
            heart_rate=92,
        )
        current_ctx = EphemeralContext(
            motion_state="walking",
            step_cadence=82,  # same cadence bucket
            ambient_noise_db=58,  # same noise bucket
            heading=84,  # same 30° bucket
            heart_rate=95,  # same HR bucket
        )

        should_inject, _ = _should_inject_telemetry_context(
            previous_signature=_build_telemetry_signature(previous_ctx),
            current_signature=_build_telemetry_signature(current_ctx),
            last_injected_ts=10.0,
            now_ts=18.0,
        )
        assert should_inject is False

    def test_telemetry_force_refresh_after_timeout(self):
        from lod.models import EphemeralContext
        from server import _build_telemetry_signature, _should_inject_telemetry_context

        ctx = EphemeralContext(motion_state="walking", step_cadence=80, ambient_noise_db=55)
        signature = _build_telemetry_signature(ctx)

        should_inject, reasons = _should_inject_telemetry_context(
            previous_signature=signature,
            current_signature=signature,
            last_injected_ts=0.0,
            now_ts=30.0,
            force_refresh_sec=25.0,
        )
        assert should_inject is True
        assert "periodic_refresh" in reasons
