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
import os
from collections import deque
from datetime import datetime, timezone
from types import SimpleNamespace
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


class TestProfileEndpoints:
    """Verify REST endpoints use the shared Firestore getter."""

    def test_get_profile_uses_shared_firestore_client(self, client):
        mock_doc = MagicMock()
        mock_doc.exists = True
        mock_doc.to_dict.return_value = {
            "preferred_name": "Ada",
            "updated_at": datetime(2026, 3, 14, tzinfo=timezone.utc),
        }
        mock_db = MagicMock()
        mock_db.collection.return_value.document.return_value.get.return_value = mock_doc

        with patch("server.get_firestore_client", return_value=mock_db) as mock_get_db:
            response = client.get("/api/profile/user-123")

        assert response.status_code == 200
        assert response.json()["preferred_name"] == "Ada"
        assert response.json()["updated_at"] == "2026-03-14T00:00:00+00:00"
        mock_get_db.assert_called_once_with()

    def test_save_profile_uses_shared_firestore_client_and_invalidates_cache(self, client):
        mock_db = MagicMock()
        mock_doc_ref = MagicMock()
        mock_db.collection.return_value.document.return_value = mock_doc_ref

        with patch("server.get_firestore_client", return_value=mock_db) as mock_get_db, \
             patch("server.session_manager.invalidate_user_profile") as mock_invalidate:
            response = client.post(
                "/api/profile/user-123",
                json={"preferred_name": "Ada", "ignored_field": "noop"},
            )

        assert response.status_code == 200
        mock_get_db.assert_called_once_with()
        mock_doc_ref.set.assert_called_once()
        args, kwargs = mock_doc_ref.set.call_args
        assert args[0]["preferred_name"] == "Ada"
        assert "ignored_field" not in args[0]
        assert kwargs["merge"] is True
        mock_invalidate.assert_called_once_with("user-123")

    def test_list_users_uses_shared_firestore_client(self, client):
        doc_a = SimpleNamespace(id="alpha")
        doc_b = SimpleNamespace(id="beta")
        mock_db = MagicMock()
        mock_db.collection.return_value.stream.return_value = [doc_b, doc_a]

        with patch("server.get_firestore_client", return_value=mock_db) as mock_get_db:
            response = client.get("/api/users")

        assert response.status_code == 200
        assert response.json() == {"users": ["alpha", "beta"], "count": 2}
        mock_get_db.assert_called_once_with()


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
        assert mgr.get_handle("s1") == "h1"
        assert mgr.has_resumable_state("s1") is True

    def test_session_cleanup_expires_resumable_state(self):
        from live_api.session_manager import SessionManager
        mgr = SessionManager()
        mgr.get_session_context("s1")
        mgr.update_handle("s1", "h1")
        mgr.remove_session("s1")
        mgr._resumable_expires_at["s1"] = 0.0
        assert mgr.get_handle("s1") is None
        assert mgr.has_resumable_state("s1") is False

    def test_session_cleanup_preserves_context_without_handle(self):
        from live_api.session_manager import SessionManager
        mgr = SessionManager()
        mgr.get_session_context("s2")
        mgr.get_ephemeral_context("s2")
        mgr.remove_session("s2")
        assert mgr.has_resumable_state("s2") is True

    def test_firestore_runtime_session_roundtrip(self):
        from live_api import session_manager as sm_mod
        from live_api.session_manager import SessionManager

        store = {}

        class FakeDoc:
            def __init__(self, doc_id: str):
                self.doc_id = doc_id

            def get(self):
                data = store.get(self.doc_id)
                return type(
                    "DocSnap",
                    (),
                    {
                        "exists": data is not None,
                        "to_dict": lambda self_: data,
                    },
                )()

            def set(self, payload):
                store[self.doc_id] = payload

            def delete(self):
                store.pop(self.doc_id, None)

        class FakeCollection:
            def document(self, doc_id: str):
                return FakeDoc(doc_id)

        class FakeFirestore:
            def collection(self, _name: str):
                return FakeCollection()

        with patch.object(sm_mod, "_get_firestore", return_value=FakeFirestore()):
            mgr = SessionManager()
            ctx = mgr.get_session_context("s3")
            ctx.current_lod = 3
            eph = mgr.get_ephemeral_context("s3")
            eph.motion_state = "walking"
            mgr.update_handle("s3", "handle_xyz")
            mgr.remove_session("s3")

            restored = SessionManager()
            assert restored.get_handle("s3") == "handle_xyz"
            assert restored.get_session_context("s3").current_lod == 3
            assert restored.get_ephemeral_context("s3").motion_state == "walking"

    def test_run_config_has_vad_params(self):
        """SL-36: RunConfig should include LOD-specific VAD settings."""
        from live_api.session_manager import SessionManager
        mgr = SessionManager()
        config = mgr.get_run_config("test", lod=1)
        assert config.streaming_mode is not None
        assert config.realtime_input_config is not None

    def test_run_config_lod3_longer_silence(self):
        """LOD 3 should have longer silence duration than LOD 1."""
        from live_api.session_manager import LOD_VAD_PRESETS, SessionManager
        assert LOD_VAD_PRESETS[3]["silence_duration_ms"] > LOD_VAD_PRESETS[1]["silence_duration_ms"]

    def test_context_window_compression_guardrail(self):
        """High-risk guardrail: keep context compression enabled for long AV sessions."""
        from live_api.session_manager import SessionManager
        mgr = SessionManager()
        config = mgr.get_run_config("test", lod=2)
        compression = config.context_window_compression
        assert compression is not None
        assert compression.trigger_tokens == 80000  # E2E-006: lowered from 100K
        assert compression.sliding_window is not None
        assert compression.sliding_window.target_tokens == 50000  # E2E-006: lowered from 80K

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

    def test_run_config_enables_aad_by_default(self):
        """Live AAD defaults to enabled unless LIVE_AAD_DISABLED is set."""
        from live_api.session_manager import SessionManager
        mgr = SessionManager()
        config = mgr.get_run_config("test_aad_default", lod=2)
        aad = config.realtime_input_config.automatic_activity_detection
        assert aad is not None
        assert aad.disabled is False

    def test_run_config_aad_override_via_env(self):
        """LIVE_AAD_DISABLED=false should re-enable server-side AAD explicitly."""
        from live_api.session_manager import SessionManager
        prev = os.environ.get("LIVE_AAD_DISABLED")
        os.environ["LIVE_AAD_DISABLED"] = "false"
        try:
            mgr = SessionManager()
            config = mgr.get_run_config("test_aad_override", lod=2)
            aad = config.realtime_input_config.automatic_activity_detection
            assert aad is not None
            assert aad.disabled is False
        finally:
            if prev is None:
                os.environ.pop("LIVE_AAD_DISABLED", None)
            else:
                os.environ["LIVE_AAD_DISABLED"] = prev


# ---------------------------------------------------------------------------
# Telemetry → LOD integration
# ---------------------------------------------------------------------------


class TestTelemetryLODIntegration:
    """Verify the telemetry → LOD engine integration path."""

    def test_telemetry_to_ephemeral_to_lod(self):
        """Full pipeline: raw telemetry JSON → EphemeralContext → LOD decision."""
        from lod import decide_lod
        from lod.models import SessionContext, UserProfile
        from telemetry.telemetry_parser import parse_telemetry_to_ephemeral

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
        from lod import decide_lod
        from lod.models import SessionContext, UserProfile
        from telemetry.telemetry_parser import parse_telemetry_to_ephemeral

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

    def test_navigation_guard_blocks_without_explicit_user_intent(self):
        from server import _allow_navigation_tool_call

        history = deque([
            {"role": "user", "text": "Please summarize what you just perceived and what I should do next."},
        ])
        allow, reason = _allow_navigation_tool_call(
            func_name="navigate_to",
            func_args={"destination": "Central Park"},
            transcript_history=history,
        )
        assert allow is False
        assert reason == "navigation_tool_requires_explicit_user_request"

    def test_navigation_guard_allows_explicit_navigation_request(self):
        from server import _allow_navigation_tool_call

        history = deque([
            {"role": "user", "text": "Use walking directions from Times Square to Central Park."},
        ])
        allow, reason = _allow_navigation_tool_call(
            func_name="get_walking_directions",
            func_args={"origin": "Times Square", "destination": "Central Park"},
            transcript_history=history,
        )
        assert allow is True
        assert reason == "explicit_navigation_intent"

    def test_memory_user_id_is_redacted_in_function_log_args(self):
        from server import _sanitize_function_args_for_log

        safe = _sanitize_function_args_for_log(
            "remember_entity",
            {"user_id": "default", "name": "Central Park"},
            "test_user",
        )
        assert safe["user_id"] == "<session_user>"
        assert safe["_session_user"] == "test_user"

    def test_activity_start_resets_stale_interrupted_state(self):
        from server import _should_reset_interrupted_on_activity_start

        assert _should_reset_interrupted_on_activity_start(
            event_name="activity_start",
            interrupted=True,
        ) is True

    def test_activity_end_does_not_reset_interrupted_state(self):
        from server import _should_reset_interrupted_on_activity_start

        assert _should_reset_interrupted_on_activity_start(
            event_name="activity_end",
            interrupted=True,
        ) is False

    @pytest.mark.asyncio
    async def test_dispatch_injects_navigation_origin_and_heading_from_metadata(self):
        from server import _dispatch_function_call

        captured = {}

        def fake_tool(**kwargs):
            captured.update(kwargs)
            return {"ok": True}

        ephemeral = SimpleNamespace(
            gps=SimpleNamespace(lat=3.14159, lng=101.6869),
            heading=270.0,
        )

        with patch("server.ALL_FUNCTIONS", {"navigate_to": fake_tool}), \
             patch("server.ALL_TOOL_RUNTIME_METADATA", {
                 "navigate_to": {
                     "gps_injection": "origin_lat_lng_heading",
                     "force_user_id": False,
                 },
             }), \
             patch("server.session_manager.get_ephemeral_context", return_value=ephemeral):
            result = await _dispatch_function_call("navigate_to", {"destination": "station"}, "s1", "u1")

        assert result == {"ok": True}
        assert captured["destination"] == "station"
        assert captured["origin_lat"] == 3.14159
        assert captured["origin_lng"] == 101.6869
        assert captured["user_heading"] == 270.0

    @pytest.mark.asyncio
    async def test_dispatch_injects_lat_lng_from_metadata(self):
        from server import _dispatch_function_call

        captured = {}

        def fake_tool(**kwargs):
            captured.update(kwargs)
            return {"ok": True}

        ephemeral = SimpleNamespace(
            gps=SimpleNamespace(lat=40.0, lng=-73.0),
            heading=None,
        )

        with patch("server.ALL_FUNCTIONS", {"maps_query": fake_tool}), \
             patch("server.ALL_TOOL_RUNTIME_METADATA", {
                 "maps_query": {
                     "gps_injection": "lat_lng",
                     "force_user_id": False,
                 },
             }), \
             patch("server.session_manager.get_ephemeral_context", return_value=ephemeral):
            result = await _dispatch_function_call("maps_query", {"question": "nearest pharmacy"}, "s1", "u1")

        assert result == {"ok": True}
        assert captured["question"] == "nearest pharmacy"
        assert captured["lat"] == 40.0
        assert captured["lng"] == -73.0

    @pytest.mark.asyncio
    async def test_dispatch_keeps_explicit_lat_lng_when_gps_missing(self):
        from server import _dispatch_function_call

        captured = {}

        def fake_tool(**kwargs):
            captured.update(kwargs)
            return {"ok": True}

        ephemeral = SimpleNamespace(gps=None, heading=None)

        with patch("server.ALL_FUNCTIONS", {"preview_destination": fake_tool}), \
             patch("server.ALL_TOOL_RUNTIME_METADATA", {
                 "preview_destination": {
                     "gps_injection": "lat_lng",
                     "force_user_id": False,
                 },
             }), \
             patch("server.session_manager.get_ephemeral_context", return_value=ephemeral):
            result = await _dispatch_function_call(
                "preview_destination",
                {"lat": 1.23, "lng": 4.56, "destination": "cafe"},
                "s1",
                "u1",
            )

        assert result == {"ok": True}
        assert captured["lat"] == 1.23
        assert captured["lng"] == 4.56
        assert captured["destination"] == "cafe"

    @pytest.mark.asyncio
    async def test_dispatch_forces_memory_user_id_from_session(self):
        from server import _dispatch_function_call

        captured = {}

        def fake_tool(**kwargs):
            captured.update(kwargs)
            return {"status": "created"}

        with patch("server.ALL_FUNCTIONS", {"remember_entity": fake_tool}), \
             patch("server.ALL_TOOL_RUNTIME_METADATA", {
                 "remember_entity": {
                     "gps_injection": None,
                     "force_user_id": True,
                 },
             }):
            result = await _dispatch_function_call(
                "remember_entity",
                {"user_id": "forged-user", "content": "Walgreens"},
                "s1",
                "real-user",
            )

        assert result == {"status": "created"}
        assert captured["user_id"] == "real-user"
        assert captured["content"] == "Walgreens"

    def test_activity_start_noop_when_not_interrupted(self):
        from server import _should_reset_interrupted_on_activity_start

        assert _should_reset_interrupted_on_activity_start(
            event_name="activity_start",
            interrupted=False,
        ) is False
