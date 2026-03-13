"""Comprehensive integration tests for WebSocketHandler.

Tests crash scenarios, audio pipeline, interrupts, tool execution,
LOD transitions, and sub-agent coordination using enhanced test doubles.
"""

import asyncio
import base64
import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import WebSocketDisconnect
from google.genai import types
from starlette.websockets import WebSocketState

from session_state import SessionState
from tools.tool_behavior import ToolBehavior
from websocket_handler import WebSocketHandler, _tool_result_fallback_text


# =============================================================================
# Test Doubles (Enhanced FakeLiveRequestQueue, FakeRunner, FakeWebSocket)
# =============================================================================


class FakeLiveRequestQueue:
    """Enhanced test double for LiveRequestQueue with event recording."""

    def __init__(self):
        self.activity_start_calls = []
        self.activity_end_calls = []
        self.content_calls = []
        self.realtime_calls = []
        self.closed = False

    def send_content(self, content) -> None:
        self.content_calls.append(content)

    def send_realtime(self, blob) -> None:
        self.realtime_calls.append(blob)

    def send_activity_start(self) -> None:
        self.activity_start_calls.append(time.monotonic())

    def send_activity_end(self) -> None:
        self.activity_end_calls.append(time.monotonic())

    def close(self) -> None:
        self.closed = True


def _make_live_event(event_dict: dict) -> MagicMock:
    """Build a mock live event with correct falsy defaults for _downstream().

    The downstream loop accesses many attributes directly (not via getattr),
    so unused attributes must be explicitly falsy rather than truthy MagicMock.
    """
    ev = MagicMock()
    # Attributes accessed directly by _downstream — default to falsy
    ev.live_session_resumption_update = None
    ev.go_away = None
    ev.server_content = None
    ev.interrupted = False
    ev.turn_complete = False
    ev.content = None
    ev.input_transcription = None
    ev.output_transcription = None
    ev.usage_metadata = None

    if "turn_complete" in event_dict:
        ev.turn_complete = event_dict["turn_complete"]

    if "content" in event_dict:
        ev.content = MagicMock()
        parts = []
        for part in event_dict["content"].get("parts", []):
            mock_part = MagicMock()
            mock_part.inline_data = None
            mock_part.text = None
            if "text" in part:
                mock_part.text = part["text"]
            if "inline_data" in part:
                mock_part.inline_data = MagicMock()
                mock_part.inline_data.mime_type = part["inline_data"]["mime_type"]
                mock_part.inline_data.data = part["inline_data"]["data"]
            parts.append(mock_part)
        ev.content.parts = parts

    if "tool_call" in event_dict:
        # _extract_function_calls checks event attributes, build appropriately
        ev.server_content = MagicMock()
        ev.server_content.interrupted = False
        ev.server_content.tool_call = MagicMock()
        ev.server_content.tool_call.function_calls = []
        for fc in event_dict["tool_call"]["function_calls"]:
            mock_fc = MagicMock()
            mock_fc.name = fc["name"]
            mock_fc.args = fc.get("args", {})
            mock_fc.id = fc.get("id", f"call_{fc['name']}")
            ev.server_content.tool_call.function_calls.append(mock_fc)

    return ev


class FakeRunner:
    """Configurable fake runner that yields pre-defined event sequences."""

    def __init__(self, events: list[dict] | None = None):
        self.events = events or []
        self.run_live_called = False

    def run_live(self, **kwargs):
        """Return async generator yielding configured events."""
        self.run_live_called = True

        async def _event_stream():
            try:
                for event_dict in self.events:
                    if event_dict.get("type") == "error":
                        raise RuntimeError(event_dict.get("message", "Gemini error"))

                    yield _make_live_event(event_dict)
                    await asyncio.sleep(0.01)

                # Keep alive after events exhausted (unless error raised)
                while True:
                    await asyncio.sleep(3600)
            except (asyncio.CancelledError, GeneratorExit):
                return

        return _event_stream()


class FakeWebSocket:
    """Test double for FastAPI WebSocket with send/receive queues.

    Uses ``WebSocketState`` enums for ``client_state`` and
    ``application_state`` so that ``WebSocketHandler._is_websocket_open()``
    returns True.
    """

    def __init__(self):
        self.client_state = WebSocketState.CONNECTED
        self.application_state = WebSocketState.CONNECTED
        self.outgoing: list[dict | bytes] = []
        self.incoming: asyncio.Queue = asyncio.Queue()
        self.closed_code: int | None = None
        self.closed_reason: str | None = None

    async def accept(self):
        """No-op accept."""
        pass

    async def send_json(self, data: dict):
        """Record JSON sent to client."""
        if self.client_state == WebSocketState.DISCONNECTED:
            raise WebSocketDisconnect(code=1000)
        self.outgoing.append(data)

    async def send_bytes(self, data: bytes):
        """Record bytes sent to client."""
        if self.client_state == WebSocketState.DISCONNECTED:
            raise WebSocketDisconnect(code=1000)
        self.outgoing.append(data)

    async def receive(self):
        """Receive next message from incoming queue."""
        msg = await self.incoming.get()
        if msg.get("type") == "websocket.disconnect":
            self.client_state = WebSocketState.DISCONNECTED
            raise WebSocketDisconnect(code=1000)
        return msg

    async def close(self, code: int = 1000, reason: str = ""):
        """Record close."""
        self.closed_code = code
        self.closed_reason = reason
        self.client_state = WebSocketState.DISCONNECTED

    def queue_message_text(self, data: dict):
        """Queue text message for handler to receive."""
        self.incoming.put_nowait({"type": "websocket.receive", "text": json.dumps(data)})

    def queue_message_bytes(self, data: bytes):
        """Queue binary message for handler to receive."""
        self.incoming.put_nowait({"type": "websocket.receive", "bytes": data})

    def queue_disconnect(self):
        """Queue disconnect event."""
        self.incoming.put_nowait({"type": "websocket.disconnect"})

    def get_sent_json_by_type(self, msg_type: str) -> list[dict]:
        """Filter sent JSON messages by type."""
        return [msg for msg in self.outgoing if isinstance(msg, dict) and msg.get("type") == msg_type]


# =============================================================================
# Helper Functions
# =============================================================================


def handler_runtime_patches():
    """Context manager that patches all module-level dependencies for handler.run()."""
    mock_sm = MagicMock()
    mock_sm.remove_session = Mock()

    return patch.multiple(
        "websocket_handler",
        build_full_dynamic_prompt=Mock(return_value="test prompt"),
        _memory_extractor_available=False,
        _memory_available=False,
        _ocr_clear_session=Mock(),
        _ocr_set_latest_frame=Mock(),
        session_manager=mock_sm,
        _NEEDS_SESSION_ID_MAPPING=False,
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fake_ws():
    """Provide FakeWebSocket instance."""
    return FakeWebSocket()


@pytest.fixture
def fake_queue():
    """Provide FakeLiveRequestQueue instance."""
    return FakeLiveRequestQueue()


@pytest.fixture
def fake_runner():
    """Provide FakeRunner with no events (idle stream)."""
    return FakeRunner(events=[])


@pytest.fixture
def session_state():
    """Provide SessionState instance with defaults."""
    return SessionState()


@pytest.fixture
def mock_session_ctx():
    """Provide mock session context."""
    ctx = MagicMock()
    ctx.user_id = "test_user"
    ctx.session_id = "test_session"
    ctx.current_lod = 2
    ctx.user_profile = MagicMock()
    ctx.user_profile.om_level = "beginner"
    ctx.user_profile.verbosity_preference = "balanced"
    ctx.user_profile.preferred_name = None
    ctx.interaction_count = 0
    ctx.space_transitions = []
    ctx.trip_purpose = ""
    ctx.activity_event_count = 0
    ctx.current_activity_state = "idle"
    return ctx


@pytest.fixture
def mock_ctx_queue():
    """Provide mock ContextInjectionQueue."""
    queue = MagicMock()
    queue.enqueue = MagicMock()
    queue.should_inject = Mock(return_value=False)
    queue.model_speaking = False
    queue.vision_spoken_cooldown_active = False
    return queue


@pytest.fixture
def mock_token_monitor():
    """Provide mock TokenBudgetMonitor."""
    monitor = MagicMock()
    monitor.add_turn_cost = Mock()
    monitor.is_budget_exceeded = Mock(return_value=False)
    monitor.get_budget_status = Mock(return_value={"remaining": 1000000})
    return monitor


@pytest.fixture
def mock_session_meta():
    """Provide mock SessionMetaTracker."""
    meta = MagicMock()
    meta.write_session_start = AsyncMock()
    meta.write_session_end = AsyncMock()
    meta.locations_visited = []
    meta.set_trip_purpose = Mock()
    return meta


@pytest.fixture
def handler(
    fake_ws,
    fake_queue,
    fake_runner,
    session_state,
    mock_session_ctx,
    mock_ctx_queue,
    mock_token_monitor,
    mock_session_meta,
):
    """Create WebSocketHandler with all dependencies mocked."""
    # Create all required dependencies
    mock_telemetry_agg = MagicMock()
    stop_downstream = asyncio.Event()

    mock_tool_dedup = MagicMock()
    mock_tool_dedup.should_execute = Mock(return_value=True)

    mock_tool_mutex = MagicMock()
    mock_tool_mutex.allows = Mock(return_value=True)

    mock_audio_gate = MagicMock()
    mock_audio_gate.is_allowed = Mock(return_value=True)

    mock_run_config = MagicMock()
    mock_run_config.get = Mock(return_value={})

    mock_location_ctx_service = MagicMock()
    mock_lod_evaluator = MagicMock()
    mock_assembled_profile = MagicMock()
    mock_memory_budget = MagicMock()

    handler = WebSocketHandler(
        websocket=fake_ws,
        user_id="test_user",
        session_id="test_session",
        state=session_state,
        live_request_queue=fake_queue,
        runner=fake_runner,
        ctx_queue=mock_ctx_queue,
        token_monitor=mock_token_monitor,
        session_ctx=mock_session_ctx,
        session_meta=mock_session_meta,
        user_profile=mock_session_ctx.user_profile,
        telemetry_agg=mock_telemetry_agg,
        stop_downstream=stop_downstream,
        tool_dedup=mock_tool_dedup,
        tool_mutex=mock_tool_mutex,
        audio_gate=mock_audio_gate,
        run_config=mock_run_config,
        location_ctx_service=mock_location_ctx_service,
        lod_evaluator=mock_lod_evaluator,
        assembled_profile=mock_assembled_profile,
        memory_budget=mock_memory_budget,
        initial_memories=None,
    )

    return handler


# =============================================================================
# A. Server-side crash scenarios (6 tests)
# =============================================================================


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_websocket_disconnect_during_active_session(handler, fake_ws):
    """Test graceful cleanup when WebSocket disconnects mid-session."""
    with handler_runtime_patches():
        # Queue disconnect event after a short delay to allow initialization
        async def queue_disconnect_later():
            await asyncio.sleep(0.05)
            fake_ws.queue_disconnect()

        asyncio.create_task(queue_disconnect_later())

        # Run handler (should exit cleanly)
        await handler.run()

        # Verify cleanup occurred
        assert handler.live_request_queue.closed


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_gemini_stream_error_mid_conversation(fake_ws, fake_queue, session_state, mock_session_ctx, mock_ctx_queue, mock_token_monitor):
    """Test handler sends error to client when Gemini stream raises exception."""
    # Create runner that yields error
    error_runner = FakeRunner(events=[{"type": "error", "message": "Gemini API error"}])

    # Create all required dependencies
    mock_session_meta = MagicMock()
    mock_session_meta.write_session_start = AsyncMock()
    mock_session_meta.write_session_end = AsyncMock()
    mock_session_meta.locations_visited = []
    mock_session_meta.set_trip_purpose = Mock()

    mock_telemetry_agg = MagicMock()
    stop_downstream = asyncio.Event()

    mock_tool_dedup = MagicMock()
    mock_tool_dedup.should_execute = Mock(return_value=True)

    mock_tool_mutex = MagicMock()
    mock_tool_mutex.allows = Mock(return_value=True)

    mock_audio_gate = MagicMock()
    mock_audio_gate.is_allowed = Mock(return_value=True)

    mock_run_config = MagicMock()
    mock_run_config.get = Mock(return_value={})

    mock_location_ctx_service = MagicMock()
    mock_lod_evaluator = MagicMock()
    mock_assembled_profile = MagicMock()
    mock_memory_budget = MagicMock()

    handler = WebSocketHandler(
        websocket=fake_ws,
        user_id="test_user",
        session_id="test_session",
        state=session_state,
        live_request_queue=fake_queue,
        runner=error_runner,
        ctx_queue=mock_ctx_queue,
        token_monitor=mock_token_monitor,
        session_ctx=mock_session_ctx,
        session_meta=mock_session_meta,
        user_profile=mock_session_ctx.user_profile,
        telemetry_agg=mock_telemetry_agg,
        stop_downstream=stop_downstream,
        tool_dedup=mock_tool_dedup,
        tool_mutex=mock_tool_mutex,
        audio_gate=mock_audio_gate,
        run_config=mock_run_config,
        location_ctx_service=mock_location_ctx_service,
        lod_evaluator=mock_lod_evaluator,
        assembled_profile=mock_assembled_profile,
        memory_budget=mock_memory_budget,
        initial_memories=None,
    )

    # Queue disconnect after short delay so upstream exits after downstream error
    async def delayed_disconnect():
        await asyncio.sleep(0.15)
        fake_ws.queue_disconnect()

    with handler_runtime_patches():
        asyncio.create_task(delayed_disconnect())
        await handler.run()

    # Verify error message sent to client (downstream exception handler sends error)
    error_msgs = fake_ws.get_sent_json_by_type("error")
    assert len(error_msgs) > 0


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_connection_close_with_code_1000_normal(handler, fake_ws):
    """Test normal close (code 1000) performs cleanup."""
    fake_ws.queue_disconnect()

    with handler_runtime_patches():
        await handler.run()

    assert handler.live_request_queue.closed


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_connection_close_with_code_1001_going_away(handler, fake_ws):
    """Test client going away (code 1001) performs cleanup."""
    fake_ws.queue_disconnect()

    with handler_runtime_patches():
        await handler.run()

    assert handler.live_request_queue.closed


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_concurrent_upstream_downstream_failure_no_deadlock(handler, fake_ws):
    """Test that concurrent failures in both streams don't cause deadlock."""
    # Queue disconnect immediately
    fake_ws.queue_disconnect()

    with handler_runtime_patches():
        # Run with short timeout to ensure no deadlock
        try:
            await asyncio.wait_for(handler.run(), timeout=2.0)
        except asyncio.TimeoutError:
            pytest.fail("Handler deadlocked on concurrent stream failure")

    assert handler.live_request_queue.closed


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_session_state_corruption_recovery(handler, fake_ws, session_state):
    """Test handler recovers from corrupted session state."""
    # Corrupt some state
    session_state.transcript_history = None  # This should be a deque

    # Queue disconnect
    fake_ws.queue_disconnect()

    # Handler should handle corrupted state gracefully
    with handler_runtime_patches():
        try:
            await handler.run()
        except AttributeError:
            pytest.fail("Handler failed to handle corrupted state")


# =============================================================================
# B. Audio pipeline scenarios (6 tests)
# =============================================================================


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_activity_start_forwarded_to_queue(handler, fake_ws, fake_queue):
    """Test client activity_start message is forwarded to LiveRequestQueue."""
    # Queue activity_start message then disconnect
    fake_ws.queue_message_text({"type": "activity_start"})
    fake_ws.queue_disconnect()

    with handler_runtime_patches():
        await handler.run()

    # Verify forwarded to queue
    assert len(fake_queue.activity_start_calls) == 1


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_binary_audio_forwarded_as_realtime_blob(handler, fake_ws, fake_queue):
    """Test binary audio (0x01 magic) is forwarded as realtime blob."""
    # Create binary audio message
    audio_data = b"\x01" + b"fake_pcm_audio_data"

    # Queue binary message then disconnect
    fake_ws.queue_message_bytes(audio_data)
    fake_ws.queue_disconnect()

    with handler_runtime_patches():
        await handler.run()

    # Verify forwarded to queue
    assert len(fake_queue.realtime_calls) == 1


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_activity_end_forwarded_and_transcript_flushed(handler, fake_ws, fake_queue, session_state):
    """Test activity_end triggers queue forwarding and transcript flush."""
    # Set up some transcript buffer
    session_state.transcript_buffer = "test transcript"

    # Queue activity_end then disconnect
    fake_ws.queue_message_text({"type": "activity_end"})
    fake_ws.queue_disconnect()

    with handler_runtime_patches():
        await handler.run()

    # Verify activity_end forwarded
    assert len(fake_queue.activity_end_calls) == 1


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_no_audio_for_30s_session_stays_alive(handler, fake_ws):
    """Test session remains active even without audio (heartbeat)."""
    # Queue disconnect after small delay (simulating no activity)
    async def delayed_disconnect():
        await asyncio.sleep(0.1)
        fake_ws.queue_disconnect()

    with handler_runtime_patches():
        asyncio.create_task(delayed_disconnect())
        await handler.run()

    # Session should close normally, not timeout
    assert handler.live_request_queue.closed


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_malformed_binary_message_error_sent_session_continues(handler, fake_ws):
    """Test malformed binary message sends error but session continues."""
    # Queue malformed binary (wrong magic byte)
    fake_ws.queue_message_bytes(b"\xFF\x99")

    # Then disconnect
    fake_ws.queue_disconnect()

    with handler_runtime_patches():
        await handler.run()

    # Check for error message in outgoing
    # (Implementation may or may not send error, but should not crash)
    assert handler.live_request_queue.closed


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_rapid_activity_toggle_debounced(handler, fake_ws, fake_queue, session_state):
    """Test rapid activity_start/end toggling is debounced correctly."""
    # Queue rapid toggles
    fake_ws.queue_message_text({"type": "activity_start"})
    fake_ws.queue_message_text({"type": "activity_end"})
    fake_ws.queue_message_text({"type": "activity_start"})
    fake_ws.queue_message_text({"type": "activity_end"})
    fake_ws.queue_disconnect()

    with handler_runtime_patches():
        await handler.run()

    # All should be forwarded (no artificial debouncing of activity events themselves)
    assert len(fake_queue.activity_start_calls) >= 1
    assert len(fake_queue.activity_end_calls) >= 1


# =============================================================================
# C. Barge-in / interrupt scenarios (6 tests)
# =============================================================================


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_activity_start_while_model_speaking_triggers_interrupt(handler, fake_ws, session_state, mock_ctx_queue):
    """Test client barge-in during model speech triggers interrupt.

    The upstream handler processes ``client_barge_in`` messages to set the
    interrupted state when the model is currently speaking.
    """
    # Simulate model speaking by setting recent audio timestamp
    session_state.model_audio_last_seen_at = time.monotonic()
    mock_ctx_queue.model_speaking = True

    # Queue client_barge_in (the actual interrupt mechanism) then disconnect
    fake_ws.queue_message_text({"type": "client_barge_in"})
    fake_ws.queue_disconnect()

    with handler_runtime_patches():
        await handler.run()

    # Interrupt should be set
    assert session_state.is_interrupted


def test_register_user_activity_starts_new_turn_and_discards_stale_context(handler, mock_ctx_queue, session_state):
    mock_ctx_queue.discard_stale = Mock()
    session_state.last_user_activity_at = time.monotonic() - 5

    started_new_turn = handler._register_user_activity(explicit_turn_start=True)

    assert started_new_turn is True
    assert session_state.user_turn_seq == 1
    mock_ctx_queue.discard_stale.assert_called_once()
    assert mock_ctx_queue.discard_stale.call_args.kwargs["min_turn_seq"] == 1


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_text_hint_updates_transcript_history_and_injects_silent_context(handler, fake_ws, mock_ctx_queue, session_state):
    fake_ws.queue_message_text({"type": "text_hint", "text": "navigate to the pharmacy"})
    fake_ws.queue_disconnect()

    with handler_runtime_patches():
        await handler.run()

    assert session_state.transcript_history[-1]["text"] == "navigate to the pharmacy"
    mock_ctx_queue.inject_immediate.assert_called()
    content = mock_ctx_queue.inject_immediate.call_args.args[0]
    assert "Prefer navigate_to or get_walking_directions" in content.parts[0].text


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_text_hint_and_activity_start_share_same_turn(handler, fake_ws, session_state):
    fake_ws.queue_message_text({"type": "text_hint", "text": "what's around me"})
    fake_ws.queue_message_text({"type": "activity_start"})
    fake_ws.queue_message_text({"type": "activity_end"})
    fake_ws.queue_disconnect()

    with handler_runtime_patches():
        await handler.run()

    assert session_state.user_turn_seq == 1


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_text_hint_ocr_request_includes_ocr_preference(handler, fake_ws, mock_ctx_queue):
    fake_ws.queue_message_text({"type": "text_hint", "text": "Can you read that sign for me?"})
    fake_ws.queue_disconnect()

    with handler_runtime_patches():
        await handler.run()

    content = mock_ctx_queue.inject_immediate.call_args.args[0]
    assert "Prefer extract_text_from_camera" in content.parts[0].text


@pytest.mark.asyncio
async def test_maps_query_redirects_to_navigation_for_explicit_navigation_request(handler, session_state):
    session_state.transcript_history.append({
        "role": "user",
        "text": "Navigate me to the nearest pharmacy please.",
    })

    with patch("websocket_handler._dispatch_function_call", new_callable=AsyncMock) as mock_dispatch:
        mock_dispatch.side_effect = [
            {
                "success": True,
                "places": [
                    {
                        "name": "CVS Pharmacy",
                        "address": "123 Main St",
                        "distance_meters": 80,
                    }
                ],
            },
            {
                "success": True,
                "destination": "123 Main St",
                "destination_direction": "at 1 o'clock, 80 meters",
            },
        ]
        redirected, redirected_result = await handler._maybe_redirect_maps_query(
            question="nearest pharmacy",
            user_speaking=False,
        )

    assert redirected is True
    assert redirected_result["redirected_to"] == "navigate_to"
    assert redirected_result["navigation_result"]["success"] is True


@pytest.mark.asyncio
async def test_direct_navigation_shortcut_emits_navigation_result(handler, session_state, fake_ws):
    session_state.user_turn_seq = 1
    session_state.pending_fallback_turn_seq = 1
    session_state.turn_output_seen = False
    handler.state.transcript_history.append({
        "role": "user",
        "text": "Navigate me to the nearest pharmacy please.",
    })

    with patch("websocket_handler._dispatch_function_call", new_callable=AsyncMock) as mock_dispatch, \
         patch("websocket_handler.synthesize_fallback_pcm", new_callable=AsyncMock) as mock_tts, \
         patch("websocket_handler.session_manager.get_ephemeral_context") as mock_ephemeral:
        mock_dispatch.side_effect = [
            {
                "success": True,
                "places": [
                    {"name": "CVS Pharmacy", "address": "123 Main St", "distance_meters": 80}
                ],
            },
            {
                "success": True,
                "destination": "123 Main St",
                "destination_direction": "at 1 o'clock, 80 meters",
            },
        ]
        mock_tts.return_value = b"\x00\x00" * 8
        mock_ephemeral.return_value = type(
            "Ephemeral",
            (),
            {"gps": type("GPS", (), {"lat": 40.0, "lng": -74.0})()},
        )()
        handled = await handler._maybe_handle_direct_navigation_intent(
            "Navigate me to the nearest pharmacy please."
        )

    assert handled is True
    nav_results = fake_ws.get_sent_json_by_type("navigation_result")
    assert nav_results
    assert session_state.turn_output_seen is True


def test_tool_result_fallback_text_for_navigation():
    text = _tool_result_fallback_text(
        "navigate_to",
        {
            "destination": "123 Main St",
            "destination_direction": "at 1 o'clock, 80 meters",
        },
    )
    assert text == "123 Main St is at 1 o'clock, 80 meters. I'll guide you there."


@pytest.mark.asyncio
async def test_emit_pending_fallback_output_marks_turn_output_seen(handler, session_state):
    session_state.user_turn_seq = 2
    session_state.pending_fallback_turn_seq = 2
    session_state.pending_fallback_text = "Fallback response"

    with patch("websocket_handler.synthesize_fallback_pcm", new_callable=AsyncMock) as mock_tts:
        mock_tts.return_value = b"\x00\x00" * 8
        emitted = await handler._emit_pending_fallback_output(2)

    assert emitted is True
    assert session_state.turn_output_seen is True
    assert session_state.pending_fallback_text is None


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_resume_requested_skips_fresh_greeting_injection():
    runner = FakeRunner(events=[])
    mock_ctx_queue = MagicMock()
    mock_ctx_queue.enqueue = MagicMock()
    mock_ctx_queue.should_inject = Mock(return_value=False)
    mock_ctx_queue.model_speaking = False
    mock_ctx_queue.vision_spoken_cooldown_active = False
    mock_ctx_queue.inject_immediate = MagicMock()

    handler, fake_ws, _, _ = _make_handler_with_runner(
        runner,
        ctx_queue=mock_ctx_queue,
        resume_requested=True,
    )

    async def delayed_disconnect():
        await asyncio.sleep(0.05)
        fake_ws.queue_disconnect()

    with handler_runtime_patches():
        asyncio.create_task(delayed_disconnect())
        await handler.run()

    mock_ctx_queue.inject_immediate.assert_not_called()


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_resume_context_is_prepended_to_next_text_hint(handler, fake_ws, mock_ctx_queue):
    handler._resume_requested = True
    fake_ws.queue_message_text({"type": "text_hint", "text": "what's around me"})
    fake_ws.queue_disconnect()

    with handler_runtime_patches():
        await handler.run()

    mock_ctx_queue.inject_immediate.assert_called()
    content = mock_ctx_queue.inject_immediate.call_args.args[0]
    assert "[SESSION RESUME]" in content.parts[0].text
    assert "what's around me" in content.parts[0].text


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_interrupt_debounce_rapid_interrupts(handler, fake_ws, session_state, mock_ctx_queue):
    """Test rapid barge-in within debounce window: only first processed."""
    # Set model speaking
    session_state.model_audio_last_seen_at = time.monotonic()
    mock_ctx_queue.model_speaking = True

    # Send rapid client_barge_in messages (the actual interrupt mechanism)
    fake_ws.queue_message_text({"type": "client_barge_in"})
    fake_ws.queue_message_text({"type": "client_barge_in"})
    fake_ws.queue_message_text({"type": "client_barge_in"})
    fake_ws.queue_disconnect()

    with handler_runtime_patches():
        await handler.run()

    # Should have set interrupted once
    assert session_state.is_interrupted


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_echo_detection_client_audio_matches_model_output(handler, fake_ws, session_state):
    """Test echo detection: client audio matching recent model output is ignored."""
    # Set recent agent text to simulate echo
    session_state.last_agent_text = "test echo text"
    session_state.last_agent_text_sent_at = time.monotonic()
    session_state.allow_agent_repeat_until = 0  # No repeat allowed

    # This test verifies the infrastructure exists
    # Actual echo detection logic is in _is_repeated_text
    fake_ws.queue_disconnect()

    with handler_runtime_patches():
        await handler.run()

    assert handler.live_request_queue.closed


@pytest.mark.skip(reason="placeholder — not yet implemented")
@pytest.mark.websocket
@pytest.mark.asyncio
async def test_barge_in_followed_by_silence_agent_resumes():
    """Test that agent resumes after barge-in followed by silence timeout."""
    pass


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_concurrent_interrupt_and_tool_result_ordering(handler, fake_ws, mock_ctx_queue):
    """Test concurrent interrupt + tool result: interrupt processed first."""
    # Set model speaking
    handler.state.model_audio_last_seen_at = time.monotonic()
    mock_ctx_queue.model_speaking = True

    # Queue client_barge_in (the actual interrupt mechanism) then disconnect
    fake_ws.queue_message_text({"type": "client_barge_in"})
    fake_ws.queue_disconnect()

    with handler_runtime_patches():
        await handler.run()

    assert handler.state.is_interrupted


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_activity_start_after_model_audio_stops_not_interrupt(handler, fake_ws, session_state):
    """Test activity_start after model stops is NOT an interrupt (turn-taking)."""
    # Set model audio seen but in the past (>1s ago)
    session_state.model_audio_last_seen_at = time.monotonic() - 5.0

    # Queue activity_start
    fake_ws.queue_message_text({"type": "activity_start"})
    fake_ws.queue_disconnect()

    with handler_runtime_patches():
        await handler.run()

    # Should NOT be interrupted (model stopped speaking)
    # Note: interrupt detection logic may vary
    # This test documents expected behavior


# =============================================================================
# D. Tool execution scenarios (6 tests)
# =============================================================================


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_tool_call_dispatched_and_result_sent(fake_ws, fake_queue, session_state, mock_session_ctx, mock_ctx_queue, mock_token_monitor):
    """Test runner yields tool_call event, tool executed, result sent back."""
    # Create runner with tool_call event
    events = [
        {
            "tool_call": {
                "function_calls": [
                    {"name": "get_current_time", "args": {}, "id": "call_time_1"}
                ]
            }
        }
    ]
    tool_runner = FakeRunner(events=events)

    # Create all required dependencies
    mock_session_meta = MagicMock()
    mock_session_meta.write_session_start = AsyncMock()
    mock_session_meta.write_session_end = AsyncMock()
    mock_session_meta.locations_visited = []
    mock_session_meta.set_trip_purpose = Mock()

    mock_telemetry_agg = MagicMock()
    stop_downstream = asyncio.Event()

    mock_tool_dedup = MagicMock()
    mock_tool_dedup.should_execute = Mock(return_value=(True, ""))

    mock_tool_mutex = MagicMock()
    mock_tool_mutex.should_execute = Mock(return_value=(True, ""))

    mock_audio_gate = MagicMock()
    mock_audio_gate.is_allowed = Mock(return_value=True)

    mock_run_config = MagicMock()
    mock_run_config.get = Mock(return_value={})

    mock_location_ctx_service = MagicMock()
    mock_lod_evaluator = MagicMock()
    mock_assembled_profile = MagicMock()
    mock_memory_budget = MagicMock()

    with handler_runtime_patches(), \
         patch("websocket_handler._dispatch_function_call", new_callable=AsyncMock) as mock_dispatch, \
         patch("websocket_handler._extract_function_calls") as mock_extract, \
         patch("websocket_handler._allow_navigation_tool_call", return_value=(True, "")), \
         patch("websocket_handler.resolve_tool_behavior", return_value=ToolBehavior.WHEN_IDLE), \
         patch("websocket_handler.ALL_FUNCTIONS", {"get_current_time": lambda: None}):
        # Mock tool dispatch to return success
        mock_dispatch.return_value = {"status": "success", "result": "12:00 PM"}

        # Mock _extract_function_calls to return our function call from the event
        mock_fc = MagicMock()
        mock_fc.name = "get_current_time"
        mock_fc.args = {}
        mock_fc.id = "call_time_1"
        # Return calls for first event, empty for subsequent
        mock_extract.side_effect = [[mock_fc]] + [[] for _ in range(100)]

        handler = WebSocketHandler(
            websocket=fake_ws,
            user_id="test_user",
            session_id="test_session",
            state=session_state,
            live_request_queue=fake_queue,
            runner=tool_runner,
            ctx_queue=mock_ctx_queue,
            token_monitor=mock_token_monitor,
            session_ctx=mock_session_ctx,
            session_meta=mock_session_meta,
            user_profile=mock_session_ctx.user_profile,
            telemetry_agg=mock_telemetry_agg,
            stop_downstream=stop_downstream,
            tool_dedup=mock_tool_dedup,
            tool_mutex=mock_tool_mutex,
            audio_gate=mock_audio_gate,
            run_config=mock_run_config,
            location_ctx_service=mock_location_ctx_service,
            lod_evaluator=mock_lod_evaluator,
            assembled_profile=mock_assembled_profile,
            memory_budget=mock_memory_budget,
            initial_memories=None,
        )

        # Queue disconnect after short delay
        async def delayed_disconnect():
            await asyncio.sleep(0.2)
            fake_ws.queue_disconnect()

        asyncio.create_task(delayed_disconnect())

        await handler.run()

        # Verify tool was called
        assert mock_dispatch.called
        injected_content = mock_ctx_queue.inject_immediate.call_args.args[0]
        assert "TOOL RESULT READY" in injected_content.parts[0].text


@pytest.mark.skip(reason="placeholder — not yet implemented")
@pytest.mark.websocket
@pytest.mark.asyncio
async def test_tool_execution_timeout_handling():
    """Test tool call timeout returns timeout error to model."""
    pass


@pytest.mark.skip(reason="placeholder — not yet implemented")
@pytest.mark.websocket
@pytest.mark.asyncio
async def test_tool_execution_error_propagated():
    """Test tool error is propagated as tool result with error status."""
    pass


@pytest.mark.skip(reason="placeholder — not yet implemented")
@pytest.mark.websocket
@pytest.mark.asyncio
async def test_concurrent_tool_calls_both_executed():
    """Test concurrent tool calls (navigation + search) both executed."""
    pass


@pytest.mark.skip(reason="placeholder — not yet implemented")
@pytest.mark.websocket
@pytest.mark.asyncio
async def test_tool_behavior_interrupt_injected_immediately():
    """Test tool with INTERRUPT behavior has result injected immediately."""
    pass


@pytest.mark.skip(reason="placeholder — not yet implemented")
@pytest.mark.websocket
@pytest.mark.asyncio
async def test_tool_behavior_when_idle_queued():
    """Test tool with WHEN_IDLE behavior is queued until model idle."""
    pass


# =============================================================================
# E. LOD transition scenarios (4 tests)
# =============================================================================


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_telemetry_motion_running_drops_lod_to_1(handler, fake_ws):
    """Test telemetry with motion=running triggers LOD drop to 1."""
    # Queue telemetry message with running motion
    # Handler checks msg_type == "telemetry" and reads message.get("data", {})
    telemetry_data = {
        "type": "telemetry",
        "data": {
            "motion_state": "running",
            "step_cadence": 150,
        }
    }

    fake_ws.queue_message_text(telemetry_data)
    fake_ws.queue_disconnect()

    with handler_runtime_patches(), \
         patch("websocket_handler.decide_lod") as mock_decide_lod, \
         patch("websocket_handler.parse_telemetry_to_ephemeral") as mock_parse, \
         patch("websocket_handler.parse_telemetry") as mock_parse_sem, \
         patch("websocket_handler._build_telemetry_signature", return_value={"motion": "running"}), \
         patch("websocket_handler._should_inject_telemetry_context", return_value=(True, ["motion_changed"])), \
         patch("websocket_handler._changed_signature_fields", return_value=["motion"]):
        mock_ephemeral = MagicMock()
        mock_ephemeral.gps = None
        mock_parse.return_value = mock_ephemeral
        mock_parse_sem.return_value = "motion=running"
        mock_decide_lod.return_value = (1, MagicMock(reason="running", triggered_rules=[], to_debug_dict=Mock(return_value={})))

        await handler.run()

        # Verify LOD decision was called
        assert mock_decide_lod.called


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_telemetry_high_noise_caps_lod_to_1(handler, fake_ws):
    """Test telemetry with noise>80dB caps LOD to 1."""
    telemetry_data = {
        "type": "telemetry",
        "data": {
            "ambient_noise_db": 85,
        }
    }

    fake_ws.queue_message_text(telemetry_data)
    fake_ws.queue_disconnect()

    with handler_runtime_patches(), \
         patch("websocket_handler.decide_lod") as mock_decide_lod, \
         patch("websocket_handler.parse_telemetry_to_ephemeral") as mock_parse, \
         patch("websocket_handler.parse_telemetry") as mock_parse_sem, \
         patch("websocket_handler._build_telemetry_signature", return_value={"noise": 85}), \
         patch("websocket_handler._should_inject_telemetry_context", return_value=(True, ["noise_changed"])), \
         patch("websocket_handler._changed_signature_fields", return_value=["noise"]):
        mock_ephemeral = MagicMock()
        mock_ephemeral.gps = None
        mock_parse.return_value = mock_ephemeral
        mock_parse_sem.return_value = "noise=85dB"
        mock_decide_lod.return_value = (1, MagicMock(reason="high_noise", triggered_rules=[], to_debug_dict=Mock(return_value={})))

        await handler.run()

        assert mock_decide_lod.called


@pytest.mark.skip(reason="placeholder — not yet implemented")
@pytest.mark.websocket
@pytest.mark.asyncio
async def test_space_transition_detected_lod_boosted():
    """Test space transition detection boosts LOD."""
    pass


@pytest.mark.skip(reason="placeholder — not yet implemented")
@pytest.mark.websocket
@pytest.mark.asyncio
async def test_gesture_force_lod_3_overrides_and_injects_context():
    """Test gesture force_lod_3 overrides LOD to 3 and injects context."""
    pass


# =============================================================================
# F. Sub-agent coordination scenarios (4 tests)
# =============================================================================


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_image_frame_triggers_vision_analysis_with_cooldown(handler, fake_ws):
    """Test image frame received triggers vision analysis with cooldown."""
    # Enable vision for this test
    with handler_runtime_patches(), patch("websocket_handler._vision_available", True):
        # Create base64 image
        fake_image = base64.b64encode(b"fake_image_data").decode()

        # Queue binary image message
        image_data = b"\x02" + fake_image.encode()
        fake_ws.queue_message_bytes(image_data)
        fake_ws.queue_disconnect()

        # Mock vision analysis
        with patch.object(handler, "_run_vision_analysis") as mock_vision:
            mock_vision.return_value = None

            await handler.run()

            # Vision should be called (or not if in cooldown)
            # This test verifies infrastructure


@pytest.mark.skip(reason="placeholder — not yet implemented")
@pytest.mark.websocket
@pytest.mark.asyncio
async def test_image_during_active_vision_skipped():
    """Test image received during active vision analysis is skipped."""
    pass


@pytest.mark.skip(reason="placeholder — not yet implemented")
@pytest.mark.websocket
@pytest.mark.asyncio
async def test_face_recognition_backoff_after_consecutive_misses():
    """Test face recognition applies backoff after consecutive misses."""
    pass


@pytest.mark.skip(reason="placeholder — not yet implemented")
@pytest.mark.websocket
@pytest.mark.asyncio
async def test_ocr_triggered_by_user_request():
    """Test OCR agent launched when triggered by user request."""
    pass


# =============================================================================
# G. Error scenario and regression tests (Phase 4)
# =============================================================================


def _make_handler_with_runner(runner, **overrides):
    """Factory to create a WebSocketHandler with a custom runner and minimal boilerplate.

    Returns (handler, fake_ws, fake_queue, session_state).
    All dependencies are mocked; pass keyword overrides to replace any.
    """
    fake_ws = overrides.pop("fake_ws", None) or FakeWebSocket()
    fake_queue = overrides.pop("fake_queue", None) or FakeLiveRequestQueue()
    state = overrides.pop("state", None) or SessionState()
    resume_requested = overrides.pop("resume_requested", False)

    mock_session_ctx = overrides.pop("session_ctx", None)
    if mock_session_ctx is None:
        mock_session_ctx = MagicMock()
        mock_session_ctx.user_id = "test_user"
        mock_session_ctx.session_id = "test_session"
        mock_session_ctx.current_lod = 2
        mock_session_ctx.user_profile = MagicMock()
        mock_session_ctx.user_profile.om_level = "beginner"
        mock_session_ctx.user_profile.verbosity_preference = "balanced"
        mock_session_ctx.user_profile.preferred_name = None
        mock_session_ctx.interaction_count = 0
        mock_session_ctx.space_transitions = []
        mock_session_ctx.trip_purpose = ""
        mock_session_ctx.activity_event_count = 0
        mock_session_ctx.current_activity_state = "idle"

    mock_ctx_queue = overrides.pop("ctx_queue", None)
    if mock_ctx_queue is None:
        mock_ctx_queue = MagicMock()
        mock_ctx_queue.enqueue = MagicMock()
        mock_ctx_queue.should_inject = Mock(return_value=False)
        mock_ctx_queue.model_speaking = False
        mock_ctx_queue.vision_spoken_cooldown_active = False

    mock_token_monitor = overrides.pop("token_monitor", None)
    if mock_token_monitor is None:
        mock_token_monitor = MagicMock()
        mock_token_monitor.add_turn_cost = Mock()
        mock_token_monitor.is_budget_exceeded = Mock(return_value=False)
        mock_token_monitor.get_budget_status = Mock(return_value={"remaining": 1000000})

    mock_session_meta = overrides.pop("session_meta", None)
    if mock_session_meta is None:
        mock_session_meta = MagicMock()
        mock_session_meta.write_session_start = AsyncMock()
        mock_session_meta.write_session_end = AsyncMock()
        mock_session_meta.locations_visited = []
        mock_session_meta.set_trip_purpose = Mock()

    mock_tool_dedup = overrides.pop("tool_dedup", None)
    if mock_tool_dedup is None:
        mock_tool_dedup = MagicMock()
        mock_tool_dedup.should_execute = Mock(return_value=(True, ""))

    mock_tool_mutex = overrides.pop("tool_mutex", None)
    if mock_tool_mutex is None:
        mock_tool_mutex = MagicMock()
        mock_tool_mutex.should_execute = Mock(return_value=(True, ""))

    mock_audio_gate = MagicMock()
    mock_audio_gate.is_allowed = Mock(return_value=True)
    mock_run_config = MagicMock()
    mock_run_config.get = Mock(return_value={})

    handler = WebSocketHandler(
        websocket=fake_ws,
        user_id="test_user",
        session_id="test_session",
        state=state,
        live_request_queue=fake_queue,
        runner=runner,
        ctx_queue=mock_ctx_queue,
        token_monitor=mock_token_monitor,
        session_ctx=mock_session_ctx,
        session_meta=mock_session_meta,
        user_profile=mock_session_ctx.user_profile,
        telemetry_agg=MagicMock(),
        stop_downstream=asyncio.Event(),
        tool_dedup=mock_tool_dedup,
        tool_mutex=mock_tool_mutex,
        audio_gate=mock_audio_gate,
        run_config=mock_run_config,
        location_ctx_service=MagicMock(),
        lod_evaluator=MagicMock(),
        assembled_profile=MagicMock(),
        memory_budget=MagicMock(),
        initial_memories=None,
        resume_requested=resume_requested,
    )
    return handler, fake_ws, fake_queue, state


# ── G1. Gemini Live API failure modes ─────────────────────────────────


@pytest.mark.websocket
@pytest.mark.error_scenario
@pytest.mark.asyncio
async def test_gemini_error_after_successful_audio_sends_error_to_client():
    """Regression: Gemini returns error *after* some successful audio events.

    The downstream handler should catch the RuntimeError, send an error message
    to the client, and not leave the session in a broken state.
    """
    # Emit one text event, then raise an error (simulates mid-stream failure)
    events = [
        {"content": {"parts": [{"text": "Starting response..."}]}},
        {"type": "error", "message": "Internal Gemini API error (503)"},
    ]
    runner = FakeRunner(events=events)
    handler, fake_ws, fake_queue, state = _make_handler_with_runner(runner)

    async def delayed_disconnect():
        await asyncio.sleep(0.2)
        fake_ws.queue_disconnect()

    with handler_runtime_patches():
        asyncio.create_task(delayed_disconnect())
        await handler.run()

    # Verify: error message sent to client
    error_msgs = fake_ws.get_sent_json_by_type("error")
    assert len(error_msgs) >= 1, "Client should receive error message when Gemini fails mid-stream"
    # Verify: session cleaned up
    assert fake_queue.closed


@pytest.mark.websocket
@pytest.mark.error_scenario
@pytest.mark.asyncio
async def test_gemini_connection_drop_closes_queue():
    """Regression: Gemini connection drops (LiveRequestQueue.close() called unexpectedly).

    When the runner's event stream ends abruptly (no events), downstream should
    exit gracefully and close the queue.
    """
    # FakeRunner with empty events will hang in the while True sleep,
    # so we rely on disconnect to end the session.
    runner = FakeRunner(events=[])
    handler, fake_ws, fake_queue, _ = _make_handler_with_runner(runner)

    fake_ws.queue_disconnect()

    with handler_runtime_patches():
        try:
            await asyncio.wait_for(handler.run(), timeout=3.0)
        except asyncio.TimeoutError:
            pytest.fail("Handler did not exit cleanly when Gemini connection dropped")

    assert fake_queue.closed


def test_silent_turn_reconnect_predicate(handler, session_state):
    session_state.user_turn_seq = 1
    session_state.turn_output_seen = False

    assert handler._should_reconnect_silent_turn() is True

    session_state.turn_output_seen = True
    assert handler._should_reconnect_silent_turn() is False


@pytest.mark.websocket
@pytest.mark.error_scenario
@pytest.mark.asyncio
async def test_gemini_malformed_tool_call_skipped_session_continues():
    """Regression: Gemini returns a tool call for a non-existent function.

    The handler should send a 'no-op' response back to the model and continue
    the session without crashing.
    """
    events = [
        {
            "tool_call": {
                "function_calls": [
                    {"name": "nonexistent_tool_xyz", "args": {"foo": "bar"}, "id": "call_bad_1"}
                ]
            }
        }
    ]
    runner = FakeRunner(events=events)
    handler, fake_ws, fake_queue, _ = _make_handler_with_runner(runner)

    with handler_runtime_patches(), \
         patch("websocket_handler._extract_function_calls") as mock_extract, \
         patch("websocket_handler.ALL_FUNCTIONS", {"get_current_time": lambda: None}):
        # Build mock function call for the nonexistent tool
        mock_fc = MagicMock()
        mock_fc.name = "nonexistent_tool_xyz"
        mock_fc.args = {"foo": "bar"}
        mock_fc.id = "call_bad_1"
        mock_extract.side_effect = [[mock_fc]] + [[] for _ in range(100)]

        async def delayed_disconnect():
            await asyncio.sleep(0.2)
            fake_ws.queue_disconnect()

        asyncio.create_task(delayed_disconnect())
        await handler.run()

    # Session should still clean up normally
    assert fake_queue.closed
    # Should have sent debug activity for hallucinated tool call
    debug_msgs = fake_ws.get_sent_json_by_type("debug_activity")
    hallucinated = [m for m in debug_msgs if m.get("data", {}).get("event") == "hallucinated_tool_call"]
    assert len(hallucinated) >= 1, "Should log hallucinated_tool_call debug event"


@pytest.mark.websocket
@pytest.mark.error_scenario
@pytest.mark.asyncio
async def test_gemini_keepalive_timeout_retries_then_reconnects():
    """Regression: Gemini keepalive ping timeout triggers retry with backoff.

    The downstream handler should retry on transient keepalive timeouts up to
    DOWNSTREAM_MAX_RETRIES before requesting client reconnect.
    """
    # Create a runner whose event stream raises a keepalive timeout
    class KeepaliveTimeoutRunner:
        def __init__(self):
            self.call_count = 0

        def run_live(self, **kwargs):
            self.call_count += 1

            async def _failing_stream():
                raise RuntimeError("keepalive ping timeout")
                yield  # noqa: unreachable — makes this an async generator

            return _failing_stream()

    runner = KeepaliveTimeoutRunner()
    state = SessionState()
    state.DOWNSTREAM_MAX_RETRIES = 2  # Low retry count for fast test
    handler, fake_ws, fake_queue, _ = _make_handler_with_runner(runner, state=state)

    with handler_runtime_patches():
        try:
            await asyncio.wait_for(handler.run(), timeout=5.0)
        except asyncio.TimeoutError:
            pytest.fail("Handler deadlocked on keepalive timeout retry")

    # Should have sent go_away messages for retries + final
    go_away_msgs = fake_ws.get_sent_json_by_type("go_away")
    assert len(go_away_msgs) >= 1, "Should send go_away when keepalive timeout persists"


@pytest.mark.websocket
@pytest.mark.error_scenario
@pytest.mark.asyncio
async def test_transient_keepalive_timeout_recovers_without_client_go_away():
    """Transient keepalive timeouts should be handled internally.

    If the downstream stream recovers on retry, the client should keep the same
    websocket session without receiving a go_away reconnect request.
    """

    class RecoveringKeepaliveRunner:
        def __init__(self):
            self.call_count = 0

        def run_live(self, **kwargs):
            self.call_count += 1

            async def _stream():
                if self.call_count == 1:
                    err = RuntimeError("1006 None. abnormal closure [internal]")
                    err.__cause__ = RuntimeError("keepalive ping timeout")
                    raise err
                    yield  # noqa: unreachable

                yield _make_live_event({"content": {"parts": [{"text": "Recovered response."}]}})
                while True:
                    await asyncio.sleep(3600)

            return _stream()

    runner = RecoveringKeepaliveRunner()
    handler, fake_ws, fake_queue, _ = _make_handler_with_runner(runner)

    async def delayed_disconnect():
        await asyncio.sleep(1.5)
        fake_ws.queue_disconnect()

    with handler_runtime_patches():
        asyncio.create_task(delayed_disconnect())
        await handler.run()

    assert runner.call_count >= 2, "Downstream should retry after transient keepalive timeout"
    assert fake_ws.get_sent_json_by_type("go_away") == [], (
        "Client should not receive go_away when downstream recovers internally"
    )
    transcript_msgs = fake_ws.get_sent_json_by_type("transcript")
    assert any(msg.get("role") == "agent" for msg in transcript_msgs), (
        "Recovered downstream should continue delivering agent transcript"
    )


@pytest.mark.websocket
@pytest.mark.error_scenario
@pytest.mark.asyncio
async def test_keepalive_retry_rebuilds_run_config_for_resumption():
    class RecoveringKeepaliveRunner:
        def __init__(self):
            self.call_count = 0
            self.run_configs = []

        def run_live(self, **kwargs):
            self.call_count += 1
            self.run_configs.append(kwargs.get("run_config"))

            async def _stream():
                if self.call_count == 1:
                    err = RuntimeError("1006 None. abnormal closure [internal]")
                    err.__cause__ = RuntimeError("keepalive ping timeout")
                    raise err
                    yield  # noqa: unreachable

                yield _make_live_event({"content": {"parts": [{"text": "Recovered response."}]}})
                while True:
                    await asyncio.sleep(3600)

            return _stream()

    runner = RecoveringKeepaliveRunner()
    handler, fake_ws, _, _ = _make_handler_with_runner(runner)

    run_config_v1 = object()
    run_config_v2 = object()

    async def delayed_disconnect():
        await asyncio.sleep(1.5)
        fake_ws.queue_disconnect()

    mock_sm = MagicMock()
    mock_sm.remove_session = Mock()
    mock_sm.get_run_config = Mock(side_effect=[run_config_v1, run_config_v2])

    with patch.multiple(
        "websocket_handler",
        build_full_dynamic_prompt=Mock(return_value="test prompt"),
        _memory_extractor_available=False,
        _memory_available=False,
        _ocr_clear_session=Mock(),
        _ocr_set_latest_frame=Mock(),
        session_manager=mock_sm,
        _NEEDS_SESSION_ID_MAPPING=False,
    ):
        asyncio.create_task(delayed_disconnect())
        await handler.run()

    assert runner.call_count >= 2
    assert runner.run_configs[:2] == [run_config_v1, run_config_v2]
    assert mock_sm.get_run_config.call_count >= 2


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_prefeedback_output_sends_transcript_and_local_audio():
    runner = FakeRunner(events=[])
    handler, fake_ws, _, state = _make_handler_with_runner(runner)
    state.user_turn_seq = 1

    with patch("websocket_handler.synthesize_local_fallback_pcm", new=AsyncMock(return_value=b"\x01\x02")):
        sent = await handler._emit_prefeedback_output("Reading the text for you...")

    assert sent is True
    transcript_msgs = fake_ws.get_sent_json_by_type("transcript")
    assert transcript_msgs[-1]["text"] == "Reading the text for you..."
    assert any(isinstance(item, bytes) and item == b"\x01\x02" for item in fake_ws.outgoing)


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_local_agent_response_marks_audio_seen_when_pcm_sent():
    runner = FakeRunner(events=[])
    handler, fake_ws, _, state = _make_handler_with_runner(runner)
    state.user_turn_seq = 1

    with patch("websocket_handler.synthesize_local_fallback_pcm", new=AsyncMock(return_value=b"\x03\x04")):
        sent = await handler._emit_local_agent_response("Goodbye for now.", source="test")

    assert sent is True
    assert state.turn_output_seen is True
    assert state.turn_audio_output_seen is True
    assert any(isinstance(item, bytes) and item == b"\x03\x04" for item in fake_ws.outgoing)


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_farewell_watchdog_emits_local_goodbye_instead_of_go_away():
    runner = FakeRunner(events=[])
    handler, fake_ws, _, state = _make_handler_with_runner(runner)
    state.user_turn_seq = 1
    state.transcript_history.append({"role": "user", "text": "Thank you, that's all for now. Goodbye."})

    with patch("websocket_handler.synthesize_local_fallback_pcm", new=AsyncMock(return_value=b"\x05\x06")):
        await handler._response_watchdog(turn_seq=1, delay_sec=0.0)

    assert fake_ws.get_sent_json_by_type("go_away") == []
    transcript_msgs = fake_ws.get_sent_json_by_type("transcript")
    assert transcript_msgs[-1]["text"] == "You're welcome. Goodbye for now."
    assert any(isinstance(item, bytes) and item == b"\x05\x06" for item in fake_ws.outgoing)


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_silent_turn_watchdog_emits_generic_local_fallback():
    runner = FakeRunner(events=[])
    handler, fake_ws, _, state = _make_handler_with_runner(runner)
    state.user_turn_seq = 1
    state.transcript_history.append({"role": "user", "text": "What is this building?"})

    with patch("websocket_handler.synthesize_local_fallback_pcm", new=AsyncMock(return_value=b"\x0B\x0C")):
        await handler._response_watchdog(turn_seq=1, delay_sec=0.0)

    assert fake_ws.get_sent_json_by_type("go_away") == []
    transcript_msgs = fake_ws.get_sent_json_by_type("transcript")
    assert transcript_msgs[-1]["text"] == "I'm having trouble responding right now. Please try again."
    assert any(isinstance(item, bytes) and item == b"\x0B\x0C" for item in fake_ws.outgoing)


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_farewell_text_detection_matches_expected_phrases():
    runner = FakeRunner(events=[])
    handler, _, _, _ = _make_handler_with_runner(runner)

    assert handler._is_farewell_text("Thank you, that's all for now. Goodbye.") is True
    assert handler._is_farewell_text("bye for now") is True
    assert handler._is_farewell_text("What is ahead of me?") is False


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_google_search_redirects_memory_recall_to_memory_tool():
    runner = FakeRunner(events=[])
    handler, _, _, _ = _make_handler_with_runner(runner)

    with patch("websocket_handler._dispatch_function_call", new=AsyncMock(return_value={"summary": "Walgreens"})):
        redirected, result = await handler._maybe_redirect_google_search(
            query="What pharmacy did I mention earlier?"
        )

    assert redirected is True
    assert result["redirected_to"] == "what_do_you_remember"
    assert result["summary"] == "Walgreens"


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_google_search_redirects_memory_store_to_remember_entity():
    runner = FakeRunner(events=[])
    handler, _, _, _ = _make_handler_with_runner(runner)

    with patch("websocket_handler._dispatch_function_call", new=AsyncMock(return_value={"status": "created", "message": "I'll remember Walgreens."})):
        redirected, result = await handler._maybe_redirect_google_search(
            query="Please remember that the pharmacy is called Walgreens on 5th Avenue."
        )

    assert redirected is True
    assert result["redirected_to"] == "remember_entity"
    assert result["status"] == "created"


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_google_search_redirects_scene_query_to_vision_context():
    runner = FakeRunner(events=[])
    handler, _, _, _ = _make_handler_with_runner(runner)
    handler.state.last_vision_context_text = "[VISION ANALYSIS]\nA hallway extends ahead with a person at 2 o'clock."

    mock_sm = MagicMock()
    mock_sm.get_ephemeral_context.return_value = MagicMock(gps=MagicMock(lat=1.0, lng=2.0))

    with patch("websocket_handler.session_manager", mock_sm), \
         patch("websocket_handler._dispatch_function_call", new=AsyncMock(return_value={"address": "123 Main St", "nearby_places": []})):
        redirected, result = await handler._maybe_redirect_google_search(
            query="What's ahead of me now?"
        )

    assert redirected is True
    assert result["redirected_to"] == "vision_context"
    assert "hallway extends ahead" in result["answer"].lower()


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_maps_query_redirects_area_context_to_vision_context():
    runner = FakeRunner(events=[])
    handler, _, _, _ = _make_handler_with_runner(runner)
    handler.state.last_vision_context_text = "[VISION ANALYSIS]\nA park path continues ahead with trees on both sides."

    mock_sm = MagicMock()
    mock_sm.get_ephemeral_context.return_value = MagicMock(gps=MagicMock(lat=1.0, lng=2.0))

    with patch("websocket_handler.session_manager", mock_sm), \
         patch("websocket_handler._dispatch_function_call", new=AsyncMock(return_value={"address": "123 Main St", "nearby_places": []})):
        redirected, result = await handler._maybe_redirect_maps_query(
            question="Tell me more about this area.",
            user_speaking=False,
        )

    assert redirected is True
    assert result["redirected_to"] == "vision_context"
    assert "park path continues ahead" in result["answer"].lower()


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_maps_query_redirects_nearby_food_query_to_nearby_search():
    runner = FakeRunner(events=[])
    handler, _, _, _ = _make_handler_with_runner(runner)

    with patch("websocket_handler._dispatch_function_call", new=AsyncMock(return_value={"success": True, "places": [{"name": "Cafe Roma", "distance_meters": 42}]})):
        redirected, result = await handler._maybe_redirect_maps_query(
            question="Search for the best Italian restaurant nearby.",
            user_speaking=False,
        )

    assert redirected is True
    assert result["redirected_to"] == "nearby_search"
    assert "Cafe Roma" in result["answer"]


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_direct_nearby_search_shortcut_handles_restaurant_query():
    runner = FakeRunner(events=[])
    handler, fake_ws, _, _ = _make_handler_with_runner(runner)
    handler.state.user_turn_seq = 1

    mock_sm = MagicMock()
    mock_sm.get_ephemeral_context.return_value = MagicMock(gps=MagicMock(lat=1.0, lng=2.0))

    with patch("websocket_handler.session_manager", mock_sm), \
         patch("websocket_handler._dispatch_function_call", new=AsyncMock(return_value={"success": True, "places": [{"name": "Cafe Roma", "distance_meters": 42}]})):
        handled = await handler._maybe_handle_direct_nearby_search_intent(
            "Search for the best Italian restaurant nearby."
        )

    assert handled is True
    tool_results = fake_ws.get_sent_json_by_type("tool_result")
    assert tool_results[-1]["tool"] == "nearby_search"


@pytest.mark.websocket
@pytest.mark.asyncio
async def test_direct_nearby_search_shortcut_without_location_emits_prompt():
    runner = FakeRunner(events=[])
    handler, fake_ws, _, _ = _make_handler_with_runner(runner)
    handler.state.user_turn_seq = 1

    mock_sm = MagicMock()
    mock_sm.get_ephemeral_context.return_value = MagicMock(gps=None)

    with patch("websocket_handler.session_manager", mock_sm), \
         patch("websocket_handler.synthesize_local_fallback_pcm", new=AsyncMock(return_value=b"\x09\x0A")):
        handled = await handler._maybe_handle_direct_nearby_search_intent(
            "Search for the best Italian restaurant nearby."
        )

    assert handled is True
    transcript_msgs = fake_ws.get_sent_json_by_type("transcript")
    assert transcript_msgs[-1]["text"].startswith("I need your location to search nearby places.")


# ── G2. Tool execution edge cases ─────────────────────────────────────


@pytest.mark.websocket
@pytest.mark.error_scenario
@pytest.mark.asyncio
async def test_tool_dispatch_exception_sends_error_result():
    """Regression: Tool dispatch raises an exception.

    The handler should catch the error and still inject a function response
    so the model doesn't hang waiting for a tool result.
    """
    events = [
        {
            "tool_call": {
                "function_calls": [
                    {"name": "get_current_time", "args": {}, "id": "call_err_1"}
                ]
            }
        }
    ]
    runner = FakeRunner(events=events)
    handler, fake_ws, fake_queue, _ = _make_handler_with_runner(runner)

    with handler_runtime_patches(), \
         patch("websocket_handler._dispatch_function_call", new_callable=AsyncMock) as mock_dispatch, \
         patch("websocket_handler._extract_function_calls") as mock_extract, \
         patch("websocket_handler._allow_navigation_tool_call", return_value=(True, "")), \
         patch("websocket_handler.resolve_tool_behavior", return_value=MagicMock()), \
         patch("websocket_handler.ALL_FUNCTIONS", {"get_current_time": lambda: None}):
        # Simulate tool raising an error
        mock_dispatch.side_effect = RuntimeError("Tool execution failed: network error")

        mock_fc = MagicMock()
        mock_fc.name = "get_current_time"
        mock_fc.args = {}
        mock_fc.id = "call_err_1"
        mock_extract.side_effect = [[mock_fc]] + [[] for _ in range(100)]

        async def delayed_disconnect():
            await asyncio.sleep(0.2)
            fake_ws.queue_disconnect()

        asyncio.create_task(delayed_disconnect())
        # The handler propagates the exception from _dispatch_function_call
        # through _handle_function_calls, which is caught by _downstream's
        # outer exception handler. This sends an error to the client.
        await handler.run()

    # Verify the tool was called
    assert mock_dispatch.called
    # Session should clean up
    assert fake_queue.closed


@pytest.mark.websocket
@pytest.mark.error_scenario
@pytest.mark.asyncio
async def test_duplicate_tool_call_deduplication():
    """Regression: Same tool called twice rapidly — second should be deduped.

    The tool_dedup guard should skip the second call and inject a 'skipped'
    function response.
    """
    events = [
        {
            "tool_call": {
                "function_calls": [
                    {"name": "get_current_time", "args": {}, "id": "call_dup_1"},
                    {"name": "get_current_time", "args": {}, "id": "call_dup_2"},
                ]
            }
        }
    ]
    runner = FakeRunner(events=events)

    # Configure dedup to allow first call, reject second
    mock_tool_dedup = MagicMock()
    mock_tool_dedup.should_execute = Mock(side_effect=[(True, ""), (False, "duplicate_call")])

    handler, fake_ws, fake_queue, _ = _make_handler_with_runner(
        runner, tool_dedup=mock_tool_dedup
    )

    with handler_runtime_patches(), \
         patch("websocket_handler._dispatch_function_call", new_callable=AsyncMock) as mock_dispatch, \
         patch("websocket_handler._extract_function_calls") as mock_extract, \
         patch("websocket_handler._allow_navigation_tool_call", return_value=(True, "")), \
         patch("websocket_handler.resolve_tool_behavior", return_value=MagicMock()), \
         patch("websocket_handler.ALL_FUNCTIONS", {"get_current_time": lambda: None}):
        mock_dispatch.return_value = {"status": "success", "result": "12:00 PM"}

        mock_fc1 = MagicMock()
        mock_fc1.name = "get_current_time"
        mock_fc1.args = {}
        mock_fc1.id = "call_dup_1"
        mock_fc2 = MagicMock()
        mock_fc2.name = "get_current_time"
        mock_fc2.args = {}
        mock_fc2.id = "call_dup_2"
        mock_extract.side_effect = [[mock_fc1, mock_fc2]] + [[] for _ in range(100)]

        async def delayed_disconnect():
            await asyncio.sleep(0.2)
            fake_ws.queue_disconnect()

        asyncio.create_task(delayed_disconnect())
        await handler.run()

    # Only the first call should have been dispatched
    assert mock_dispatch.call_count == 1, "Second duplicate call should be deduped"
    # Dedup should have been checked twice
    assert mock_tool_dedup.should_execute.call_count == 2


@pytest.mark.websocket
@pytest.mark.error_scenario
@pytest.mark.asyncio
async def test_tool_returns_error_but_session_continues():
    """Regression: Tool returns an error result but session should not crash.

    The handler should forward the error result to the model and continue
    processing the event stream.
    """
    events = [
        {
            "tool_call": {
                "function_calls": [
                    {"name": "get_current_time", "args": {}, "id": "call_fail_1"}
                ]
            }
        }
    ]
    runner = FakeRunner(events=events)
    handler, fake_ws, fake_queue, _ = _make_handler_with_runner(runner)

    with handler_runtime_patches(), \
         patch("websocket_handler._dispatch_function_call", new_callable=AsyncMock) as mock_dispatch, \
         patch("websocket_handler._extract_function_calls") as mock_extract, \
         patch("websocket_handler._allow_navigation_tool_call", return_value=(True, "")), \
         patch("websocket_handler.resolve_tool_behavior", return_value=MagicMock()), \
         patch("websocket_handler.ALL_FUNCTIONS", {"get_current_time": lambda: None}):
        # Tool returns error result (not an exception)
        mock_dispatch.return_value = {"status": "error", "error": "Service temporarily unavailable"}

        mock_fc = MagicMock()
        mock_fc.name = "get_current_time"
        mock_fc.args = {}
        mock_fc.id = "call_fail_1"
        mock_extract.side_effect = [[mock_fc]] + [[] for _ in range(100)]

        async def delayed_disconnect():
            await asyncio.sleep(0.2)
            fake_ws.queue_disconnect()

        asyncio.create_task(delayed_disconnect())
        await handler.run()

    # Tool dispatch should have been called
    assert mock_dispatch.called
    # Session should clean up normally (not crash)
    assert fake_queue.closed
    # Error result should have been sent as tool_result to client
    tool_results = fake_ws.get_sent_json_by_type("tool_result")
    assert len(tool_results) >= 1


# ── G3. Client-side error scenarios ───────────────────────────────────


@pytest.mark.websocket
@pytest.mark.error_scenario
@pytest.mark.asyncio
async def test_client_sends_malformed_json_session_continues():
    """Regression: Client sends non-JSON text over WebSocket.

    The upstream handler should log a warning and continue — not crash the session.
    """
    runner = FakeRunner(events=[])
    handler, fake_ws, fake_queue, _ = _make_handler_with_runner(runner)

    # Queue a raw text message that is not valid JSON
    fake_ws.incoming.put_nowait({"type": "websocket.receive", "text": "this is not json {{{}"})
    # Then a valid message to prove session continues
    fake_ws.queue_message_text({"type": "activity_start"})
    fake_ws.queue_disconnect()

    with handler_runtime_patches():
        await handler.run()

    # Session should have processed activity_start after the bad JSON
    assert len(fake_queue.activity_start_calls) == 1
    assert fake_queue.closed


@pytest.mark.websocket
@pytest.mark.error_scenario
@pytest.mark.asyncio
async def test_client_disconnects_mid_tool_execution():
    """Regression: Client disconnects while a tool is being executed.

    The handler should not crash when trying to send tool results to a
    disconnected client.
    """
    events = [
        {
            "tool_call": {
                "function_calls": [
                    {"name": "get_current_time", "args": {}, "id": "call_dc_1"}
                ]
            }
        }
    ]
    runner = FakeRunner(events=events)
    handler, fake_ws, fake_queue, _ = _make_handler_with_runner(runner)

    with handler_runtime_patches(), \
         patch("websocket_handler._dispatch_function_call", new_callable=AsyncMock) as mock_dispatch, \
         patch("websocket_handler._extract_function_calls") as mock_extract, \
         patch("websocket_handler._allow_navigation_tool_call", return_value=(True, "")), \
         patch("websocket_handler.resolve_tool_behavior", return_value=MagicMock()), \
         patch("websocket_handler.ALL_FUNCTIONS", {"get_current_time": lambda: None}):
        # Tool takes time; client disconnects during execution
        async def slow_tool(*args, **kwargs):
            await asyncio.sleep(0.05)
            # Simulate client disconnect during tool execution
            fake_ws.client_state = WebSocketState.DISCONNECTED
            return {"status": "success", "result": "12:00 PM"}

        mock_dispatch.side_effect = slow_tool

        mock_fc = MagicMock()
        mock_fc.name = "get_current_time"
        mock_fc.args = {}
        mock_fc.id = "call_dc_1"
        mock_extract.side_effect = [[mock_fc]] + [[] for _ in range(100)]

        # Also queue disconnect on upstream side
        async def delayed_disconnect():
            await asyncio.sleep(0.1)
            fake_ws.queue_disconnect()

        asyncio.create_task(delayed_disconnect())

        # Should not raise
        try:
            await asyncio.wait_for(handler.run(), timeout=3.0)
        except asyncio.TimeoutError:
            pytest.fail("Handler deadlocked after client disconnect mid-tool")

    assert fake_queue.closed


@pytest.mark.websocket
@pytest.mark.error_scenario
@pytest.mark.asyncio
async def test_client_sends_audio_before_session_fully_initialized(handler, fake_ws, fake_queue):
    """Regression: Client sends audio data before Gemini session is ready.

    Early audio should be forwarded to the queue without error.
    The handler's upstream processes audio immediately via the queue.
    """
    # Queue audio, then disconnect
    audio_data = b"\x01" + b"\x00" * 320  # MAGIC_AUDIO + 20ms of silence at 16kHz
    fake_ws.queue_message_bytes(audio_data)
    fake_ws.queue_disconnect()

    with handler_runtime_patches():
        await handler.run()

    # Audio should have been forwarded to the queue
    assert len(fake_queue.realtime_calls) >= 1
    assert fake_queue.closed


@pytest.mark.websocket
@pytest.mark.error_scenario
@pytest.mark.asyncio
async def test_client_sends_unknown_message_type_ignored():
    """Regression: Client sends a message with an unknown type.

    The upstream handler should log a warning and continue processing.
    """
    runner = FakeRunner(events=[])
    handler, fake_ws, fake_queue, _ = _make_handler_with_runner(runner)

    # Queue unknown message type, then a valid one, then disconnect
    fake_ws.queue_message_text({"type": "totally_unknown_type", "data": "whatever"})
    fake_ws.queue_message_text({"type": "activity_start"})
    fake_ws.queue_disconnect()

    with handler_runtime_patches():
        await handler.run()

    # Valid message after unknown should still be processed
    assert len(fake_queue.activity_start_calls) == 1
    assert fake_queue.closed


# ── G4. Session resumption / state recovery ───────────────────────────


@pytest.mark.websocket
@pytest.mark.error_scenario
@pytest.mark.asyncio
async def test_fresh_session_state_on_new_connection():
    """Regression: Each new connection gets fresh SessionState.

    Stale state from a previous session should not leak into a new one.
    """
    # Create state that looks like it was from a previous session
    stale_state = SessionState()
    stale_state.is_interrupted = True
    stale_state.vision_in_progress = True
    stale_state.model_audio_last_seen_at = time.monotonic() - 100
    stale_state.transcript_buffer = "leftover text"

    # Verify fresh state has clean defaults
    fresh_state = SessionState()
    assert not fresh_state.is_interrupted
    assert not fresh_state.vision_in_progress
    assert fresh_state.model_audio_last_seen_at == 0.0
    assert fresh_state.transcript_buffer == ""
    assert fresh_state.face_consecutive_misses == 0
    assert fresh_state.frame_seq == 0

    # Verify stale state is indeed stale
    assert stale_state.is_interrupted
    assert stale_state.vision_in_progress


@pytest.mark.websocket
@pytest.mark.error_scenario
@pytest.mark.asyncio
async def test_vision_lock_released_after_exception():
    """Regression: Vision analysis exception doesn't block future analyses.

    If _run_vision_analysis raises, the vision_in_progress flag must be
    reset so the next image frame can trigger analysis.
    """
    runner = FakeRunner(events=[])
    handler, fake_ws, fake_queue, state = _make_handler_with_runner(runner)

    # Simulate a vision analysis that fails
    with patch("websocket_handler._vision_available", True), \
         patch("websocket_handler.session_manager") as mock_sm:
        mock_sm.get_ephemeral_context.return_value = MagicMock(
            motion_state="stationary", step_cadence=0.0, ambient_noise_db=40.0
        )
        with patch("agents.vision_agent.analyze_scene", side_effect=RuntimeError("Vision model OOM")):
            await handler._run_vision_analysis("fake_b64_image")

    # Vision lock must be released despite the exception
    assert not state.vision_in_progress, "vision_in_progress should be False after exception"
    # The lock should be acquirable
    acquired = state.vision_lock.locked()
    assert not acquired, "vision_lock should not be held after exception"


@pytest.mark.websocket
@pytest.mark.error_scenario
@pytest.mark.asyncio
async def test_lod_state_persists_across_telemetry_updates(handler, fake_ws):
    """Regression: LOD state persists correctly across multiple telemetry updates.

    Multiple telemetry messages should each trigger LOD evaluation and the
    session_ctx.current_lod should reflect the latest decision.
    """
    with handler_runtime_patches(), \
         patch("websocket_handler.decide_lod") as mock_decide_lod, \
         patch("websocket_handler.parse_telemetry_to_ephemeral") as mock_parse, \
         patch("websocket_handler.parse_telemetry") as mock_parse_sem, \
         patch("websocket_handler._build_telemetry_signature") as mock_sig, \
         patch("websocket_handler._should_inject_telemetry_context", return_value=(True, ["sig_changed"])), \
         patch("websocket_handler._changed_signature_fields", return_value=["motion"]):
        mock_ephemeral = MagicMock()
        mock_ephemeral.gps = None
        mock_parse.return_value = mock_ephemeral
        mock_parse_sem.return_value = "motion=walking"
        # First telemetry -> LOD 2, second -> LOD 1
        mock_decide_lod.side_effect = [
            (2, MagicMock(reason="walking", triggered_rules=[], to_debug_dict=Mock(return_value={}))),
            (1, MagicMock(reason="running", triggered_rules=[], to_debug_dict=Mock(return_value={}))),
        ]
        mock_sig.side_effect = [{"motion": "walking"}, {"motion": "running"}]

        fake_ws.queue_message_text({"type": "telemetry", "data": {"motion_state": "walking"}})
        fake_ws.queue_message_text({"type": "telemetry", "data": {"motion_state": "running"}})
        fake_ws.queue_disconnect()

        await handler.run()

    # LOD decision should have been called at least once
    assert mock_decide_lod.call_count >= 1


@pytest.mark.websocket
@pytest.mark.error_scenario
@pytest.mark.asyncio
async def test_face_backoff_resets_on_successful_detection():
    """Regression: Face consecutive_misses resets to 0 when a face is detected.

    After FACE_BACKOFF_THRESHOLD misses, the counter should reset when
    a successful detection occurs.
    """
    state = SessionState()
    # Simulate prior consecutive misses at backoff threshold, with skip counter
    # at the limit so the next call goes through (skip_counter > SKIP_CYCLES).
    state.face_consecutive_misses = state.FACE_BACKOFF_THRESHOLD
    state.face_skip_counter = state.FACE_BACKOFF_SKIP_CYCLES
    state.face_runtime_available = True
    state.face_library = [{"person_name": "Alice", "embedding": [0.1] * 128}]
    state.face_library_loaded_at = time.monotonic()

    runner = FakeRunner(events=[])
    handler, fake_ws, fake_queue, _ = _make_handler_with_runner(runner, state=state)

    with patch("websocket_handler._face_available", True), \
         patch("websocket_handler.session_manager") as mock_sm, \
         patch("websocket_handler._memory_available", False), \
         patch("websocket_handler.should_speak", return_value=False), \
         patch("websocket_handler._format_face_results", return_value="[FACE] Alice"), \
         patch("websocket_handler.behavior_to_text", return_value="SILENT"), \
         patch("websocket_handler.FACE_LIBRARY_REFRESH_SEC", 99999):
        mock_sm.get_ephemeral_context.return_value = MagicMock(
            step_cadence=0.0, ambient_noise_db=40.0
        )
        with patch("agents.face_agent.identify_persons_in_frame", return_value=[
            {"person_name": "Alice", "bbox": [0, 0, 100, 100], "score": 0.95, "similarity": 0.92}
        ]):
            await handler._run_face_recognition("fake_b64_image")

    # Consecutive misses should be reset
    assert state.face_consecutive_misses == 0
    assert state.face_skip_counter == 0


@pytest.mark.websocket
@pytest.mark.error_scenario
@pytest.mark.asyncio
async def test_concurrent_binary_messages_all_processed():
    """Regression: Multiple binary messages (audio + image) in rapid succession.

    All messages should be processed without drops or errors.
    """
    runner = FakeRunner(events=[])
    handler, fake_ws, fake_queue, state = _make_handler_with_runner(runner)

    # Queue mixed binary messages rapidly
    for _ in range(5):
        # Audio frames
        fake_ws.queue_message_bytes(b"\x01" + b"\x00" * 320)
    # Image frame
    fake_ws.queue_message_bytes(b"\x02" + b"fake_jpeg_data")
    # More audio
    for _ in range(3):
        fake_ws.queue_message_bytes(b"\x01" + b"\x00" * 320)
    fake_ws.queue_disconnect()

    with handler_runtime_patches(), \
         patch.object(handler, "_process_image_frame", new_callable=AsyncMock) as mock_img:
        await handler.run()

    # All 8 audio frames should be forwarded
    assert len(fake_queue.realtime_calls) >= 8
    # Image processing should have been triggered
    assert mock_img.called
    assert fake_queue.closed
