"""WebSocket contract tests for activity_start/activity_end observability."""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from lod.models import UserProfile


class FakeLiveRequestQueue:
    """Test double for ADK LiveRequestQueue used by websocket_endpoint."""

    instances: list["FakeLiveRequestQueue"] = []

    def __init__(self) -> None:
        self.activity_start_calls = 0
        self.activity_end_calls = 0
        self.content_calls = 0
        self.realtime_calls = 0
        self.closed = False
        FakeLiveRequestQueue.instances.append(self)

    def send_content(self, _content) -> None:
        self.content_calls += 1

    def send_realtime(self, _blob) -> None:
        self.realtime_calls += 1

    def send_activity_start(self) -> None:
        self.activity_start_calls += 1

    def send_activity_end(self) -> None:
        self.activity_end_calls += 1

    def close(self) -> None:
        self.closed = True


class FakeRunner:
    """Returns an empty async event stream so downstream exits cleanly."""

    @staticmethod
    def run_live(**_kwargs):
        async def _empty_stream():
            if False:
                yield None

        return _empty_stream()


def _receive_until_type(ws, message_type: str, max_reads: int = 6) -> dict:
    for _ in range(max_reads):
        payload = ws.receive_json()
        if payload.get("type") == message_type:
            return payload
    pytest.fail(f"Did not receive websocket payload type={message_type}")


@pytest.fixture
def patched_server(monkeypatch):
    import server

    FakeLiveRequestQueue.instances.clear()
    monkeypatch.setattr(
        server.session_manager,
        "load_user_profile",
        AsyncMock(return_value=UserProfile.default()),
    )
    monkeypatch.setattr(server.session_manager, "remove_session", lambda _session_id: None)
    monkeypatch.setattr(server, "LiveRequestQueue", FakeLiveRequestQueue)
    monkeypatch.setattr(server, "runner", FakeRunner())
    return server


def test_activity_start_contract_updates_state_and_emits_debug_event(patched_server):
    with TestClient(patched_server.app) as client:
        with client.websocket_connect("/ws/test_user/ws_activity_start") as ws:
            _ = _receive_until_type(ws, "session_ready")
            ws.send_text('{"type":"activity_start"}')

            debug_event = _receive_until_type(ws, "debug_activity")
            data = debug_event["data"]
            assert data["event"] == "activity_start"
            assert data["state"] == "user_speaking"
            assert data["queue_status"] == "forwarded"

            session_ctx = patched_server.session_manager.get_session_context("ws_activity_start")
            assert session_ctx.current_activity_state == "user_speaking"
            assert session_ctx.last_activity_event == "activity_start"
            assert session_ctx.activity_event_count >= 1

    queue = FakeLiveRequestQueue.instances[-1]
    assert queue.activity_start_calls == 1


def test_activity_end_contract_updates_state_and_emits_debug_event(patched_server):
    with TestClient(patched_server.app) as client:
        with client.websocket_connect("/ws/test_user/ws_activity_end") as ws:
            _ = _receive_until_type(ws, "session_ready")
            ws.send_text('{"type":"activity_start"}')
            _ = _receive_until_type(ws, "debug_activity")

            ws.send_text('{"type":"activity_end"}')
            debug_event = _receive_until_type(ws, "debug_activity")
            data = debug_event["data"]
            assert data["event"] == "activity_end"
            assert data["state"] == "idle"
            assert data["queue_status"] == "forwarded"

            session_ctx = patched_server.session_manager.get_session_context("ws_activity_end")
            assert session_ctx.current_activity_state == "idle"
            assert session_ctx.last_activity_event == "activity_end"
            assert session_ctx.activity_event_count >= 2

    queue = FakeLiveRequestQueue.instances[-1]
    assert queue.activity_start_calls == 1
    assert queue.activity_end_calls == 1
