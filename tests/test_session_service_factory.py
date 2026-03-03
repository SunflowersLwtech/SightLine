"""Tests for session service factory mode selection."""

from __future__ import annotations

import sys
import types

import pytest

from live_api.session_manager import create_session_service


def _install_sessions_stub(
    monkeypatch: pytest.MonkeyPatch,
    *,
    include_vertex: bool = True,
    include_database: bool = True,
) -> tuple[type, type, type]:
    """Install a stub google.adk.sessions module for deterministic factory tests."""
    sessions_module = types.ModuleType("google.adk.sessions")

    class VertexAiSessionService:
        def __init__(self, project: str, location: str, agent_engine_id: str | None = None):
            self.project = project
            self.location = location
            self.agent_engine_id = agent_engine_id

    class DatabaseSessionService:
        def __init__(self, db_url: str):
            self.db_url = db_url

    class InMemorySessionService:
        pass

    if include_vertex:
        sessions_module.VertexAiSessionService = VertexAiSessionService
    if include_database:
        sessions_module.DatabaseSessionService = DatabaseSessionService
    sessions_module.InMemorySessionService = InMemorySessionService

    monkeypatch.setitem(sys.modules, "google.adk.sessions", sessions_module)
    return VertexAiSessionService, DatabaseSessionService, InMemorySessionService


def test_factory_uses_database_when_vertex_flag_disabled(monkeypatch: pytest.MonkeyPatch):
    _, DatabaseSessionService, _ = _install_sessions_stub(monkeypatch)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")
    monkeypatch.setenv("AGENT_ENGINE_ID", "engine-should-be-ignored")

    service = create_session_service()

    assert isinstance(service, DatabaseSessionService)
    assert service.db_url == "sqlite:///sightline_sessions.db"


def test_factory_uses_vertex_when_vertex_flag_enabled(monkeypatch: pytest.MonkeyPatch):
    VertexAiSessionService, _, _ = _install_sessions_stub(monkeypatch)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
    monkeypatch.setenv("AGENT_ENGINE_ID", "engine-123")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-project")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "us-central1")

    service = create_session_service()

    assert isinstance(service, VertexAiSessionService)
    assert service.project == "demo-project"
    assert service.location == "us-central1"
    assert service.agent_engine_id == "engine-123"


def test_factory_falls_back_to_in_memory_when_database_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
    _, _, InMemorySessionService = _install_sessions_stub(
        monkeypatch,
        include_vertex=False,
        include_database=False,
    )
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")

    service = create_session_service()

    assert isinstance(service, InMemorySessionService)
