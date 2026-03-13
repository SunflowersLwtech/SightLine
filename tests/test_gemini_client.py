"""Tests for shared Gemini client factories."""

from __future__ import annotations

from unittest.mock import patch, sentinel

import pytest

import gemini_client


@pytest.fixture(autouse=True)
def reset_client_caches():
    gemini_client.reset_gemini_clients_for_testing()
    yield
    gemini_client.reset_gemini_clients_for_testing()


def test_get_gemini_api_client_is_cached(monkeypatch):
    monkeypatch.delenv("_GOOGLE_AI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    with patch("gemini_client.genai.Client", return_value=sentinel.client) as mock_client:
        first = gemini_client.get_gemini_api_client()
        second = gemini_client.get_gemini_api_client()

    assert first is sentinel.client
    assert second is sentinel.client
    mock_client.assert_called_once_with(api_key="test-key", vertexai=False)


def test_get_gemini_vertex_client_uses_project_and_region(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-project")
    monkeypatch.setenv("GOOGLE_CLOUD_REGION", "asia-southeast1")

    with patch("gemini_client.genai.Client", return_value=sentinel.vertex_client) as mock_client:
        client = gemini_client.get_gemini_vertex_client()

    assert client is sentinel.vertex_client
    mock_client.assert_called_once_with(
        vertexai=True,
        project="demo-project",
        location="asia-southeast1",
    )


def test_get_gemini_api_client_raises_when_key_missing(monkeypatch):
    monkeypatch.delenv("_GOOGLE_AI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="Gemini API key not configured"):
        gemini_client.get_gemini_api_client()
