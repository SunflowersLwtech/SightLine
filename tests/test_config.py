"""Tests for runtime config helpers."""

from __future__ import annotations

from config import (
    DEFAULT_GOOGLE_CLOUD_PROJECT,
    DEFAULT_GOOGLE_CLOUD_REGION,
    DEFAULT_SESSION_DB_URL,
    get_google_cloud_project,
    get_google_cloud_region,
    get_session_db_url,
)


def test_google_cloud_project_env_override(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-project")

    assert get_google_cloud_project() == "demo-project"


def test_google_cloud_project_defaults_when_missing(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

    assert get_google_cloud_project() == DEFAULT_GOOGLE_CLOUD_PROJECT


def test_google_cloud_region_falls_back_to_location(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_REGION", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "asia-southeast1")

    assert get_google_cloud_region() == "asia-southeast1"


def test_google_cloud_region_defaults_when_missing(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_REGION", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)

    assert get_google_cloud_region() == DEFAULT_GOOGLE_CLOUD_REGION


def test_session_db_url_defaults_and_override(monkeypatch):
    monkeypatch.delenv("SESSION_DB_URL", raising=False)
    assert get_session_db_url() == DEFAULT_SESSION_DB_URL

    monkeypatch.setenv("SESSION_DB_URL", "sqlite:///custom.db")
    assert get_session_db_url() == "sqlite:///custom.db"
