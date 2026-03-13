"""Runtime configuration helpers for SightLine backend modules."""

from __future__ import annotations

import os

DEFAULT_GOOGLE_CLOUD_PROJECT = "sightline-hackathon"
DEFAULT_GOOGLE_CLOUD_REGION = "us-central1"
DEFAULT_SESSION_DB_URL = "sqlite:///sightline_sessions.db"


def _get_env(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    trimmed = value.strip()
    return trimmed or default


def get_google_cloud_project() -> str:
    """Return the configured Google Cloud project ID."""
    return _get_env("GOOGLE_CLOUD_PROJECT", DEFAULT_GOOGLE_CLOUD_PROJECT)


def get_google_cloud_region() -> str:
    """Return the configured Google Cloud region.

    ``GOOGLE_CLOUD_LOCATION`` remains supported for backwards compatibility
    with current deployment configuration.
    """
    region = os.getenv("GOOGLE_CLOUD_REGION", "").strip()
    if region:
        return region
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "").strip()
    if location:
        return location
    return DEFAULT_GOOGLE_CLOUD_REGION


def get_session_db_url() -> str:
    """Return the configured session database URL."""
    return _get_env("SESSION_DB_URL", DEFAULT_SESSION_DB_URL)


__all__ = [
    "DEFAULT_GOOGLE_CLOUD_PROJECT",
    "DEFAULT_GOOGLE_CLOUD_REGION",
    "DEFAULT_SESSION_DB_URL",
    "get_google_cloud_project",
    "get_google_cloud_region",
    "get_session_db_url",
]
