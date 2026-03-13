"""Shared Gemini client factories."""

from __future__ import annotations

import os
from functools import lru_cache

from google import genai

from config import get_google_cloud_project, get_google_cloud_region


def _get_gemini_api_key() -> str:
    api_key = os.environ.get("_GOOGLE_AI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "Gemini API key not configured. "
            "Set _GOOGLE_AI_API_KEY or GOOGLE_API_KEY environment variable."
        )
    return api_key


@lru_cache(maxsize=1)
def get_gemini_api_client() -> genai.Client:
    """Return a cached Gemini client using Developer API key auth."""
    return genai.Client(api_key=_get_gemini_api_key(), vertexai=False)


@lru_cache(maxsize=1)
def get_gemini_vertex_client() -> genai.Client:
    """Return a cached Gemini client using Vertex AI auth."""
    return genai.Client(
        vertexai=True,
        project=get_google_cloud_project(),
        location=get_google_cloud_region(),
    )


def reset_gemini_clients_for_testing() -> None:
    """Clear shared client caches for deterministic tests."""
    get_gemini_api_client.cache_clear()
    get_gemini_vertex_client.cache_clear()


__all__ = [
    "get_gemini_api_client",
    "get_gemini_vertex_client",
    "reset_gemini_clients_for_testing",
]
