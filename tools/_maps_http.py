"""Shared REST helper for Google Maps Platform APIs.

Used by Routes API, Places API (New), Address Validation, and Street View.
All requests use X-Goog-Api-Key + optional X-Goog-FieldMask headers.
"""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger("sightline.tools._maps_http")

_API_KEY: str | None = None


def _get_api_key() -> str:
    global _API_KEY
    if _API_KEY is None:
        _API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")
    if not _API_KEY:
        raise RuntimeError("GOOGLE_MAPS_API_KEY environment variable not set")
    return _API_KEY


def maps_rest_post(
    url: str,
    body: dict,
    field_mask: str | None = None,
    timeout: float = 15.0,
) -> dict:
    """POST to a Google Maps REST API with standard auth headers.

    Args:
        url: Full API endpoint URL.
        body: JSON request body.
        field_mask: Optional X-Goog-FieldMask header value (controls billing).
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON response dict.

    Raises:
        httpx.HTTPStatusError: On 4xx/5xx responses.
    """
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": _get_api_key(),
    }
    if field_mask:
        headers["X-Goog-FieldMask"] = field_mask

    resp = httpx.post(url, json=body, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def maps_rest_get(
    url: str,
    params: dict | None = None,
    timeout: float = 15.0,
) -> httpx.Response:
    """GET from a Google Maps REST API with API key as query param.

    Returns the raw httpx.Response (useful for binary data like Street View images).
    """
    if params is None:
        params = {}
    params["key"] = _get_api_key()
    resp = httpx.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp
