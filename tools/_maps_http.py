"""Shared REST helper for Google Maps Platform APIs.

Used by Routes API, Places API (New), Address Validation, and Street View.
All requests use X-Goog-Api-Key + optional X-Goog-FieldMask headers.
"""

from __future__ import annotations

import logging
import os
import time

import httpx

logger = logging.getLogger("sightline.tools._maps_http")

_API_KEY: str | None = None
_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_SEC = 0.35


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

    resp = _request_with_retry(
        method="POST",
        url=url,
        json=body,
        headers=headers,
        timeout=timeout,
    )
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
    resp = _request_with_retry(
        method="GET",
        url=url,
        params=params,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp


def _request_with_retry(
    *,
    method: str,
    url: str,
    timeout: float,
    **kwargs,
) -> httpx.Response:
    last_exc: Exception | None = None
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            resp = httpx.request(method, url, timeout=timeout, **kwargs)
            if resp.status_code in {429, 500, 502, 503, 504} and attempt < _RETRY_ATTEMPTS:
                backoff = _RETRY_BACKOFF_SEC * attempt
                logger.warning(
                    "Maps API transient HTTP %s for %s %s; retrying in %.2fs (%d/%d)",
                    resp.status_code,
                    method,
                    url,
                    backoff,
                    attempt,
                    _RETRY_ATTEMPTS,
                )
                time.sleep(backoff)
                continue
            resp.raise_for_status()
            return resp
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            last_exc = exc
            if attempt >= _RETRY_ATTEMPTS:
                raise
            backoff = _RETRY_BACKOFF_SEC * attempt
            logger.warning(
                "Maps API %s for %s %s; retrying in %.2fs (%d/%d)",
                type(exc).__name__,
                method,
                url,
                backoff,
                attempt,
                _RETRY_ATTEMPTS,
            )
            time.sleep(backoff)
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            if exc.response.status_code not in {429, 500, 502, 503, 504} or attempt >= _RETRY_ATTEMPTS:
                raise
            backoff = _RETRY_BACKOFF_SEC * attempt
            logger.warning(
                "Maps API HTTPStatusError %s for %s %s; retrying in %.2fs (%d/%d)",
                exc.response.status_code,
                method,
                url,
                backoff,
                attempt,
                _RETRY_ATTEMPTS,
            )
            time.sleep(backoff)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Maps API request failed unexpectedly: {method} {url}")
