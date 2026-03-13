"""Shared API helpers extracted from ``server.py``."""

from __future__ import annotations

import json


def _coerce_bool(value: object, default: bool = False) -> bool:
    """Parse bool-like JSON values safely for request handling."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(value, (int, float)):
        return value != 0
    return default


def _json_safe(value):
    """Best-effort conversion for JSON payloads sent over WebSocket."""
    try:
        json.dumps(value)
        return value
    except TypeError:
        return json.loads(json.dumps(value, default=str))

