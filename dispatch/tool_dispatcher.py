"""Function-call dispatch extracted from ``server.py``."""

from __future__ import annotations

import asyncio
import logging

from memory.memory_tools import MEMORY_FUNCTIONS
from tools import ALL_FUNCTIONS, ALL_TOOL_RUNTIME_METADATA

logger = logging.getLogger("sightline.server")

DEFAULT_TOOL_TIMEOUT_SEC = 5.0
TOOL_TIMEOUTS_SEC = {
    "navigate_to": 8.0,
    "preview_destination": 10.0,
    "validate_address": 5.0,
    "google_search": 8.0,
    "maps_query": 8.0,
}

# ---------------------------------------------------------------------------
# Tool result truncation (E2E-006)
# ---------------------------------------------------------------------------

_MAX_TOOL_RESULT_CHARS = 4000


def _sanitize_function_args_for_log(func_name: str, func_args: dict, user_id: str) -> dict:
    """Sanitize function-call logging to avoid leaking/echoing forged user IDs."""
    safe_args = dict(func_args)
    if func_name in MEMORY_FUNCTIONS:
        if "user_id" in safe_args:
            safe_args["user_id"] = "<session_user>"
        safe_args["_session_user"] = user_id
    return safe_args


def _truncate_tool_result(result: dict, max_chars: int = _MAX_TOOL_RESULT_CHARS) -> dict:
    """Truncate oversized string values in tool results to prevent token overflow."""
    truncated = {}
    for k, v in result.items():
        if isinstance(v, str) and len(v) > max_chars:
            truncated[k] = v[:max_chars] + "\u2026 [truncated]"
        elif isinstance(v, dict):
            truncated[k] = _truncate_tool_result(v, max_chars)
        elif isinstance(v, list):
            truncated[k] = [
                _truncate_tool_result(item, max_chars) if isinstance(item, dict) else item
                for item in v
            ]
        else:
            truncated[k] = v
    return truncated


def _extract_function_calls(event) -> list:
    """Extract function calls from ADK event objects across SDK schema changes."""
    getter = getattr(event, "get_function_calls", None)
    if callable(getter):
        try:
            calls = getter() or []
            if calls:
                return list(calls)
        except Exception:
            logger.debug("event.get_function_calls() failed; trying legacy access path", exc_info=True)

    # Legacy fallback (older assumptions in downstream loop).
    actions = getattr(event, "actions", None)
    if not actions:
        return []
    legacy_calls = getattr(actions, "function_calls", None)
    if not legacy_calls:
        return []
    return list(legacy_calls)


async def _dispatch_function_call(
    func_name: str,
    func_args: dict,
    session_id: str,
    user_id: str,
    *,
    session_manager,
) -> dict:
    """Dispatch a function call from Gemini to the appropriate tool.

    Uses the unified ALL_FUNCTIONS dict for dispatch.  Navigation tools
    get automatic GPS/heading injection from ephemeral context.

    Returns the tool result as a dict to be sent back as function response.
    """
    if func_name not in ALL_FUNCTIONS:
        logger.warning("Unknown function call: %s (should have been caught upstream)", func_name)
        return {
            "status": "unavailable",
            "message": f"'{func_name}' does not exist. Use only the tools listed in your instructions.",
        }

    runtime_metadata = ALL_TOOL_RUNTIME_METADATA.get(
        func_name,
        {"gps_injection": None, "force_user_id": False},
    )
    gps_injection = runtime_metadata.get("gps_injection")
    ephemeral = session_manager.get_ephemeral_context(session_id) if gps_injection else None
    if ephemeral and ephemeral.gps:
        if gps_injection == "origin_lat_lng_heading":
            func_args.setdefault("origin_lat", ephemeral.gps.lat)
            func_args.setdefault("origin_lng", ephemeral.gps.lng)
            func_args.setdefault("user_heading", ephemeral.heading)
        elif gps_injection == "lat_lng":
            func_args.setdefault("lat", ephemeral.gps.lat)
            func_args.setdefault("lng", ephemeral.gps.lng)

    if runtime_metadata.get("force_user_id"):
        func_args["user_id"] = user_id

    logger.info(
        "Function call: %s(%s)",
        func_name,
        _sanitize_function_args_for_log(func_name, func_args, user_id),
    )

    # OCR tool: intercept and run async OCR pipeline with latest camera frame
    if func_name == "extract_text_from_camera":
        from tools.ocr_tool import _latest_frames

        frame_b64 = _latest_frames.get(session_id)
        if not frame_b64:
            return {
                "text": "",
                "text_type": "unknown",
                "items": [],
                "confidence": 0.0,
                "message": "No camera frame available. The camera may not be active.",
            }
        try:
            from agents.ocr_agent import extract_text as _ocr_extract

            hint = func_args.get("context_hint", "")
            result = await _ocr_extract(frame_b64, context_hint=hint, safety_only=False)
            if isinstance(result, dict) and not result.get("text"):
                result = {
                    **result,
                    "message": (
                        str(result.get("message") or "").strip()
                        or "I couldn't read the text clearly. Please pan the camera a little and try again."
                    ),
                }
            return result
        except Exception:
            logger.exception("OCR tool dispatch failed")
            return {
                "text": "",
                "text_type": "unknown",
                "items": [],
                "confidence": 0.0,
                "error": "OCR extraction failed",
            }

    try:
        timeout_sec = TOOL_TIMEOUTS_SEC.get(func_name, DEFAULT_TOOL_TIMEOUT_SEC)
        raw_result = await asyncio.wait_for(
            asyncio.to_thread(ALL_FUNCTIONS[func_name], **func_args),
            timeout=timeout_sec,
        )
        return _truncate_tool_result(raw_result) if isinstance(raw_result, dict) else raw_result
    except TimeoutError:
        logger.warning(
            "Tool %s timed out after %.1fs",
            func_name,
            TOOL_TIMEOUTS_SEC.get(func_name, DEFAULT_TOOL_TIMEOUT_SEC),
        )
        return {
            "error": "tool_timeout",
            "tool": func_name,
            "message": f"The {func_name} tool timed out. Please try again.",
            "retryable": True,
        }
    except Exception:
        logger.exception("Tool %s raised an exception", func_name)
        return {
            "error": "tool_execution_failed",
            "tool": func_name,
            "message": f"The {func_name} tool encountered an internal error. Try again or use a different approach.",
        }

