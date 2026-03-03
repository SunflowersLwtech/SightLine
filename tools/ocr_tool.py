"""OCR tool wrapper for Orchestrator function calling.

Provides extract_text_from_camera as a callable tool that the Orchestrator
can invoke when the user asks to read text. The actual OCR is performed by
agents/ocr_agent.py — this module only provides the function-calling contract
and the latest-frame storage mechanism.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("sightline.tools.ocr_tool")

# Per-session latest frame storage.  Keys are session_id strings,
# values are base64-encoded image strings.  Updated by server.py
# on each camera frame received.  Thread-safe in asyncio (single-thread).
_latest_frames: dict[str, str] = {}


def set_latest_frame(session_id: str, image_base64: str) -> None:
    """Store the latest camera frame for a session (called by server.py)."""
    _latest_frames[session_id] = image_base64


def clear_session(session_id: str) -> None:
    """Clean up frame storage when session ends."""
    _latest_frames.pop(session_id, None)


def extract_text_from_camera(session_id: str = "", context_hint: str = "") -> dict[str, Any]:
    """Read text from the current camera view.

    This is a synchronous stub — the actual async OCR is dispatched by
    server.py's _dispatch_function_call using the stored latest frame.
    The function exists to satisfy ADK's function-calling contract.

    Args:
        session_id: Injected by server.py dispatch (not user-provided).
        context_hint: Optional hint about what kind of text to look for.

    Returns:
        OCR result dict (text, text_type, items, confidence).
    """
    # This should never be called directly — server.py intercepts the
    # function call and runs the async OCR pipeline.  If it does get
    # called, return a placeholder.
    return {
        "text": "",
        "text_type": "unknown",
        "items": [],
        "confidence": 0.0,
        "message": "OCR dispatch handled by server pipeline.",
    }


OCR_TOOL_FUNCTIONS = {
    "extract_text_from_camera": extract_text_from_camera,
}

OCR_TOOL_DECLARATIONS = [
    {
        "name": "extract_text_from_camera",
        "description": (
            "Read and extract text from the current camera view. "
            "Use when the user asks 'what does it say?', 'read this for me', "
            "'any text here?', 'what's written there?', or similar text-reading requests. "
            "Do NOT call this proactively — only when the user explicitly requests text reading. "
            "Safety-critical text (danger signs, warnings) is detected automatically."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "context_hint": {
                    "type": "string",
                    "description": "Optional hint about what kind of text to look for (e.g., 'menu', 'sign', 'document').",
                },
            },
            "required": [],
        },
    },
]
