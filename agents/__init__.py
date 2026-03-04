"""SightLine agent package.

Use lazy imports to avoid pulling heavy runtime dependencies at package import
time (e.g. face/vision stacks) and to keep tests isolated.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "create_orchestrator_agent",
    "analyze_scene",
    "extract_text",
    "identify_persons_in_frame",
    "orchestrator",
    "vision_agent",
    "ocr_agent",
    "face_agent",
]


def __getattr__(name: str) -> Any:
    if name == "create_orchestrator_agent":
        return import_module("agents.orchestrator").create_orchestrator_agent
    if name == "analyze_scene":
        return import_module("agents.vision_agent").analyze_scene
    if name == "extract_text":
        return import_module("agents.ocr_agent").extract_text
    if name == "identify_persons_in_frame":
        return import_module("agents.face_agent").identify_persons_in_frame
    if name in {"orchestrator", "vision_agent", "ocr_agent", "face_agent"}:
        return import_module(f"agents.{name}")
    raise AttributeError(f"module 'agents' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
