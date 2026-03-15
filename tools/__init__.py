"""SightLine tool registry and manifest helpers.

This module is the single source of truth for:
- runtime function dispatch
- callable tool ordering for the orchestrator
- tool declarations shown to clients
- per-tool category metadata used in the tools manifest
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from tools.accessibility import (
    ACCESSIBILITY_FUNCTIONS,
    ACCESSIBILITY_TOOL_DECLARATIONS,
    get_accessibility_info,
)
from tools.emergency import (
    EMERGENCY_FUNCTIONS,
    EMERGENCY_TOOL_DECLARATIONS,
    get_emergency_help,
)
from tools.face_tools import (
    MAX_FACE_SAMPLES,
    MIN_FACE_SAMPLES,
    clear_face_library,
    delete_all_faces,
    delete_face,
    list_faces,
    load_face_library,
    register_face,
)
from tools.maps_grounding import (
    MAPS_GROUNDING_FUNCTIONS,
    MAPS_GROUNDING_TOOL_DECLARATIONS,
    maps_query,
)
from tools.navigation import (
    ACTIVE_NAVIGATION_TOOLS,
    LOCATION_QUERY_TOOLS,
    NAVIGATION_FUNCTIONS,
    NAVIGATION_TOOL_DECLARATIONS,
    bearing_between,
    bearing_to_clock,
    format_clock_direction,
    get_location_info,
    get_walking_directions,
    navigate_to,
    nearby_search,
    preview_destination,
    reverse_geocode,
    validate_address,
)
from tools.ocr_tool import (
    OCR_TOOL_DECLARATIONS,
    OCR_TOOL_FUNCTIONS,
    extract_text_from_camera,
)
from tools.plus_codes import (
    PLUS_CODES_FUNCTIONS,
    PLUS_CODES_TOOL_DECLARATIONS,
    convert_to_plus_code,
    resolve_plus_code,
)
from tools.search import (
    SEARCH_FUNCTIONS,
    SEARCH_TOOL_DECLARATIONS,
    google_search,
)

try:
    from memory.memory_tools import MEMORY_FUNCTIONS, MEMORY_TOOL_DECLARATIONS
except ImportError:
    MEMORY_FUNCTIONS: dict = {}
    MEMORY_TOOL_DECLARATIONS: list[dict[str, Any]] = []
from tools.tool_behavior import ToolBehavior, behavior_to_text, resolve_tool_behavior

GPSInjectionMode = Literal[None, "lat_lng", "origin_lat_lng_heading"]


class ToolRuntimeMetadata(TypedDict):
    gps_injection: GPSInjectionMode
    force_user_id: bool


def identify_person(
    description: str,
    user_id: str = "",
    image_base64: str | None = None,
    behavior: ToolBehavior = ToolBehavior.SILENT,
) -> dict[str, Any]:
    """No-op stub — face recognition runs automatically from camera frames.

    The realtime frame matching pipeline is executed in ``server.py`` via
    ``agents.face_agent.identify_persons_in_frame``.  This function exists
    only to satisfy the function-calling contract; it should rarely be called.
    """
    return {
        "status": "no_op",
        "message": (
            "Face recognition runs automatically from camera frames. "
            "Results are injected into your context as [FACE ID] entries. "
            "Do not announce this to the user."
        ),
    }

# Face tool declarations for Gemini Live API function calling
FACE_TOOL_DECLARATIONS = [
    {
        "name": "identify_person",
        "description": (
            "Acknowledge that face recognition runs automatically in the background. "
            "Do NOT call this tool — face results are injected into your context "
            "automatically as [FACE ID] entries. If called, returns a no-op confirmation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Brief description of the person's appearance and position",
                },
            },
            "required": ["description"],
        },
    },
]

FACE_FUNCTIONS = {
    "identify_person": identify_person,
}

# Aggregate all tool declarations and function maps
CALLABLE_TOOL_DECLARATIONS = (
    NAVIGATION_TOOL_DECLARATIONS
    + SEARCH_TOOL_DECLARATIONS
    + MEMORY_TOOL_DECLARATIONS
    + PLUS_CODES_TOOL_DECLARATIONS
    + ACCESSIBILITY_TOOL_DECLARATIONS
    + MAPS_GROUNDING_TOOL_DECLARATIONS
    + OCR_TOOL_DECLARATIONS
    + EMERGENCY_TOOL_DECLARATIONS
)
PASSIVE_TOOL_DECLARATIONS = FACE_TOOL_DECLARATIONS
ALL_TOOL_DECLARATIONS = CALLABLE_TOOL_DECLARATIONS + PASSIVE_TOOL_DECLARATIONS

ALL_FUNCTIONS = {
    **NAVIGATION_FUNCTIONS,
    **SEARCH_FUNCTIONS,
    **FACE_FUNCTIONS,
    **MEMORY_FUNCTIONS,
    **PLUS_CODES_FUNCTIONS,
    **ACCESSIBILITY_FUNCTIONS,
    **MAPS_GROUNDING_FUNCTIONS,
    **OCR_TOOL_FUNCTIONS,
    **EMERGENCY_FUNCTIONS,
}

CALLABLE_TOOL_ORDER = [str(decl["name"]) for decl in CALLABLE_TOOL_DECLARATIONS]
CALLABLE_TOOL_NAMES = set(CALLABLE_TOOL_ORDER)
PASSIVE_TOOL_NAMES = {str(decl["name"]) for decl in PASSIVE_TOOL_DECLARATIONS}

ALL_TOOL_CATEGORIES: dict[str, str] = {
    "navigate_to": "navigation",
    "get_location_info": "navigation",
    "nearby_search": "navigation",
    "reverse_geocode": "navigation",
    "get_walking_directions": "navigation",
    "preview_destination": "navigation",
    "validate_address": "navigation",
    "google_search": "search",
    "identify_person": "face",
    "resolve_plus_code": "plus_codes",
    "convert_to_plus_code": "plus_codes",
    "get_accessibility_info": "accessibility",
    "maps_query": "maps_grounding",
    "preload_memory": "memory",
    "remember_entity": "memory",
    "what_do_you_remember": "memory",
    "forget_entity": "memory",
    "forget_recent_memory": "memory",
    "extract_text_from_camera": "ocr",
    "get_emergency_help": "emergency",
}


def _runtime_metadata(
    *,
    gps_injection: GPSInjectionMode = None,
    force_user_id: bool = False,
) -> ToolRuntimeMetadata:
    return {
        "gps_injection": gps_injection,
        "force_user_id": force_user_id,
    }


ALL_TOOL_RUNTIME_METADATA: dict[str, ToolRuntimeMetadata] = {
    name: _runtime_metadata() for name in ALL_FUNCTIONS
}
ALL_TOOL_RUNTIME_METADATA.update({
    "navigate_to": _runtime_metadata(gps_injection="origin_lat_lng_heading"),
    "get_location_info": _runtime_metadata(gps_injection="lat_lng"),
    "nearby_search": _runtime_metadata(gps_injection="lat_lng"),
    "reverse_geocode": _runtime_metadata(gps_injection="lat_lng"),
    "convert_to_plus_code": _runtime_metadata(gps_injection="lat_lng"),
    "preview_destination": _runtime_metadata(gps_injection="lat_lng"),
    "get_accessibility_info": _runtime_metadata(gps_injection="lat_lng"),
    "maps_query": _runtime_metadata(gps_injection="lat_lng"),
    "get_emergency_help": _runtime_metadata(gps_injection="lat_lng"),
})
for name in MEMORY_FUNCTIONS:
    ALL_TOOL_RUNTIME_METADATA[name] = _runtime_metadata(force_user_id=True)


def build_tool_manifest_entries(
    *,
    lod: int,
    is_user_speaking: bool = False,
) -> list[dict[str, Any]]:
    """Build tool manifest entries for the current session state."""
    entries: list[dict[str, Any]] = []
    for decl in ALL_TOOL_DECLARATIONS:
        name = str(decl["name"])
        entries.append(
            {
                "name": name,
                "category": ALL_TOOL_CATEGORIES.get(name, "unknown"),
                "behavior": behavior_to_text(
                    resolve_tool_behavior(
                        tool_name=name,
                        lod=lod,
                        is_user_speaking=is_user_speaking,
                    )
                ),
                "callable": name in CALLABLE_TOOL_NAMES,
                "source": "automatic" if name in PASSIVE_TOOL_NAMES else "function_call",
                "description": decl.get("description", ""),
            }
        )
    return entries

__all__ = [
    # Navigation
    "navigate_to",
    "get_location_info",
    "nearby_search",
    "reverse_geocode",
    "get_walking_directions",
    "preview_destination",
    "validate_address",
    "bearing_between",
    "bearing_to_clock",
    "format_clock_direction",
    "NAVIGATION_TOOL_DECLARATIONS",
    "NAVIGATION_FUNCTIONS",
    "ACTIVE_NAVIGATION_TOOLS",
    "LOCATION_QUERY_TOOLS",
    # Search
    "google_search",
    "SEARCH_TOOL_DECLARATIONS",
    "SEARCH_FUNCTIONS",
    # Face
    "register_face",
    "delete_face",
    "delete_all_faces",
    "clear_face_library",
    "list_faces",
    "load_face_library",
    "MIN_FACE_SAMPLES",
    "MAX_FACE_SAMPLES",
    "identify_person",
    "ToolBehavior",
    "resolve_tool_behavior",
    "behavior_to_text",
    "FACE_TOOL_DECLARATIONS",
    "FACE_FUNCTIONS",
    # Plus Codes
    "resolve_plus_code",
    "convert_to_plus_code",
    "PLUS_CODES_TOOL_DECLARATIONS",
    "PLUS_CODES_FUNCTIONS",
    # Accessibility
    "get_accessibility_info",
    "ACCESSIBILITY_TOOL_DECLARATIONS",
    "ACCESSIBILITY_FUNCTIONS",
    # Maps Grounding
    "maps_query",
    "MAPS_GROUNDING_TOOL_DECLARATIONS",
    "MAPS_GROUNDING_FUNCTIONS",
    "extract_text_from_camera",
    # Emergency
    "get_emergency_help",
    "EMERGENCY_TOOL_DECLARATIONS",
    "EMERGENCY_FUNCTIONS",
    # Memory
    "MEMORY_FUNCTIONS",
    "MEMORY_TOOL_DECLARATIONS",
    # Aggregated
    "CALLABLE_TOOL_DECLARATIONS",
    "PASSIVE_TOOL_DECLARATIONS",
    "CALLABLE_TOOL_ORDER",
    "CALLABLE_TOOL_NAMES",
    "PASSIVE_TOOL_NAMES",
    "ALL_TOOL_CATEGORIES",
    "ALL_TOOL_DECLARATIONS",
    "ALL_FUNCTIONS",
    "ALL_TOOL_RUNTIME_METADATA",
    "build_tool_manifest_entries",
]
