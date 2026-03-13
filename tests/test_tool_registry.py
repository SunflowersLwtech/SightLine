"""Tests for shared tool registry consistency."""

from tools import (
    ALL_FUNCTIONS,
    ALL_TOOL_DECLARATIONS,
    CALLABLE_TOOL_NAMES,
    CALLABLE_TOOL_ORDER,
    PASSIVE_TOOL_NAMES,
    build_tool_manifest_entries,
)


def test_callable_tool_order_matches_callable_tool_set():
    assert len(CALLABLE_TOOL_ORDER) == len(set(CALLABLE_TOOL_ORDER))
    assert set(CALLABLE_TOOL_ORDER) == CALLABLE_TOOL_NAMES


def test_callable_tool_names_have_runtime_functions():
    for name in CALLABLE_TOOL_NAMES:
        assert name in ALL_FUNCTIONS


def test_declarations_cover_callable_and_passive_tools():
    declared = {str(decl["name"]) for decl in ALL_TOOL_DECLARATIONS}
    assert CALLABLE_TOOL_NAMES.issubset(declared)
    assert PASSIVE_TOOL_NAMES.issubset(declared)


def test_manifest_marks_passive_face_tool_as_automatic():
    manifest = build_tool_manifest_entries(lod=2)
    by_name = {entry["name"]: entry for entry in manifest}

    assert by_name["identify_person"]["callable"] is False
    assert by_name["identify_person"]["source"] == "automatic"
    assert by_name["preload_memory"]["callable"] is True
    assert by_name["preload_memory"]["description"]


def test_manifest_uses_runtime_behavior_policy_for_navigation():
    manifest_lod1 = build_tool_manifest_entries(lod=1)
    manifest_lod2 = build_tool_manifest_entries(lod=2)
    by_name_lod1 = {entry["name"]: entry for entry in manifest_lod1}
    by_name_lod2 = {entry["name"]: entry for entry in manifest_lod2}

    assert by_name_lod1["navigate_to"]["behavior"] == "INTERRUPT"
    assert by_name_lod2["navigate_to"]["behavior"] == "WHEN_IDLE"
