"""Tests for tool behavior routing policy."""

from tools.tool_behavior import ToolBehavior, resolve_tool_behavior


def test_identify_person_is_silent():
    behavior = resolve_tool_behavior("identify_person", lod=2, is_user_speaking=False)
    assert behavior == ToolBehavior.SILENT


def test_navigation_interrupts_only_in_lod1():
    assert resolve_tool_behavior("navigate_to", lod=1, is_user_speaking=False) == ToolBehavior.INTERRUPT
    assert resolve_tool_behavior("navigate_to", lod=2, is_user_speaking=False) == ToolBehavior.WHEN_IDLE


def test_search_defaults_to_when_idle():
    behavior = resolve_tool_behavior("google_search", lod=2, is_user_speaking=False)
    assert behavior == ToolBehavior.WHEN_IDLE


def test_unknown_tool_defaults_to_when_idle():
    behavior = resolve_tool_behavior("some_unknown_tool", lod=2, is_user_speaking=False)
    assert behavior == ToolBehavior.WHEN_IDLE
