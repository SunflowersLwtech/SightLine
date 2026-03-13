"""Unit tests for tool call deduplication (E2E-002 / E2E-003)."""

import time
from unittest.mock import patch

import pytest

from tools.dedup import AudioGate, MutualExclusionFilter, ToolCallDeduplicator

# ---------------------------------------------------------------------------
# ToolCallDeduplicator
# ---------------------------------------------------------------------------


class TestToolCallDeduplicator:
    def test_first_call_allowed(self):
        d = ToolCallDeduplicator(cooldown_sec=5.0)
        ok, reason = d.should_execute("nearby_search", {"lat": 1.0, "lng": 2.0})
        assert ok is True
        assert reason == "ok"

    def test_duplicate_within_cooldown_blocked(self):
        d = ToolCallDeduplicator(cooldown_sec=5.0)
        d.should_execute("nearby_search", {"lat": 1.0, "lng": 2.0})
        ok, reason = d.should_execute("nearby_search", {"lat": 1.0, "lng": 2.0})
        assert ok is False
        assert "repeat" in reason or "duplicate" in reason

    def test_different_args_blocked_within_same_turn(self):
        d = ToolCallDeduplicator(cooldown_sec=5.0)
        d.should_execute("nearby_search", {"lat": 1.0, "lng": 2.0})
        ok, reason = d.should_execute("nearby_search", {"lat": 3.0, "lng": 4.0})
        assert ok is False
        assert "repeat" in reason or "duplicate" in reason

    def test_extract_text_from_camera_blocked_within_same_turn(self):
        d = ToolCallDeduplicator(cooldown_sec=5.0)
        d.should_execute("extract_text_from_camera", {"hint": "menu"})
        ok, reason = d.should_execute("extract_text_from_camera", {"hint": "price"})
        assert ok is False
        assert "repeat" in reason or "duplicate" in reason

    def test_non_target_tool_different_args_still_allowed(self):
        d = ToolCallDeduplicator(cooldown_sec=5.0)
        ok1, _ = d.should_execute("remember_entity", {"name": "Alice"})
        ok2, _ = d.should_execute("remember_entity", {"name": "Bob"})
        assert ok1 is True
        assert ok2 is True

    def test_after_cooldown_allowed(self):
        d = ToolCallDeduplicator(cooldown_sec=0.1)
        d.should_execute("navigate_to", {"destination": "cafe"})
        time.sleep(0.15)
        d.reset()
        ok, reason = d.should_execute("navigate_to", {"destination": "cafe"})
        assert ok is True

    def test_non_target_tool_always_allowed(self):
        d = ToolCallDeduplicator(cooldown_sec=5.0)
        ok, _ = d.should_execute("remember_entity", {"name": "Alice"})
        assert ok is True
        ok, _ = d.should_execute("remember_entity", {"name": "Alice"})
        assert ok is True

    def test_reset_clears_history(self):
        d = ToolCallDeduplicator(cooldown_sec=5.0)
        d.should_execute("navigate_to", {"destination": "cafe"})
        d.reset()
        ok, _ = d.should_execute("navigate_to", {"destination": "cafe"})
        assert ok is True


# ---------------------------------------------------------------------------
# MutualExclusionFilter
# ---------------------------------------------------------------------------


class TestMutualExclusionFilter:
    def test_first_in_group_allowed(self):
        m = MutualExclusionFilter()
        ok, reason = m.should_execute("nearby_search")
        assert ok is True

    def test_second_in_same_group_blocked(self):
        m = MutualExclusionFilter()
        m.should_execute("nearby_search")
        ok, reason = m.should_execute("maps_query")
        assert ok is False
        assert "mutex" in reason

    def test_same_tool_twice_blocked(self):
        """Same tool in the same mutex batch is blocked after first call."""
        m = MutualExclusionFilter()
        m.should_execute("nearby_search")
        ok, reason = m.should_execute("nearby_search")
        assert ok is False
        assert "mutex" in reason

    def test_different_groups_allowed(self):
        m = MutualExclusionFilter()
        m.should_execute("nearby_search")  # place_search group
        ok, _ = m.should_execute("navigate_to")  # navigation group
        assert ok is True

    def test_non_grouped_tool_always_allowed(self):
        m = MutualExclusionFilter()
        ok, _ = m.should_execute("google_search")
        assert ok is True

    def test_reset_clears_state(self):
        m = MutualExclusionFilter()
        m.should_execute("nearby_search")
        m.reset()
        ok, _ = m.should_execute("maps_query")
        assert ok is True


# ---------------------------------------------------------------------------
# AudioGate
# ---------------------------------------------------------------------------


class TestAudioGate:
    def test_default_not_muted(self):
        g = AudioGate()
        assert g.should_mute is False

    def test_enter_mutes(self):
        g = AudioGate()
        g.enter()
        assert g.should_mute is True

    def test_exit_unmutes(self):
        g = AudioGate()
        g.enter()
        g.exit()
        assert g.should_mute is False
