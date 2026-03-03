"""Tests for context.habit_detector module."""

import time
from unittest.mock import MagicMock, patch

import pytest

from context.habit_detector import HabitDetector, ProactiveHint, _MIN_OCCURRENCES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detector(sessions: list[dict]) -> HabitDetector:
    """Create a HabitDetector with mocked Firestore returning given sessions."""
    with patch.object(HabitDetector, "_try_init"):
        detector = HabitDetector("test_user")
    detector._firestore = MagicMock()
    detector._load_sessions = lambda: sessions
    return detector


def _make_session(
    locations: list[str] | None = None,
    time_context: str = "unknown",
    lod_overrides: list[dict] | None = None,
) -> dict:
    return {
        "start_time": time.time(),
        "locations_visited": locations or [],
        "time_context": time_context,
        "lod_overrides": lod_overrides or [],
    }


# ---------------------------------------------------------------------------
# Location habits
# ---------------------------------------------------------------------------


class TestLocationHabits:
    def test_detects_frequent_location(self):
        sessions = [
            _make_session(locations=["Starbucks"], time_context="morning_commute")
            for _ in range(5)
        ]
        detector = _make_detector(sessions)
        hints = detector.detect()

        location_hints = [h for h in hints if h.hint_type == "location_habit"]
        assert len(location_hints) >= 1
        assert any("Starbucks" in h.description for h in location_hints)

    def test_ignores_infrequent_location(self):
        sessions = [
            _make_session(locations=["Random Place"], time_context="work_hours")
            for _ in range(2)  # below _MIN_OCCURRENCES
        ]
        detector = _make_detector(sessions)
        hints = detector.detect()

        location_hints = [h for h in hints if h.hint_type == "location_habit"]
        assert len(location_hints) == 0

    def test_multiple_location_habits(self):
        sessions = []
        for _ in range(4):
            sessions.append(_make_session(locations=["Starbucks"], time_context="morning_commute"))
        for _ in range(3):
            sessions.append(_make_session(locations=["Gym"], time_context="evening"))

        detector = _make_detector(sessions)
        hints = detector.detect()

        location_hints = [h for h in hints if h.hint_type == "location_habit"]
        assert len(location_hints) == 2
        locations = {h.location for h in location_hints}
        assert locations == {"Starbucks", "Gym"}

    def test_empty_sessions(self):
        detector = _make_detector([])
        hints = detector.detect()
        assert hints == []


# ---------------------------------------------------------------------------
# LOD preferences
# ---------------------------------------------------------------------------


class TestLODPreferences:
    def test_detects_consistent_lod_override(self):
        sessions = [
            _make_session(
                locations=["Restaurant"],
                lod_overrides=[{"direction": "up"}],
            )
            for _ in range(4)
        ]
        detector = _make_detector(sessions)
        hints = detector.detect()

        lod_hints = [h for h in hints if h.hint_type == "lod_preference"]
        assert len(lod_hints) >= 1
        assert any("more detail" in h.description for h in lod_hints)

    def test_ignores_inconsistent_overrides(self):
        sessions = [
            _make_session(
                locations=["Place"],
                lod_overrides=[{"direction": "up"}],
            ),
            _make_session(
                locations=["Place"],
                lod_overrides=[{"direction": "down"}],
            ),
        ]
        detector = _make_detector(sessions)
        hints = detector.detect()

        lod_hints = [h for h in hints if h.hint_type == "lod_preference"]
        assert len(lod_hints) == 0  # Neither direction exceeds _MIN_OCCURRENCES


# ---------------------------------------------------------------------------
# Confidence sorting
# ---------------------------------------------------------------------------


class TestConfidenceSorting:
    def test_hints_sorted_by_confidence(self):
        sessions = []
        # 10 visits to Starbucks (high confidence)
        for _ in range(10):
            sessions.append(_make_session(locations=["Starbucks"], time_context="morning_commute"))
        # 3 visits to Gym (low confidence)
        for _ in range(3):
            sessions.append(_make_session(locations=["Gym"], time_context="evening"))

        detector = _make_detector(sessions)
        hints = detector.detect()

        if len(hints) >= 2:
            assert hints[0].confidence >= hints[1].confidence
