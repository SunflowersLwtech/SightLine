"""Tests for SpatialChangeDetector."""

import pytest

from context.spatial_change_detector import SpatialChange, SpatialChangeDetector


@pytest.fixture
def detector():
    return SpatialChangeDetector()


class TestSpatialChangeDetector:
    """Tests for rule-based spatial change detection."""

    def test_empty_previous_returns_no_changes(self, detector):
        current = {"safety_warnings": [], "people_count": 2, "spatial_objects": []}
        assert detector.detect({}, current, "stationary") == []

    def test_empty_current_returns_no_changes(self, detector):
        previous = {"safety_warnings": [], "people_count": 2, "spatial_objects": []}
        assert detector.detect(previous, {}, "stationary") == []

    def test_new_safety_warning_detected(self, detector):
        previous = {"safety_warnings": ["stairs at 12 o'clock"], "people_count": 0, "spatial_objects": []}
        current = {
            "safety_warnings": ["stairs at 12 o'clock", "vehicle at 3 o'clock"],
            "people_count": 0,
            "spatial_objects": [],
        }
        changes = detector.detect(previous, current, "walking")
        assert len(changes) == 1
        assert changes[0].change_type == "hazard_appeared"
        assert changes[0].severity == "safety"
        assert "vehicle" in changes[0].details

    def test_person_approaching_detected(self, detector):
        previous = {"safety_warnings": [], "people_count": 1, "spatial_objects": []}
        current = {
            "safety_warnings": [],
            "people_count": 2,
            "spatial_objects": [
                {"label": "person", "distance_estimate": "1m", "salience": "interaction"},
            ],
        }
        changes = detector.detect(previous, current, "walking")
        assert any(c.change_type == "new_person_approaching" for c in changes)

    def test_person_left_detected(self, detector):
        previous = {"safety_warnings": [], "people_count": 5, "spatial_objects": []}
        current = {"safety_warnings": [], "people_count": 2, "spatial_objects": []}
        changes = detector.detect(previous, current, "stationary")
        assert any(c.change_type == "person_left" for c in changes)

    def test_person_decrease_by_one_not_detected(self, detector):
        """Only >=2 decrease triggers person_left."""
        previous = {"safety_warnings": [], "people_count": 3, "spatial_objects": []}
        current = {"safety_warnings": [], "people_count": 2, "spatial_objects": []}
        changes = detector.detect(previous, current, "stationary")
        assert not any(c.change_type == "person_left" for c in changes)

    def test_layout_change_when_stationary(self, detector):
        previous = {
            "safety_warnings": [],
            "people_count": 0,
            "spatial_objects": [
                {"label": "chair"}, {"label": "table"}, {"label": "door"},
            ],
        }
        current = {
            "safety_warnings": [],
            "people_count": 0,
            "spatial_objects": [
                {"label": "car"}, {"label": "tree"}, {"label": "sidewalk"},
            ],
        }
        changes = detector.detect(previous, current, "stationary")
        assert any(c.change_type == "layout_change" for c in changes)

    def test_layout_change_suppressed_when_walking(self, detector):
        """Camera bounce during walking suppresses layout_change."""
        previous = {
            "safety_warnings": [],
            "people_count": 0,
            "spatial_objects": [{"label": "chair"}, {"label": "table"}],
        }
        current = {
            "safety_warnings": [],
            "people_count": 0,
            "spatial_objects": [{"label": "car"}, {"label": "tree"}],
        }
        changes = detector.detect(previous, current, "walking")
        assert not any(c.change_type == "layout_change" for c in changes)

    def test_severity_ordering(self, detector):
        """Safety changes should appear before significant before minor."""
        previous = {
            "safety_warnings": [],
            "people_count": 5,
            "spatial_objects": [{"label": "a"}, {"label": "b"}],
        }
        current = {
            "safety_warnings": ["obstacle ahead"],
            "people_count": 2,
            "spatial_objects": [{"label": "c"}, {"label": "d"}],
        }
        changes = detector.detect(previous, current, "stationary")
        severities = [c.severity for c in changes]
        # safety should come first
        if "safety" in severities and "minor" in severities:
            assert severities.index("safety") < severities.index("minor")

    def test_no_changes_when_same_state(self, detector):
        state = {
            "safety_warnings": ["stairs at 12"],
            "people_count": 3,
            "spatial_objects": [{"label": "person"}, {"label": "chair"}],
        }
        changes = detector.detect(state, state, "stationary")
        assert len(changes) == 0
