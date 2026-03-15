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

    # -----------------------------------------------------------------------
    # Rule 5: Approaching vehicle
    # -----------------------------------------------------------------------

    def test_vehicle_approaching_detected(self, detector):
        """Vehicle that moves from far to close should trigger alert."""
        previous = {
            "safety_warnings": [],
            "people_count": 0,
            "spatial_objects": [
                {"label": "vehicle", "distance_estimate": "5m", "clock_position": 3},
            ],
        }
        current = {
            "safety_warnings": [],
            "people_count": 0,
            "spatial_objects": [
                {"label": "vehicle", "distance_estimate": "2m", "clock_position": 3, "motion_direction": "approaching"},
            ],
        }
        changes = detector.detect(previous, current, "walking")
        vehicle_changes = [c for c in changes if c.change_type == "vehicle_approaching"]
        assert len(vehicle_changes) == 1
        assert vehicle_changes[0].severity == "safety"
        assert vehicle_changes[0].urgency == "approaching"
        assert "3 o'clock" in vehicle_changes[0].details

    def test_vehicle_far_away_not_flagged(self, detector):
        """Vehicle at 5m should not trigger vehicle_approaching."""
        previous = {
            "safety_warnings": [],
            "people_count": 0,
            "spatial_objects": [
                {"label": "vehicle", "distance_estimate": "far"},
            ],
        }
        current = {
            "safety_warnings": [],
            "people_count": 0,
            "spatial_objects": [
                {"label": "vehicle", "distance_estimate": "5m"},
            ],
        }
        changes = detector.detect(previous, current, "walking")
        assert not any(c.change_type == "vehicle_approaching" for c in changes)

    # -----------------------------------------------------------------------
    # Rule 6: Sudden obstacle in path
    # -----------------------------------------------------------------------

    def test_sudden_obstacle_at_12_oclock(self, detector):
        """New obstacle appearing at 12 o'clock within 2m triggers alert."""
        previous = {
            "safety_warnings": [],
            "people_count": 0,
            "spatial_objects": [],
        }
        current = {
            "safety_warnings": [],
            "people_count": 0,
            "spatial_objects": [
                {"label": "pole", "clock_position": 12, "distance_estimate": "1m", "salience": "safety"},
            ],
        }
        changes = detector.detect(previous, current, "walking")
        obstacle_changes = [c for c in changes if c.change_type == "sudden_obstacle"]
        assert len(obstacle_changes) == 1
        assert "pole" in obstacle_changes[0].details

    def test_obstacle_at_3_oclock_not_flagged(self, detector):
        """Obstacles not at 11-1 o'clock should not trigger sudden_obstacle."""
        previous = {
            "safety_warnings": [],
            "people_count": 0,
            "spatial_objects": [],
        }
        current = {
            "safety_warnings": [],
            "people_count": 0,
            "spatial_objects": [
                {"label": "pole", "clock_position": 3, "distance_estimate": "1m", "salience": "safety"},
            ],
        }
        changes = detector.detect(previous, current, "walking")
        assert not any(c.change_type == "sudden_obstacle" for c in changes)

    def test_obstacle_far_away_not_flagged(self, detector):
        """Obstacles at 12 o'clock but >2m should not trigger."""
        previous = {
            "safety_warnings": [],
            "people_count": 0,
            "spatial_objects": [],
        }
        current = {
            "safety_warnings": [],
            "people_count": 0,
            "spatial_objects": [
                {"label": "pole", "clock_position": 12, "distance_estimate": "5m", "salience": "safety"},
            ],
        }
        changes = detector.detect(previous, current, "walking")
        assert not any(c.change_type == "sudden_obstacle" for c in changes)

    # -----------------------------------------------------------------------
    # Rule 7: Person very close
    # -----------------------------------------------------------------------

    def test_person_very_close_detected(self, detector):
        """Person at within_reach who wasn't before should trigger alert."""
        previous = {
            "safety_warnings": [],
            "people_count": 1,
            "spatial_objects": [
                {"label": "person", "distance_estimate": "2m"},
            ],
        }
        current = {
            "safety_warnings": [],
            "people_count": 1,
            "spatial_objects": [
                {"label": "person", "distance_estimate": "within_reach", "clock_position": 12},
            ],
        }
        changes = detector.detect(previous, current, "walking")
        close_changes = [c for c in changes if c.change_type == "person_very_close"]
        assert len(close_changes) == 1
        assert close_changes[0].urgency == "immediate"

    def test_person_already_close_not_re_flagged(self, detector):
        """Person already at within_reach should not re-trigger."""
        state = {
            "safety_warnings": [],
            "people_count": 1,
            "spatial_objects": [
                {"label": "person", "distance_estimate": "within_reach"},
            ],
        }
        changes = detector.detect(state, state, "stationary")
        assert not any(c.change_type == "person_very_close" for c in changes)

    # -----------------------------------------------------------------------
    # Urgency field
    # -----------------------------------------------------------------------

    def test_urgency_default_is_awareness(self):
        change = SpatialChange(
            change_type="hazard_appeared",
            severity="safety",
            details="test",
        )
        assert change.urgency == "awareness"

    def test_urgency_immediate_for_within_reach(self, detector):
        previous = {
            "safety_warnings": [],
            "people_count": 0,
            "spatial_objects": [],
        }
        current = {
            "safety_warnings": [],
            "people_count": 0,
            "spatial_objects": [
                {"label": "bollard", "clock_position": 12, "distance_estimate": "within_reach", "salience": "safety"},
            ],
        }
        changes = detector.detect(previous, current, "walking")
        obstacle = [c for c in changes if c.change_type == "sudden_obstacle"]
        assert len(obstacle) == 1
        assert obstacle[0].urgency == "immediate"
