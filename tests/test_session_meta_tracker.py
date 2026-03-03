"""Tests for SessionMetaTracker (in-memory accumulation, no Firestore)."""

import time
from unittest.mock import patch

import telemetry.session_meta_tracker as _smt_mod


def _real_class():
    """Access SessionMetaTracker at call time to dodge import-time MagicMock pollution."""
    return _smt_mod.SessionMetaTracker


class TestLodDistribution:
    """Verify that LOD elapsed-seconds tracking works correctly."""

    def test_lod_distribution_tracks_time(self):
        """Elapsed seconds are accumulated per LOD level."""
        tracker = _real_class()(user_id="u1", session_id="s1")

        # Simulate starting at LOD 2, spending 10s there
        tracker._lod_start_time = time.monotonic() - 10
        tracker._current_lod = 2
        tracker.record_lod_time(3)  # switch to LOD 3

        assert tracker._lod_distribution["lod2"] >= 9  # allow 1s tolerance
        assert tracker._current_lod == 3

        # Simulate 5s at LOD 3
        tracker._lod_start_time = time.monotonic() - 5
        tracker.record_lod_time(1)

        assert tracker._lod_distribution["lod3"] >= 4
        assert tracker._current_lod == 1

    def test_lod_distribution_accumulates(self):
        """Multiple transitions to the same LOD accumulate time."""
        tracker = _real_class()(user_id="u1", session_id="s1")
        tracker._current_lod = 1
        tracker._lod_start_time = time.monotonic() - 3
        tracker.record_lod_time(2)  # 3s at LOD 1

        tracker._lod_start_time = time.monotonic() - 2
        tracker.record_lod_time(1)  # 2s at LOD 2

        tracker._lod_start_time = time.monotonic() - 4
        tracker.record_lod_time(2)  # 4s at LOD 1 (cumulative)

        assert tracker._lod_distribution["lod1"] >= 6  # 3 + 4 = 7, allow tolerance
        assert tracker._lod_distribution["lod2"] >= 1


class TestInteractionCounter:
    """Verify interaction counting."""

    def test_interaction_counter(self):
        tracker = _real_class()(user_id="u1", session_id="s1")
        assert tracker._total_interactions == 0

        tracker.record_interaction()
        tracker.record_interaction()
        tracker.record_interaction()

        assert tracker._total_interactions == 3

    def test_interaction_counter_starts_at_zero(self):
        tracker = _real_class()(user_id="u1", session_id="s1")
        assert tracker._total_interactions == 0


class TestSpaceTransitions:
    """Verify space transitions are recorded and filtered."""

    def test_space_transition(self):
        tracker = _real_class()(user_id="u1", session_id="s1")
        tracker.space_transitions = ["outdoor->indoor", "indoor->elevator"]

        doc = tracker.build_end_doc()
        assert doc["space_transitions"] == ["outdoor->indoor", "indoor->elevator"]

    def test_unknown_filtered(self):
        """Transitions containing 'unknown' are filtered out."""
        tracker = _real_class()(user_id="u1", session_id="s1")
        tracker.space_transitions = ["outdoor->indoor", "unknown", "indoor->cafe"]

        doc = tracker.build_end_doc()
        assert "unknown" not in doc["space_transitions"]
        assert len(doc["space_transitions"]) == 2


class TestBuildEndDoc:
    """Verify the output dict structure of build_end_doc."""

    def test_build_end_doc_structure(self):
        tracker = _real_class()(user_id="u1", session_id="s1")
        tracker.set_trip_purpose("Coffee run")
        tracker.space_transitions = ["outdoor->cafe"]
        tracker.record_interaction()
        tracker.record_interaction()

        doc = tracker.build_end_doc()

        assert "end_time" in doc
        assert doc["trip_purpose"] == "Coffee run"
        assert isinstance(doc["lod_distribution"], dict)
        assert "lod1" in doc["lod_distribution"]
        assert "lod2" in doc["lod_distribution"]
        assert "lod3" in doc["lod_distribution"]
        assert doc["space_transitions"] == ["outdoor->cafe"]
        assert doc["total_interactions"] == 2

    def test_build_end_doc_empty_session(self):
        """An empty session still produces a valid document structure."""
        tracker = _real_class()(user_id="u1", session_id="s1")
        doc = tracker.build_end_doc()

        assert doc["trip_purpose"] == ""
        assert doc["total_interactions"] == 0
        assert doc["space_transitions"] == []
        assert isinstance(doc["lod_distribution"], dict)


class TestSetTripPurpose:
    """Verify trip purpose setter."""

    def test_set_trip_purpose(self):
        tracker = _real_class()(user_id="u1", session_id="s1")
        tracker.set_trip_purpose("Going to the office")
        assert tracker._trip_purpose == "Going to the office"

    def test_default_trip_purpose_empty(self):
        tracker = _real_class()(user_id="u1", session_id="s1")
        assert tracker._trip_purpose == ""
