"""Tests for memory.memory_ranking module."""

import time

import pytest

from memory.memory_ranking import (
    ENTITY_OVERLAP_WEIGHT,
    IMPORTANCE_WEIGHT,
    LOCATION_WEIGHT,
    RECENCY_WEIGHT,
    RELEVANCE_WEIGHT,
    rank_memories,
    score_memories,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_memory(
    content: str = "test",
    relevance_score: float = 0.5,
    importance: float = 0.5,
    age_hours: float = 0.0,
) -> dict:
    """Create a memory dict with the given parameters."""
    return {
        "content": content,
        "relevance_score": relevance_score,
        "importance": importance,
        "timestamp": time.time() - (age_hours * 3600),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRankMemories:
    """Tests for the rank_memories function."""

    def test_sorts_by_composite_score(self):
        memories = [
            _make_memory("low", relevance_score=0.1, importance=0.1),
            _make_memory("high", relevance_score=0.9, importance=0.9),
            _make_memory("mid", relevance_score=0.5, importance=0.5),
        ]
        ranked = rank_memories(memories)
        assert ranked[0]["content"] == "high"
        assert ranked[-1]["content"] == "low"

    def test_relevance_dimension_affects_ranking(self):
        memories = [
            _make_memory("low_rel", relevance_score=0.1, importance=0.5),
            _make_memory("high_rel", relevance_score=0.9, importance=0.5),
        ]
        ranked = rank_memories(memories)
        assert ranked[0]["content"] == "high_rel"

    def test_recency_dimension_recent_beats_old(self):
        memories = [
            _make_memory("old", relevance_score=0.5, importance=0.5, age_hours=168),  # 7 days
            _make_memory("recent", relevance_score=0.5, importance=0.5, age_hours=0),
        ]
        ranked = rank_memories(memories)
        assert ranked[0]["content"] == "recent"
        # Verify recency score for recent is higher
        assert ranked[0]["_recency"] > ranked[1]["_recency"]

    def test_importance_dimension_affects_ranking(self):
        # Same relevance, same age, different importance
        memories = [
            _make_memory("low_imp", relevance_score=0.5, importance=0.1),
            _make_memory("high_imp", relevance_score=0.5, importance=0.9),
        ]
        ranked = rank_memories(memories)
        assert ranked[0]["content"] == "high_imp"

    def test_max_results_truncation(self):
        memories = [_make_memory(f"mem_{i}") for i in range(10)]
        ranked = rank_memories(memories, max_results=3)
        assert len(ranked) == 3

    def test_max_results_larger_than_input(self):
        memories = [_make_memory("only")]
        ranked = rank_memories(memories, max_results=5)
        assert len(ranked) == 1

    def test_empty_memories_list(self):
        ranked = rank_memories([])
        assert ranked == []

    def test_composite_score_present(self):
        memories = [_make_memory("test")]
        ranked = rank_memories(memories)
        assert "_composite_score" in ranked[0]
        assert "_recency" in ranked[0]

    def test_recency_decay_half_life(self):
        """Verify that a 24-hour-old memory has roughly half the recency score."""
        memories = [
            _make_memory("now", age_hours=0),
            _make_memory("one_day", age_hours=24),
        ]
        ranked = rank_memories(memories, max_results=2)
        now_mem = next(m for m in ranked if m["content"] == "now")
        day_mem = next(m for m in ranked if m["content"] == "one_day")

        # Recency of 24h old should be ~0.5 of now
        assert abs(day_mem["_recency"] - 0.5 * now_mem["_recency"]) < 0.01

    def test_default_values_for_missing_fields(self):
        """Memories missing optional fields should use defaults."""
        mem = {"content": "bare minimum"}
        ranked = rank_memories([mem])
        assert len(ranked) == 1
        assert "_composite_score" in ranked[0]

    def test_weights_sum_to_one(self):
        total = RELEVANCE_WEIGHT + RECENCY_WEIGHT + IMPORTANCE_WEIGHT + LOCATION_WEIGHT + ENTITY_OVERLAP_WEIGHT
        assert abs(total - 1.0) < 1e-9

    def test_original_fields_preserved(self):
        memories = [{"content": "test", "category": "place", "extra": 42}]
        ranked = rank_memories(memories)
        assert ranked[0]["category"] == "place"
        assert ranked[0]["extra"] == 42


class TestLayeredDecay:
    """Tests for layer-aware half-life decay."""

    def test_episodic_decays_faster_than_semantic(self):
        """Episodic (7-day half-life) should rank lower than semantic (90-day)
        at the same age."""
        age = 168  # 7 days in hours
        memories = [
            {**_make_memory("episodic", age_hours=age), "half_life_days": 7},
            {**_make_memory("semantic", age_hours=age), "half_life_days": 90},
        ]
        ranked = rank_memories(memories, max_results=2)
        # Semantic decays slower, so it should rank higher
        assert ranked[0]["content"] == "semantic"
        assert ranked[0]["_recency"] > ranked[1]["_recency"]

    def test_procedural_near_permanent(self):
        """Procedural (999-day half-life) should barely decay even after months."""
        age = 720  # 30 days in hours
        memories = [
            {**_make_memory("proc", age_hours=age), "half_life_days": 999},
        ]
        ranked = rank_memories(memories, max_results=1)
        # 30 days / 999 days ~ 0.03 half-lives → recency ~ 0.98
        assert ranked[0]["_recency"] > 0.95

    def test_legacy_default_half_life(self):
        """Memories without half_life_days use 1-day (legacy 24hr decay)."""
        memories = [_make_memory("legacy", age_hours=24)]
        ranked = rank_memories(memories)
        # half_life_days=1 → 24hr old = 0.5 recency
        assert abs(ranked[0]["_recency"] - 0.5) < 0.01

    def test_backward_compat_no_new_fields(self):
        """Old memories without memory_layer/entity_refs still rank correctly."""
        memories = [
            _make_memory("old_a", relevance_score=0.9, importance=0.8),
            _make_memory("old_b", relevance_score=0.2, importance=0.3),
        ]
        ranked = rank_memories(memories)
        assert ranked[0]["content"] == "old_a"


class TestLocationScoring:
    """Tests for location-aware ranking dimension."""

    def test_matching_location_boosts_score(self):
        class FakeLocation:
            matched_entity_id = "place_001"

        memories = [
            {**_make_memory("at_place", relevance_score=0.5), "location_ref": "place_001"},
            {**_make_memory("elsewhere", relevance_score=0.5), "location_ref": "place_999"},
        ]
        ranked = rank_memories(memories, current_location=FakeLocation())
        assert ranked[0]["content"] == "at_place"
        assert ranked[0]["_location_score"] == 1.0
        assert ranked[1]["_location_score"] == 0.3

    def test_no_location_context_neutral(self):
        memories = [
            {**_make_memory("a", relevance_score=0.5), "location_ref": "place_001"},
        ]
        ranked = rank_memories(memories)
        assert ranked[0]["_location_score"] == 0.5


class TestEntityOverlapScoring:
    """Tests for entity overlap ranking dimension."""

    def test_entity_overlap_boosts_score(self):
        memories = [
            {**_make_memory("with_entity", relevance_score=0.5), "entity_refs": ["e001", "e002"]},
            {**_make_memory("no_entity", relevance_score=0.5), "entity_refs": []},
        ]
        ranked = rank_memories(memories, visible_entity_ids=["e001"])
        assert ranked[0]["content"] == "with_entity"
        assert ranked[0]["_entity_score"] > ranked[1]["_entity_score"]

    def test_full_overlap(self):
        memories = [
            {**_make_memory("full", relevance_score=0.5), "entity_refs": ["e001"]},
        ]
        ranked = rank_memories(memories, visible_entity_ids=["e001", "e002"])
        assert ranked[0]["_entity_score"] == 1.0

    def test_no_visible_entities_neutral(self):
        memories = [
            {**_make_memory("a"), "entity_refs": ["e001"]},
        ]
        ranked = rank_memories(memories)
        assert ranked[0]["_entity_score"] == 0.5


class TestScoreMemories:
    """Tests for the score_memories backward-compatibility alias."""

    def test_score_memories_returns_same_as_rank(self):
        memories = [
            _make_memory("a", relevance_score=0.9),
            _make_memory("b", relevance_score=0.1),
        ]
        ranked = rank_memories(memories)
        scored = score_memories(memories)
        assert [m["content"] for m in ranked] == [m["content"] for m in scored]

    def test_score_memories_accepts_context(self):
        memories = [_make_memory("test")]
        result = score_memories(memories, context="some context")
        assert len(result) == 1
