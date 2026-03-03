"""Tests for context.scene_matcher module."""

import time
from unittest.mock import MagicMock, patch

import pytest

from context.scene_matcher import (
    SceneMatch,
    SceneMatcher,
    ScenePattern,
    _COSINE_THRESHOLD,
    _cosine_similarity,
)


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(_cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_zero_vector(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert _cosine_similarity(a, b) == 0.0


# ---------------------------------------------------------------------------
# ScenePattern dataclass
# ---------------------------------------------------------------------------


class TestScenePattern:
    def test_to_dict_roundtrip(self):
        p = ScenePattern(
            pattern_id="p1",
            scene_embedding=[0.1, 0.2],
            preferred_lod=3,
            adjustment_count=5,
            location_name="Cafe",
        )
        d = p.to_dict()
        restored = ScenePattern.from_dict("p1", d)
        assert restored.preferred_lod == 3
        assert restored.location_name == "Cafe"

    def test_from_dict_defaults(self):
        p = ScenePattern.from_dict("x", {})
        assert p.preferred_lod == 2
        assert p.satisfaction_score == 0.5


# ---------------------------------------------------------------------------
# SceneMatcher
# ---------------------------------------------------------------------------


def _make_matcher(patterns: list[ScenePattern] | None = None) -> SceneMatcher:
    """Create a SceneMatcher with mocked Firestore."""
    with patch.object(SceneMatcher, "_try_init"):
        matcher = SceneMatcher("test_user")
    matcher._firestore = MagicMock()
    matcher._load_patterns = lambda: patterns or []
    return matcher


class TestSceneMatcher:
    def test_match_similar_scene(self):
        # Create a stored pattern with a known embedding
        stored = ScenePattern(
            pattern_id="p1",
            scene_embedding=[1.0, 0.0, 0.0, 0.0],
            preferred_lod=3,
            location_name="Cafe",
        )
        matcher = _make_matcher([stored])

        # Query with similar embedding
        result = matcher.match([0.99, 0.05, 0.0, 0.0])

        assert result.matched is True
        assert result.similarity >= _COSINE_THRESHOLD
        assert result.suggested_lod == 3

    def test_no_match_different_scene(self):
        stored = ScenePattern(
            pattern_id="p1",
            scene_embedding=[1.0, 0.0, 0.0, 0.0],
            preferred_lod=3,
        )
        matcher = _make_matcher([stored])

        # Query with orthogonal embedding
        result = matcher.match([0.0, 1.0, 0.0, 0.0])

        assert result.matched is False

    def test_no_patterns_returns_unmatched(self):
        matcher = _make_matcher([])
        result = matcher.match([1.0, 0.0])
        assert result.matched is False

    def test_empty_embedding_returns_unmatched(self):
        matcher = _make_matcher([])
        result = matcher.match([])
        assert result.matched is False

    def test_best_match_wins(self):
        patterns = [
            ScenePattern(pattern_id="p1", scene_embedding=[1.0, 0.0], preferred_lod=2),
            ScenePattern(pattern_id="p2", scene_embedding=[0.9, 0.1], preferred_lod=3),
        ]
        matcher = _make_matcher(patterns)

        # Query closest to p1
        result = matcher.match([1.0, 0.0])
        assert result.matched is True
        assert result.suggested_lod == 2

    def test_no_firestore_returns_unmatched(self):
        with patch.object(SceneMatcher, "_try_init"):
            matcher = SceneMatcher("u1")
        matcher._firestore = None

        result = matcher.match([1.0, 0.0])
        assert result.matched is False


class TestSceneMatcherStore:
    def test_store_pattern(self):
        with patch.object(SceneMatcher, "_try_init"):
            matcher = SceneMatcher("test_user")
        matcher._firestore = MagicMock()
        coll = matcher._firestore.collection.return_value.document.return_value.collection.return_value

        pattern = ScenePattern(
            scene_embedding=[0.1] * 10,
            preferred_lod=3,
            location_name="Cafe",
        )

        with patch("google.cloud.firestore_v1.vector.Vector", lambda x: x):
            result = matcher.store_pattern(pattern)

        assert result is not None

    def test_store_pattern_no_firestore(self):
        with patch.object(SceneMatcher, "_try_init"):
            matcher = SceneMatcher("u1")
        matcher._firestore = None

        result = matcher.store_pattern(ScenePattern())
        assert result is None
