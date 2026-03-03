"""Tests for memory.memory_extractor module.

Covers confidence threshold filtering, category validation,
text-based duplicate detection fallback, and candidate validation.
"""

from unittest.mock import MagicMock, patch

import pytest

from memory.memory_extractor import MemoryExtractor, _VALID_CATEGORIES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_extractor(**kwargs) -> MemoryExtractor:
    return MemoryExtractor(**kwargs)


def _make_mock_bank(existing_memories=None):
    bank = MagicMock()
    bank.retrieve_memories.return_value = existing_memories or []
    bank.store_memory.return_value = "new_id"
    bank.delete_memory.return_value = True
    return bank


def _make_budget(exhausted=False):
    budget = MagicMock()
    budget.exhausted = exhausted
    budget.try_write.return_value = True
    return budget


# ---------------------------------------------------------------------------
# Confidence threshold filtering (Task 2.3)
# ---------------------------------------------------------------------------


class TestConfidenceThreshold:
    """Tests for per-category confidence thresholds."""

    def test_default_threshold_is_075(self):
        ext = _make_extractor()
        assert ext.confidence_threshold == 0.75

    def test_general_memory_below_threshold_skipped(self):
        ext = _make_extractor()
        bank = _make_mock_bank()
        budget = _make_budget()

        # Candidate with confidence 0.6 < 0.75 should be skipped
        candidates = [{"content": "Low conf", "category": "general", "importance": 0.5, "confidence": 0.6}]
        with patch.object(ext, "_call_extraction_model", return_value=candidates):
            count = ext.extract_and_store("u1", "s1", [{"role": "user", "text": "hello world test"}], bank, budget)

        assert count == 0
        bank.store_memory.assert_not_called()

    def test_general_memory_above_threshold_stored(self):
        ext = _make_extractor()
        bank = _make_mock_bank()
        budget = _make_budget()

        candidates = [{"content": "Good memory", "category": "general", "importance": 0.5, "confidence": 0.8}]
        with patch.object(ext, "_call_extraction_model", return_value=candidates):
            with patch.object(ext, "_find_duplicate", return_value=None):
                count = ext.extract_and_store("u1", "s1", [{"role": "user", "text": "hello world test"}], bank, budget)

        assert count == 1

    def test_person_category_requires_09(self):
        ext = _make_extractor()
        bank = _make_mock_bank()
        budget = _make_budget()

        # Person with confidence 0.85 should be skipped (needs 0.9)
        candidates = [{"content": "David is a coworker", "category": "person", "importance": 0.7, "confidence": 0.85}]
        with patch.object(ext, "_call_extraction_model", return_value=candidates):
            count = ext.extract_and_store("u1", "s1", [{"role": "user", "text": "hello world test"}], bank, budget)

        assert count == 0

    def test_person_category_at_09_stored(self):
        ext = _make_extractor()
        bank = _make_mock_bank()
        budget = _make_budget()

        candidates = [{"content": "David is a coworker", "category": "person", "importance": 0.7, "confidence": 0.95}]
        with patch.object(ext, "_call_extraction_model", return_value=candidates):
            with patch.object(ext, "_find_duplicate", return_value=None):
                count = ext.extract_and_store("u1", "s1", [{"role": "user", "text": "hello world test"}], bank, budget)

        assert count == 1


# ---------------------------------------------------------------------------
# Category validation (Task 2.4)
# ---------------------------------------------------------------------------


class TestCategoryValidation:
    """Tests for _validate_candidate method."""

    def test_valid_candidate_passes(self):
        ext = _make_extractor()
        result = ext._validate_candidate({
            "content": "User likes coffee",
            "category": "preference",
            "importance": 0.8,
            "confidence": 0.9,
        })
        assert result is not None
        assert result["content"] == "User likes coffee"
        assert result["category"] == "preference"
        assert result["importance"] == 0.8
        assert result["confidence"] == 0.9

    def test_empty_content_rejected(self):
        ext = _make_extractor()
        result = ext._validate_candidate({"content": "", "category": "general"})
        assert result is None

    def test_whitespace_only_content_rejected(self):
        ext = _make_extractor()
        result = ext._validate_candidate({"content": "   ", "category": "general"})
        assert result is None

    def test_unknown_category_defaults_to_general(self):
        ext = _make_extractor()
        result = ext._validate_candidate({
            "content": "Some fact",
            "category": "unknown_category",
            "importance": 0.5,
            "confidence": 0.8,
        })
        assert result is not None
        assert result["category"] == "general"

    def test_all_valid_categories_accepted(self):
        ext = _make_extractor()
        for cat in _VALID_CATEGORIES:
            result = ext._validate_candidate({
                "content": f"Test {cat}",
                "category": cat,
                "importance": 0.5,
                "confidence": 0.8,
            })
            assert result is not None
            assert result["category"] == cat

    def test_importance_clamped_to_0_1(self):
        ext = _make_extractor()
        result = ext._validate_candidate({
            "content": "Fact",
            "category": "general",
            "importance": 1.5,
            "confidence": 0.8,
        })
        assert result["importance"] == 1.0

        result = ext._validate_candidate({
            "content": "Fact",
            "category": "general",
            "importance": -0.3,
            "confidence": 0.8,
        })
        assert result["importance"] == 0.0

    def test_confidence_clamped_to_0_1(self):
        ext = _make_extractor()
        result = ext._validate_candidate({
            "content": "Fact",
            "category": "general",
            "importance": 0.5,
            "confidence": 2.0,
        })
        assert result["confidence"] == 1.0

    def test_missing_fields_use_defaults(self):
        ext = _make_extractor()
        result = ext._validate_candidate({"content": "Only content"})
        assert result is not None
        assert result["category"] == "general"
        assert result["importance"] == 0.5
        assert result["confidence"] == 0.0

    def test_invalid_type_returns_none(self):
        ext = _make_extractor()
        result = ext._validate_candidate({"content": "ok", "importance": "not_a_number"})
        assert result is None


# ---------------------------------------------------------------------------
# Text-based duplicate detection fallback (Task 2.2)
# ---------------------------------------------------------------------------


class TestTextDuplicateDetection:
    """Tests for _find_duplicate with text fallback."""

    def test_text_similarity_high_overlap(self):
        ext = _make_extractor()
        sim = ext._text_similarity(
            "The pharmacy is on Main Street",
            "The pharmacy is on Main Street near the park",
        )
        assert sim > 0.5

    def test_text_similarity_no_overlap(self):
        ext = _make_extractor()
        sim = ext._text_similarity("hello world", "foo bar baz")
        assert sim == 0.0

    def test_text_similarity_empty_string(self):
        ext = _make_extractor()
        assert ext._text_similarity("", "hello") == 0.0
        assert ext._text_similarity("hello", "") == 0.0

    def test_find_duplicate_uses_text_fallback_when_embedding_fails(self):
        """When embedding returns zero vector, text similarity should still detect duplicates."""
        ext = _make_extractor()
        existing = [
            {"content": "the pharmacy is on main street", "memory_id": "mem1"},
        ]
        # Mock embedding to always return zero vector
        with patch("memory.memory_extractor._compute_embedding", return_value=[0.0] * 2048):
            result = ext._find_duplicate("the pharmacy is on main street near park", existing)

        assert result is not None
        assert result["memory_id"] == "mem1"

    def test_find_duplicate_returns_none_for_dissimilar_text(self):
        ext = _make_extractor()
        existing = [
            {"content": "The pharmacy is on Main Street", "memory_id": "mem1"},
        ]
        with patch("memory.memory_extractor._compute_embedding", return_value=[0.0] * 2048):
            result = ext._find_duplicate("My dog is named Buddy", existing)

        assert result is None

    def test_find_duplicate_empty_existing(self):
        ext = _make_extractor()
        result = ext._find_duplicate("anything", [])
        assert result is None

    def test_find_duplicate_prefers_vector_similarity(self):
        """When embeddings work, vector similarity should be used."""
        ext = _make_extractor()
        existing = [
            {"content": "User prefers dark roast coffee", "memory_id": "mem1"},
        ]
        # Mock embedding to return non-zero vectors with high cosine sim
        with patch("memory.memory_extractor._compute_embedding", return_value=[0.9] * 2048):
            with patch("memory.memory_extractor._cosine_similarity", return_value=0.95):
                result = ext._find_duplicate("User prefers dark roast", existing)

        assert result is not None
        assert result["memory_id"] == "mem1"


# ---------------------------------------------------------------------------
# Embedding cache optimization (Fix 4)
# ---------------------------------------------------------------------------


class TestEmbeddingCache:
    """Tests for _precompute_existing_embeddings and cached _find_duplicate."""

    def test_precompute_caches_all_memories(self):
        """_precompute_existing_embeddings returns embeddings keyed by memory_id."""
        ext = _make_extractor()
        memories = [
            {"memory_id": "m1", "content": "User likes coffee"},
            {"memory_id": "m2", "content": "User takes the 8am bus"},
            {"memory_id": "m3", "content": ""},  # empty content, should be skipped
            {"content": "No id field"},  # no memory_id, should be skipped
        ]
        fake_emb = [0.5] * 2048
        with patch("memory.memory_extractor._compute_embedding", return_value=fake_emb) as mock_emb:
            cache = ext._precompute_existing_embeddings(memories)

        assert "m1" in cache
        assert "m2" in cache
        assert "m3" not in cache  # empty content skipped
        assert len(cache) == 2
        assert mock_emb.call_count == 2
        assert cache["m1"] == fake_emb
        assert cache["m2"] == fake_emb

    def test_find_duplicate_uses_cache(self):
        """When existing_embeddings cache is provided, _compute_embedding is NOT called for cached memories."""
        ext = _make_extractor()
        existing = [
            {"content": "User likes coffee", "memory_id": "m1"},
            {"content": "User takes the 8am bus", "memory_id": "m2"},
        ]
        # Pre-populated cache
        cached_emb = [0.9] * 2048
        existing_embeddings = {"m1": cached_emb, "m2": cached_emb}

        call_count = 0
        def mock_compute(text):
            nonlocal call_count
            call_count += 1
            return [0.9] * 2048

        with patch("memory.memory_extractor._compute_embedding", side_effect=mock_compute):
            with patch("memory.memory_extractor._cosine_similarity", return_value=0.95):
                result = ext._find_duplicate("User likes tea", existing, existing_embeddings=existing_embeddings)

        # _compute_embedding should only be called once — for the candidate content, not for existing memories
        assert call_count == 1
        assert result is not None
