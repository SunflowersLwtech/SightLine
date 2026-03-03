"""Tests for context.profile_assembler module."""

import time

import pytest

from context.profile_assembler import ProfileAssembler, _MAX_CHARS
from lod.models import UserProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_profile(**kwargs) -> UserProfile:
    defaults = {
        "user_id": "test",
        "vision_status": "totally_blind",
        "blindness_onset": "congenital",
        "has_white_cane": True,
        "tts_speed": 1.5,
        "verbosity_preference": "concise",
        "om_level": "intermediate",
        "travel_frequency": "weekly",
        "preferred_name": "Alex",
    }
    defaults.update(kwargs)
    return UserProfile(**defaults)


class FakeLocationCtx:
    def __init__(self, **kwargs):
        self.place_name = kwargs.get("place_name", "Blue Bottle Coffee")
        self.place_type = kwargs.get("place_type", "cafe")
        self.is_indoor = kwargs.get("is_indoor", True)
        self.familiarity_score = kwargs.get("familiarity_score", 0.8)
        self.address = kwargs.get("address", "123 Main St")
        self.matched_entity_id = kwargs.get("matched_entity_id", "p001")


class FakeEntity:
    def __init__(self, name="David", entity_type="person", attributes=None):
        self.name = name
        self.entity_type = entity_type
        self.attributes = attributes or {}


def _make_memory(content, memory_layer="semantic", category="general", age_hours=0):
    return {
        "content": content,
        "memory_layer": memory_layer,
        "category": category,
        "timestamp": time.time() - (age_hours * 3600),
        "importance": 0.5,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProfileAssembler:
    def test_basic_profile_only(self):
        assembler = ProfileAssembler()
        result = assembler.assemble(_make_profile())

        assert "## User Profile" in result
        assert "totally_blind" in result
        assert "white cane" in result
        assert "Alex" in result

    def test_with_location_context(self):
        assembler = ProfileAssembler()
        result = assembler.assemble(
            _make_profile(),
            location_ctx=FakeLocationCtx(),
        )

        assert "## Current Location" in result
        assert "Blue Bottle Coffee" in result
        assert "cafe" in result
        assert "well-known" in result

    def test_with_entities(self):
        assembler = ProfileAssembler()
        entities = [
            FakeEntity("David", "person", {"role": "coworker"}),
            FakeEntity("Starbucks", "place"),
        ]
        result = assembler.assemble(_make_profile(), entities=entities)

        assert "## Known People & Places Nearby" in result
        assert "David" in result
        assert "coworker" in result

    def test_with_procedural_memories(self):
        assembler = ProfileAssembler()
        memories = [
            _make_memory("Prefers LOD 3 at restaurants", memory_layer="procedural"),
            _make_memory("Always takes bus 42", category="routine"),
        ]
        result = assembler.assemble(_make_profile(), memories=memories)

        assert "## Habits & Preferences" in result
        assert "LOD 3 at restaurants" in result
        assert "bus 42" in result

    def test_with_episodic_memories(self):
        assembler = ProfileAssembler()
        memories = [
            _make_memory("Visited museum yesterday", memory_layer="episodic", age_hours=24),
        ]
        result = assembler.assemble(_make_profile(), memories=memories)

        assert "## Recent Events" in result
        assert "museum yesterday" in result

    def test_old_episodic_excluded(self):
        assembler = ProfileAssembler()
        memories = [
            _make_memory("Old event", memory_layer="episodic", age_hours=200),  # >7 days
        ]
        result = assembler.assemble(_make_profile(), memories=memories)

        assert "## Recent Events" not in result

    def test_output_within_budget(self):
        assembler = ProfileAssembler()
        # Create many memories
        memories = [
            _make_memory(f"Memory {i} with lots of detail " * 10, memory_layer="procedural")
            for i in range(20)
        ]
        result = assembler.assemble(
            _make_profile(),
            location_ctx=FakeLocationCtx(),
            entities=[FakeEntity(f"Person{i}") for i in range(10)],
            memories=memories,
        )

        assert len(result) <= _MAX_CHARS

    def test_empty_location_name_skipped(self):
        assembler = ProfileAssembler()
        result = assembler.assemble(
            _make_profile(),
            location_ctx=FakeLocationCtx(place_name=""),
        )
        assert "## Current Location" not in result

    def test_color_description_disabled(self):
        assembler = ProfileAssembler()
        result = assembler.assemble(
            _make_profile(blindness_onset="congenital", color_description=False),
        )
        assert "Color descriptions: DISABLED" in result

    def test_no_memories_no_memory_sections(self):
        assembler = ProfileAssembler()
        result = assembler.assemble(_make_profile())
        assert "## Habits" not in result
        assert "## Recent Events" not in result

    def test_unfamiliar_location(self):
        assembler = ProfileAssembler()
        result = assembler.assemble(
            _make_profile(),
            location_ctx=FakeLocationCtx(familiarity_score=0.1),
        )
        assert "first or rare visit" in result
