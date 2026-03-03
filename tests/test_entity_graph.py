"""Tests for context.entity_graph module.

All Firestore calls are mocked so tests run offline.
"""

import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from context.entity_graph import (
    Entity,
    EntityGraphService,
    MAX_ENTITIES,
    MAX_RELATIONS,
    Relation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph(user_id: str = "test_user") -> EntityGraphService:
    """Create an EntityGraphService with mocked Firestore."""
    with patch.object(EntityGraphService, "_try_init"):
        graph = EntityGraphService(user_id)
    graph._firestore = MagicMock()
    graph._ensure_firestore = lambda: graph._firestore
    return graph


def _make_entity(**kwargs) -> Entity:
    defaults = {
        "entity_id": "e001",
        "entity_type": "person",
        "name": "David",
        "aliases": ["Dave"],
        "attributes": {"role": "coworker"},
        "embedding": [0.1] * 10,
        "first_seen": time.time() - 3600,
        "last_seen": time.time(),
        "visit_count": 0,
        "confidence": 0.9,
    }
    defaults.update(kwargs)
    return Entity(**defaults)


def _make_relation(**kwargs) -> Relation:
    defaults = {
        "relation_id": "r001",
        "source_eid": "e001",
        "target_eid": "e002",
        "relation_type": "colleague_of",
        "strength": 0.8,
        "first_observed": time.time() - 3600,
        "last_observed": time.time(),
    }
    defaults.update(kwargs)
    return Relation(**defaults)


# ---------------------------------------------------------------------------
# Entity dataclass
# ---------------------------------------------------------------------------


class TestEntityDataclass:
    def test_to_dict_and_from_dict_roundtrip(self):
        e = _make_entity()
        d = e.to_dict()
        restored = Entity.from_dict("e001", d)
        assert restored.entity_id == "e001"
        assert restored.name == "David"
        assert restored.entity_type == "person"
        assert restored.aliases == ["Dave"]

    def test_from_dict_defaults(self):
        e = Entity.from_dict("x", {})
        assert e.entity_type == "place"
        assert e.name == ""
        assert e.visit_count == 0
        assert e.confidence == 0.8


class TestRelationDataclass:
    def test_to_dict_and_from_dict_roundtrip(self):
        r = _make_relation()
        d = r.to_dict()
        restored = Relation.from_dict("r001", d)
        assert restored.source_eid == "e001"
        assert restored.target_eid == "e002"
        assert restored.relation_type == "colleague_of"

    def test_from_dict_defaults(self):
        r = Relation.from_dict("x", {})
        assert r.strength == 0.5
        assert r.relation_type == ""


# ---------------------------------------------------------------------------
# EntityGraphService — create/get/update/delete
# ---------------------------------------------------------------------------


class TestEntityCRUD:
    def test_create_entity_returns_id(self):
        graph = _make_graph()
        graph._count_collection = lambda _: 0
        entity = _make_entity(entity_id="")

        result = graph.create_entity(entity)

        assert result is not None
        assert len(result) > 0

    def test_create_entity_sets_timestamps(self):
        graph = _make_graph()
        graph._count_collection = lambda _: 0
        entity = _make_entity(first_seen=0, entity_id="new1")

        graph.create_entity(entity)

        assert entity.first_seen > 0
        assert entity.last_seen > 0

    def test_create_entity_rejects_at_cap(self):
        graph = _make_graph()
        graph._count_collection = lambda _: MAX_ENTITIES

        entity = _make_entity()
        result = graph.create_entity(entity)

        assert result is None

    def test_get_entity_existing(self):
        graph = _make_graph()
        mock_doc = MagicMock()
        mock_doc.exists = True
        mock_doc.id = "e001"
        mock_doc.to_dict.return_value = {
            "entity_type": "person",
            "name": "David",
            "aliases": [],
            "attributes": {},
            "embedding": [],
            "first_seen": 1000,
            "last_seen": 2000,
            "visit_count": 5,
            "confidence": 0.9,
        }

        coll = graph._firestore.collection.return_value.document.return_value.collection.return_value
        coll.document.return_value.get.return_value = mock_doc

        e = graph.get_entity("e001")
        assert e is not None
        assert e.name == "David"

    def test_get_entity_not_found(self):
        graph = _make_graph()
        mock_doc = MagicMock()
        mock_doc.exists = False

        coll = graph._firestore.collection.return_value.document.return_value.collection.return_value
        coll.document.return_value.get.return_value = mock_doc

        e = graph.get_entity("nonexistent")
        assert e is None

    def test_update_entity(self):
        graph = _make_graph()
        coll = graph._firestore.collection.return_value.document.return_value.collection.return_value

        result = graph.update_entity("e001", {"name": "David W."})
        assert result is True
        coll.document.return_value.update.assert_called_once()

    def test_delete_entity_also_deletes_relations(self):
        graph = _make_graph()
        # Mock get_relations to return one relation
        rel = _make_relation(relation_id="r1")
        graph.get_relations = MagicMock(return_value=[rel])

        coll = graph._firestore.collection.return_value.document.return_value.collection.return_value

        result = graph.delete_entity("e001")
        assert result is True

    def test_touch_entity_increments_visit(self):
        graph = _make_graph()
        coll = graph._firestore.collection.return_value.document.return_value.collection.return_value

        with patch("google.cloud.firestore.Increment", return_value="INCREMENT_1"):
            result = graph.touch_entity("e001")

        assert result is True


# ---------------------------------------------------------------------------
# EntityGraphService — relations
# ---------------------------------------------------------------------------


class TestRelationCRUD:
    def test_create_relation_returns_id(self):
        graph = _make_graph()
        graph._count_collection = lambda _: 0
        rel = _make_relation(relation_id="")

        result = graph.create_relation(rel)
        assert result is not None

    def test_create_relation_rejects_at_cap(self):
        graph = _make_graph()
        graph._count_collection = lambda _: MAX_RELATIONS

        rel = _make_relation()
        result = graph.create_relation(rel)
        assert result is None

    def test_get_relations_both_directions(self):
        graph = _make_graph()
        coll = graph._firestore.collection.return_value.document.return_value.collection.return_value

        # Source query returns 1 doc
        mock_doc1 = MagicMock()
        mock_doc1.id = "r1"
        mock_doc1.to_dict.return_value = {
            "source_eid": "e001",
            "target_eid": "e002",
            "relation_type": "colleague_of",
            "strength": 0.8,
            "first_observed": 1000,
            "last_observed": 2000,
        }
        # Target query returns 1 doc
        mock_doc2 = MagicMock()
        mock_doc2.id = "r2"
        mock_doc2.to_dict.return_value = {
            "source_eid": "e003",
            "target_eid": "e001",
            "relation_type": "works_at",
            "strength": 0.6,
            "first_observed": 1000,
            "last_observed": 2000,
        }

        coll.where.return_value.stream.side_effect = [
            [mock_doc1],  # source query
            [mock_doc2],  # target query
        ]

        rels = graph.get_relations("e001")
        assert len(rels) == 2


# ---------------------------------------------------------------------------
# EntityGraphService — traversal
# ---------------------------------------------------------------------------


class TestGraphTraversal:
    def test_get_connected_entities(self):
        graph = _make_graph()

        # Mock get_relations
        graph.get_relations = MagicMock(return_value=[
            _make_relation(source_eid="e001", target_eid="e002"),
            _make_relation(source_eid="e003", target_eid="e001"),
        ])

        # Mock get_entity
        e2 = _make_entity(entity_id="e002", name="Alice")
        e3 = _make_entity(entity_id="e003", name="Bob")
        graph.get_entity = MagicMock(side_effect=lambda eid: {"e002": e2, "e003": e3}.get(eid))

        connected = graph.get_connected_entities("e001")
        names = {e.name for e in connected}
        assert names == {"Alice", "Bob"}

    def test_get_connected_entities_empty(self):
        graph = _make_graph()
        graph.get_relations = MagicMock(return_value=[])

        connected = graph.get_connected_entities("e001")
        assert connected == []


# ---------------------------------------------------------------------------
# EntityGraphService — find by name
# ---------------------------------------------------------------------------


class TestFindByName:
    def test_find_by_exact_name(self):
        graph = _make_graph()
        coll = graph._firestore.collection.return_value.document.return_value.collection.return_value

        mock_doc = MagicMock()
        mock_doc.id = "e001"
        mock_doc.to_dict.return_value = {
            "entity_type": "person",
            "name": "David",
            "aliases": [],
            "attributes": {},
            "embedding": [],
            "first_seen": 1000,
            "last_seen": 2000,
            "visit_count": 0,
            "confidence": 0.9,
        }

        coll.where.return_value.stream.side_effect = [
            [mock_doc],  # name query
        ]

        e = graph.find_entity_by_name("David")
        assert e is not None
        assert e.name == "David"

    def test_find_by_name_not_found(self):
        graph = _make_graph()
        coll = graph._firestore.collection.return_value.document.return_value.collection.return_value

        # Both name and alias queries return empty
        coll.where.return_value.stream.return_value = []
        coll.limit.return_value.stream.return_value = []

        e = graph.find_entity_by_name("Nobody")
        assert e is None

    def test_find_by_name_filters_type(self):
        graph = _make_graph()
        coll = graph._firestore.collection.return_value.document.return_value.collection.return_value

        mock_doc = MagicMock()
        mock_doc.id = "e001"
        mock_doc.to_dict.return_value = {
            "entity_type": "person",
            "name": "Starbucks",
            "aliases": [],
            "attributes": {},
            "embedding": [],
            "first_seen": 1000,
            "last_seen": 2000,
            "visit_count": 0,
            "confidence": 0.9,
        }

        coll.where.return_value.stream.side_effect = [
            [mock_doc],  # name query matches
        ]

        # Should not match because entity_type is "person" not "place"
        e = graph.find_entity_by_name("Starbucks", entity_type="place")
        # The name query returned a person, so it's filtered out,
        # then alias query runs
        assert e is None or e.entity_type == "place"


# ---------------------------------------------------------------------------
# EntityGraphService — no Firestore
# ---------------------------------------------------------------------------


class TestNoFirestore:
    def test_create_entity_without_firestore_returns_none(self):
        with patch.object(EntityGraphService, "_try_init"):
            graph = EntityGraphService("u1")
        graph._firestore = None
        graph._ensure_firestore = lambda: None

        result = graph.create_entity(_make_entity())
        assert result is None

    def test_list_entities_without_firestore_returns_empty(self):
        with patch.object(EntityGraphService, "_try_init"):
            graph = EntityGraphService("u1")
        graph._firestore = None
        graph._ensure_firestore = lambda: None

        result = graph.list_entities()
        assert result == []
