"""Entity graph — Firestore CRUD for entities and relations.

Firestore paths:
    user_profiles/{uid}/entities/{eid}
    user_profiles/{uid}/relations/{rid}

Design constraints:
    - 1-hop traversal only (Firestore is not a graph DB)
    - Cap: 200 entities / 500 relations per user
    - Embeddings: 2048-D (same model as memory system)
"""

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "sightline-hackathon")

MAX_ENTITIES = 200
MAX_RELATIONS = 500


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Entity:
    """A named thing in the user's world (person, place, org, event)."""

    entity_id: str = ""
    entity_type: str = "place"  # person | place | organization | event
    name: str = ""
    aliases: list[str] = field(default_factory=list)
    attributes: dict = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    visit_count: int = 0
    confidence: float = 0.8

    def to_dict(self) -> dict:
        return {
            "entity_type": self.entity_type,
            "name": self.name,
            "aliases": self.aliases,
            "attributes": self.attributes,
            "embedding": self.embedding,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "visit_count": self.visit_count,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, eid: str, data: dict) -> "Entity":
        return cls(
            entity_id=eid,
            entity_type=data.get("entity_type", "place"),
            name=data.get("name", ""),
            aliases=data.get("aliases", []),
            attributes=data.get("attributes", {}),
            embedding=data.get("embedding", []),
            first_seen=float(data.get("first_seen", 0)),
            last_seen=float(data.get("last_seen", 0)),
            visit_count=int(data.get("visit_count", 0)),
            confidence=float(data.get("confidence", 0.8)),
        )


@dataclass
class Relation:
    """A directed edge between two entities."""

    relation_id: str = ""
    source_eid: str = ""
    target_eid: str = ""
    relation_type: str = ""  # works_at, colleague_of, visits_regularly, ...
    strength: float = 0.5  # 0-1, decays / reinforced
    first_observed: float = 0.0
    last_observed: float = 0.0

    def to_dict(self) -> dict:
        return {
            "source_eid": self.source_eid,
            "target_eid": self.target_eid,
            "relation_type": self.relation_type,
            "strength": self.strength,
            "first_observed": self.first_observed,
            "last_observed": self.last_observed,
        }

    @classmethod
    def from_dict(cls, rid: str, data: dict) -> "Relation":
        return cls(
            relation_id=rid,
            source_eid=data.get("source_eid", ""),
            target_eid=data.get("target_eid", ""),
            relation_type=data.get("relation_type", ""),
            strength=float(data.get("strength", 0.5)),
            first_observed=float(data.get("first_observed", 0)),
            last_observed=float(data.get("last_observed", 0)),
        )


# ---------------------------------------------------------------------------
# Graph service
# ---------------------------------------------------------------------------


class EntityGraphService:
    """CRUD + 1-hop traversal for the per-user entity graph."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self._firestore = None
        self._try_init()

    def _try_init(self):
        try:
            from google.cloud import firestore
            self._firestore = firestore.Client(project=PROJECT_ID)
        except Exception as e:
            logger.warning("Firestore init failed for EntityGraph user=%s: %s", self.user_id, e)

    def _ensure_firestore(self):
        if self._firestore is None:
            self._try_init()
        return self._firestore

    def _user_doc(self):
        return self._firestore.collection("user_profiles").document(self.user_id)

    def _entities_coll(self):
        return self._user_doc().collection("entities")

    def _relations_coll(self):
        return self._user_doc().collection("relations")

    # -- Entity CRUD --------------------------------------------------------

    def create_entity(self, entity: Entity) -> Optional[str]:
        """Create or overwrite an entity. Returns entity_id or None."""
        if not self._ensure_firestore():
            return None
        try:
            count = self._count_collection(self._entities_coll())
            if count >= MAX_ENTITIES:
                logger.warning("Entity cap (%d) reached for user %s", MAX_ENTITIES, self.user_id)
                return None

            now = time.time()
            if not entity.entity_id:
                entity.entity_id = uuid.uuid4().hex[:16]
            if entity.first_seen == 0:
                entity.first_seen = now
            entity.last_seen = now

            data = entity.to_dict()
            # Store embedding as Firestore Vector if non-empty
            if entity.embedding:
                from google.cloud.firestore_v1.vector import Vector
                data["embedding"] = Vector(entity.embedding)

            self._entities_coll().document(entity.entity_id).set(data)
            logger.info("Created entity %s (%s) for user %s", entity.entity_id, entity.name, self.user_id)
            return entity.entity_id
        except Exception:
            logger.error("Failed to create entity for user %s", self.user_id, exc_info=True)
            return None

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Fetch a single entity by ID."""
        if not self._ensure_firestore():
            return None
        try:
            doc = self._entities_coll().document(entity_id).get()
            if not doc.exists:
                return None
            return Entity.from_dict(doc.id, doc.to_dict())
        except Exception:
            logger.error("Failed to get entity %s", entity_id, exc_info=True)
            return None

    def find_entity_by_name(self, name: str, entity_type: str | None = None) -> Optional[Entity]:
        """Find an entity by exact name or alias match."""
        if not self._ensure_firestore():
            return None
        try:
            name_lower = name.lower()
            coll = self._entities_coll()
            # Check name field first
            for doc in coll.where("name", "==", name).stream():
                e = Entity.from_dict(doc.id, doc.to_dict())
                if entity_type and e.entity_type != entity_type:
                    continue
                return e
            # Check aliases via array_contains
            for doc in coll.where("aliases", "array_contains", name).stream():
                e = Entity.from_dict(doc.id, doc.to_dict())
                if entity_type and e.entity_type != entity_type:
                    continue
                return e
            # Case-insensitive fallback: scan recent entities
            for doc in coll.limit(MAX_ENTITIES).stream():
                data = doc.to_dict()
                e = Entity.from_dict(doc.id, data)
                if entity_type and e.entity_type != entity_type:
                    continue
                all_names = [e.name.lower()] + [a.lower() for a in e.aliases]
                if name_lower in all_names:
                    return e
            return None
        except Exception:
            logger.error("Failed to find entity by name '%s'", name, exc_info=True)
            return None

    def update_entity(self, entity_id: str, updates: dict) -> bool:
        """Merge-update fields on an existing entity."""
        if not self._ensure_firestore():
            return False
        try:
            updates["last_seen"] = time.time()
            self._entities_coll().document(entity_id).update(updates)
            return True
        except Exception:
            logger.error("Failed to update entity %s", entity_id, exc_info=True)
            return False

    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and all relations referencing it."""
        if not self._ensure_firestore():
            return False
        try:
            # Delete relations first
            for rel in self.get_relations(entity_id):
                self._relations_coll().document(rel.relation_id).delete()
            self._entities_coll().document(entity_id).delete()
            logger.info("Deleted entity %s for user %s", entity_id, self.user_id)
            return True
        except Exception:
            logger.error("Failed to delete entity %s", entity_id, exc_info=True)
            return False

    def list_entities(self, entity_type: str | None = None, limit: int = 50) -> list[Entity]:
        """List entities, optionally filtered by type."""
        if not self._ensure_firestore():
            return []
        try:
            coll = self._entities_coll()
            if entity_type:
                query = coll.where("entity_type", "==", entity_type).limit(limit)
            else:
                query = coll.order_by("last_seen", direction="DESCENDING").limit(limit)
            return [Entity.from_dict(doc.id, doc.to_dict()) for doc in query.stream()]
        except Exception:
            logger.error("Failed to list entities for user %s", self.user_id, exc_info=True)
            return []

    def touch_entity(self, entity_id: str) -> bool:
        """Bump last_seen and increment visit_count (for places)."""
        if not self._ensure_firestore():
            return False
        try:
            from google.cloud import firestore
            self._entities_coll().document(entity_id).update({
                "last_seen": time.time(),
                "visit_count": firestore.Increment(1),
            })
            return True
        except Exception:
            logger.error("Failed to touch entity %s", entity_id, exc_info=True)
            return False

    # -- Relation CRUD ------------------------------------------------------

    def create_relation(self, relation: Relation) -> Optional[str]:
        """Create a relation between two entities."""
        if not self._ensure_firestore():
            return None
        try:
            count = self._count_collection(self._relations_coll())
            if count >= MAX_RELATIONS:
                logger.warning("Relation cap (%d) reached for user %s", MAX_RELATIONS, self.user_id)
                return None

            now = time.time()
            if not relation.relation_id:
                relation.relation_id = uuid.uuid4().hex[:16]
            if relation.first_observed == 0:
                relation.first_observed = now
            relation.last_observed = now

            self._relations_coll().document(relation.relation_id).set(relation.to_dict())
            logger.info("Created relation %s for user %s", relation.relation_id, self.user_id)
            return relation.relation_id
        except Exception:
            logger.error("Failed to create relation for user %s", self.user_id, exc_info=True)
            return None

    def get_relations(self, entity_id: str) -> list[Relation]:
        """Get all relations where entity_id is source OR target (1-hop)."""
        if not self._ensure_firestore():
            return []
        try:
            coll = self._relations_coll()
            results = []
            # Source relations
            for doc in coll.where("source_eid", "==", entity_id).stream():
                results.append(Relation.from_dict(doc.id, doc.to_dict()))
            # Target relations
            for doc in coll.where("target_eid", "==", entity_id).stream():
                results.append(Relation.from_dict(doc.id, doc.to_dict()))
            return results
        except Exception:
            logger.error("Failed to get relations for entity %s", entity_id, exc_info=True)
            return []

    def delete_relation(self, relation_id: str) -> bool:
        """Delete a specific relation."""
        if not self._ensure_firestore():
            return False
        try:
            self._relations_coll().document(relation_id).delete()
            return True
        except Exception:
            logger.error("Failed to delete relation %s", relation_id, exc_info=True)
            return False

    def get_connected_entities(self, entity_id: str) -> list[Entity]:
        """1-hop traversal: get all entities connected to the given entity."""
        relations = self.get_relations(entity_id)
        connected_ids = set()
        for rel in relations:
            if rel.source_eid == entity_id:
                connected_ids.add(rel.target_eid)
            else:
                connected_ids.add(rel.source_eid)

        entities = []
        for eid in connected_ids:
            e = self.get_entity(eid)
            if e:
                entities.append(e)
        return entities

    # -- Helpers ------------------------------------------------------------

    def _count_collection(self, coll_ref) -> int:
        """Count documents in a collection (bounded scan)."""
        try:
            count = 0
            for _ in coll_ref.limit(max(MAX_ENTITIES, MAX_RELATIONS) + 1).stream():
                count += 1
            return count
        except Exception:
            return 0
