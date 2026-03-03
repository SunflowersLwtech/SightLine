"""Scene matcher — embedding-based scene warm-start.

Matches the current scene (location + entities + time) against stored
scene patterns to provide a warm-start LOD and proactive hints.

Firestore path: user_profiles/{uid}/scene_patterns/{pid}
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "sightline-hackathon")

_COSINE_THRESHOLD = 0.8  # minimum similarity for a match
_MAX_PATTERNS = 50


@dataclass
class ScenePattern:
    """A stored scene pattern with learned preferences."""

    pattern_id: str = ""
    scene_embedding: list[float] = field(default_factory=list)
    preferred_lod: int = 2
    adjustment_count: int = 0  # user overrides → signal
    satisfaction_score: float = 0.5  # inferred from user behavior
    location_name: str = ""
    time_context: str = ""
    last_seen: float = 0.0

    def to_dict(self) -> dict:
        return {
            "scene_embedding": self.scene_embedding,
            "preferred_lod": self.preferred_lod,
            "adjustment_count": self.adjustment_count,
            "satisfaction_score": self.satisfaction_score,
            "location_name": self.location_name,
            "time_context": self.time_context,
            "last_seen": self.last_seen,
        }

    @classmethod
    def from_dict(cls, pid: str, data: dict) -> "ScenePattern":
        return cls(
            pattern_id=pid,
            scene_embedding=data.get("scene_embedding", []),
            preferred_lod=int(data.get("preferred_lod", 2)),
            adjustment_count=int(data.get("adjustment_count", 0)),
            satisfaction_score=float(data.get("satisfaction_score", 0.5)),
            location_name=data.get("location_name", ""),
            time_context=data.get("time_context", ""),
            last_seen=float(data.get("last_seen", 0)),
        )


@dataclass
class SceneMatch:
    """Result of a scene matching attempt."""

    matched: bool = False
    pattern: Optional[ScenePattern] = None
    similarity: float = 0.0
    suggested_lod: int = 2


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    va = np.array(a, dtype=np.float64)
    vb = np.array(b, dtype=np.float64)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


class SceneMatcher:
    """Matches current scene against stored patterns for warm-start."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self._firestore = None
        self._try_init()

    def _try_init(self):
        try:
            from google.cloud import firestore
            self._firestore = firestore.Client(project=PROJECT_ID)
        except Exception as e:
            logger.warning("Firestore init failed for SceneMatcher user=%s: %s", self.user_id, e)

    def _scene_patterns_coll(self):
        return (
            self._firestore.collection("user_profiles")
            .document(self.user_id)
            .collection("scene_patterns")
        )

    def match(self, current_scene_embedding: list[float]) -> SceneMatch:
        """Match current scene against stored patterns.

        Args:
            current_scene_embedding: 2048-D embedding of the current scene.

        Returns:
            SceneMatch with matched=True if similarity > threshold.
        """
        if not self._firestore or not current_scene_embedding:
            return SceneMatch()

        try:
            patterns = self._load_patterns()
            if not patterns:
                return SceneMatch()

            best_match: Optional[ScenePattern] = None
            best_sim = 0.0

            for pattern in patterns:
                if not pattern.scene_embedding:
                    continue
                sim = _cosine_similarity(current_scene_embedding, pattern.scene_embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_match = pattern

            if best_match and best_sim >= _COSINE_THRESHOLD:
                return SceneMatch(
                    matched=True,
                    pattern=best_match,
                    similarity=best_sim,
                    suggested_lod=best_match.preferred_lod,
                )

            return SceneMatch(matched=False, similarity=best_sim)

        except Exception:
            logger.debug("Scene matching failed", exc_info=True)
            return SceneMatch()

    def store_pattern(self, pattern: ScenePattern) -> Optional[str]:
        """Store a new scene pattern to Firestore."""
        if not self._firestore:
            return None
        try:
            import uuid
            if not pattern.pattern_id:
                pattern.pattern_id = uuid.uuid4().hex[:16]
            pattern.last_seen = time.time()

            data = pattern.to_dict()
            # Store embedding as Vector if available
            if pattern.scene_embedding:
                from google.cloud.firestore_v1.vector import Vector
                data["scene_embedding"] = Vector(pattern.scene_embedding)

            self._scene_patterns_coll().document(pattern.pattern_id).set(data)
            return pattern.pattern_id
        except Exception:
            logger.error("Failed to store scene pattern", exc_info=True)
            return None

    def update_pattern(self, pattern_id: str, updates: dict) -> bool:
        """Update an existing scene pattern."""
        if not self._firestore:
            return False
        try:
            self._scene_patterns_coll().document(pattern_id).update(updates)
            return True
        except Exception:
            logger.error("Failed to update scene pattern %s", pattern_id, exc_info=True)
            return False

    def _load_patterns(self) -> list[ScenePattern]:
        """Load all scene patterns for this user."""
        try:
            coll = self._scene_patterns_coll()
            query = coll.order_by("last_seen", direction="DESCENDING").limit(_MAX_PATTERNS)
            return [ScenePattern.from_dict(doc.id, doc.to_dict()) for doc in query.stream()]
        except Exception:
            logger.debug("Failed to load scene patterns", exc_info=True)
            return []
