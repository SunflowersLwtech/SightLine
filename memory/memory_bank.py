"""SightLine long-term memory service.

Wraps Firestore-based storage for persistent cross-session memory.
Uses Gemini embeddings for semantic retrieval via Firestore vector search.

Firestore collection: user_profiles/{user_id}/memories/{memory_id}
Embedding model: gemini-embedding-001 (truncated to 2048-D)
"""

import logging
import os
import re
import time
import uuid
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "sightline-hackathon")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIM = 2048


def _compute_embedding(text: str) -> list[float]:
    """Compute a 2048-D embedding for the given text using google-genai.

    Returns a zero vector on failure so callers can proceed gracefully.
    """
    normalized = (text or "").strip()
    if not normalized:
        logger.debug("Embedding input text is empty; using zero vector")
        return [0.0] * EMBEDDING_DIM

    try:
        from google import genai

        api_key = os.environ.get("_GOOGLE_AI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
        client = genai.Client(api_key=api_key, vertexai=False)
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=normalized,
            config={"output_dimensionality": EMBEDDING_DIM},
        )
        return result.embeddings[0].values
    except Exception:
        logger.warning("Embedding computation failed; using zero vector", exc_info=True)
        return [0.0] * EMBEDDING_DIM


class MemoryBankService:
    """Firestore-backed long-term memory for a single user.

    Provides persistent memory storage across sessions with semantic
    retrieval via embeddings.
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self._firestore = None
        self._memories_cache: list[dict] = []
        self._try_init()

    def _try_init(self):
        """Single non-blocking attempt to initialize Firestore backend."""
        try:
            from google.cloud import firestore

            self._firestore = firestore.Client(project=PROJECT_ID)
            logger.info("MemoryBankService initialized for user %s", self.user_id)
        except Exception as e:
            logger.warning(
                "Firestore init failed for user %s: %s (memories will be EPHEMERAL until retry)",
                self.user_id, e,
            )

    def _ensure_firestore(self):
        """Lazy re-init: retry Firestore connection on first actual use if init failed."""
        if self._firestore is None:
            self._try_init()
        return self._firestore

    def _memories_collection(self):
        """Return the memories subcollection reference for this user."""
        return (
            self._firestore.collection("user_profiles")
            .document(self.user_id)
            .collection("memories")
        )

    # Layer half-lives (days): episodic decays fast, procedural is near-permanent
    _LAYER_HALF_LIVES: dict[str, float] = {
        "episodic": 7,
        "semantic": 90,
        "procedural": 999,
    }

    def store_memory(
        self,
        content: str,
        category: str = "general",
        importance: float = 0.5,
        memory_layer: str = "semantic",
        entity_refs: list[str] | None = None,
        location_ref: str | None = None,
    ) -> Optional[str]:
        """Store a memory with metadata and embedding.

        Args:
            content: The memory text content.
            category: Classification (e.g. "general", "place", "person").
            importance: Importance score from 0 to 1.
            memory_layer: One of "episodic", "semantic", "procedural".
            entity_refs: List of entity IDs this memory references.
            location_ref: Entity ID of a place this memory is associated with.

        Returns:
            The Firestore document ID, or None on failure.
        """
        half_life = self._LAYER_HALF_LIVES.get(memory_layer, 90)
        now = time.time()

        if not self._ensure_firestore():
            # Ephemeral fallback: store in cache only
            memory_id = uuid.uuid4().hex
            self._memories_cache.append({
                "memory_id": memory_id,
                "content": content,
                "category": category,
                "importance": importance,
                "timestamp": now,
                "memory_layer": memory_layer,
                "entity_refs": entity_refs or [],
                "location_ref": location_ref,
                "half_life_days": half_life,
                "access_count": 0,
                "last_accessed": now,
            })
            return memory_id

        try:
            from google.cloud.firestore_v1.vector import Vector

            embedding = _compute_embedding(content)
            doc_data = {
                "content": content,
                "category": category,
                "importance": importance,
                "timestamp": now,
                "embedding": Vector(embedding),
                "memory_layer": memory_layer,
                "entity_refs": entity_refs or [],
                "location_ref": location_ref,
                "half_life_days": half_life,
                "access_count": 0,
                "last_accessed": now,
            }

            doc_ref = self._memories_collection().document()
            doc_ref.set(doc_data)
            logger.info("Stored memory %s for user %s", doc_ref.id, self.user_id)
            return doc_ref.id
        except Exception:
            logger.error("Failed to store memory for user %s", self.user_id, exc_info=True)
            return None

    def retrieve_memories(
        self,
        context: str,
        top_k: int = 3,
        location_context=None,
        visible_entity_ids: list[str] | None = None,
    ) -> list[dict]:
        """Retrieve relevant memories using vector search with text fallback.

        First attempts Firestore vector nearest-neighbor search. If that
        fails (e.g. index not ready), falls back to fetching recent
        memories and doing client-side text matching.

        Results are re-ranked using multi-dimensional scoring from
        memory_ranking.

        Args:
            context: Current conversation context or query string.
            top_k: Maximum number of memories to return.
            location_context: Optional LocationContext for location-aware ranking.
            visible_entity_ids: Entity IDs currently relevant (scene/conversation).

        Returns:
            List of memory dicts with content, category, importance,
            timestamp, relevance_score, and memory_id.
        """
        from memory.memory_ranking import rank_memories

        rank_kwargs = {
            "query_context": context,
            "max_results": top_k,
            "current_location": location_context,
            "visible_entity_ids": visible_entity_ids,
        }

        if not self._ensure_firestore():
            results = self._retrieve_from_cache(context, top_k * 2)
            return rank_memories(results, **rank_kwargs)

        # Try Firestore vector search first
        try:
            results = self._vector_search(context, top_k * 2)
            return rank_memories(results, **rank_kwargs)
        except Exception:
            logger.debug("Vector search unavailable, falling back to text match", exc_info=True)

        # Fallback: fetch recent memories and rank by text overlap
        results = self._text_fallback(context, top_k * 2)
        return rank_memories(results, **rank_kwargs)

    def _vector_search(self, context: str, top_k: int) -> list[dict]:
        """Firestore find_nearest vector search."""
        from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
        from google.cloud.firestore_v1.vector import Vector

        query_embedding = _compute_embedding(context)
        coll = self._memories_collection()

        vector_query = coll.find_nearest(
            vector_field="embedding",
            query_vector=Vector(query_embedding),
            distance_measure=DistanceMeasure.COSINE,
            limit=top_k,
            distance_result_field="vector_distance",
        )

        results = []
        for doc in vector_query.stream():
            data = doc.to_dict()
            # Firestore COSINE distance: 0 = identical, 2 = opposite.
            # Convert to similarity: 1 - (distance / 2) → range [0, 1].
            raw_distance = float(data.pop("vector_distance", 0.4))
            relevance = max(0.0, min(1.0, 1.0 - raw_distance / 2.0))
            results.append({
                "memory_id": doc.id,
                "content": data.get("content", ""),
                "category": data.get("category", "general"),
                "importance": float(data.get("importance", 0.5)),
                "timestamp": float(data.get("timestamp", 0)),
                "relevance_score": relevance,
            })

        return results

    def _text_fallback(self, context: str, top_k: int) -> list[dict]:
        """Fetch recent memories and score by keyword overlap."""
        coll = self._memories_collection()
        query = coll.order_by("timestamp", direction="DESCENDING").limit(20)

        results = []
        context_words = set(context.lower().split())

        for doc in query.stream():
            data = doc.to_dict()
            content = data.get("content", "")
            content_words = set(content.lower().split())

            # Simple Jaccard-like relevance score
            if context_words and content_words:
                overlap = len(context_words & content_words)
                union = len(context_words | content_words)
                relevance = overlap / union if union > 0 else 0.0
            else:
                relevance = 0.1

            results.append({
                "memory_id": doc.id,
                "content": content,
                "category": data.get("category", "general"),
                "importance": float(data.get("importance", 0.5)),
                "timestamp": float(data.get("timestamp", 0)),
                "relevance_score": relevance,
            })

        # Sort by relevance and return top_k
        results.sort(key=lambda m: m["relevance_score"], reverse=True)
        return results[:top_k]

    def _retrieve_from_cache(self, context: str, top_k: int) -> list[dict]:
        """Retrieve from in-memory cache (ephemeral fallback)."""
        context_words = set(context.lower().split())
        scored = []

        for mem in self._memories_cache:
            content_words = set(mem["content"].lower().split())
            if context_words and content_words:
                overlap = len(context_words & content_words)
                union = len(context_words | content_words)
                relevance = overlap / union if union > 0 else 0.0
            else:
                relevance = 0.1

            scored.append({**mem, "relevance_score": relevance})

        scored.sort(key=lambda m: m["relevance_score"], reverse=True)
        return scored[:top_k]

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory by ID.

        Args:
            memory_id: The Firestore document ID.

        Returns:
            True if the document existed and was deleted, False otherwise.
        """
        if not self._ensure_firestore():
            before = len(self._memories_cache)
            self._memories_cache = [
                m for m in self._memories_cache if m.get("memory_id") != memory_id
            ]
            return len(self._memories_cache) < before

        try:
            doc_ref = self._memories_collection().document(memory_id)
            doc = doc_ref.get()
            if not doc.exists:
                return False
            doc_ref.delete()
            logger.info("Deleted memory %s for user %s", memory_id, self.user_id)
            return True
        except Exception:
            logger.error("Failed to delete memory %s", memory_id, exc_info=True)
            return False

    def delete_recent_memories(self, minutes: int = 30) -> int:
        """Delete memories created within the last N minutes.

        Args:
            minutes: How far back to delete.

        Returns:
            Number of memories deleted.
        """
        cutoff = time.time() - (minutes * 60)

        if not self._ensure_firestore():
            before = len(self._memories_cache)
            self._memories_cache = [
                m for m in self._memories_cache
                if m.get("timestamp", 0) < cutoff
            ]
            return before - len(self._memories_cache)

        try:
            coll = self._memories_collection()
            query = coll.where("timestamp", ">=", cutoff)
            count = 0
            for doc in query.stream():
                doc.reference.delete()
                count += 1
            logger.info(
                "Deleted %d recent memories (last %d min) for user %s",
                count, minutes, self.user_id,
            )
            return count
        except Exception:
            logger.error("Failed to delete recent memories", exc_info=True)
            return 0


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

_bank_instances: dict[str, MemoryBankService] = {}
_bank_last_accessed: dict[str, float] = {}


def _get_bank(user_id: str) -> MemoryBankService:
    """Get or create a MemoryBankService for the given user.

    Re-attempts Firestore init if a cached instance has no connection.
    """
    _bank_last_accessed[user_id] = time.time()
    if user_id in _bank_instances:
        bank = _bank_instances[user_id]
        if bank._firestore is None:
            bank._try_init()
        return bank
    _bank_instances[user_id] = MemoryBankService(user_id)
    return _bank_instances[user_id]


def evict_stale_banks(max_age_sec: int = 3600) -> int:
    """Remove cached MemoryBankService instances not accessed within max_age_sec.

    Returns the number of evicted instances.
    """
    now = time.time()
    stale = [
        uid for uid, ts in _bank_last_accessed.items()
        if now - ts > max_age_sec
    ]
    for uid in stale:
        _bank_instances.pop(uid, None)
        _bank_last_accessed.pop(uid, None)
    if stale:
        logger.info("Evicted %d stale MemoryBankService instances", len(stale))
    return len(stale)


def _sanitize_memory_content(text: str) -> str:
    """Remove prompt-injection patterns from memory content."""
    text = re.sub(r'(?i)ignore\s+(all\s+)?previous\s+instructions?', '[REDACTED]', text)
    text = re.sub(r'(?i)you\s+are\s+now\s+', 'the user mentioned ', text)
    text = re.sub(r'(?i)system\s*:\s*', '', text)
    return text.strip()


def load_relevant_memories(user_id: str, context: str, top_k: int = 3) -> list[str]:
    """Load top-K relevant memories for prompt injection.

    Returns a list of sanitized memory content strings, ranked by relevance.
    """
    bank = _get_bank(user_id)
    results = bank.retrieve_memories(context, top_k=top_k)
    return [_sanitize_memory_content(m["content"]) for m in results]


