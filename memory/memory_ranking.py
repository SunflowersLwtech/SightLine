"""Memory retrieval ranking with multi-dimensional scoring.

Dimensions (5D):
- relevance: Semantic similarity to current context (0.30)
- recency: How recent the memory is, layer-aware half-lives (0.20)
- importance: User-assigned or system-inferred importance (0.15)
- location: Proximity to current location via entity refs (0.20)
- entity_overlap: Shared entities with current scene (0.15)

Backward compatible: callers without location/entity params get the same
results as before (location + entity scores default to neutral 0.5).
"""

import time

# Dimension weights — must sum to 1.0
RELEVANCE_WEIGHT = 0.30
RECENCY_WEIGHT = 0.20
IMPORTANCE_WEIGHT = 0.15
LOCATION_WEIGHT = 0.20
ENTITY_OVERLAP_WEIGHT = 0.15


def rank_memories(
    memories: list[dict],
    query_context: str = "",
    max_results: int = 3,
    current_location=None,
    visible_entity_ids: list[str] | None = None,
) -> list[dict]:
    """Rank memories by composite score across five dimensions.

    Each memory dict should have:
      - content: str
      - timestamp: float (unix epoch)
      - importance: float (0-1)
      - relevance_score: float (0-1, from semantic search)
      - half_life_days: float (optional, default 1 = legacy 24hr decay)
      - entity_refs: list[str] (optional, entity IDs referenced)
      - location_ref: str (optional, place entity ID)

    Args:
        memories: List of memory dicts from vector search or cache.
        query_context: Current conversation context (unused in scoring,
            kept for API compatibility).
        max_results: Maximum memories to return.
        current_location: Optional LocationContext with matched_entity_id.
        visible_entity_ids: Entity IDs currently relevant to the scene.

    Returns:
        Sorted list (highest score first), limited to max_results.
    """
    scored = []
    now = time.time()

    # Pre-compute location entity ID from LocationContext if available
    location_entity_id = None
    if current_location is not None:
        location_entity_id = getattr(current_location, "matched_entity_id", None)

    visible_set = set(visible_entity_ids) if visible_entity_ids else set()

    for mem in memories:
        relevance = float(mem.get("relevance_score", 0.5))
        importance = float(mem.get("importance", 0.5))

        # Recency: exponential decay with layer-aware half-life
        ts = float(mem.get("timestamp", now))
        age_hours = (now - ts) / 3600
        half_life_days = float(mem.get("half_life_days", 1))  # 1 day = legacy
        recency = 2 ** (-age_hours / (half_life_days * 24))

        # Location score: 1.0 if memory references current location, else 0.5
        mem_location_ref = mem.get("location_ref")
        if location_entity_id and mem_location_ref:
            location_score = 1.0 if mem_location_ref == location_entity_id else 0.3
        else:
            location_score = 0.5  # neutral when no location context

        # Entity overlap score: fraction of memory's entity_refs in visible set
        mem_entity_refs = mem.get("entity_refs", [])
        if visible_set and mem_entity_refs:
            overlap = len(set(mem_entity_refs) & visible_set)
            entity_score = min(1.0, overlap / max(1, len(mem_entity_refs)))
        elif visible_set and not mem_entity_refs:
            entity_score = 0.3  # penalise memories with no entity refs when scene has entities
        else:
            entity_score = 0.5  # neutral when no entity context

        composite = (
            RELEVANCE_WEIGHT * relevance
            + RECENCY_WEIGHT * recency
            + IMPORTANCE_WEIGHT * importance
            + LOCATION_WEIGHT * location_score
            + ENTITY_OVERLAP_WEIGHT * entity_score
        )

        scored.append({
            **mem,
            "_composite_score": composite,
            "_recency": recency,
            "_location_score": location_score,
            "_entity_score": entity_score,
        })

    scored.sort(key=lambda m: m["_composite_score"], reverse=True)
    return scored[:max_results]


def score_memories(memories: list[dict], context: str = "") -> list[dict]:
    """Alias for rank_memories (backward compatibility)."""
    return rank_memories(memories, query_context=context)
