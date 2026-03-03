"""Memory auto-extraction at session end.

Analyzes the session transcript using Gemini Flash to extract key memories
(preferences, experiences, people, locations, routines) and stores them
in the user's memory bank with conflict detection.
"""

import json
import logging
import os
from typing import Optional

import numpy as np

from memory.memory_bank import MemoryBankService, _compute_embedding, EMBEDDING_DIM
from memory.memory_budget import MemoryBudgetTracker

logger = logging.getLogger(__name__)

EXTRACTION_MODEL = "gemini-3-flash-preview"

EXTRACTION_PROMPT = """\
You are a memory extraction system for SightLine, an AI assistant for visually impaired users.

Analyze the following conversation transcript and extract key facts the user would want \
remembered across sessions. Focus on:

1. **preference** — User preferences (e.g., "prefers clock-position directions", "likes detailed descriptions")
2. **experience** — Notable experiences or events (e.g., "visited Central Park on 2024-03-15")
3. **person** — People mentioned by name and their relationship (e.g., "David is the user's coworker")
4. **location** — Important locations (e.g., "user's office is at 123 Main St")
5. **routine** — Regular habits or routines (e.g., "takes the 8am bus to work")

For each extracted memory, provide:
- "content": A concise factual statement (1-2 sentences max)
- "category": One of [preference, experience, person, location, routine]
- "importance": Float 0-1 (how important is this for future sessions?)
- "confidence": Float 0-1 (how confident are you this is a real, extractable fact?)
- "memory_layer": One of [episodic, semantic, procedural] — episodic for one-time events/experiences, \
semantic for facts about people/places/things, procedural for habits/preferences/routines
- "entity_names": Array of entity names (people, places, organizations) referenced in this memory. \
Empty array if none.

Return a JSON array of objects. If no meaningful memories can be extracted, return an empty array [].

Only extract facts that are clearly stated or strongly implied. Do NOT speculate.

Transcript:
{transcript}
"""


_VALID_CATEGORIES = {"preference", "experience", "person", "location", "routine", "general"}

_VALID_LAYERS = {"episodic", "semantic", "procedural"}

# Default layer mapping: category → memory_layer
_CATEGORY_TO_LAYER: dict[str, str] = {
    "preference": "procedural",
    "experience": "episodic",
    "person": "semantic",
    "location": "semantic",
    "routine": "procedural",
    "general": "semantic",
}


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    va = np.array(a, dtype=np.float64)
    vb = np.array(b, dtype=np.float64)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


class MemoryExtractor:
    """Extracts and stores key memories from a session transcript."""

    def __init__(self, similarity_threshold: float = 0.85, confidence_threshold: float = 0.75):
        self.similarity_threshold = similarity_threshold
        self.confidence_threshold = confidence_threshold

    def extract_and_store(
        self,
        user_id: str,
        session_id: str,
        transcript_history: list[dict],
        memory_bank: MemoryBankService,
        budget: MemoryBudgetTracker,
    ) -> int:
        """Extract memories from a session transcript and store them.

        Args:
            user_id: The user identifier.
            session_id: The session identifier.
            transcript_history: List of {"role": "user"|"agent", "text": "..."} dicts.
            memory_bank: The user's MemoryBankService instance.
            budget: The session's MemoryBudgetTracker.

        Returns:
            Number of memories stored (new + updated).
        """
        if not transcript_history:
            logger.info("No transcript history for session %s, skipping extraction", session_id)
            return 0

        transcript_text = self._format_transcript(transcript_history)
        if len(transcript_text.strip()) < 20:
            logger.info("Transcript too short for session %s, skipping extraction", session_id)
            return 0

        # Extract candidate memories via Gemini Flash
        candidates = self._call_extraction_model(transcript_text)
        if not candidates:
            logger.info("No memories extracted for session %s", session_id)
            return 0

        # Fetch existing memories for conflict detection
        existing_memories = memory_bank.retrieve_memories(context="", top_k=50)
        existing_embeddings = self._precompute_existing_embeddings(existing_memories)

        stored_count = 0
        for candidate in candidates:
            confidence = float(candidate.get("confidence", 0))
            category = str(candidate.get("category", "general")).strip().lower()
            required_confidence = self.confidence_threshold
            if category in ("person", "stress_trigger"):
                required_confidence = 0.9
            if confidence < required_confidence:
                logger.debug(
                    "Skipping (conf=%.2f < %.2f): %s",
                    confidence, required_confidence, candidate.get("content", "")[:80],
                )
                continue

            if budget.exhausted:
                logger.info("Memory budget exhausted, stopping extraction for session %s", session_id)
                break

            if not budget.try_write():
                logger.info("Memory budget denied write, stopping extraction for session %s", session_id)
                break

            content = candidate["content"]
            category = candidate.get("category", "general")
            importance = float(candidate.get("importance", 0.5))
            memory_layer = candidate.get("memory_layer", "semantic")
            entity_names = candidate.get("entity_names", [])

            # Resolve entity names to IDs if entity graph is available
            entity_refs = self._resolve_entity_names(user_id, entity_names)

            # Conflict detection: check if a similar memory already exists
            duplicate = self._find_duplicate(content, existing_memories, existing_embeddings=existing_embeddings)
            if duplicate is not None:
                # Update existing memory instead of creating new one
                memory_id = duplicate.get("memory_id")
                if memory_id:
                    memory_bank.delete_memory(memory_id)
                    new_id = memory_bank.store_memory(
                        content, category, importance,
                        memory_layer=memory_layer,
                        entity_refs=entity_refs,
                    )
                    if new_id:
                        logger.info(
                            "Updated memory %s -> %s for user %s (conflict resolved)",
                            memory_id, new_id, user_id,
                        )
                        stored_count += 1
            else:
                new_id = memory_bank.store_memory(
                    content, category, importance,
                    memory_layer=memory_layer,
                    entity_refs=entity_refs,
                )
                if new_id:
                    logger.info("Stored new memory %s for user %s", new_id, user_id)
                    stored_count += 1

        logger.info(
            "Memory extraction complete for session %s: %d memories stored",
            session_id, stored_count,
        )
        return stored_count

    def _format_transcript(self, transcript_history: list[dict]) -> str:
        """Format transcript history into a readable string."""
        lines = []
        for entry in transcript_history:
            role = entry.get("role", "unknown")
            text = entry.get("text", "")
            if text.strip():
                label = "User" if role == "user" else "Assistant"
                lines.append(f"{label}: {text}")
        return "\n".join(lines)

    def _call_extraction_model(self, transcript_text: str) -> list[dict]:
        """Call Gemini Flash to extract memories from transcript."""
        try:
            from google import genai

            api_key = os.environ.get("_GOOGLE_AI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
            client = genai.Client(api_key=api_key, vertexai=False)
            prompt = EXTRACTION_PROMPT.format(transcript=transcript_text)

            response = client.models.generate_content(
                model=EXTRACTION_MODEL,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0.2,
                },
            )

            raw_text = response.text.strip()
            candidates = json.loads(raw_text)
            if not isinstance(candidates, list):
                logger.warning("Extraction model returned non-list: %s", type(candidates))
                return []
            validated = []
            for raw in candidates:
                v = self._validate_candidate(raw)
                if v is not None:
                    validated.append(v)
            return validated
        except Exception:
            logger.exception("Memory extraction model call failed")
            return []

    def _validate_candidate(self, raw: dict) -> Optional[dict]:
        """Validate and normalize a candidate memory dict."""
        try:
            content = str(raw.get("content", "")).strip()
            if not content:
                return None
            category = str(raw.get("category", "general")).strip().lower()
            if category not in _VALID_CATEGORIES:
                logger.warning("Unknown category %r, defaulting to 'general'", category)
                category = "general"
            importance = max(0.0, min(1.0, float(raw.get("importance", 0.5))))
            confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0))))

            # Memory layer: use LLM output if valid, else infer from category
            memory_layer = str(raw.get("memory_layer", "")).strip().lower()
            if memory_layer not in _VALID_LAYERS:
                memory_layer = _CATEGORY_TO_LAYER.get(category, "semantic")

            # Entity names referenced in this memory
            entity_names = raw.get("entity_names", [])
            if not isinstance(entity_names, list):
                entity_names = []
            entity_names = [str(n).strip() for n in entity_names if str(n).strip()]

            return {
                "content": content,
                "category": category,
                "importance": importance,
                "confidence": confidence,
                "memory_layer": memory_layer,
                "entity_names": entity_names,
            }
        except (TypeError, ValueError) as e:
            logger.warning("Invalid candidate: %s", e)
            return None

    def _precompute_existing_embeddings(self, existing_memories: list[dict]) -> dict[str, list[float]]:
        """Pre-compute embeddings for all existing memories. Returns dict keyed by memory_id."""
        cache: dict[str, list[float]] = {}
        for mem in existing_memories:
            mid = mem.get("memory_id", "")
            content = mem.get("content", "")
            if mid and content:
                try:
                    cache[mid] = _compute_embedding(content)
                except Exception:
                    logger.debug("Failed to compute embedding for memory %s", mid, exc_info=True)
        return cache

    def _find_duplicate(
        self,
        content: str,
        existing_memories: list[dict],
        existing_embeddings: dict[str, list[float]] | None = None,
    ) -> Optional[dict]:
        """Check if content is semantically similar to an existing memory.

        Uses vector cosine similarity when embeddings are available,
        falls back to Jaccard text similarity when embeddings fail.

        Args:
            content: The candidate memory content.
            existing_memories: List of existing memory dicts.
            existing_embeddings: Optional pre-computed embeddings keyed by memory_id.

        Returns the matching memory dict if similarity > threshold, else None.
        """
        if not existing_memories:
            return None

        new_embedding = _compute_embedding(content)
        embedding_valid = not all(v == 0.0 for v in new_embedding[:10])

        for mem in existing_memories:
            existing_content = mem.get("content", "")
            if not existing_content:
                continue
            # Try vector similarity
            if embedding_valid:
                mid = mem.get("memory_id", "")
                if existing_embeddings and mid in existing_embeddings:
                    existing_embedding = existing_embeddings[mid]
                else:
                    existing_embedding = _compute_embedding(existing_content)
                if not all(v == 0.0 for v in existing_embedding[:10]):
                    sim = _cosine_similarity(new_embedding, existing_embedding)
                    if sim > self.similarity_threshold:
                        logger.debug(
                            "Found duplicate (vec sim=%.3f): '%s' ~ '%s'",
                            sim, content[:50], existing_content[:50],
                        )
                        return mem
            # Text fallback (Jaccard)
            if self._text_similarity(content, existing_content) > 0.7:
                logger.debug(
                    "Found duplicate (text sim): '%s' ~ '%s'",
                    content[:50], existing_content[:50],
                )
                return mem

        return None

    def _resolve_entity_names(self, user_id: str, entity_names: list[str]) -> list[str]:
        """Resolve entity names to IDs via the entity graph. Best-effort."""
        if not entity_names:
            return []
        try:
            from context.entity_graph import EntityGraphService
            graph = EntityGraphService(user_id)
            refs = []
            for name in entity_names:
                entity = graph.find_entity_by_name(name)
                if entity:
                    refs.append(entity.entity_id)
            return refs
        except Exception:
            logger.debug("Entity name resolution unavailable", exc_info=True)
            return []

    def _text_similarity(self, a: str, b: str) -> float:
        """Jaccard word-overlap similarity between two strings."""
        a_words = set(a.lower().split())
        b_words = set(b.lower().split())
        if not a_words or not b_words:
            return 0.0
        return len(a_words & b_words) / len(a_words | b_words)
