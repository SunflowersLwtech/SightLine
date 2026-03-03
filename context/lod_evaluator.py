"""LLM-based LOD micro-adjustment via Gemini Flash.

Provides an async evaluator that can suggest ±1 LOD adjustments based
on rich context that the rule engine cannot see (location familiarity,
entity graph, memory patterns).

Invariants:
    - Adjustment capped at ±1
    - Hard timeout: 500ms
    - Debounce: max once per 30 seconds

Model: gemini-3-flash-preview (lowest latency)
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

EVALUATOR_MODEL = "gemini-3-flash-preview"
_TIMEOUT_S = 0.5  # 500ms hard limit
_DEBOUNCE_S = 30.0  # minimum interval between evaluations

_EVALUATION_PROMPT = """\
You are the LOD micro-adjuster for SightLine, an AI assistant for blind users.

Current state:
- Rule-engine LOD: {baseline_lod}
- Location: {location_summary}
- Familiarity: {familiarity}
- Relevant entities: {entities}
- Recent memories: {memories}
- User profile: vision={vision_status}, O&M={om_level}, verbosity={verbosity}

Based on this context, should the LOD be adjusted?

Rules:
- You can only suggest KEEP, UP (+1), or DOWN (-1)
- UP means more detail (higher LOD number)
- DOWN means less detail (lower LOD number)
- Only suggest change if the context clearly warrants it
- Err on the side of KEEP

Respond with EXACTLY one line in this format:
DECISION: KEEP|UP|DOWN
REASON: <one sentence explanation>
"""


@dataclass
class LODAdjustment:
    """Result of an LLM LOD evaluation."""

    delta: int  # -1, 0, or +1
    reason: str
    confidence: float


class LODEvaluator:
    """Async LLM evaluator for LOD micro-adjustment."""

    def __init__(self):
        self._last_eval_time: float = 0.0
        self._last_result: LODAdjustment | None = None

    async def evaluate(
        self,
        baseline_lod: int,
        location_ctx=None,
        relevant_memories: list[dict] | None = None,
        user_profile=None,
        visible_entities: list | None = None,
    ) -> LODAdjustment:
        """Evaluate whether LOD should be micro-adjusted.

        Returns LODAdjustment with delta in {-1, 0, +1}.
        On timeout or error, returns delta=0 (KEEP).
        """
        # LOD 1 is already the minimum — no further adjustment
        if baseline_lod == 1:
            return LODAdjustment(delta=0, reason="LOD 1 — already minimum", confidence=1.0)

        # Debounce
        now = time.time()
        if now - self._last_eval_time < _DEBOUNCE_S:
            if self._last_result:
                return self._last_result
            return LODAdjustment(delta=0, reason="Debounce — too soon", confidence=1.0)

        # Build prompt
        prompt = self._build_prompt(
            baseline_lod, location_ctx, relevant_memories, user_profile, visible_entities
        )

        # Call LLM with timeout
        try:
            result = await asyncio.wait_for(
                self._call_llm(prompt),
                timeout=_TIMEOUT_S,
            )
            self._last_eval_time = now
            self._last_result = result
            return result
        except asyncio.TimeoutError:
            logger.debug("LOD evaluator timed out (%.1fs limit)", _TIMEOUT_S)
            return LODAdjustment(delta=0, reason="Timeout — using rule-engine result", confidence=1.0)
        except Exception:
            logger.debug("LOD evaluator failed", exc_info=True)
            return LODAdjustment(delta=0, reason="Error — using rule-engine result", confidence=1.0)

    def _build_prompt(self, baseline_lod, location_ctx, memories, profile, entities) -> str:
        location_summary = "unknown"
        familiarity = "0.5"
        if location_ctx:
            place = getattr(location_ctx, "place_name", "")
            if place:
                location_summary = f"{place} ({getattr(location_ctx, 'place_type', 'unknown')})"
            familiarity = f"{getattr(location_ctx, 'familiarity_score', 0.5):.2f}"

        entity_names = []
        if entities:
            entity_names = [getattr(e, "name", "?") for e in entities[:5]]

        memory_snippets = []
        if memories:
            memory_snippets = [m.get("content", "")[:60] for m in memories[:3]]

        vision_status = "totally_blind"
        om_level = "intermediate"
        verbosity = "concise"
        if profile:
            vision_status = getattr(profile, "vision_status", vision_status)
            om_level = getattr(profile, "om_level", om_level)
            verbosity = getattr(profile, "verbosity_preference", verbosity)

        return _EVALUATION_PROMPT.format(
            baseline_lod=baseline_lod,
            location_summary=location_summary,
            familiarity=familiarity,
            entities=", ".join(entity_names) if entity_names else "none",
            memories="; ".join(memory_snippets) if memory_snippets else "none",
            vision_status=vision_status,
            om_level=om_level,
            verbosity=verbosity,
        )

    async def _call_llm(self, prompt: str) -> LODAdjustment:
        """Call Gemini Flash and parse response."""
        from google import genai

        api_key = os.environ.get("_GOOGLE_AI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
        client = genai.Client(api_key=api_key, vertexai=False)

        response = await asyncio.to_thread(
            client.models.generate_content,
            model=EVALUATOR_MODEL,
            contents=prompt,
            config={"temperature": 0.1, "max_output_tokens": 100},
        )

        return self._parse_response(response.text)

    def _parse_response(self, text: str) -> LODAdjustment:
        """Parse LLM response into LODAdjustment."""
        text = text.strip().upper()

        delta = 0
        reason = "LLM evaluation"

        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("DECISION:"):
                decision = line.split(":", 1)[1].strip()
                if "UP" in decision:
                    delta = 1
                elif "DOWN" in decision:
                    delta = -1
                else:
                    delta = 0
            elif line.startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()

        # Clamp to ±1
        delta = max(-1, min(1, delta))

        return LODAdjustment(delta=delta, reason=reason, confidence=0.8)
