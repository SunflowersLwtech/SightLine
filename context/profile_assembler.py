"""Profile assembler — unified user context document for prompt injection.

Assembles a structured text block (~800 tokens) from all context sources:
  - User Profile (static: vision, O&M, preferences)
  - Known People (from entity graph, relevant to current scene)
  - Current Location (from LocationContext)
  - Habits & Preferences (procedural memories)
  - Recent Events (episodic memories, last 7 days)

Inspired by Gemini's ``user_context`` design: structured text > complex
retrieval pipelines. The LOD engine needs deterministic signals, not
just text blobs, so we keep sections machine-readable.
"""

import logging
import time
from typing import Optional

from lod.models import UserProfile

logger = logging.getLogger(__name__)

# Target budget: ~800 tokens ≈ ~3200 chars
_MAX_CHARS = 3200
_MAX_MEMORIES_PER_SECTION = 5
_EPISODIC_LOOKBACK_DAYS = 7


class ProfileAssembler:
    """Assembles a unified context document from all user signals."""

    def assemble(
        self,
        profile: UserProfile,
        location_ctx=None,
        entities: list | None = None,
        memories: list[dict] | None = None,
    ) -> str:
        """Build the full context text.

        Args:
            profile: User's static profile.
            location_ctx: Current LocationContext (or None).
            entities: Entities relevant to current scene.
            memories: All retrieved memories (already ranked).

        Returns:
            Structured text block for system prompt injection.
        """
        sections: list[str] = []

        # Section 1: User Profile (always present)
        sections.append(self._build_profile_section(profile))

        # Section 2: Current Location
        if location_ctx:
            loc_section = self._build_location_section(location_ctx)
            if loc_section:
                sections.append(loc_section)

        # Section 3: Known People / Entities
        if entities:
            ent_section = self._build_entities_section(entities)
            if ent_section:
                sections.append(ent_section)

        # Section 4 & 5: Memories split by layer
        if memories:
            proc_section = self._build_procedural_section(memories)
            if proc_section:
                sections.append(proc_section)

            epi_section = self._build_episodic_section(memories)
            if epi_section:
                sections.append(epi_section)

        result = "\n\n".join(sections)

        # Truncate to budget if needed
        if len(result) > _MAX_CHARS:
            result = result[:_MAX_CHARS - 3] + "..."
            logger.debug("Profile assembly truncated to %d chars", _MAX_CHARS)

        return result

    def _build_profile_section(self, profile: UserProfile) -> str:
        aids = []
        if profile.has_guide_dog:
            aids.append("guide dog")
        if profile.has_white_cane:
            aids.append("white cane")

        onset = "congenital" if profile.blindness_onset == "congenital" else "acquired"
        lines = [
            "## User Profile",
            f"- Vision: {profile.vision_status} ({onset})",
            f"- Mobility aids: {', '.join(aids) if aids else 'none'}",
            f"- TTS speed: {profile.tts_speed}x",
            f"- O&M level: {profile.om_level} (travel: {profile.travel_frequency})",
            f"- Verbosity preference: {profile.verbosity_preference}",
            f"- Description priority: {profile.description_priority}",
        ]

        if profile.preferred_name:
            name = " ".join(profile.preferred_name.strip().split())
            lines.append(f"- Preferred name: {name}")

        if profile.blindness_onset == "congenital" and not profile.color_description:
            lines.append(
                "- Color descriptions: DISABLED — use tactile/spatial/sound analogies instead"
            )

        return "\n".join(lines)

    def _build_location_section(self, location_ctx) -> Optional[str]:
        place = getattr(location_ctx, "place_name", "")
        if not place:
            return None

        lines = ["## Current Location"]
        lines.append(f"- Place: {place}")

        place_type = getattr(location_ctx, "place_type", "")
        if place_type and place_type != "unknown":
            lines.append(f"- Type: {place_type}")

        is_indoor = getattr(location_ctx, "is_indoor", None)
        if is_indoor is not None:
            lines.append(f"- Indoor: {'yes' if is_indoor else 'no'}")

        familiarity = getattr(location_ctx, "familiarity_score", 0)
        if familiarity > 0:
            if familiarity >= 0.8:
                lines.append("- Familiarity: well-known (regular visit)")
            elif familiarity >= 0.4:
                lines.append("- Familiarity: somewhat familiar")
            else:
                lines.append("- Familiarity: first or rare visit")

        address = getattr(location_ctx, "address", "")
        if address:
            lines.append(f"- Address: {address}")

        return "\n".join(lines)

    def _build_entities_section(self, entities: list) -> Optional[str]:
        if not entities:
            return None

        lines = ["## Known People & Places Nearby"]
        for e in entities[:_MAX_MEMORIES_PER_SECTION]:
            name = getattr(e, "name", "unknown")
            etype = getattr(e, "entity_type", "")
            attrs = getattr(e, "attributes", {})

            desc_parts = [name]
            if etype:
                desc_parts.append(f"({etype})")
            # Include key attributes like "role", "relationship"
            for key in ("role", "relationship", "description"):
                if key in attrs:
                    desc_parts.append(f"— {attrs[key]}")
                    break

            lines.append(f"- {' '.join(desc_parts)}")

        return "\n".join(lines)

    def _build_procedural_section(self, memories: list[dict]) -> Optional[str]:
        """Habits & preferences (procedural layer)."""
        procedural = [
            m for m in memories
            if m.get("memory_layer") == "procedural"
               or m.get("category") in ("preference", "routine")
        ]
        if not procedural:
            return None

        lines = ["## Habits & Preferences"]
        for m in procedural[:_MAX_MEMORIES_PER_SECTION]:
            lines.append(f"- {m.get('content', '')}")

        return "\n".join(lines)

    def _build_episodic_section(self, memories: list[dict]) -> Optional[str]:
        """Recent events (episodic layer, last 7 days)."""
        now = time.time()
        cutoff = now - (_EPISODIC_LOOKBACK_DAYS * 86400)

        episodic = [
            m for m in memories
            if (m.get("memory_layer") == "episodic"
                or m.get("category") == "experience")
            and float(m.get("timestamp", 0)) > cutoff
        ]
        if not episodic:
            return None

        lines = ["## Recent Events"]
        for m in episodic[:_MAX_MEMORIES_PER_SECTION]:
            lines.append(f"- {m.get('content', '')}")

        return "\n".join(lines)
