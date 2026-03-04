"""Tool call deduplication for SightLine.

Prevents duplicate and mutually exclusive tool calls within a single
Gemini response batch, reducing token waste and cascade failures.

Three components:
- ToolCallDeduplicator: time-window dedup for same function + same args
- MutualExclusionFilter: only one tool per mutex group per batch
- AudioGate: mute flag during tool execution to prevent re-triggers
"""

from __future__ import annotations

import hashlib
import json
import logging
import time

logger = logging.getLogger("sightline.tools.dedup")


class ToolCallDeduplicator:
    """Skip repeated calls to the same tool with the same args within a cooldown window."""

    SINGLE_CALL_TOOLS = {
        "navigate_to", "nearby_search", "maps_query",
        "google_search", "get_walking_directions", "get_location_info",
    }

    def __init__(self, cooldown_sec: float = 8.0) -> None:
        self._cooldown_sec = cooldown_sec
        self._recent: dict[str, float] = {}  # fingerprint → timestamp

    def _fingerprint(self, func_name: str, args: dict) -> str:
        """Create a stable fingerprint for function + args."""
        key = json.dumps({"f": func_name, "a": args}, sort_keys=True, default=str)
        return hashlib.md5(key.encode()).hexdigest()

    def should_execute(self, func_name: str, args: dict) -> tuple[bool, str]:
        """Check if a tool call should execute.

        Returns:
            (should_execute, reason) tuple.
        """
        if func_name not in self.SINGLE_CALL_TOOLS:
            return True, "not_dedup_target"

        fp = self._fingerprint(func_name, args)
        now = time.monotonic()
        last_time = self._recent.get(fp)

        if last_time is not None and (now - last_time) < self._cooldown_sec:
            elapsed = now - last_time
            logger.info(
                "Dedup: skipping %s (same call %.1fs ago, cooldown=%.1fs)",
                func_name, elapsed, self._cooldown_sec,
            )
            return False, f"duplicate_within_{self._cooldown_sec}s"

        self._recent[fp] = now
        # Prune old entries
        cutoff = now - self._cooldown_sec * 2
        self._recent = {k: v for k, v in self._recent.items() if v > cutoff}
        return True, "ok"

    def reset(self) -> None:
        """Clear all tracked calls (e.g. on new session)."""
        self._recent.clear()


_MUTEX_GROUPS = {
    "place_search": {"nearby_search", "maps_query"},
    "navigation":   {"navigate_to", "get_walking_directions"},
}


class MutualExclusionFilter:
    """Within a single batch of function calls, only the first tool from each
    mutual exclusion group is allowed to execute."""

    def __init__(self) -> None:
        self._fired: dict[str, str] = {}  # group_name → first tool that fired

    def should_execute(self, func_name: str) -> tuple[bool, str]:
        """Check if a tool call is allowed given mutual exclusion rules.

        Returns:
            (should_execute, reason) tuple.
        """
        for group_name, members in _MUTEX_GROUPS.items():
            if func_name not in members:
                continue
            first = self._fired.get(group_name)
            if first is not None and first != func_name:
                logger.info(
                    "Mutex: skipping %s (group '%s' already used by %s)",
                    func_name, group_name, first,
                )
                return False, f"mutex_{group_name}_already_used_by_{first}"
            self._fired[group_name] = func_name
        return True, "ok"

    def reset(self) -> None:
        """Reset for a new batch of function calls."""
        self._fired.clear()


class AudioGate:
    """Simple flag to indicate tool execution is in progress.

    When active, the upstream handler can mute audio input to prevent
    the model from re-triggering tool calls based on stale audio.
    """

    def __init__(self) -> None:
        self._active: bool = False

    def enter(self) -> None:
        """Mark tool execution as started."""
        self._active = True

    def exit(self) -> None:
        """Mark tool execution as finished."""
        self._active = False

    @property
    def should_mute(self) -> bool:
        """Whether audio input should be muted."""
        return self._active
