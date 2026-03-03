"""Session metadata tracker for runtime telemetry.

Accumulates session-level metrics in memory and writes them to
Firestore at session start/end. All operations are non-blocking
and exception-safe — a tracker failure must never crash a session.

Firestore path: user_profiles/{user_id}/sessions_meta/{session_id}
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Module-level singleton — intentionally shared across all tracker instances
_firestore_client = None


@dataclass
class SessionMetaTracker:
    """Tracks session-level metrics and persists to Firestore.

    Designed for zero overhead on the hot path: all state is accumulated
    in-memory via simple counter increments. Firestore writes happen only
    at session start and end, via ``asyncio.to_thread`` to avoid blocking.
    """

    user_id: str
    session_id: str

    # LOD time tracking
    _lod_distribution: dict[str, int] = field(default_factory=lambda: {"lod1": 0, "lod2": 0, "lod3": 0})
    _current_lod: int = 2
    _lod_start_time: float = field(default_factory=time.monotonic)

    # Interaction counter
    _total_interactions: int = 0

    # Session metadata (set at end)
    space_transitions: list[str] = field(default_factory=list)
    _trip_purpose: str = ""

    # Phase 6 ACE fields (populated at session end, best-effort)
    locations_visited: list[str] = field(default_factory=list)
    entities_seen: list[str] = field(default_factory=list)
    lod_overrides: list[dict] = field(default_factory=list)

    def record_lod_time(self, new_lod: int) -> None:
        """Record elapsed time at the current LOD, then switch to new_lod.

        Call this whenever LOD changes. Elapsed seconds at the previous
        level are accumulated in ``_lod_distribution``.
        """
        now = time.monotonic()
        elapsed = int(now - self._lod_start_time)
        key = f"lod{self._current_lod}"
        self._lod_distribution[key] = self._lod_distribution.get(key, 0) + elapsed
        self._current_lod = new_lod
        self._lod_start_time = now

    def record_interaction(self) -> None:
        """Increment the interaction counter (user transcript turn or gesture)."""
        self._total_interactions += 1

    def set_trip_purpose(self, purpose: str) -> None:
        """Set the trip purpose string."""
        self._trip_purpose = purpose

    def _flush_current_lod(self) -> None:
        """Flush elapsed time for the current LOD level."""
        now = time.monotonic()
        elapsed = int(now - self._lod_start_time)
        key = f"lod{self._current_lod}"
        self._lod_distribution[key] = self._lod_distribution.get(key, 0) + elapsed
        self._lod_start_time = now

    def build_end_doc(self) -> dict:
        """Build the Firestore document dict for session end.

        Returns a dict suitable for Firestore ``update()`` — does NOT
        include ``start_time`` (that's written at session start).
        """
        self._flush_current_lod()
        doc = {
            "end_time": "SERVER_TIMESTAMP",  # replaced by caller
            "trip_purpose": self._trip_purpose,
            "lod_distribution": dict(self._lod_distribution),
            "space_transitions": [t for t in self.space_transitions if t != "unknown"],
            "total_interactions": self._total_interactions,
        }
        if self.locations_visited:
            doc["locations_visited"] = self.locations_visited[:20]
        if self.entities_seen:
            doc["entities_seen"] = self.entities_seen[:50]
        if self.lod_overrides:
            doc["lod_overrides"] = self.lod_overrides[:50]
        return doc

    # -- Firestore I/O (async, non-blocking) --------------------------------

    def _get_firestore(self):
        """Return a lazily-initialized Firestore client (module-level singleton)."""
        global _firestore_client
        if _firestore_client is None:
            from google.cloud import firestore
            project = os.getenv("GOOGLE_CLOUD_PROJECT", "sightline-hackathon")
            _firestore_client = firestore.Client(project=project)
        return _firestore_client

    def _get_doc_ref(self):
        """Return the Firestore document reference for this session."""
        db = self._get_firestore()
        return (
            db.collection("user_profiles")
            .document(self.user_id)
            .collection("sessions_meta")
            .document(self.session_id)
        )

    async def write_session_start(self) -> None:
        """Write the session start document to Firestore."""
        try:
            await asyncio.to_thread(self._write_start_sync)
        except Exception:
            logger.warning(
                "Failed to write session_meta start for session %s",
                self.session_id,
                exc_info=True,
            )

    async def write_session_end(self) -> None:
        """Write session end data to Firestore."""
        try:
            await asyncio.to_thread(self._write_end_sync)
        except Exception:
            logger.warning(
                "Failed to write session_meta end for session %s",
                self.session_id,
                exc_info=True,
            )

    def _write_start_sync(self) -> None:
        """Synchronous Firestore write for session start."""
        from google.cloud import firestore

        doc_ref = self._get_doc_ref()
        doc_ref.set({
            "start_time": firestore.SERVER_TIMESTAMP,
            "end_time": None,
            "trip_purpose": "",
            "lod_distribution": {"lod1": 0, "lod2": 0, "lod3": 0},
            "space_transitions": [],
            "total_interactions": 0,
        })
        logger.info(
            "session_meta start written: user=%s session=%s",
            self.user_id, self.session_id,
        )

    def _write_end_sync(self) -> None:
        """Synchronous Firestore write for session end."""
        from google.cloud import firestore

        self._flush_current_lod()
        doc_ref = self._get_doc_ref()
        update_data = {
            "end_time": firestore.SERVER_TIMESTAMP,
            "trip_purpose": self._trip_purpose,
            "lod_distribution": dict(self._lod_distribution),
            "space_transitions": [t for t in self.space_transitions if t != "unknown"],
            "total_interactions": self._total_interactions,
        }
        # Phase 6 ACE fields (only written when present)
        if self.locations_visited:
            update_data["locations_visited"] = self.locations_visited[:20]
        if self.entities_seen:
            update_data["entities_seen"] = self.entities_seen[:50]
        if self.lod_overrides:
            update_data["lod_overrides"] = self.lod_overrides[:50]
        doc_ref.update(update_data)
        logger.info(
            "session_meta end written: user=%s session=%s interactions=%d",
            self.user_id, self.session_id, self._total_interactions,
        )
