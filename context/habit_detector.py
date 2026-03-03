"""Habit detector — pattern mining from session history.

Scans the last 30 ``sessions_meta`` documents for behavioral patterns:
  - Frequent location+time combos → "visits Starbucks Monday mornings"
  - Consistent LOD overrides → "always wants LOD 3 at restaurants"
  - Proactive action acceptance → "always wants prices read at food places"

Output: proactive hints for session prompt injection + procedural memories.

Firestore path: user_profiles/{uid}/sessions_meta/{sid}
"""

import logging
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "sightline-hackathon")

_MAX_SESSIONS = 30
_MIN_OCCURRENCES = 3  # minimum times a pattern must appear to be considered a habit


@dataclass
class ProactiveHint:
    """A learned behavioral pattern to inject into the session prompt."""

    hint_type: str  # "location_habit" | "lod_preference" | "action_preference"
    description: str  # human-readable for prompt injection
    confidence: float = 0.0  # 0-1, based on occurrence frequency
    location: str = ""  # place name if location-specific
    time_context: str = ""  # time of day if time-specific


class HabitDetector:
    """Pattern mining from session history."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self._firestore = None
        self._try_init()

    def _try_init(self):
        try:
            from google.cloud import firestore
            self._firestore = firestore.Client(project=PROJECT_ID)
        except Exception as e:
            logger.warning("Firestore init failed for HabitDetector user=%s: %s", self.user_id, e)

    def detect(self) -> list[ProactiveHint]:
        """Scan recent sessions and extract behavioral patterns.

        Returns a list of ProactiveHints, sorted by confidence (highest first).
        """
        sessions = self._load_sessions()
        if not sessions:
            return []

        hints: list[ProactiveHint] = []
        hints.extend(self._detect_location_habits(sessions))
        hints.extend(self._detect_lod_preferences(sessions))

        # Sort by confidence and return
        hints.sort(key=lambda h: h.confidence, reverse=True)
        return hints

    def _load_sessions(self) -> list[dict]:
        """Load recent sessions_meta from Firestore."""
        if not self._firestore:
            return []
        try:
            coll = (
                self._firestore.collection("user_profiles")
                .document(self.user_id)
                .collection("sessions_meta")
            )
            query = coll.order_by("start_time", direction="DESCENDING").limit(_MAX_SESSIONS)
            return [doc.to_dict() for doc in query.stream()]
        except Exception:
            logger.debug("Failed to load sessions_meta", exc_info=True)
            return []

    def _detect_location_habits(self, sessions: list[dict]) -> list[ProactiveHint]:
        """Detect frequent location + time combinations."""
        combo_counter: Counter = Counter()

        for s in sessions:
            locations = s.get("locations_visited", [])
            time_ctx = s.get("time_context", "unknown")
            for loc in locations:
                if isinstance(loc, str) and loc.strip():
                    combo_counter[(loc.strip(), time_ctx)] += 1

        hints = []
        for (location, time_ctx), count in combo_counter.items():
            if count >= _MIN_OCCURRENCES:
                confidence = min(1.0, count / _MAX_SESSIONS)
                time_desc = f" during {time_ctx}" if time_ctx != "unknown" else ""
                hints.append(ProactiveHint(
                    hint_type="location_habit",
                    description=f"User regularly visits {location}{time_desc} ({count} times in recent sessions)",
                    confidence=confidence,
                    location=location,
                    time_context=time_ctx,
                ))

        return hints

    def _detect_lod_preferences(self, sessions: list[dict]) -> list[ProactiveHint]:
        """Detect consistent LOD override patterns at locations."""
        # Track: location → list of (override_direction, count)
        location_overrides: dict[str, Counter] = defaultdict(Counter)

        for s in sessions:
            overrides = s.get("lod_overrides", [])
            locations = s.get("locations_visited", [])
            primary_location = locations[0] if locations else "unknown"

            for override in overrides:
                if isinstance(override, dict):
                    direction = override.get("direction", "")
                    if direction in ("up", "down"):
                        location_overrides[primary_location][direction] += 1

        hints = []
        for location, directions in location_overrides.items():
            for direction, count in directions.items():
                if count >= _MIN_OCCURRENCES:
                    confidence = min(1.0, count / (_MAX_SESSIONS * 0.5))
                    lod_desc = "more detail" if direction == "up" else "less detail"
                    hints.append(ProactiveHint(
                        hint_type="lod_preference",
                        description=f"User typically requests {lod_desc} at {location} ({count} overrides)",
                        confidence=confidence,
                        location=location,
                    ))

        return hints
