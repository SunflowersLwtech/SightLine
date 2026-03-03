"""SightLine LOD Decision Engine.

Rule-based (non-LLM) engine that fuses three context layers
(Ephemeral, Session, Profile) into a LOD 1/2/3 decision in <1 ms.

Decision priority (high -> low):
    1. Motion-state baseline (experience-driven)
    2. Ambient noise adjustment
    3. Space transition boost
    4. User verbosity preference
    5. O&M level adjustment
    6. Explicit user override (final)

Every decision produces an explainable ``LODDecisionLog``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from lod.models import EphemeralContext, SessionContext, UserProfile

logger = logging.getLogger("sightline.lod")

# ---------------------------------------------------------------------------
# LOD Decision Log — explainable audit trail (SL-39)
# ---------------------------------------------------------------------------


@dataclass
class LODDecisionLog:
    """Complete explainable record of a single LOD decision."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Input snapshot
    motion_state: str = ""
    step_cadence: float = 0.0
    ambient_noise_db: float = 70.0
    heart_rate: float | None = None
    space_transition: bool = False
    verbosity_preference: str = "concise"
    om_level: str = "intermediate"
    travel_frequency: str = "weekly"
    user_override: str | None = None  # "detail" | "stop" | None

    # Decision trace
    triggered_rules: list[str] = field(default_factory=list)
    base_lod_before_adjustments: int = 2

    # Output
    previous_lod: int = 2
    final_lod: int = 2
    reason: str = ""

    def to_debug_dict(self) -> dict:
        """Compact dict for DebugOverlay / WebSocket ``debug_lod`` event."""
        return {
            "lod": self.final_lod,
            "prev": self.previous_lod,
            "reason": self.reason,
            "rules": self.triggered_rules,
            "hr": self.heart_rate,
            "motion": self.motion_state,
            "cadence": self.step_cadence,
            "noise_db": self.ambient_noise_db,
            "watch_heading": getattr(self, "watch_heading", None),
            "stability": getattr(self, "stability_score", None),
        }


# ---------------------------------------------------------------------------
# Speech-cost thresholds (§3.2 "发声有成本")
# ---------------------------------------------------------------------------

# Base threshold that info_value must exceed to trigger speech
BASE_SPEECH_THRESHOLD = 3.5

INFO_VALUES: dict[str, float] = {
    "navigation": 8.0,
    "face_recognition": 7.0,
    "spatial_description": 5.0,
    "object_enumeration": 3.0,
    "atmosphere": 1.0,
}


def should_speak(
    info_type: str,
    current_lod: int,
    step_cadence: float = 0.0,
    ambient_noise_db: float = 70.0,
) -> bool:
    """Determine whether the agent should vocalise this information."""
    info_value = INFO_VALUES.get(info_type, 1.0)

    movement_penalty = (step_cadence / 60.0) * 2.0
    noise_penalty = max(0.0, (ambient_noise_db - 60) * 0.1)
    # Stillness penalty: reduce proactive speech when user is stationary
    stillness_penalty = 1.5 if step_cadence < 5.0 else 0.0
    threshold = BASE_SPEECH_THRESHOLD + movement_penalty + noise_penalty + stillness_penalty

    return info_value > threshold


# ---------------------------------------------------------------------------
# Core LOD decision function
# ---------------------------------------------------------------------------


def decide_lod(
    ephemeral: EphemeralContext,
    session: SessionContext,
    profile: UserProfile,
) -> tuple[int, LODDecisionLog]:
    """Fuse three context layers and return (lod, log).

    Returns
    -------
    (lod, log) : tuple[int, LODDecisionLog]
        lod in {1, 2, 3}; log contains full decision trace.
    """
    def _to_float(value, default: float) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _to_opt_float(value) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    motion_state = getattr(ephemeral, "motion_state", "stationary") or "stationary"
    step_cadence = _to_float(getattr(ephemeral, "step_cadence", 0.0), 0.0)
    _raw_noise = getattr(ephemeral, "ambient_noise_db", None)
    if _raw_noise is None:
        logger.debug("ambient_noise_db missing; defaulting to conservative 70dB")
        ambient_noise_db = 70.0
    else:
        ambient_noise_db = _to_float(_raw_noise, 70.0)
    heart_rate = _to_opt_float(getattr(ephemeral, "heart_rate", None))
    _raw_gesture = getattr(ephemeral, "user_gesture", None)
    if isinstance(_raw_gesture, str) and _raw_gesture.strip():
        user_gesture = _raw_gesture.strip().lower()
        if user_gesture not in ("lod_up", "lod_down", "tap", "shake"):
            logger.warning("Unknown user_gesture: %r", user_gesture)
            user_gesture = None
    else:
        user_gesture = None

    recent_space_transition = bool(getattr(session, "recent_space_transition", False))
    user_requested_detail = bool(getattr(session, "user_requested_detail", False))
    user_said_stop = bool(getattr(session, "user_said_stop", False))
    previous_lod = int(getattr(session, "current_lod", 2) or 2)

    verbosity_preference_raw = getattr(profile, "verbosity_preference", "concise") or "concise"
    verbosity_preference = str(verbosity_preference_raw).strip().lower()
    om_level = getattr(profile, "om_level", "intermediate") or "intermediate"
    travel_frequency = getattr(profile, "travel_frequency", "weekly") or "weekly"

    log = LODDecisionLog(
        motion_state=motion_state,
        step_cadence=step_cadence,
        ambient_noise_db=ambient_noise_db,
        heart_rate=heart_rate,
        space_transition=recent_space_transition,
        verbosity_preference=verbosity_preference,
        om_level=om_level,
        travel_frequency=travel_frequency,
        previous_lod=previous_lod,
    )

    # ── Rule 1: Motion-state baseline (experience-driven) ───────────
    if motion_state == "running" or step_cadence >= 120:
        base_lod = 1  # brief — user is busy moving fast
        log.triggered_rules.append("Rule1:running→LOD1(brief)")
    elif motion_state == "walking":
        base_lod = 2  # standard — user can listen while walking
        log.triggered_rules.append("Rule1:walking→LOD2")
    elif motion_state == "cycling":
        base_lod = 1  # brief — user is busy
        log.triggered_rules.append("Rule1:cycling→LOD1(brief)")
    elif motion_state == "in_vehicle":
        base_lod = 3  # detailed — user has attention available
        log.triggered_rules.append("Rule1:in_vehicle→LOD3")
    else:  # stationary
        base_lod = 3  # detailed — user is relaxed
        log.triggered_rules.append("Rule1:stationary→LOD3")

    log.base_lod_before_adjustments = base_lod

    # ── Rule 1b: Time-of-day adjustment ─────────────────────────────
    time_context = getattr(ephemeral, "time_context", "unknown") or "unknown"
    if time_context in ("morning_commute", "late_night") and base_lod > 1:
        base_lod = max(1, base_lod - 1)
        log.triggered_rules.append(f"Rule1b:{time_context}→-1")

    # ── Rule 1c: Watch stability adjustment (experience-driven) ─────
    stability = _to_float(getattr(ephemeral, "watch_stability_score", 1.0), 1.0)
    if stability < 0.4 and base_lod > 1:
        base_lod = max(1, base_lod - 1)
        log.triggered_rules.append(f"Rule1c:unstable({stability:.2f})→-1(brief)")

    # ── Rule 2: Ambient noise adjustment ────────────────────────────
    if ambient_noise_db > 80:
        if base_lod > 1:
            log.triggered_rules.append(f"Rule2:noise={ambient_noise_db:.0f}dB>80→cap_LOD1")
        base_lod = min(base_lod, 1)

    # ── Rule 3: Space transition boost ────────────────────────────────
    if recent_space_transition:
        if base_lod < 2:
            log.triggered_rules.append("Rule3:space_transition→boost_LOD2")
        base_lod = max(base_lod, 2)

    # ── Rule 4: User verbosity preference ─────────────────────────────
    if verbosity_preference == "concise":
        if base_lod >= 3:
            base_lod -= 1
            log.triggered_rules.append("Rule4:concise_pref→-1")
    elif verbosity_preference == "detailed":
        prev = base_lod
        base_lod = min(3, base_lod + 1)
        if base_lod != prev:
            log.triggered_rules.append("Rule4:detailed_pref→+1")

    # ── Rule 5: O&M expert adjustment ─────────────────────────────────
    if om_level == "advanced" and travel_frequency == "daily":
        prev = base_lod
        base_lod = max(1, base_lod - 1)
        if base_lod != prev:
            log.triggered_rules.append("Rule5:advanced_daily→-1")

    # ── Rule 5b: Familiarity-based adjustment ─────────────────────────
    familiarity = _to_float(getattr(session, "familiarity_score", 0.5), 0.5)
    if familiarity > 0.8 and base_lod > 1:
        base_lod = max(1, base_lod - 1)
        log.triggered_rules.append(f"Rule5b:familiar({familiarity:.2f})→-1")
    elif familiarity < 0.2 and base_lod < 3:
        base_lod = min(3, base_lod + 1)
        log.triggered_rules.append(f"Rule5b:unfamiliar({familiarity:.2f})→+1")

    # ── Rule 6: Explicit user override + gesture (final) ──────────────
    if user_gesture == "lod_up":
        prev = base_lod
        base_lod = min(3, base_lod + 1)
        if base_lod != prev:
            log.triggered_rules.append("Gesture:lod_up→+1")
    elif user_gesture == "lod_down":
        prev = base_lod
        base_lod = max(1, base_lod - 1)
        if base_lod != prev:
            log.triggered_rules.append("Gesture:lod_down→-1")
    elif user_requested_detail:
        base_lod = 3
        log.triggered_rules.append("Rule6:user_requested_detail→LOD3")
        log.user_override = "detail"
    elif user_said_stop:
        base_lod = 1
        log.triggered_rules.append("Rule6:user_said_stop→LOD1")
        log.user_override = "stop"

    # ── Finalise ──────────────────────────────────────────────────────
    log.final_lod = base_lod
    if log.triggered_rules:
        log.reason = " + ".join(log.triggered_rules) + f" → LOD {base_lod}"
    else:
        log.reason = f"default → LOD {base_lod}"

    if base_lod != previous_lod:
        logger.info(
            "LOD %d → %d  (%s)",
            previous_lod,
            base_lod,
            log.reason,
        )

    return base_lod, log
