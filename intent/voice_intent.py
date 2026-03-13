"""Voice intent helpers extracted from ``server.py``."""

from __future__ import annotations

import re
from collections import deque


def _normalize_text_for_dedupe(text: str) -> str:
    """Normalize free text for repeat suppression checks."""
    lowered = (text or "").strip().lower()
    if not lowered:
        return ""
    compact = re.sub(r"\s+", " ", lowered)
    compact = re.sub(r"[^\w\s]", "", compact, flags=re.UNICODE)
    return compact.strip()


def _is_repeated_text(
    text: str,
    *,
    previous_text: str,
    now_ts: float,
    previous_ts: float,
    cooldown_sec: float,
    min_chars: int = 6,
) -> bool:
    """Return True when the same meaningful text repeats inside cooldown."""
    if not previous_text:
        return False
    if now_ts < previous_ts:
        return False
    normalized = _normalize_text_for_dedupe(text)
    previous_normalized = _normalize_text_for_dedupe(previous_text)
    if len(normalized) < min_chars or len(previous_normalized) < min_chars:
        return False
    if normalized != previous_normalized:
        return False
    return (now_ts - previous_ts) < cooldown_sec


def _should_reset_interrupted_on_activity_start(
    *,
    event_name: str,
    interrupted: bool,
) -> bool:
    """Return True when a fresh user turn should clear stale interrupted state."""
    return event_name == "activity_start" and interrupted


_DETAIL_PHRASES = {"tell me more", "more detail", "describe more", "what else", "elaborate"}
_STOP_PHRASES = {"stop", "be quiet", "shut up", "enough", "stop talking", "quiet"}
_NAVIGATION_INTENT_PHRASES = {
    "navigate",
    "navigation",
    "direction",
    "directions",
    "route",
    "way to",
    "how do i get",
    "take me to",
    "go to",
    "walk to",
    "walking guidance",
    "head to",
    "guide me to",
    "turn by turn",
}
_LOCATION_QUERY_PHRASES = {
    "around me", "around here", "nearby", "near me", "near here",
    "what's here", "what is here", "where am i", "where are we",
    "what's around", "what is around", "what's close", "find",
    "search for", "is there a", "any", "closest", "look up",
}


def _detect_voice_intent(text: str) -> str | None:
    """Detect user intent from transcribed speech for LOD flag setting."""
    lower = text.strip().lower()
    for phrase in _DETAIL_PHRASES:
        if phrase in lower:
            return "detail"
    for phrase in _STOP_PHRASES:
        if phrase in lower:
            return "stop"
    return None


def _has_navigation_intent(text: str) -> bool:
    """Heuristic detector for explicit user navigation intent."""
    lowered = _normalize_text_for_dedupe(text)
    if not lowered:
        return False
    return any(phrase in lowered for phrase in _NAVIGATION_INTENT_PHRASES)


def _has_location_query_intent(text: str) -> bool:
    """Heuristic detector for implicit location query intent."""
    lowered = _normalize_text_for_dedupe(text)
    if not lowered:
        return False
    return any(phrase in lowered for phrase in _LOCATION_QUERY_PHRASES)


def _recent_user_utterances(
    transcript_history: deque,
    *,
    max_items: int = 3,
) -> list[str]:
    """Return latest non-empty user utterances from transcript history."""
    user_texts: list[str] = []
    for entry in reversed(transcript_history):
        if entry.get("role") != "user":
            continue
        text = str(entry.get("text", "")).strip()
        if not text:
            continue
        user_texts.append(text)
        if len(user_texts) >= max_items:
            break
    return user_texts


def _allow_navigation_tool_call(
    *,
    func_name: str,
    func_args: dict,
    transcript_history: deque,
) -> tuple[bool, str]:
    """Gate navigation calls with two tiers:

    Tier 1 (ACTIVE_NAVIGATION_TOOLS): navigate_to, get_walking_directions
        → require explicit navigation intent in recent user utterances.

    Tier 2 (LOCATION_QUERY_TOOLS): get_location_info, nearby_search, etc.
        → allowed with explicit OR implicit location query intent,
          or general question patterns (what/where/find/any).
    """
    from tools.navigation import (
        ACTIVE_NAVIGATION_TOOLS,
        LOCATION_QUERY_TOOLS,
        NAVIGATION_FUNCTIONS,
    )

    if func_name not in NAVIGATION_FUNCTIONS:
        return True, "not_navigation_tool"

    recent_user = _recent_user_utterances(transcript_history, max_items=3)
    if not recent_user:
        return False, "no_recent_user_transcript"

    # Tier 2: Location queries — explicit or implicit intent
    if func_name in LOCATION_QUERY_TOOLS:
        if any(_has_navigation_intent(t) for t in recent_user):
            return True, "explicit_navigation_intent"
        if any(_has_location_query_intent(t) for t in recent_user):
            return True, "implicit_location_query_intent"
        # General question patterns also allowed for location queries
        if any(
            "?" in t or any(w in t.lower() for w in ("what", "where", "find", "any"))
            for t in recent_user
        ):
            return True, "general_query_intent"

    # Tier 1: Active navigation — explicit intent only
    if func_name in ACTIVE_NAVIGATION_TOOLS:
        if any(_has_navigation_intent(t) for t in recent_user):
            return True, "explicit_navigation_intent"
        # Destination followup heuristic
        destination = str(func_args.get("destination", "")).strip().lower()
        if destination:
            for text in recent_user:
                lowered = text.lower()
                if destination in lowered and any(
                    hint in lowered for hint in ("how", "way", "get", "go", "route", "direction")
                ):
                    return True, "destination_followup_navigation_intent"

    return False, "navigation_tool_requires_explicit_user_request"

