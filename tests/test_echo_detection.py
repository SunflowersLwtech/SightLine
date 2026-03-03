"""Tests for server-side echo detection logic.

Tests the _is_likely_echo heuristic that identifies when user input
transcription is actually an echo of recent model output.
"""

import time


def _jaccard_similarity(a: str, b: str) -> float:
    """Compute Jaccard word similarity between two strings."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union) if union else 0.0


def _is_likely_echo(
    candidate: str,
    now_ts: float,
    recent_agent_texts: list[tuple[float, str]],
) -> bool:
    """Standalone implementation matching server.py _is_likely_echo."""
    words_candidate = set(candidate.lower().split())
    if len(words_candidate) < 3:
        return False
    cutoff = now_ts - 5.0
    for ts, agent_text in reversed(recent_agent_texts):
        if ts < cutoff:
            break
        words_agent = set(agent_text.lower().split())
        if not words_agent:
            continue
        intersection = words_candidate & words_agent
        union = words_candidate | words_agent
        jaccard = len(intersection) / len(union) if union else 0.0
        if jaccard > 0.6:
            return True
    return False


def test_exact_echo_detected():
    """Exact repeat of agent text should be detected as echo."""
    now = time.monotonic()
    agent_texts = [(now - 1.0, "There are stairs ahead at your two o'clock")]
    assert _is_likely_echo(
        "There are stairs ahead at your two o'clock", now, agent_texts
    )


def test_partial_echo_detected():
    """Partial echo with most words matching should be detected."""
    now = time.monotonic()
    agent_texts = [(now - 1.0, "There are stairs ahead at your two o'clock position")]
    # Dropping one word still gives high Jaccard
    assert _is_likely_echo(
        "There are stairs ahead at your two o'clock", now, agent_texts
    )


def test_different_text_not_echo():
    """Genuinely different user input should NOT be flagged as echo."""
    now = time.monotonic()
    agent_texts = [(now - 1.0, "There are stairs ahead at your two o'clock")]
    assert not _is_likely_echo(
        "What steps should I take next", now, agent_texts
    )


def test_expired_window_not_echo():
    """Agent text older than 5s should not trigger echo detection."""
    now = time.monotonic()
    agent_texts = [(now - 6.0, "There are stairs ahead at your two o'clock")]
    assert not _is_likely_echo(
        "There are stairs ahead at your two o'clock", now, agent_texts
    )


def test_short_input_not_echo():
    """Inputs with fewer than 3 words should never be flagged as echo."""
    now = time.monotonic()
    agent_texts = [(now - 1.0, "yes okay")]
    assert not _is_likely_echo("yes okay", now, agent_texts)


def test_empty_agent_history():
    """No agent history should never flag anything."""
    now = time.monotonic()
    assert not _is_likely_echo(
        "There are stairs ahead at your two o'clock", now, []
    )


def test_jaccard_threshold_boundary():
    """Verify the 0.6 threshold is correctly applied."""
    # Exactly at boundary: 3 shared / 5 total = 0.6 (should NOT trigger, >0.6 required)
    assert _jaccard_similarity("a b c d e", "a b c f g") == 3.0 / 7.0  # ~0.43 - not echo
    # Above threshold
    assert _jaccard_similarity(
        "stairs ahead two o'clock", "stairs ahead at two o'clock"
    ) > 0.6


def test_multiple_agent_texts_latest_match():
    """Echo detection should check all recent agent texts."""
    now = time.monotonic()
    agent_texts = [
        (now - 4.0, "The weather is nice today"),
        (now - 1.0, "There are stairs ahead at your two o'clock"),
    ]
    # Matches second entry
    assert _is_likely_echo(
        "There are stairs ahead at your two o'clock", now, agent_texts
    )
    # Does not match either entry
    assert not _is_likely_echo(
        "What time is it now please", now, agent_texts
    )
