"""Helpers for recovering interrupted Gemini Live downstream streams."""

from __future__ import annotations

_RETRYABLE_TRANSPORT_TOKENS = (
    "keepalive ping timeout",
    "abnormal closure [internal]",
    "1011 (internal error)",
    "1006",
    "connectionclosed",
    "connection closed",
    "connection reset",
    "broken pipe",
    "stream closed",
    "stream interrupted",
    "eoferror",
    "unexpected eof",
)

def flatten_exception_text(exc: BaseException) -> str:
    """Flatten an exception chain into lowercase text for coarse classification."""
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(type(current).__name__)
        text = str(current).strip()
        if text:
            parts.append(text)
        for arg in getattr(current, "args", ()):
            if isinstance(arg, str) and arg.strip():
                parts.append(arg.strip())
        current = current.__cause__ or current.__context__
    return " | ".join(parts).lower()


def is_retryable_transport_error(exc_text: str) -> bool:
    """Whether a downstream exception looks like a transient transport drop."""
    if not exc_text:
        return False
    if "1000" in exc_text:
        return False
    if "normal closure" in exc_text and "abnormal closure" not in exc_text:
        return False
    return any(token in exc_text for token in _RETRYABLE_TRANSPORT_TOKENS)


def compute_retry_backoff(retry_attempt: int) -> float:
    """Use a small capped backoff for internal downstream restarts."""
    return min(3.0, 0.8 * max(1, retry_attempt))
