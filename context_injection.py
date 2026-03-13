"""Context injection queue primitives extracted from ``server.py``."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from dataclasses import field as dc_field
from enum import Enum

from google.genai import types

logger = logging.getLogger("sightline.server")

QUEUE_MAX_AGE_SEC = 15.0
VISION_SPOKEN_COOLDOWN_SEC = 12.0
QUEUE_FLUSH_CHECK_INTERVAL_SEC = 1.0


class TokenBudgetMonitor:
    """Track and log token usage from Gemini usage_metadata events."""

    _WARN_THRESHOLD = 0.70   # 70% utilization
    _CRIT_THRESHOLD = 0.85   # 85% utilization
    _CONTEXT_LIMIT = 128_000  # Gemini Live API context window

    def __init__(self) -> None:
        self._last_total: int = 0
        self._warned: bool = False
        self._critical: bool = False

    def update(self, usage_metadata) -> None:
        """Extract token counts from usage_metadata and log utilization."""
        if usage_metadata is None:
            return
        total = getattr(usage_metadata, "total_token_count", 0) or 0
        if total <= 0:
            return
        self._last_total = total
        ratio = total / self._CONTEXT_LIMIT

        if ratio >= self._CRIT_THRESHOLD and not self._critical:
            self._critical = True
            logger.critical(
                "Token budget CRITICAL: %d / %d (%.0f%%)",
                total, self._CONTEXT_LIMIT, ratio * 100,
            )
        elif ratio >= self._WARN_THRESHOLD and not self._warned:
            self._warned = True
            logger.warning(
                "Token budget WARNING: %d / %d (%.0f%%)",
                total, self._CONTEXT_LIMIT, ratio * 100,
            )

    @property
    def last_total(self) -> int:
        return self._last_total


@dataclass
class _QueuedItem:
    """A single queued context injection waiting for the model to finish."""

    category: str
    text: str
    priority: int  # lower = more important
    speak: bool
    turn_seq: int = 0
    enqueued_at: float = dc_field(default_factory=time.monotonic)


class ModelState(str, Enum):
    """State machine for model generation/playback lifecycle.

    Transitions:
        IDLE → GENERATING  (on inject_immediate / flush that produces audio)
        GENERATING → DRAINING  (on first audio chunk received from Gemini)
        DRAINING → IDLE  (on turn_complete + quiet period confirmed)
        Any → IDLE  (on interrupt / barge-in)

    Flush is ONLY allowed in IDLE state.  This eliminates the 5 overlapping
    guard timers (generation ramp-up, deferred flush, iOS playback drain,
    model audio freshness, playback started tracking) and replaces them with
    a single, deterministic state check.
    """

    IDLE = "idle"
    GENERATING = "generating"
    DRAINING = "draining"


class ContextInjectionQueue:
    """Queue that batches non-urgent send_content() calls.

    When the model is generating or draining audio, enqueued items are held
    and merged into a single Content message when state returns to IDLE.
    Bypass items (safety, gestures, function responses) skip the queue.

    State machine replaces the previous multi-timer approach:
    - IDLE: Model is silent, flush is allowed
    - GENERATING: send_content sent, awaiting first audio chunk
    - DRAINING: Audio chunks flowing / iOS playing back, no flush

    Thread safety: All public methods are intentionally synchronous (no
    ``await`` inside mutation paths).  In asyncio cooperative scheduling,
    dict reads/writes cannot be preempted mid-operation, so no explicit
    lock is needed.
    """

    # How long to stay in GENERATING before timing out to IDLE.
    # Covers the case where Gemini never produces audio for a silent context.
    _GENERATING_TIMEOUT_SEC = 5.0

    # Safety-net timeout for DRAINING state (iOS playback stall).
    _DRAINING_TIMEOUT_SEC = 8.0

    BATCH_WINDOW_SEC = 0.2  # 200ms collection window for sub-agent results

    def __init__(self, live_request_queue) -> None:
        self._lrq = live_request_queue
        self._queue: dict[str, _QueuedItem] = {}
        self._state = ModelState.IDLE
        self._state_entered_at: float = time.monotonic()
        self._vision_spoken_at: float = 0.0
        self._bg_task: asyncio.Task | None = None
        self._stopped = False
        self._deferred_flush_handle: asyncio.TimerHandle | None = None
        self._first_turn: bool = True

    # -- State machine -------------------------------------------------------

    def _transition_to(self, new_state: ModelState) -> None:
        """Transition to a new state, cancelling any pending flush timer."""
        old = self._state
        if old == new_state:
            return
        self._state = new_state
        self._state_entered_at = time.monotonic()
        self._cancel_deferred_flush()
        logger.debug("State: %s → %s", old.value, new_state.value)

        # Auto-schedule flush when entering IDLE with pending items
        if new_state == ModelState.IDLE and self._queue:
            self._schedule_deferred_flush()

    @property
    def state(self) -> ModelState:
        return self._state

    # -- Public API (backwards-compatible) -----------------------------------

    def set_model_speaking(self, speaking: bool) -> None:
        """Called when model audio chunks start/stop.

        speaking=True  → transition to DRAINING (audio is flowing)
        speaking=False → handled by on_turn_complete (don't go IDLE here;
                         wait for quiet period confirmation)
        """
        if speaking:
            self._transition_to(ModelState.DRAINING)

    def set_model_audio_timestamp(self, ts: float) -> None:
        """Record that a model audio chunk was just received."""
        # Any audio chunk means we're in DRAINING (audio actively flowing)
        if self._state == ModelState.GENERATING:
            self._transition_to(ModelState.DRAINING)

    def set_ios_playback_drained(self, drained: bool) -> None:
        """Mark whether iOS has finished playing all buffered audio."""
        if drained and self._state == ModelState.DRAINING:
            # iOS finished playback — transition to IDLE
            self._transition_to(ModelState.IDLE)
        # If not drained and we're IDLE (shouldn't happen often), stay put

    @property
    def model_speaking(self) -> bool:
        """Backwards-compatible property: True when model is active."""
        return self._state != ModelState.IDLE

    def on_turn_complete(self) -> None:
        """Called on turn_complete event.  Transitions to IDLE.

        The downstream handler adds a quiet period check (Step 3) before
        calling this, so we can safely go to IDLE here.
        """
        self._transition_to(ModelState.IDLE)

    # -- Immediate bypass (safety / gestures / function responses) -----------

    def inject_immediate(self, content: "types.Content", is_function_response: bool = False) -> None:
        """Send directly to LiveRequestQueue, bypassing the queue.

        Args:
            content: The Content to send.
            is_function_response: If True, skip state transition to GENERATING.
                Function responses go through LiveClientToolResponse path which
                doesn't immediately trigger model generation — the state
                transition would block subsequent context flushes.
        """
        self._lrq.send_content(content)
        if not is_function_response:
            # We just sent content → model will start generating
            self._transition_to(ModelState.GENERATING)

    # -- Queued injection ----------------------------------------------------

    def enqueue(
        self,
        category: str,
        text: str,
        priority: int = 5,
        speak: bool = True,
        turn_seq: int = 0,
    ) -> None:
        """Queue a context injection.  Always queues; never sends immediately."""
        self._queue[category] = _QueuedItem(
            category=category,
            text=text,
            priority=priority,
            speak=speak,
            turn_seq=turn_seq,
        )
        logger.info(
            "Queued [%s] (priority=%d, speak=%s, queue_size=%d, state=%s)",
            category,
            priority,
            speak,
            len(self._queue),
            self._state.value,
        )
        # Only schedule flush if IDLE and no timer pending
        if self._state == ModelState.IDLE and self._deferred_flush_handle is None:
            self._schedule_deferred_flush()

    def discard_stale(
        self,
        *,
        min_turn_seq: int,
        categories: set[str] | None = None,
    ) -> list[str]:
        """Drop queued items that belong to older user turns.

        This prevents silent control messages or delayed sub-agent results from
        leaking into a newer user request and causing cross-turn carryover.
        """
        dropped: list[str] = []
        kept: dict[str, _QueuedItem] = {}
        for category, item in self._queue.items():
            should_drop = item.turn_seq < min_turn_seq
            if categories is not None:
                should_drop = should_drop and category in categories
            if should_drop:
                dropped.append(category)
                continue
            kept[category] = item

        if not dropped:
            return []

        self._queue = kept
        if not self._queue:
            self._cancel_deferred_flush()
        logger.info(
            "Discarded %d stale queued item(s) for turn %d: %s",
            len(dropped),
            min_turn_seq,
            ", ".join(sorted(dropped)),
        )
        return dropped

    # -- Deferred flush (batching window) ------------------------------------

    def _schedule_deferred_flush(self, delay: float = BATCH_WINDOW_SEC):
        self._cancel_deferred_flush()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("No running event loop; flushing immediately")
            self.flush()
            return
        self._deferred_flush_handle = loop.call_later(delay, self._deferred_flush_callback)
        logger.debug("Scheduled deferred flush in %.1fs", delay)

    def _deferred_flush_callback(self):
        self._deferred_flush_handle = None
        if self._state != ModelState.IDLE:
            logger.debug("Deferred flush skipped — state=%s", self._state.value)
            return
        if self._queue:
            logger.info("Deferred flush firing (%d items)", len(self._queue))
            self.flush()

    def _cancel_deferred_flush(self):
        if self._deferred_flush_handle is not None:
            self._deferred_flush_handle.cancel()
            self._deferred_flush_handle = None

    def schedule_flush_after(self, delay: float):
        """Schedule a flush after a fixed delay (used for post-greeting pause)."""
        self._cancel_deferred_flush()
        self._schedule_deferred_flush(delay=delay)

    def flush_or_defer_first_turn(
        self,
        first_turn_delay: float = 2.5,
        camera_active: bool = False,
    ) -> None:
        """Flush queued items, always deferring slightly to let audio drain.

        When camera is active, use a longer delay (4.0s) to give the user more
        time to speak before vision context is flushed.
        """
        if self._first_turn:
            self._first_turn = False
            self.schedule_flush_after(first_turn_delay)
            logger.info("Post-greeting pause: flush deferred %.1fs", first_turn_delay)
        else:
            delay = 4.0 if camera_active else 2.5
            self.schedule_flush_after(delay)

    # -- Flush ---------------------------------------------------------------

    def flush(self, force: bool = False) -> bool:
        """Merge and send all queued items as one Content message.

        Returns True if anything was flushed.

        When all items are speak=False and force is False, skip send_content()
        to avoid triggering model audio response for silent-only context.
        Items stay in queue; check_max_age() will force-flush after 15s.
        """
        self._cancel_deferred_flush()  # prevent double-flush
        if not self._queue:
            return False

        items = sorted(self._queue.values(), key=lambda it: it.priority)
        all_silent = all(not it.speak for it in items)

        if all_silent and not force:
            logger.info(
                "Skipped silent-only flush (%d items); awaiting speech-worthy item or max-age",
                len(items),
            )
            return False

        self._queue.clear()

        merged_parts: list[str] = []
        for it in items:
            merged_parts.append(it.text)

        # Multi-source fusion hint: when vision+face+OCR arrive together,
        # instruct the model to combine them into one coherent response.
        categories = {it.category for it in items}
        if len(categories) > 1 and not all_silent:
            fusion_hint = (
                "[MULTI-SOURCE UPDATE: Combine the following naturally into "
                "one coherent response. Do not present each source separately.]\n"
            )
            merged_text = fusion_hint + "\n\n".join(merged_parts)
        else:
            merged_text = "\n\n".join(merged_parts)
        if all_silent:
            merged_text = "<<<INTERNAL_CONTEXT>>>\n" + merged_text + "\n<<<END_INTERNAL_CONTEXT>>>"

        content = types.Content(
            parts=[types.Part(text=merged_text)],
            role="user",
        )
        self._lrq.send_content(content)
        if not all_silent:
            # Will produce audio → enter GENERATING
            self._transition_to(ModelState.GENERATING)
        logger.info(
            "Flushed %d queued items (all_silent=%s, state=%s)",
            len(items),
            all_silent,
            self._state.value,
        )
        return True

    def check_max_age(self) -> bool:
        """Force-flush if the oldest item exceeds QUEUE_MAX_AGE_SEC.

        Also handles state timeouts:
        - GENERATING for too long without audio → force to IDLE
        - DRAINING for too long without iOS drain → force to IDLE
        """
        now = time.monotonic()
        elapsed = now - self._state_entered_at

        # State timeout: GENERATING without audio chunks
        if self._state == ModelState.GENERATING and elapsed > self._GENERATING_TIMEOUT_SEC:
            logger.warning("GENERATING timeout (%.1fs) — forcing to IDLE", elapsed)
            self._transition_to(ModelState.IDLE)

        # State timeout: DRAINING without iOS drain confirmation
        if self._state == ModelState.DRAINING and elapsed > self._DRAINING_TIMEOUT_SEC:
            logger.warning("DRAINING timeout (%.1fs) — forcing to IDLE", elapsed)
            self._transition_to(ModelState.IDLE)

        if self._state != ModelState.IDLE:
            return False
        if not self._queue:
            return False
        oldest = min(it.enqueued_at for it in self._queue.values())
        if (now - oldest) > QUEUE_MAX_AGE_SEC:
            logger.info("Max-age flush triggered (oldest=%.1fs)", now - oldest)
            return self.flush(force=True)
        return False

    # -- Vision spoken cooldown -----------------------------------------------

    def record_vision_spoken(self) -> None:
        """Record that the model just spoke about a vision result."""
        self._vision_spoken_at = time.monotonic()

    @property
    def vision_spoken_cooldown_active(self) -> bool:
        """True if a vision result was spoken recently."""
        return (time.monotonic() - self._vision_spoken_at) < VISION_SPOKEN_COOLDOWN_SEC

    # -- Background flush task ------------------------------------------------

    def start_background_flush_task(self) -> None:
        """Start the periodic max-age checker."""
        if self._bg_task is None:
            self._bg_task = asyncio.create_task(self._background_flush_loop())

    async def _background_flush_loop(self) -> None:
        try:
            while not self._stopped:
                await asyncio.sleep(QUEUE_FLUSH_CHECK_INTERVAL_SEC)
                self.check_max_age()
        except asyncio.CancelledError:
            pass

    def stop(self) -> None:
        """Stop the background flush task."""
        self._stopped = True
        self._cancel_deferred_flush()
        if self._bg_task and not self._bg_task.done():
            self._bg_task.cancel()
