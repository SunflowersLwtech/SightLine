"""SightLine backend server.

FastAPI application with WebSocket endpoint for real-time bidirectional
communication between the iOS client and the Gemini Live API via Google ADK.

Phase 3 additions:
- Vision Sub-Agent (async scene analysis with LOD-adaptive prompting)
- OCR Sub-Agent (async text extraction)
- Face recognition pipeline (InsightFace + Firestore face library)
- Function calling tools (navigation, search, face ID)
- Tool behavior strategy (INTERRUPT / WHEN_IDLE / SILENT)
- Firestore UserProfile loading on session start
"""

import asyncio
import base64
from collections import deque
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.runners import Runner
from google.genai import types
from starlette.websockets import WebSocketState

# ---------------------------------------------------------------------------
# WebSocket message type constants — eliminates raw string literals
# ---------------------------------------------------------------------------


class MessageType(str, Enum):
    """All downstream WebSocket message types sent to the iOS client."""
    TRANSCRIPT = "transcript"
    SESSION_READY = "session_ready"
    SESSION_RESUMPTION = "session_resumption"
    LOD_UPDATE = "lod_update"
    INTERRUPTED = "interrupted"
    GO_AWAY = "go_away"
    ERROR = "error"
    FRAME_ACK = "frame_ack"
    TOOL_EVENT = "tool_event"
    TOOL_RESULT = "tool_result"
    CAPABILITY_DEGRADED = "capability_degraded"
    IDENTITY_UPDATE = "identity_update"
    PERSON_IDENTIFIED = "person_identified"
    VISION_RESULT = "vision_result"
    VISION_DEBUG = "vision_debug"
    OCR_RESULT = "ocr_result"
    OCR_DEBUG = "ocr_debug"
    FACE_DEBUG = "face_debug"
    FACE_LIBRARY_RELOADED = "face_library_reloaded"
    FACE_LIBRARY_CLEARED = "face_library_cleared"
    DEBUG_LOD = "debug_lod"
    DEBUG_ACTIVITY = "debug_activity"
    NAVIGATION_RESULT = "navigation_result"
    SEARCH_RESULT = "search_result"
    PROFILE_UPDATED_ACK = "profile_updated_ack"
    TOOLS_MANIFEST = "tools_manifest"


from agents.orchestrator import create_orchestrator_agent
from live_api.session_manager import (
    SessionManager,
    build_vad_runtime_update_message,
    build_vad_runtime_update_payload,
    create_session_service,
    supports_runtime_vad_reconfiguration,
)
from lod import (
    build_full_dynamic_prompt,
    build_lod_update_message,
    decide_lod,
    on_lod_change,
)
from lod.lod_engine import should_speak
from lod.telemetry_aggregator import TelemetryAggregator
from telemetry.telemetry_parser import parse_telemetry, parse_telemetry_to_ephemeral
from telemetry.session_meta_tracker import SessionMetaTracker
from session_state import SessionState

# ---------------------------------------------------------------------------
# Environment & logging
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).parent / ".env")

# Vertex AI SDK auto-reads GOOGLE_API_KEY / GEMINI_API_KEY from env.
# When VERTEXAI=TRUE, this conflicts with project/location (mutually exclusive).
# Move the API key to a SDK-invisible env var so sub-agents can still read it.
if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").upper() == "TRUE":
    _api_key = os.environ.pop("GOOGLE_API_KEY", "") or os.environ.pop("GEMINI_API_KEY", "")
    os.environ.pop("GEMINI_API_KEY", None)
    os.environ.pop("GOOGLE_API_KEY", None)
    if _api_key:
        os.environ["_GOOGLE_AI_API_KEY"] = _api_key

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sightline.server")

# ---------------------------------------------------------------------------
# App globals
# ---------------------------------------------------------------------------

LIVE_MODEL = os.getenv("GEMINI_LIVE_MODEL", "gemini-live-2.5-flash-native-audio")
PORT = int(os.getenv("PORT", "8100"))
SESSION_TIMEOUT_SEC = int(os.getenv("SESSION_TIMEOUT", "3600"))
WS_INACTIVITY_TIMEOUT_SEC = int(os.getenv("WS_INACTIVITY_TIMEOUT", str(SESSION_TIMEOUT_SEC)))

app = FastAPI(title="SightLine Backend", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

session_service = create_session_service()
session_manager = SessionManager()

# Create the ADK agent and runner once at module level.
agent = create_orchestrator_agent(model_name=LIVE_MODEL)
runner = Runner(
    agent=agent,
    app_name="sightline",
    session_service=session_service,
    auto_create_session=True,
)

# Detect if session service needs ID mapping (Vertex AI generates its own IDs)
_NEEDS_SESSION_ID_MAPPING = type(session_service).__name__ in (
    "VertexAiSessionService", "VertexAISessionService"
)


def _coerce_bool(value: object, default: bool = False) -> bool:
    """Parse bool-like JSON values safely for request handling."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(value, (int, float)):
        return value != 0
    return default


TELEMETRY_FORCE_REFRESH_SEC = 60.0
QUEUE_MAX_AGE_SEC = 15.0
VISION_SPOKEN_COOLDOWN_SEC = 12.0
QUEUE_FLUSH_CHECK_INTERVAL_SEC = 1.0
AGENT_TEXT_REPEAT_SUPPRESS_SEC = 6.0
VISION_REPEAT_SUPPRESS_SEC = 8.0
OCR_REPEAT_SUPPRESS_SEC = 8.0
VISION_PREFEEDBACK_COOLDOWN_SEC = 12.0
OCR_PREFEEDBACK_COOLDOWN_SEC = 15.0
PASSIVE_SPEECH_GUARD_SEC = 8.0
DEFAULT_TOOL_TIMEOUT_SEC = 5.0
STARTUP_HABIT_DETECT_TIMEOUT_SEC = float(os.getenv("STARTUP_HABIT_DETECT_TIMEOUT_SEC", "2.0"))
TOOL_TIMEOUTS_SEC = {
    "navigate_to": 8.0,
    "preview_destination": 10.0,
    "validate_address": 5.0,
    "google_search": 5.0,
    "maps_query": 5.0,
}

# Regex to strip internal context tags before sending to client
_INTERNAL_TAG_RE = re.compile(
    r"\[(?:VISION ANALYSIS|OCR RESULT|FACE RECOGNITION|NAVIGATION|SEARCH RESULT|"
    r"ACCESSIBILITY|MEMORY CONTEXT|ENTITY UPDATE|LOCATION INFO|DEPTH MAP)\]",
    re.IGNORECASE,
)

_MEANINGFUL_TELEMETRY_FIELDS = {
    "motion_state",
    "hr_bucket",
    "noise_bucket",
    "cadence_bucket",
    "heading_bucket",
    "gps_bucket",
    "time_context",
    "device_type",
}


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


# ---------------------------------------------------------------------------
# Token Budget Monitor (E2E-006) — logs context window utilization
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Context Injection Queue — batches non-urgent context to avoid interrupting
# the model mid-speech (clientContent unconditionally interrupts generation).
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field as dc_field


@dataclass
class _QueuedItem:
    """A single queued context injection waiting for the model to finish."""
    category: str
    text: str
    priority: int  # lower = more important
    speak: bool
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

    def __init__(self, live_request_queue: "LiveRequestQueue") -> None:
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
    ) -> None:
        """Queue a context injection.  Always queues; never sends immediately."""
        self._queue[category] = _QueuedItem(
            category=category, text=text, priority=priority, speak=speak,
        )
        logger.info("Queued [%s] (priority=%d, speak=%s, queue_size=%d, state=%s)",
                     category, priority, speak, len(self._queue), self._state.value)
        # Only schedule flush if IDLE and no timer pending
        if self._state == ModelState.IDLE and self._deferred_flush_handle is None:
            self._schedule_deferred_flush()

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

    def flush_or_defer_first_turn(self, first_turn_delay: float = 2.5,
                                   camera_active: bool = False) -> None:
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
            logger.info("Skipped silent-only flush (%d items); awaiting speech-worthy item or max-age",
                         len(items))
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
        logger.info("Flushed %d queued items (all_silent=%s, state=%s)",
                     len(items), all_silent, self._state.value)
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
            logger.info("Max-age flush triggered (oldest=%.1fs)",
                        now - oldest)
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


def _heart_rate_bucket(heart_rate: float | None) -> str:
    if heart_rate is None or heart_rate <= 0:
        return "unknown"
    if heart_rate > 100:
        return "elevated"
    return "normal"


def _noise_bucket(noise_db: float) -> str:
    if noise_db < 40:
        return "quiet"
    if noise_db < 65:
        return "moderate"
    if noise_db < 80:
        return "loud"
    return "very_loud"


def _cadence_bucket(step_cadence: float) -> str:
    if step_cadence <= 0:
        return "still"
    if step_cadence < 60:
        return "slow"
    if step_cadence < 120:
        return "walk"
    return "fast"


def _heading_bucket(heading: float | None) -> int | None:
    if heading is None:
        return None
    return int((heading % 360) // 30)


def _gps_bucket(gps) -> tuple[float, float] | None:
    if gps is None:
        return None
    try:
        return (round(float(gps.lat), 3), round(float(gps.lng), 3))
    except (TypeError, ValueError, AttributeError):
        return None


def _build_telemetry_signature(ephemeral_ctx) -> dict[str, object]:
    """Build coarse signature to detect meaningful telemetry changes."""
    heading_value = getattr(ephemeral_ctx, "heading", None)
    heading_bucket = _heading_bucket(heading_value if heading_value not in (None, 0.0) else None)
    return {
        "motion_state": getattr(ephemeral_ctx, "motion_state", "unknown"),
        "hr_bucket": _heart_rate_bucket(getattr(ephemeral_ctx, "heart_rate", None)),
        "noise_bucket": _noise_bucket(float(getattr(ephemeral_ctx, "ambient_noise_db", 50.0) or 50.0)),
        "cadence_bucket": _cadence_bucket(float(getattr(ephemeral_ctx, "step_cadence", 0.0) or 0.0)),
        "heading_bucket": heading_bucket,
        "gps_bucket": _gps_bucket(getattr(ephemeral_ctx, "gps", None)),
        "time_context": getattr(ephemeral_ctx, "time_context", "unknown"),
        "device_type": getattr(ephemeral_ctx, "device_type", "phone_only"),
    }


def _changed_signature_fields(
    previous_signature: dict[str, object] | None,
    current_signature: dict[str, object],
) -> list[str]:
    if previous_signature is None:
        return ["initial"]
    changed: list[str] = []
    for key, value in current_signature.items():
        if previous_signature.get(key) != value:
            changed.append(key)
    return changed


def _should_inject_telemetry_context(
    *,
    previous_signature: dict[str, object] | None,
    current_signature: dict[str, object],
    last_injected_ts: float,
    now_ts: float,
    force_refresh_sec: float = TELEMETRY_FORCE_REFRESH_SEC,
) -> tuple[bool, list[str]]:
    """Decide if telemetry context should be injected into the model."""
    changed = _changed_signature_fields(previous_signature, current_signature)
    if previous_signature is None:
        return True, changed

    meaningful_change = [field for field in changed if field in _MEANINGFUL_TELEMETRY_FIELDS]
    if meaningful_change:
        return True, meaningful_change

    if now_ts - last_injected_ts >= force_refresh_sec:
        return True, ["periodic_refresh"]

    return False, changed

# ---------------------------------------------------------------------------
# Sub-agent & tool imports (lazy to handle missing deps gracefully)
# ---------------------------------------------------------------------------

_vision_available = False
_ocr_available = False
_face_available = False

try:
    from agents.vision_agent import analyze_scene
    _vision_available = True
except ImportError:
    logger.warning("Vision agent not available (missing dependencies)")

try:
    from agents.ocr_agent import extract_text
    _ocr_available = True
except ImportError:
    logger.warning("OCR agent not available (missing dependencies)")

try:
    from agents.face_agent import identify_persons_in_frame
    from tools.face_tools import load_face_library
    _face_available = True
except ImportError:
    logger.warning("Face agent not available (missing dependencies)")

FACE_LIBRARY_REFRESH_SEC: float = 60.0

from tools import ALL_FUNCTIONS, ALL_TOOL_DECLARATIONS
from tools.navigation import NAVIGATION_FUNCTIONS
from tools.search import SEARCH_FUNCTIONS
from tools.plus_codes import PLUS_CODES_FUNCTIONS
from tools.accessibility import ACCESSIBILITY_FUNCTIONS
from tools.maps_grounding import MAPS_GROUNDING_FUNCTIONS
from tools.ocr_tool import set_latest_frame as _ocr_set_latest_frame, clear_session as _ocr_clear_session
from memory.memory_tools import MEMORY_FUNCTIONS
from tools.tool_behavior import ToolBehavior, behavior_to_text, resolve_tool_behavior

# ---------------------------------------------------------------------------
# Tool category / behavior mapping for tools_manifest
# ---------------------------------------------------------------------------

_TOOL_CATEGORY_MAP: dict[str, tuple[str, str]] = {
    "navigate_to": ("navigation", "INTERRUPT"),
    "get_location_info": ("navigation", "WHEN_IDLE"),
    "nearby_search": ("navigation", "WHEN_IDLE"),
    "reverse_geocode": ("navigation", "WHEN_IDLE"),
    "get_walking_directions": ("navigation", "WHEN_IDLE"),
    "preview_destination": ("navigation", "WHEN_IDLE"),
    "validate_address": ("navigation", "WHEN_IDLE"),
    "google_search": ("search", "WHEN_IDLE"),
    "identify_person": ("face", "SILENT"),
    "resolve_plus_code": ("plus_codes", "WHEN_IDLE"),
    "convert_to_plus_code": ("plus_codes", "WHEN_IDLE"),
    "get_accessibility_info": ("accessibility", "WHEN_IDLE"),
    "maps_query": ("maps_grounding", "WHEN_IDLE"),
    "preload_memory": ("memory", "SILENT"),
    "remember_entity": ("memory", "SILENT"),
    "what_do_you_remember": ("memory", "WHEN_IDLE"),
    "forget_entity": ("memory", "SILENT"),
    "forget_recent_memory": ("memory", "SILENT"),
    "extract_text_from_camera": ("ocr", "WHEN_IDLE"),
}

# ---------------------------------------------------------------------------
# Memory system (Phase 4, SL-71)
# ---------------------------------------------------------------------------

_memory_available = False
_memory_extractor_available = False
try:
    from memory.memory_bank import load_relevant_memories, MemoryBankService, evict_stale_banks
    from memory.memory_budget import MemoryBudgetTracker, MEMORY_WRITE_BUDGET
    _memory_available = True
except ImportError:
    logger.warning("Memory module not available")

    def load_relevant_memories(user_id: str, context: str, top_k: int = 3) -> list[str]:
        return []

try:
    from memory.memory_extractor import MemoryExtractor
    _memory_extractor_available = True
except ImportError:
    logger.warning("Memory extractor not available")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    """Health check endpoint for Cloud Run readiness probes."""
    return {
        "status": "ok",
        "model": LIVE_MODEL,
        "phase": 6,
        "capabilities": {
            "vision": _vision_available,
            "ocr": _ocr_available,
            "face": _face_available,
            "plus_codes": True,
            "elevation": True,
            "street_view": True,
            "address_validation": True,
            "accessibility": True,
            "maps_grounding": bool(os.getenv("GOOGLE_MAPS_API_KEY")),
            "weather": True,
            "haptics": True,
            "depth": True,
        },
    }


# ---------------------------------------------------------------------------
# REST API — Face Registration (Phase 5, SL-P2-①)
# ---------------------------------------------------------------------------


@app.post("/api/face/register")
async def api_register_face(request: Request) -> JSONResponse:
    """Register a face via REST (for iOS FaceRegistrationView).

    Body JSON:
        user_id: str
        person_name: str
        relationship: str
        image_base64: str  (JPEG base64-encoded)
        photo_index: int (optional, default 0)
        consent_confirmed: bool (optional, default false)
        store_reference_photo: bool (optional, default false)

    Returns the face_id and metadata on success.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    user_id = body.get("user_id")
    person_name = body.get("person_name")
    relationship = body.get("relationship", "")
    image_base64 = body.get("image_base64")
    photo_index = body.get("photo_index", 0)
    consent_confirmed = _coerce_bool(body.get("consent_confirmed"), default=False)
    store_reference_photo = _coerce_bool(body.get("store_reference_photo"), default=False)

    if not all([user_id, person_name, image_base64]):
        return JSONResponse(
            {"error": "Missing required fields: user_id, person_name, image_base64"},
            status_code=400,
        )

    if store_reference_photo and not consent_confirmed:
        return JSONResponse(
            {"error": "consent_confirmed must be true when store_reference_photo is enabled"},
            status_code=400,
        )

    if not _face_available:
        return JSONResponse(
            {"error": "Face recognition is not available on this server"},
            status_code=503,
        )

    try:
        from tools.face_tools import register_face
        result = await asyncio.to_thread(
            register_face,
            user_id=user_id,
            person_name=person_name,
            relationship=relationship,
            image_base64=image_base64,
            photo_index=photo_index,
            consent_confirmed=consent_confirmed,
            store_reference_photo=store_reference_photo,
        )
        logger.info("REST face register: %s for user %s", result.get("face_id"), user_id)
        return JSONResponse(result, status_code=201)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    except Exception as e:
        logger.exception("Face registration failed")
        return JSONResponse({"error": f"Registration failed: {str(e)}"}, status_code=500)


@app.get("/api/face/list/{user_id}")
async def api_list_faces(user_id: str) -> JSONResponse:
    """List all registered faces for a user (without embeddings)."""
    if not _face_available:
        return JSONResponse({"error": "Face recognition not available"}, status_code=503)

    try:
        from tools.face_tools import list_faces
        faces = await asyncio.to_thread(list_faces, user_id)
        return JSONResponse({"faces": faces, "count": len(faces)})
    except Exception as e:
        logger.exception("List faces failed for user %s", user_id)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.delete("/api/face/{user_id}/{face_id}")
async def api_delete_face(user_id: str, face_id: str) -> JSONResponse:
    """Delete a single face entry from the library."""
    if not _face_available:
        return JSONResponse({"error": "Face recognition not available"}, status_code=503)

    try:
        from tools.face_tools import delete_face
        deleted = await asyncio.to_thread(delete_face, user_id, face_id)
        if deleted:
            return JSONResponse({"status": "deleted", "face_id": face_id})
        return JSONResponse({"error": "Face not found"}, status_code=404)
    except Exception as e:
        logger.exception("Delete face failed")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.delete("/api/face/{user_id}")
async def api_clear_face_library(user_id: str, request: Request) -> JSONResponse:
    """Clear all faces or delete a specific person from the user's library.

    Query params:
        person_name: Optional. If provided, only delete that person's entries.
    """
    if not _face_available:
        return JSONResponse({"error": "Face recognition not available"}, status_code=503)

    person_name = request.query_params.get("person_name")

    try:
        from tools.face_tools import delete_all_faces
        count = await asyncio.to_thread(delete_all_faces, user_id, person_name)
        if person_name:
            return JSONResponse({
                "status": "deleted",
                "person_name": person_name,
                "deleted_count": count,
            })
        return JSONResponse({"status": "cleared", "deleted_count": count})
    except Exception as e:
        logger.exception("Clear face library failed")
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# REST API — User Profile (Phase 5, SL-P2-③)
# ---------------------------------------------------------------------------


@app.get("/api/profile/{user_id}")
async def api_get_profile(user_id: str) -> JSONResponse:
    """Get the UserProfile from Firestore."""
    try:
        from google.cloud import firestore as _fs
        db = _fs.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT", "sightline-hackathon"))
        doc = db.collection("user_profiles").document(user_id).get()
        if not doc.exists:
            return JSONResponse({"error": "Profile not found"}, status_code=404)
        data = doc.to_dict()
        # Convert timestamps to ISO strings
        for key in ("created_at", "updated_at"):
            if key in data and hasattr(data[key], "isoformat"):
                data[key] = data[key].isoformat()
        return JSONResponse(data)
    except Exception as e:
        logger.exception("Get profile failed for %s", user_id)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/profile/{user_id}")
async def api_save_profile(user_id: str, request: Request) -> JSONResponse:
    """Create or update a UserProfile in Firestore.

    Body JSON — any of:
        vision_status: str (totally_blind / low_vision)
        blindness_onset: str (congenital / acquired)
        onset_age: int | null
        has_guide_dog: bool
        has_white_cane: bool
        tts_speed: float
        verbosity_preference: str (concise / detailed)
        language: str
        description_priority: str (spatial / object)
        color_description: bool
        om_level: str (beginner / intermediate / advanced)
        travel_frequency: str (daily / weekly / rarely)
        preferred_name: str
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    ALLOWED_FIELDS = {
        "vision_status", "blindness_onset", "onset_age",
        "has_guide_dog", "has_white_cane", "tts_speed",
        "verbosity_preference", "language", "description_priority",
        "color_description", "om_level", "travel_frequency", "preferred_name",
    }
    filtered = {k: v for k, v in body.items() if k in ALLOWED_FIELDS}
    if not filtered:
        return JSONResponse({"error": "No valid fields provided"}, status_code=400)

    try:
        from google.cloud import firestore as _fs
        db = _fs.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT", "sightline-hackathon"))
        doc_ref = db.collection("user_profiles").document(user_id)
        filtered["updated_at"] = _fs.SERVER_TIMESTAMP
        # Merge so we don't overwrite fields not included in this request
        doc_ref.set(filtered, merge=True)
        session_manager.invalidate_user_profile(user_id)
        logger.info("REST profile save for user %s: %s (cache invalidated)", user_id, list(filtered.keys()))
        return JSONResponse({"status": "saved", "user_id": user_id, "fields": list(filtered.keys())})
    except Exception as e:
        logger.exception("Save profile failed for %s", user_id)
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# User list endpoint (for demo user switching)
# ---------------------------------------------------------------------------


@app.get("/api/users")
async def api_list_users() -> JSONResponse:
    """List all user IDs from Firestore."""
    try:
        from google.cloud import firestore as _fs
        db = _fs.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT", "sightline-hackathon"))
        docs = db.collection("user_profiles").stream()
        user_ids = sorted(doc.id for doc in docs)
        return JSONResponse({"users": user_ids, "count": len(user_ids)})
    except Exception as e:
        logger.exception("List users failed")
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# Voice intent detection (detail / stop flags for LOD engine)
# ---------------------------------------------------------------------------

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


_LOCATION_QUERY_PHRASES = {
    "around me", "around here", "nearby", "near me", "near here",
    "what's here", "what is here", "where am i", "where are we",
    "what's around", "what is around", "what's close", "find",
    "search for", "is there a", "any", "closest", "look up",
}


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
    from tools.navigation import ACTIVE_NAVIGATION_TOOLS, LOCATION_QUERY_TOOLS

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


def _sanitize_function_args_for_log(func_name: str, func_args: dict, user_id: str) -> dict:
    """Sanitize function-call logging to avoid leaking/echoing forged user IDs."""
    safe_args = dict(func_args)
    if func_name in MEMORY_FUNCTIONS:
        if "user_id" in safe_args:
            safe_args["user_id"] = "<session_user>"
        safe_args["_session_user"] = user_id
    return safe_args


# ---------------------------------------------------------------------------
# Function calling dispatcher
# ---------------------------------------------------------------------------


def _json_safe(value):
    """Best-effort conversion for JSON payloads sent over WebSocket."""
    try:
        json.dumps(value)
        return value
    except TypeError:
        return json.loads(json.dumps(value, default=str))


# ---------------------------------------------------------------------------
# Tool result truncation (E2E-006)
# ---------------------------------------------------------------------------

_MAX_TOOL_RESULT_CHARS = 4000


def _truncate_tool_result(result: dict, max_chars: int = _MAX_TOOL_RESULT_CHARS) -> dict:
    """Truncate oversized string values in tool results to prevent token overflow."""
    truncated = {}
    for k, v in result.items():
        if isinstance(v, str) and len(v) > max_chars:
            truncated[k] = v[:max_chars] + "\u2026 [truncated]"
        elif isinstance(v, dict):
            truncated[k] = _truncate_tool_result(v, max_chars)
        elif isinstance(v, list):
            truncated[k] = [
                _truncate_tool_result(item, max_chars) if isinstance(item, dict) else item
                for item in v
            ]
        else:
            truncated[k] = v
    return truncated


def _extract_function_calls(event) -> list:
    """Extract function calls from ADK event objects across SDK schema changes."""
    getter = getattr(event, "get_function_calls", None)
    if callable(getter):
        try:
            calls = getter() or []
            if calls:
                return list(calls)
        except Exception:
            logger.debug("event.get_function_calls() failed; trying legacy access path", exc_info=True)

    # Legacy fallback (older assumptions in downstream loop).
    actions = getattr(event, "actions", None)
    if not actions:
        return []
    legacy_calls = getattr(actions, "function_calls", None)
    if not legacy_calls:
        return []
    return list(legacy_calls)


async def _dispatch_function_call(
    func_name: str,
    func_args: dict,
    session_id: str,
    user_id: str,
) -> dict:
    """Dispatch a function call from Gemini to the appropriate tool.

    Uses the unified ALL_FUNCTIONS dict for dispatch.  Navigation tools
    get automatic GPS/heading injection from ephemeral context.

    Returns the tool result as a dict to be sent back as function response.
    """
    if func_name not in ALL_FUNCTIONS:
        logger.warning("Unknown function call: %s (should have been caught upstream)", func_name)
        return {
            "status": "unavailable",
            "message": f"'{func_name}' does not exist. Use only the tools listed in your instructions.",
        }

    # Navigation tools: inject current GPS/heading from ephemeral context
    if func_name in NAVIGATION_FUNCTIONS:
        ephemeral = session_manager.get_ephemeral_context(session_id)
        if func_name == "navigate_to" and ephemeral.gps:
            func_args.setdefault("origin_lat", ephemeral.gps.lat)
            func_args.setdefault("origin_lng", ephemeral.gps.lng)
            func_args.setdefault("user_heading", ephemeral.heading)
        elif func_name in ("get_location_info", "nearby_search", "reverse_geocode") and ephemeral.gps:
            func_args.setdefault("lat", ephemeral.gps.lat)
            func_args.setdefault("lng", ephemeral.gps.lng)

    # Plus Codes: inject GPS for convert_to_plus_code
    if func_name in PLUS_CODES_FUNCTIONS:
        ephemeral = session_manager.get_ephemeral_context(session_id)
        if func_name == "convert_to_plus_code" and ephemeral.gps:
            func_args.setdefault("lat", ephemeral.gps.lat)
            func_args.setdefault("lng", ephemeral.gps.lng)

    # preview_destination: inject GPS if not explicitly provided
    if func_name == "preview_destination":
        ephemeral = session_manager.get_ephemeral_context(session_id)
        if ephemeral.gps:
            func_args.setdefault("lat", ephemeral.gps.lat)
            func_args.setdefault("lng", ephemeral.gps.lng)

    # Accessibility: inject GPS
    if func_name in ACCESSIBILITY_FUNCTIONS:
        ephemeral = session_manager.get_ephemeral_context(session_id)
        if ephemeral.gps:
            func_args.setdefault("lat", ephemeral.gps.lat)
            func_args.setdefault("lng", ephemeral.gps.lng)

    # Maps grounding: inject GPS
    if func_name in MAPS_GROUNDING_FUNCTIONS:
        ephemeral = session_manager.get_ephemeral_context(session_id)
        if ephemeral.gps:
            func_args.setdefault("lat", ephemeral.gps.lat)
            func_args.setdefault("lng", ephemeral.gps.lng)

    # Memory tools: hard-set user_id from session (security: prevents cross-user access)
    if func_name in MEMORY_FUNCTIONS:
        func_args["user_id"] = user_id

    logger.info(
        "Function call: %s(%s)",
        func_name,
        _sanitize_function_args_for_log(func_name, func_args, user_id),
    )

    # OCR tool: intercept and run async OCR pipeline with latest camera frame
    if func_name == "extract_text_from_camera":
        from tools.ocr_tool import _latest_frames
        frame_b64 = _latest_frames.get(session_id)
        if not frame_b64:
            return {
                "text": "",
                "text_type": "unknown",
                "items": [],
                "confidence": 0.0,
                "message": "No camera frame available. The camera may not be active.",
            }
        try:
            from agents.ocr_agent import extract_text as _ocr_extract
            hint = func_args.get("context_hint", "")
            result = await _ocr_extract(frame_b64, context_hint=hint, safety_only=False)
            return result
        except Exception:
            logger.exception("OCR tool dispatch failed")
            return {
                "text": "",
                "text_type": "unknown",
                "items": [],
                "confidence": 0.0,
                "error": "OCR extraction failed",
            }

    try:
        timeout_sec = TOOL_TIMEOUTS_SEC.get(func_name, DEFAULT_TOOL_TIMEOUT_SEC)
        raw_result = await asyncio.wait_for(
            asyncio.to_thread(ALL_FUNCTIONS[func_name], **func_args),
            timeout=timeout_sec,
        )
        return _truncate_tool_result(raw_result) if isinstance(raw_result, dict) else raw_result
    except TimeoutError:
        logger.warning("Tool %s timed out after %.1fs", func_name, TOOL_TIMEOUTS_SEC.get(func_name, DEFAULT_TOOL_TIMEOUT_SEC))
        return {
            "error": "tool_timeout",
            "tool": func_name,
            "message": f"The {func_name} tool timed out. Please try again.",
            "retryable": True,
        }
    except Exception:
        logger.exception("Tool %s raised an exception", func_name)
        return {
            "error": "tool_execution_failed",
            "tool": func_name,
            "message": f"The {func_name} tool encountered an internal error. Try again or use a different approach.",
        }


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@app.websocket("/ws/{user_id}/{session_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str, session_id: str) -> None:
    """Main WebSocket endpoint for bidirectional audio/vision streaming.

    Manages the lifecycle of a Gemini Live API session through the ADK runner,
    forwarding upstream messages from the iOS client and downstream events
    from the model.

    Phase 3: Integrates sub-agents (vision, OCR, face), function calling
    tools (navigation, search), and Firestore UserProfile loading.
    """
    await websocket.accept()
    raw_session_id = session_id
    session_id = session_id.strip().lower()
    if session_id != raw_session_id:
        logger.info(
            "Normalized session id for backend compatibility: %s -> %s",
            raw_session_id,
            session_id,
        )
    logger.info("WebSocket connected: user=%s session=%s", user_id, session_id)

    stop_downstream = asyncio.Event()
    resume_handle = (websocket.query_params.get("resume_handle") or "").strip()
    if resume_handle:
        session_manager.update_handle(session_id, resume_handle)
        logger.info("Received resume handle from client for session %s", session_id)

    # -- Per-session LOD state -----------------------------------------------
    telemetry_agg = TelemetryAggregator()
    session_ctx = session_manager.get_session_context(session_id)
    user_profile_task = asyncio.create_task(session_manager.load_user_profile(user_id))
    initial_memories_task = None
    if _memory_available:
        initial_memories_task = asyncio.create_task(asyncio.to_thread(
            load_relevant_memories,
            user_id,
            session_ctx.active_task or session_ctx.trip_purpose or "",
            3,
        ))
    _initial_face_library_task = None
    if _face_available:
        _initial_face_library_task = asyncio.create_task(asyncio.to_thread(load_face_library, user_id))

    user_profile = await user_profile_task
    session_meta = SessionMetaTracker(user_id=user_id, session_id=session_id)

    # -- Per-session mutable state (replaces nonlocal variables) -------------
    state = SessionState(
        face_runtime_available=_face_available,
        face_library_task=_initial_face_library_task,
    )

    # -- Dependencies for handler ---------------------------------------------
    from websocket_handler import WebSocketHandler

    live_request_queue = LiveRequestQueue()
    ctx_queue = ContextInjectionQueue(live_request_queue)
    ctx_queue.start_background_flush_task()
    token_monitor = TokenBudgetMonitor()

    # Tool call deduplication (E2E-002 / E2E-003)
    from tools.dedup import ToolCallDeduplicator, MutualExclusionFilter, AudioGate
    tool_dedup = ToolCallDeduplicator()
    tool_mutex = MutualExclusionFilter()
    audio_gate = AudioGate()

    run_config = session_manager.get_run_config(
        session_id, lod=session_ctx.current_lod,
        language_code=user_profile.language,
    )

    # -- Phase 6: Context Engine initialisation ---------------------------------
    _location_ctx_service = None
    _lod_evaluator = None
    _assembled_profile = None
    try:
        from context.location_context import LocationContextService
        from context.lod_evaluator import LODEvaluator
        from context.profile_assembler import ProfileAssembler
        from context.habit_detector import HabitDetector

        _location_ctx_service = LocationContextService(user_id)
        _lod_evaluator = LODEvaluator()

        state.proactive_hints = []

        async def _load_proactive_hints_background() -> None:
            try:
                hints = await asyncio.wait_for(
                    asyncio.to_thread(HabitDetector(user_id).detect),
                    timeout=STARTUP_HABIT_DETECT_TIMEOUT_SEC,
                )
                state.proactive_hints = hints or []
                if state.proactive_hints:
                    logger.info(
                        "Detected %d habits for user=%s (top: %s)",
                        len(state.proactive_hints), user_id,
                        state.proactive_hints[0].description[:60],
                    )
            except TimeoutError:
                logger.warning(
                    "Habit detection timed out after %.1fs for user %s",
                    STARTUP_HABIT_DETECT_TIMEOUT_SEC,
                    user_id,
                )
            except Exception:
                logger.debug("Habit detection skipped", exc_info=True)

        _assembler = ProfileAssembler()
        _assembled_profile = _assembler.assemble(
            profile=user_profile,
            location_ctx=None,
            entities=None,
            memories=None,
        )
        asyncio.create_task(_load_proactive_hints_background())
    except Exception:
        logger.debug("Context engine init skipped (import error)", exc_info=True)

    # Resolve initial memories before handing off to handler
    initial_memories = await initial_memories_task if initial_memories_task is not None else []
    memory_budget = MemoryBudgetTracker() if _memory_available else None

    # -- Hand off to WebSocketHandler ------------------------------------------
    handler = WebSocketHandler(
        websocket=websocket,
        user_id=user_id,
        session_id=session_id,
        state=state,
        live_request_queue=live_request_queue,
        runner=runner,
        ctx_queue=ctx_queue,
        token_monitor=token_monitor,
        session_ctx=session_ctx,
        session_meta=session_meta,
        user_profile=user_profile,
        telemetry_agg=telemetry_agg,
        stop_downstream=stop_downstream,
        tool_dedup=tool_dedup,
        tool_mutex=tool_mutex,
        audio_gate=audio_gate,
        run_config=run_config,
        location_ctx_service=_location_ctx_service,
        lod_evaluator=_lod_evaluator,
        assembled_profile=_assembled_profile,
        memory_budget=memory_budget,
        initial_memories=initial_memories,
    )
    await handler.run()


# ---------------------------------------------------------------------------
# Sub-agent result formatters
# ---------------------------------------------------------------------------


def _format_vision_result(result: dict, lod: int) -> str:
    """Format vision analysis result for Gemini context injection."""
    parts = ["[VISION ANALYSIS]"]

    # Safety warnings always first
    warnings = result.get("safety_warnings", [])
    for w in warnings:
        parts.append(f"SAFETY: {w}")

    # Navigation info as compact spatial summary
    nav = result.get("navigation_info", {})
    if lod >= 2:
        nav_items = []
        for key in ("entrances", "paths", "landmarks"):
            items = nav.get(key, [])
            if items:
                nav_items.extend(items)
        if nav_items:
            parts.append("Spatial: " + " | ".join(nav_items))

    desc = result.get("scene_description", "")
    if desc:
        parts.append(f"Scene: {desc}")

    text = result.get("detected_text")
    if text and lod >= 2:
        parts.append(f"Text spotted: {text}")

    count = result.get("people_count", 0)
    if count > 0 and lod >= 2:
        if count == 1:
            parts.append("1 person nearby")
        else:
            parts.append(f"{count} people nearby")

    return "\n".join(parts)


def _format_face_results(known_faces: list[dict]) -> str:
    """Format face recognition results for SILENT context injection."""
    parts = ["[FACE ID]"]
    for face in known_faces:
        name = face["person_name"]
        rel = face.get("relationship", "")
        sim = face.get("similarity", 0)
        desc = f"{name}"
        if rel:
            desc += f" ({rel})"
        if sim >= 0.85:
            desc += " — high confidence"
        elif sim >= 0.70:
            desc += " — moderate confidence, verify if possible"
        else:
            desc += " — low confidence, do not announce unless user asks"
        parts.append(desc)
    return "\n".join(parts)


def _format_ocr_result(result: dict) -> str:
    """Format OCR result for Gemini context injection."""
    parts = ["[OCR RESULT]"]

    text_type = result.get("text_type", "unknown")
    confidence = result.get("confidence", 1.0)

    # Context-aware type labels
    type_hints = {
        "menu": "Menu text detected — read items with prices:",
        "sign": "Sign text detected:",
        "document": "Document text detected:",
        "label": "Label text detected:",
    }
    parts.append(type_hints.get(text_type, f"Text detected ({text_type}):"))

    if confidence < 0.5:
        parts.append("(Note: text quality is poor, some characters may be inaccurate)")

    items = result.get("items", [])
    if items:
        for item in items:
            parts.append(f"  - {item}")
    else:
        text = result.get("text", "")
        if text:
            parts.append(text)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
