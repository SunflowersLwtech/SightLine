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
VISION_SPOKEN_COOLDOWN_SEC = 25.0
QUEUE_FLUSH_CHECK_INTERVAL_SEC = 1.0
AGENT_TEXT_REPEAT_SUPPRESS_SEC = 14.0
VISION_REPEAT_SUPPRESS_SEC = 18.0
OCR_REPEAT_SUPPRESS_SEC = 20.0
VISION_PREFEEDBACK_COOLDOWN_SEC = 12.0
OCR_PREFEEDBACK_COOLDOWN_SEC = 15.0

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

    BATCH_WINDOW_SEC = 0.4  # 400ms collection window for sub-agent results

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

    def inject_immediate(self, content: "types.Content") -> None:
        """Send directly to LiveRequestQueue, bypassing the queue."""
        self._lrq.send_content(content)
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

        merged_text = "\n\n".join(merged_parts)
        if all_silent:
            merged_text = "[CONTEXT UPDATE - DO NOT SPEAK]\n" + merged_text

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
    logger.info("Function call: %s(%s)", func_name, func_args)

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
        return await asyncio.to_thread(ALL_FUNCTIONS[func_name], **func_args)
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
    user_profile = await session_manager.load_user_profile(user_id)
    session_meta = SessionMetaTracker(user_id=user_id, session_id=session_id)

    # -- Per-session face library cache (Phase 3) ----------------------------
    face_library: list[dict] = []
    _face_library_loaded_at: float = 0.0
    if _face_available:
        try:
            face_library = load_face_library(user_id)
            _face_library_loaded_at = time.monotonic()
            logger.info("Loaded %d face(s) for user %s", len(face_library), user_id)
        except Exception:
            logger.exception("Failed to load face library for user %s", user_id)

    # -- WebSocket write lock (prevent interleaved frames from concurrent coroutines)
    _ws_write_lock = asyncio.Lock()

    # -- Vision analysis state -----------------------------------------------
    _vision_lock = asyncio.Lock()
    _vision_in_progress = False
    _last_vision_time = 0.0
    _frame_seq = 0
    _last_vision_context_text = ""
    _last_vision_context_sent_at = 0.0
    _last_vision_prefeedback_at = 0.0
    _last_ocr_context_text = ""
    _last_ocr_context_sent_at = 0.0
    _last_ocr_prefeedback_at = 0.0
    _last_telemetry_signature: dict[str, object] | None = None
    _last_telemetry_context_sent_at = 0.0
    _last_agent_text = ""
    _last_agent_text_sent_at = 0.0
    _allow_agent_repeat_until = 0.0

    # Echo detection state (P0_FIX_3)
    _recent_agent_texts: list[tuple[float, str]] = []

    _model_audio_last_seen_at: float = 0.0

    # Interrupt state — shared between _upstream (client barge-in) and
    # _downstream (Gemini-side interrupted events).
    _is_interrupted: bool = False
    _last_interrupt_at: float = 0.0
    _INTERRUPT_DEBOUNCE_SEC = 1.0

    # Adaptive face detection: skip cycles when no faces are consistently detected
    _face_consecutive_misses: int = 0
    _FACE_BACKOFF_THRESHOLD = 3       # consecutive 0-face before slowing
    _FACE_BACKOFF_SKIP_CYCLES = 2     # skip N cycles after threshold hit
    _face_skip_counter: int = 0

    # User activity tracking (for telemetry / future use)
    _last_user_activity_at: float = time.monotonic()

    # Vision spoken tracking for context injection queue
    _turn_had_vision_content = False

    # Output transcription buffering (avoid flooding client with fragments)
    _transcript_buffer: str = ""
    _transcript_buffer_last_update: float = 0.0
    _TRANSCRIPT_FLUSH_TIMEOUT_SEC = 1.5

    _SENTENCE_BOUNDARY_RE = re.compile(r"[。！？.!?\n]")

    def _has_sentence_boundary(text: str) -> bool:
        """Return True if text contains a sentence-ending punctuation mark."""
        return bool(_SENTENCE_BOUNDARY_RE.search(text))

    async def _flush_transcript_buffer() -> bool:
        """Flush the accumulated transcript buffer to the client."""
        nonlocal _transcript_buffer, _transcript_buffer_last_update
        text = _transcript_buffer.strip()
        _transcript_buffer = ""
        _transcript_buffer_last_update = 0.0
        if not text:
            return True
        # Track for echo detection
        now_mono = time.monotonic()
        _recent_agent_texts.append((now_mono, text))
        cutoff = now_mono - 10.0
        while _recent_agent_texts and _recent_agent_texts[0][0] < cutoff:
            _recent_agent_texts.pop(0)
        return await _forward_agent_transcript(text)

    def _is_websocket_open() -> bool:
        return (
            websocket.client_state == WebSocketState.CONNECTED
            and websocket.application_state == WebSocketState.CONNECTED
        )

    async def _safe_send_json(payload: dict) -> bool:
        if not _is_websocket_open():
            stop_downstream.set()
            return False
        try:
            async with _ws_write_lock:
                await websocket.send_json(payload)
            return True
        except (WebSocketDisconnect, RuntimeError):
            stop_downstream.set()
            return False

    async def _safe_send_bytes(payload: bytes) -> bool:
        if not _is_websocket_open():
            stop_downstream.set()
            return False
        try:
            async with _ws_write_lock:
                await websocket.send_bytes(payload)
            return True
        except (WebSocketDisconnect, RuntimeError):
            stop_downstream.set()
            return False

    async def _forward_agent_transcript(text: str) -> bool:
        """Forward agent transcript with short-window duplicate suppression."""
        nonlocal _last_agent_text, _last_agent_text_sent_at
        now_mono = time.monotonic()
        can_repeat = now_mono <= _allow_agent_repeat_until
        is_repeat = _is_repeated_text(
            text,
            previous_text=_last_agent_text,
            now_ts=now_mono,
            previous_ts=_last_agent_text_sent_at,
            cooldown_sec=AGENT_TEXT_REPEAT_SUPPRESS_SEC,
        )
        if is_repeat and not can_repeat:
            logger.debug("Suppressed repeated downstream transcript: %s", text[:120])
            return True
        sent = await _safe_send_json({
            "type": MessageType.TRANSCRIPT,
            "text": text,
            "role": "agent",
        })
        if sent:
            _last_agent_text = text
            _last_agent_text_sent_at = now_mono
        return sent

    def _is_likely_echo(candidate: str, now_ts: float) -> bool:
        """Check if candidate text is likely an echo of recent agent output.

        When the model was recently speaking (within 3s), uses relaxed
        thresholds (1 word min, Jaccard >0.35, 8s window) to catch echoes
        more aggressively. Otherwise uses normal thresholds (3 words min,
        Jaccard >0.6, 5s window).
        """
        words_candidate = set(candidate.lower().split())
        # Relaxed mode when model was speaking recently
        model_speaking = (now_ts - _model_audio_last_seen_at) < 3.0

        min_words = 1 if model_speaking else 3
        jaccard_threshold = 0.35 if model_speaking else 0.6
        window_sec = 8.0 if model_speaking else 5.0

        if len(words_candidate) < min_words:
            return False
        cutoff = now_ts - window_sec
        for ts, agent_text in reversed(_recent_agent_texts):
            if ts < cutoff:
                break
            words_agent = set(agent_text.lower().split())
            if not words_agent:
                continue
            intersection = words_candidate & words_agent
            union = words_candidate | words_agent
            jaccard = len(intersection) / len(union) if union else 0.0
            if jaccard > jaccard_threshold:
                return True
        return False

    async def _emit_tool_event(
        tool: str,
        behavior: ToolBehavior | str,
        *,
        status: str,
        data: dict | None = None,
    ) -> None:
        payload: dict = {
            "type": MessageType.TOOL_EVENT,
            "tool": tool,
            "behavior": behavior_to_text(behavior),
            "status": status,
        }
        if data:
            payload["data"] = _json_safe(data)
        await _safe_send_json(payload)

    async def _emit_capability_degraded(
        capability: str,
        reason: str,
        recoverable: bool = True,
    ) -> None:
        """Notify iOS client that a sub-agent capability is degraded."""
        await _safe_send_json({
            "type": MessageType.CAPABILITY_DEGRADED,
            "capability": capability,
            "reason": reason,
            "recoverable": recoverable,
        })

    async def _emit_identity_event(
        *,
        person_name: str,
        matched: bool,
        similarity: float = 0.0,
        source: str = "face_pipeline",
    ) -> None:
        payload = {
            "type": MessageType.IDENTITY_UPDATE,
            "person_name": person_name,
            "matched": matched,
            "similarity": similarity,
            "source": source,
            "behavior": behavior_to_text(ToolBehavior.SILENT),
        }
        await _safe_send_json(payload)
        if matched:
            await _safe_send_json({
                "type": MessageType.PERSON_IDENTIFIED,
                "person_name": person_name,
                "similarity": similarity,
                "source": source,
                "behavior": behavior_to_text(ToolBehavior.SILENT),
            })

    def _build_tools_manifest() -> dict:
        """Build a tools_manifest payload for the iOS Dev Console."""
        tools_list = [
            {
                "name": decl["name"],
                "category": _TOOL_CATEGORY_MAP.get(decl["name"], ("unknown", "WHEN_IDLE"))[0],
                "behavior": _TOOL_CATEGORY_MAP.get(decl["name"], ("unknown", "WHEN_IDLE"))[1],
                "description": decl.get("description", ""),
            }
            for decl in ALL_TOOL_DECLARATIONS
        ]
        # Include memory tools (not in ALL_TOOL_DECLARATIONS but in ALL_FUNCTIONS)
        for mem_name in ("preload_memory", "remember_entity", "what_do_you_remember", "forget_entity", "forget_recent_memory"):
            if mem_name in _TOOL_CATEGORY_MAP:
                cat, beh = _TOOL_CATEGORY_MAP[mem_name]
                tools_list.append({
                    "name": mem_name,
                    "category": cat,
                    "behavior": beh,
                    "description": "",
                })

        # Context modules — check which were successfully initialised
        _entity_graph_available = False
        try:
            from context.entity_graph import EntityGraphService  # noqa: F811
            _entity_graph_available = True
        except ImportError:
            pass

        context_modules = [
            {"name": "LocationContext", "status": "ready" if _location_ctx_service is not None else "unavailable"},
            {"name": "LODEvaluator", "status": "ready" if _lod_evaluator is not None else "unavailable"},
            {"name": "ProfileAssembler", "status": "ready" if _assembled_profile is not None else "unavailable"},
            {"name": "HabitDetector", "status": "ready" if _proactive_hints is not None else "unavailable"},
            {"name": "SceneMatcher", "status": "ready" if _location_ctx_service is not None else "unavailable"},
            {"name": "EntityGraph", "status": "ready" if _entity_graph_available else "unavailable"},
        ]

        return {
            "type": MessageType.TOOLS_MANIFEST,
            "tools": tools_list,
            "context_modules": context_modules,
            "sub_agents": {
                "vision": "ready",
                "ocr": "ready",
                "face": "ready" if _face_available else "unavailable",
            },
        }

    # Notify client immediately so the iOS layer knows the
    # WebSocket is live before the Gemini connection is ready.
    if not await _safe_send_json({"type": MessageType.SESSION_READY}):
        logger.info("WebSocket closed before session_ready: user=%s session=%s", user_id, session_id)
        return

    asyncio.create_task(session_meta.write_session_start())

    live_request_queue = LiveRequestQueue()
    ctx_queue = ContextInjectionQueue(live_request_queue)
    ctx_queue.start_background_flush_task()

    run_config = session_manager.get_run_config(
        session_id, lod=session_ctx.current_lod,
        language_code=user_profile.language,
    )

    # -- Phase 6: Context Engine initialisation ---------------------------------
    _location_ctx_service = None
    _lod_evaluator = None
    _assembled_profile = None
    _current_location_ctx = None  # latest LocationContext from GPS
    _proactive_hints = None  # set by HabitDetector inside try block
    try:
        from context.location_context import LocationContextService
        from context.lod_evaluator import LODEvaluator
        from context.profile_assembler import ProfileAssembler
        from context.habit_detector import HabitDetector
        from context.scene_matcher import SceneMatcher

        _location_ctx_service = LocationContextService(user_id)
        _lod_evaluator = LODEvaluator()

        # Detect habits from session history (background, best-effort)
        _proactive_hints = []
        try:
            _habit_detector = HabitDetector(user_id)
            _proactive_hints = _habit_detector.detect()
            if _proactive_hints:
                logger.info(
                    "Detected %d habits for user=%s (top: %s)",
                    len(_proactive_hints), user_id,
                    _proactive_hints[0].description[:60] if _proactive_hints else "",
                )
        except Exception:
            logger.debug("Habit detection skipped", exc_info=True)

        # Assemble initial profile
        _assembler = ProfileAssembler()
        _assembled_profile = _assembler.assemble(
            profile=user_profile,
            location_ctx=None,  # no GPS yet at session start
            entities=None,
            memories=None,
        )
    except Exception:
        logger.debug("Context engine init skipped (import error)", exc_info=True)

    # Send tools manifest so iOS Dev Console shows tool/context status
    await _safe_send_json(_build_tools_manifest())

    # -- E-7: Initial LOD context injection at session start -----------------
    # Inject the full dynamic system prompt so the model has LOD context
    # immediately, rather than waiting for the first telemetry tick.
    _initial_ephemeral = session_manager.get_ephemeral_context(session_id)
    _initial_memories = load_relevant_memories(
        user_id,
        session_ctx.active_task or session_ctx.trip_purpose or "",
        top_k=3,
    )
    _initial_prompt = build_full_dynamic_prompt(
        lod=session_ctx.current_lod,
        profile=user_profile,
        ephemeral_semantic="",
        session=session_ctx,
        memories=_initial_memories if _initial_memories else None,
        assembled_profile=_assembled_profile,
    )
    # -- Merge system prompt + greeting into ONE send_content call -----------
    # Two separate inject_immediate() calls cause Gemini to generate two
    # overlapping audio responses (one for the system prompt, one for the
    # greeting).  By combining them into a single Content message with
    # [DO NOT SPEAK] on the context part, Gemini absorbs the instructions
    # silently and only speaks the greeting — one audio output.
    _greeting_parts: list[str] = [
        "[SESSION START] Greet the user briefly (1-2 sentences).",
        "Let them know you're ready to help.",
    ]
    if user_profile and user_profile.preferred_name:
        _greeting_parts.append(
            f"Address them as '{user_profile.preferred_name}'."
        )
    _greeting_parts.append("Keep it natural and concise — no instructions or tutorials.")
    _combined_content = types.Content(
        parts=[
            types.Part(text="[CONTEXT UPDATE - DO NOT SPEAK]\n" + _initial_prompt),
            types.Part(text=" ".join(_greeting_parts)),
        ],
        role="user",
    )
    ctx_queue.inject_immediate(_combined_content)
    # Pre-mark model as speaking so early telemetry/LOD injections are queued,
    # not flushed to Gemini mid-greeting.  The real audio-chunk callback
    # (set_model_speaking(True)) confirms this ~200ms later; turn_complete
    # resets it when the greeting finishes.
    # inject_immediate already transitions state to GENERATING
    logger.info(
        "Injected combined context (LOD %d) + greeting for session %s",
        session_ctx.current_lod, session_id,
    )

    # -- Track client camera state for context injection ----------------------
    _client_camera_active = False
    _camera_activated_at: float = 0.0
    _CAMERA_GRACE_PERIOD_SEC: float = 12.0
    _first_vision_after_camera: bool = True  # First vision post-grace period is silent

    # -- Throttle raw frames to Gemini Live API ------------------------------
    _last_frame_to_gemini_at: float = 0.0
    _FRAME_TO_GEMINI_INTERVAL: float = 2.0

    # -- LOD engine helpers --------------------------------------------------

    async def _send_lod_update(
        new_lod: int,
        ephemeral_ctx,
        reason: str,
    ) -> None:
        """Build and inject a [LOD UPDATE] message into the Live session."""
        # SL-71: Preload relevant memories for prompt injection
        memories = await _load_session_memories(
            context_hint=session_ctx.active_task or session_ctx.trip_purpose or ""
        )
        lod_message = build_lod_update_message(
            lod=new_lod,
            ephemeral=ephemeral_ctx,
            session=session_ctx,
            profile=user_profile,
            reason=reason,
            memories=memories,
            assembled_profile=_assembled_profile,
            location_ctx=_current_location_ctx,
        )
        ctx_queue.enqueue(
            category="lod",
            text=lod_message,
            priority=3,
            speak=False,
        )
        logger.info("Injected [LOD UPDATE] -> LOD %d (%s)", new_lod, reason)

    # -- Per-session memory state (Phase 4) -----------------------------------
    memory_top3: list[str] = []
    memory_top3_detailed: list[dict] = []
    # Per-session budget tracker — intentionally not persisted to Firestore.
    # Each session starts with a fresh budget to prevent runaway writes from
    # accumulating across reconnections.  Persistence would require atomic
    # read-modify-write on Firestore and adds complexity without clear benefit.
    memory_budget = MemoryBudgetTracker() if _memory_available else None
    transcript_history: list[dict] = []

    async def _load_session_memories(context_hint: str = "") -> list[str]:
        """Load relevant memories for this user session."""
        nonlocal memory_top3, memory_top3_detailed
        try:
            if _memory_available:
                bank = MemoryBankService(user_id)
                raw_results = bank.retrieve_memories(context_hint, top_k=3)
                memory_top3 = [m["content"] for m in raw_results][:3]
                memory_top3_detailed = [{
                    "content": m.get("content", "")[:120],
                    "category": m.get("category", "general"),
                    "importance": round(float(m.get("importance", 0.5)), 2),
                    "score": round(float(m.get("_composite_score", 0)), 3),
                } for m in raw_results][:3]
                return memory_top3
            return []
        except Exception:
            logger.exception("Failed to load memories for user %s", user_id)
            return []

    async def _sync_runtime_vad_update(new_lod: int) -> dict:
        """Inject a best-effort runtime VAD update marker into the live session."""
        supported, reason = supports_runtime_vad_reconfiguration()
        payload = build_vad_runtime_update_payload(new_lod)
        payload["runtime_hot_reconfig_supported"] = supported
        payload["runtime_note"] = "transport_hot_update_applied" if supported else reason

        ctx_queue.enqueue(
            category="vad",
            text=build_vad_runtime_update_message(new_lod),
            priority=7,
            speak=False,
        )

        if supported:
            logger.info("Injected runtime VAD update payload for LOD %d: %s", new_lod, payload)
        else:
            logger.warning(
                "Runtime VAD transport hot-update unavailable (%s); injected sync marker only: %s",
                reason,
                payload,
            )
        return payload

    async def _notify_ios_lod_change(
        new_lod: int,
        reason: str,
        debug_dict: dict,
        vad_update: dict | None = None,
    ) -> None:
        """Send LOD change notification to iOS client."""
        await _safe_send_json({
            "type": MessageType.LOD_UPDATE,
            "lod": new_lod,
            "reason": reason,
        })
        # SL-77: Include memory_top3 in debug_lod for DebugOverlay
        debug_dict["memory_top3"] = memory_top3
        debug_dict["memory_top3_detailed"] = memory_top3_detailed
        if vad_update:
            debug_dict["vad_update"] = vad_update
        await _safe_send_json({
            "type": MessageType.DEBUG_LOD,
            "data": debug_dict,
        })

    async def _emit_activity_debug_event(
        *,
        event_name: str,
        queue_status: str,
        queue_note: str = "",
        source: str = "ios_client",
    ) -> None:
        """Emit an observable activity event for iOS debug overlay."""
        ts = datetime.now(timezone.utc)
        is_activity_start = event_name == "activity_start"
        session_ctx.current_activity_state = "user_speaking" if is_activity_start else "idle"
        session_ctx.last_activity_event = event_name
        session_ctx.last_activity_event_ts = ts
        session_ctx.last_activity_source = source
        session_ctx.activity_event_count += 1

        await _safe_send_json({
            "type": MessageType.DEBUG_ACTIVITY,
            "data": {
                "event": event_name,
                "state": session_ctx.current_activity_state,
                "source": source,
                "queue_status": queue_status,
                "queue_note": queue_note,
                "timestamp": ts.isoformat(),
                "event_count": session_ctx.activity_event_count,
            },
        })

    # -- Sub-agent helpers (Phase 3) -----------------------------------------

    async def _run_vision_analysis(image_base64: str) -> None:
        """Run async vision analysis and inject results into Live session."""
        nonlocal _vision_in_progress, _last_vision_context_text
        nonlocal _last_vision_context_sent_at, _last_vision_prefeedback_at
        nonlocal _first_vision_after_camera
        if not _vision_available:
            await _emit_tool_event(
                "analyze_scene",
                ToolBehavior.WHEN_IDLE,
                status="unavailable",
                data={"reason": "vision_agent_unavailable"},
            )
            return

        # asyncio.Lock is safe here: the lock is only held across synchronous
        # flag reads/writes (no await between acquire and release), so there is
        # no race window.  The lock guards against concurrent task scheduling
        # at the outer await boundary, not against OS-thread preemption.
        async with _vision_lock:
            if _vision_in_progress:
                return
            _vision_in_progress = True

        # Camera activation grace period: suppress non-safety vision for 8s
        # after camera opens to prevent immediate narration cascade.
        now_mono = time.monotonic()
        if _camera_activated_at > 0 and (now_mono - _camera_activated_at) < _CAMERA_GRACE_PERIOD_SEC:
            logger.info("Suppressed vision: camera activation grace period (%.1fs)",
                        now_mono - _camera_activated_at)
            async with _vision_lock:
                _vision_in_progress = False
            return

        # B-5: Pre-feedback — immediate audio cue before analysis starts
        # Suppress during camera activation grace period
        if (now_mono - _last_vision_prefeedback_at >= VISION_PREFEEDBACK_COOLDOWN_SEC
                and (_camera_activated_at <= 0 or now_mono - _camera_activated_at >= _CAMERA_GRACE_PERIOD_SEC)):
            await _forward_agent_transcript("Let me look at that for you...")
            _last_vision_prefeedback_at = now_mono

        try:
            ctx_dict = {
                "space_type": session_ctx.space_type,
                "trip_purpose": session_ctx.trip_purpose,
                "active_task": session_ctx.active_task,
                "motion_state": session_manager.get_ephemeral_context(session_id).motion_state,
            }
            result = await analyze_scene(image_base64, session_ctx.current_lod, ctx_dict)
            vision_text = _format_vision_result(result, session_ctx.current_lod)
            vision_repeat_window = VISION_REPEAT_SUPPRESS_SEC
            now_mono = time.monotonic()
            repeated = _is_repeated_text(
                vision_text,
                previous_text=_last_vision_context_text,
                now_ts=now_mono,
                previous_ts=_last_vision_context_sent_at,
                cooldown_sec=vision_repeat_window,
            )
            if not repeated:
                await _safe_send_json({
                    "type": MessageType.VISION_RESULT,
                    "summary": result.get("scene_description", ""),
                    "behavior": behavior_to_text(ToolBehavior.WHEN_IDLE),
                    "data": _json_safe(result),
                })
                _last_vision_context_text = vision_text
                _last_vision_context_sent_at = now_mono
            else:
                logger.debug("Suppressed repeated vision summary within %.1fs window", vision_repeat_window)

            await _safe_send_json({
                "type": MessageType.VISION_DEBUG,
                "data": {
                    "bounding_boxes": _json_safe(result.get("bounding_boxes", [])),
                    "confidence": float(result.get("confidence", 0.0)),
                    "lod": session_ctx.current_lod,
                },
            })
            await _emit_tool_event(
                "analyze_scene",
                ToolBehavior.WHEN_IDLE,
                status="completed",
                data={
                    "confidence": float(result.get("confidence", 0.0)),
                    "repeat_suppressed": repeated,
                },
            )

            if result.get("confidence", 0) > 0 and not repeated:
                if ctx_queue.vision_spoken_cooldown_active:
                    logger.info("Suppressed vision: vision_spoken_cooldown active")
                else:
                    info_type = "spatial_description"
                    ephemeral = session_manager.get_ephemeral_context(session_id)
                    speak = should_speak(
                        info_type=info_type,
                        current_lod=session_ctx.current_lod,
                        step_cadence=getattr(ephemeral, "step_cadence", 0.0) or 0.0,
                        ambient_noise_db=getattr(ephemeral, "ambient_noise_db", 50.0) or 50.0,
                    )

                    # First vision after camera activation is always silent
                    # to avoid unprompted scene narration
                    if _first_vision_after_camera:
                        speak = False
                        _first_vision_after_camera = False  # type: ignore[has-type]
                        logger.info("First vision post-camera: forced silent injection")

                    if not speak:
                        vision_text = "[SILENT - context only, do not speak aloud]\n" + vision_text
                    ctx_queue.enqueue(
                        category="vision",
                        text=vision_text,
                        priority=5,
                        speak=speak,
                    )
                    # Structural cooldown: activate when vision is injected with speak=True
                    # (language-agnostic — no longer relies on English keyword detection)
                    if speak:
                        ctx_queue.record_vision_spoken()
                    logger.info("Injected [VISION ANALYSIS] (LOD %d, confidence %.2f, speak=%s)",
                                session_ctx.current_lod, result.get("confidence", 0), speak)
        except Exception as exc:
            logger.exception("Vision analysis failed")
            await _emit_tool_event(
                "analyze_scene",
                ToolBehavior.WHEN_IDLE,
                status="error",
                data={"reason": "vision_analysis_failed"},
            )
            await _emit_capability_degraded("vision", str(exc)[:200])
        finally:
            async with _vision_lock:
                _vision_in_progress = False

    async def _run_face_recognition(image_base64: str) -> None:
        """Run face recognition and inject results as SILENT context."""
        nonlocal face_library, _face_library_loaded_at, _face_consecutive_misses, _face_skip_counter
        if not _face_available:
            await _emit_tool_event(
                "identify_person",
                ToolBehavior.SILENT,
                status="unavailable",
                data={"reason": "face_agent_unavailable"},
            )
            return

        # Adaptive backoff: skip cycles when no faces are consistently detected
        if _face_consecutive_misses >= _FACE_BACKOFF_THRESHOLD:
            _face_skip_counter += 1
            if _face_skip_counter <= _FACE_BACKOFF_SKIP_CYCLES:
                logger.debug("Face detection skipped (backoff: %d misses)", _face_consecutive_misses)
                return
            _face_skip_counter = 0  # run this cycle, then skip again

        # Periodic refresh of face library
        now_mono = time.monotonic()
        if now_mono - _face_library_loaded_at >= FACE_LIBRARY_REFRESH_SEC:
            try:
                face_library = load_face_library(user_id)
                _face_library_loaded_at = now_mono
                logger.info("Refreshed face library (%d faces) for user %s", len(face_library), user_id)
            except Exception:
                logger.exception("Failed to refresh face library for user %s", user_id)

        try:
            results = await asyncio.to_thread(
                identify_persons_in_frame,
                image_base64,
                user_id,
                face_library,
                ToolBehavior.SILENT,
            )
            await _safe_send_json({
                "type": MessageType.FACE_DEBUG,
                "data": {
                    "face_boxes": _json_safe([
                        {
                            "bbox": item.get("bbox", []),
                            "label": item.get("person_name", "unknown"),
                            "score": float(item.get("score", 0.0)),
                            "similarity": float(item.get("similarity", 0.0)),
                        }
                        for item in results
                    ]),
                },
            })
            # Update adaptive frequency counters
            if len(results) == 0:
                _face_consecutive_misses += 1
            else:
                _face_consecutive_misses = 0
                _face_skip_counter = 0

            known = [r for r in results if r["person_name"] != "unknown"]
            await _emit_tool_event(
                "identify_person",
                ToolBehavior.SILENT,
                status="completed",
                data={"detections": len(results), "known": len(known)},
            )
            if known:
                ephemeral = session_manager.get_ephemeral_context(session_id)
                speak = should_speak(
                    info_type="face_recognition",
                    current_lod=session_ctx.current_lod,
                    step_cadence=getattr(ephemeral, "step_cadence", 0.0) or 0.0,
                    ambient_noise_db=getattr(ephemeral, "ambient_noise_db", 50.0) or 50.0,
                )

                face_text = _format_face_results(known)
                if _memory_available:
                    for person in known:
                        pname = person.get("person_name", "")
                        if pname and pname != "unknown":
                            try:
                                person_memories = load_relevant_memories(user_id, f"person {pname}", top_k=2)
                                if person_memories:
                                    face_text += f"\nMemories about {pname}:"
                                    for mem in person_memories:
                                        face_text += f"\n- {mem}"
                            except Exception:
                                logger.debug("Failed to load memories for person %s", pname, exc_info=True)
                if not speak:
                    face_text = "[SILENT - context only, do not speak aloud]\n" + face_text
                ctx_queue.enqueue(
                    category="face",
                    text=face_text,
                    priority=4,
                    speak=speak,
                )
                logger.info("Injected [FACE ID] (speak=%s): %s",
                            speak, ", ".join(r["person_name"] for r in known))
                best = max(known, key=lambda item: float(item.get("similarity", 0.0)))
                await _emit_identity_event(
                    person_name=str(best.get("person_name", "unknown")),
                    matched=True,
                    similarity=float(best.get("similarity", 0.0)),
                    source="face_match",
                )
            elif results:
                await _emit_identity_event(
                    person_name="unknown",
                    matched=False,
                    similarity=0.0,
                    source="face_detected_no_match",
                )
        except Exception as exc:
            logger.exception("Face recognition failed")
            await _emit_tool_event(
                "identify_person",
                ToolBehavior.SILENT,
                status="error",
                data={"reason": "face_recognition_failed"},
            )
            await _emit_capability_degraded("face", str(exc)[:200])

    async def _run_ocr_analysis(image_base64: str, safety_only: bool = False) -> None:
        """Run OCR and inject results into Live session context."""
        nonlocal _last_ocr_context_text, _last_ocr_context_sent_at, _last_ocr_prefeedback_at
        if not _ocr_available:
            await _emit_tool_event(
                "extract_text",
                ToolBehavior.WHEN_IDLE,
                status="unavailable",
                data={"reason": "ocr_agent_unavailable"},
            )
            return

        # B-5: Pre-feedback — immediate audio cue before analysis starts
        # Skip pre-feedback for safety-only scans (LOD 1/2)
        if not safety_only:
            now_mono = time.monotonic()
            if now_mono - _last_ocr_prefeedback_at >= OCR_PREFEEDBACK_COOLDOWN_SEC:
                await _forward_agent_transcript("Reading the text for you...")
                _last_ocr_prefeedback_at = now_mono

        try:
            # Build context hint from session state
            hint = ""
            if session_ctx.space_type:
                hint = f"User is in a {session_ctx.space_type} environment."
            if session_ctx.active_task:
                hint += f" Currently: {session_ctx.active_task}."

            result = await extract_text(image_base64, context_hint=hint, safety_only=safety_only)
            ocr_text = _format_ocr_result(result)
            repeat_window = OCR_REPEAT_SUPPRESS_SEC
            now_mono = time.monotonic()
            repeated = _is_repeated_text(
                ocr_text,
                previous_text=_last_ocr_context_text,
                now_ts=now_mono,
                previous_ts=_last_ocr_context_sent_at,
                cooldown_sec=repeat_window,
            )
            if not repeated:
                await _safe_send_json({
                    "type": MessageType.OCR_RESULT,
                    "summary": result.get("text", ""),
                    "behavior": behavior_to_text(ToolBehavior.WHEN_IDLE),
                    "data": _json_safe(result),
                })
                _last_ocr_context_text = ocr_text
                _last_ocr_context_sent_at = now_mono
            else:
                logger.debug("Suppressed repeated OCR summary within %.1fs window", repeat_window)

            await _safe_send_json({
                "type": MessageType.OCR_DEBUG,
                "data": {
                    "text_regions": _json_safe(result.get("text_regions", [])),
                    "text_type": result.get("text_type", "unknown"),
                    "confidence": float(result.get("confidence", 0.0)),
                },
            })
            await _emit_tool_event(
                "extract_text",
                ToolBehavior.WHEN_IDLE,
                status="completed",
                data={
                    "confidence": float(result.get("confidence", 0.0)),
                    "repeat_suppressed": repeated,
                },
            )

            if result.get("confidence", 0) > 0.3 and result.get("text") and not repeated:
                ephemeral = session_manager.get_ephemeral_context(session_id)
                info_type = "object_enumeration"
                speak = should_speak(
                    info_type=info_type,
                    current_lod=session_ctx.current_lod,
                    step_cadence=getattr(ephemeral, "step_cadence", 0.0) or 0.0,
                    ambient_noise_db=getattr(ephemeral, "ambient_noise_db", 50.0) or 50.0,
                )

                if not speak:
                    ocr_text = "[SILENT - context only, do not speak aloud]\n" + ocr_text
                ctx_queue.enqueue(
                    category="ocr",
                    text=ocr_text,
                    priority=5,
                    speak=speak,
                )
                logger.info("Injected [OCR RESULT] (%s, confidence %.2f, speak=%s)",
                            result.get("text_type", "unknown"), result.get("confidence", 0), speak)
        except Exception as exc:
            logger.exception("OCR analysis failed")
            await _emit_tool_event(
                "extract_text",
                ToolBehavior.WHEN_IDLE,
                status="error",
                data={"reason": "ocr_analysis_failed"},
            )
            await _emit_capability_degraded("ocr", str(exc)[:200])

    # -- Upstream handler ----------------------------------------------------

    # Binary frame magic bytes for audio/image binary protocol (Phase 5)
    _MAGIC_AUDIO = 0x01
    _MAGIC_IMAGE = 0x02

    async def _upstream() -> None:
        """Read messages from the iOS client and forward to the Live API.

        Supports both legacy JSON text messages and optimized binary frames.
        Binary protocol: first byte is magic byte (0x01=audio, 0x02=image),
        remaining bytes are raw payload. This eliminates ~33% Base64 overhead.

        Handles upstream message types: audio, image, telemetry,
        activity_start, activity_end, gesture.
        """
        nonlocal _last_vision_time, _frame_seq, _allow_agent_repeat_until, _client_camera_active
        nonlocal _last_user_activity_at, _is_interrupted, _last_interrupt_at, _model_audio_last_seen_at
        nonlocal _last_frame_to_gemini_at

        try:
            while True:
                # Use low-level receive() with inactivity timeout.
                # If no message arrives within WS_INACTIVITY_TIMEOUT_SEC,
                # the connection is considered stale (e.g. silent disconnect).
                try:
                    ws_message = await asyncio.wait_for(
                        websocket.receive(),
                        timeout=WS_INACTIVITY_TIMEOUT_SEC,
                    )
                except asyncio.TimeoutError:
                    logger.info(
                        "WebSocket inactivity timeout (%ds): user=%s session=%s",
                        WS_INACTIVITY_TIMEOUT_SEC, user_id, session_id,
                    )
                    stop_downstream.set()
                    try:
                        await websocket.close(code=1000, reason="inactivity_timeout")
                    except Exception:
                        pass
                    return

                # --- Binary frame (optimized path) ---
                if "bytes" in ws_message and ws_message["bytes"]:
                    raw_bytes: bytes = ws_message["bytes"]
                    if len(raw_bytes) < 2:
                        continue

                    magic = raw_bytes[0]
                    payload = raw_bytes[1:]

                    if magic == _MAGIC_AUDIO:
                        blob = types.Blob(
                            data=payload,
                            mime_type="audio/pcm;rate=16000",
                        )
                        live_request_queue.send_realtime(blob)
                        continue

                    elif magic == _MAGIC_IMAGE:
                        _frame_seq += 1
                        # Store latest frame for on-demand OCR tool
                        _ocr_set_latest_frame(session_id, base64.b64encode(payload).decode("ascii"))
                        now_frame = time.monotonic()
                        if now_frame - _last_frame_to_gemini_at >= _FRAME_TO_GEMINI_INTERVAL:
                            blob = types.Blob(
                                data=payload,
                                mime_type="image/jpeg",
                            )
                            live_request_queue.send_realtime(blob)
                            _last_frame_to_gemini_at = now_frame

                        # Trigger async sub-agents (same logic as JSON path)
                        import time as _time
                        now = _time.monotonic()
                        lod = session_ctx.current_lod
                        # Vision intervals widened: LOD1: 10s, LOD2: 8s, LOD3: 5s
                        vision_interval = {1: 10.0, 2: 8.0, 3: 5.0}.get(lod, 8.0)
                        queued_agents: list[str] = []
                        if now - _last_vision_time >= vision_interval:
                            _last_vision_time = now
                            image_b64 = base64.b64encode(payload).decode("ascii")
                            await _emit_tool_event(
                                "analyze_scene", ToolBehavior.WHEN_IDLE, status="queued",
                            )
                            queued_agents.append("vision")
                            asyncio.create_task(_run_vision_analysis(image_b64))

                            # Face recognition: only at LOD 2/3 (match JSON path)
                            if lod >= 2:
                                await _emit_tool_event(
                                    "identify_person", ToolBehavior.SILENT, status="queued",
                                )
                                queued_agents.append("face")
                                await _emit_identity_event(
                                    person_name="unknown",
                                    matched=False,
                                    similarity=0.0,
                                    source="queued",
                                )
                                asyncio.create_task(_run_face_recognition(image_b64))

                            # OCR: safety-only at LOD 1 with 15s interval only.
                            # Full OCR is now user-intent driven (Orchestrator tool).
                            if lod == 1 and now - getattr(_run_ocr_analysis, '_last_safety_ocr_at', 0) >= 15.0:
                                _run_ocr_analysis._last_safety_ocr_at = now  # type: ignore[attr-defined]
                                await _emit_tool_event(
                                    "extract_text", ToolBehavior.WHEN_IDLE, status="queued",
                                )
                                queued_agents.append("ocr")
                                asyncio.create_task(_run_ocr_analysis(image_b64, safety_only=True))
                        await _safe_send_json({
                            "type": MessageType.FRAME_ACK,
                            "frame_id": _frame_seq,
                            "queued_agents": queued_agents,
                        })
                        continue

                    else:
                        logger.warning("Unknown binary magic byte: 0x%02x", magic)
                        continue

                # --- Text frame (JSON, legacy + control messages) ---
                raw_text = ws_message.get("text")
                if not raw_text:
                    # Check for disconnect
                    if ws_message.get("type") == "websocket.disconnect":
                        break
                    continue

                try:
                    message = json.loads(raw_text)
                except json.JSONDecodeError:
                    logger.warning("Received non-JSON text message, ignoring")
                    continue

                if message.get("type") == "audio":
                    audio_bytes = base64.b64decode(message["data"])
                    blob = types.Blob(
                        data=audio_bytes,
                        mime_type="audio/pcm;rate=16000",
                    )
                    live_request_queue.send_realtime(blob)

                elif message.get("type") == "image":
                    image_bytes = base64.b64decode(message["data"])
                    mime_type = message.get("mimeType", "image/jpeg")
                    _frame_seq += 1
                    # Store latest frame for on-demand OCR tool
                    _ocr_set_latest_frame(session_id, message["data"])
                    # Throttle raw frames to Gemini (sub-agents still run at LOD intervals)
                    now_frame = time.monotonic()
                    if now_frame - _last_frame_to_gemini_at >= _FRAME_TO_GEMINI_INTERVAL:
                        blob = types.Blob(
                            data=image_bytes,
                            mime_type=mime_type,
                        )
                        live_request_queue.send_realtime(blob)
                        _last_frame_to_gemini_at = now_frame

                    # Phase 3: Trigger async sub-agents on image frames
                    import time as _time
                    now = _time.monotonic()
                    lod = session_ctx.current_lod

                    # Vision analysis: LOD-aware frequency (widened)
                    # LOD 1: every 10s, LOD 2: every 8s, LOD 3: every 5s
                    vision_interval = {1: 10.0, 2: 8.0, 3: 5.0}.get(lod, 8.0)
                    queued_agents: list[str] = []
                    if now - _last_vision_time >= vision_interval:
                        _last_vision_time = now
                        image_b64 = message["data"]
                        await _emit_tool_event(
                            "analyze_scene",
                            ToolBehavior.WHEN_IDLE,
                            status="queued",
                        )
                        queued_agents.append("vision")
                        # Fire-and-forget async tasks
                        asyncio.create_task(_run_vision_analysis(image_b64))

                        # Face recognition: only at LOD 2/3
                        if lod >= 2:
                            await _emit_tool_event(
                                "identify_person",
                                ToolBehavior.SILENT,
                                status="queued",
                            )
                            queued_agents.append("face")
                            await _emit_identity_event(
                                person_name="unknown",
                                matched=False,
                                similarity=0.0,
                                source="queued",
                            )
                            asyncio.create_task(_run_face_recognition(image_b64))

                        # OCR: safety-only at LOD 1 with 15s interval only.
                        # Full OCR is now user-intent driven (Orchestrator tool).
                        if lod == 1 and now - getattr(_run_ocr_analysis, '_last_safety_ocr_at', 0) >= 15.0:
                            _run_ocr_analysis._last_safety_ocr_at = now  # type: ignore[attr-defined]
                            await _emit_tool_event(
                                "extract_text",
                                ToolBehavior.WHEN_IDLE,
                                status="queued",
                            )
                            queued_agents.append("ocr")
                            asyncio.create_task(_run_ocr_analysis(image_b64, safety_only=True))
                    await _safe_send_json({
                        "type": MessageType.FRAME_ACK,
                        "frame_id": _frame_seq,
                        "queued_agents": queued_agents,
                    })

                elif message.get("type") == "camera_failure":
                    # SL-76: Camera hardware failure path
                    camera_error = (
                        message.get("error")
                        or message.get("reason")
                        or "camera_unavailable"
                    )
                    logger.warning("Camera failure reported: %s", camera_error)
                    await _emit_capability_degraded(
                        "camera",
                        camera_error,
                        recoverable=message.get("recoverable", True),
                    )

                elif message.get("type") == "telemetry":
                    telemetry_data = message.get("data", {})
                    await _process_telemetry(telemetry_data)

                elif message.get("type") == "gesture":
                    gesture = message.get("gesture")
                    _last_user_activity_at = time.monotonic()
                    session_meta.record_interaction()
                    if gesture in ("lod_up", "lod_down"):
                        ephemeral_ctx = session_manager.get_ephemeral_context(session_id)
                        ephemeral_ctx.user_gesture = gesture
                        await _process_lod_decision(ephemeral_ctx)
                        ephemeral_ctx.user_gesture = None
                        session_manager.update_ephemeral_context(session_id, ephemeral_ctx)

                    elif isinstance(gesture, str) and gesture.startswith("force_lod_"):
                        try:
                            forced_lod = int(gesture.rsplit("_", 1)[-1])
                        except (TypeError, ValueError):
                            logger.warning("Invalid force_lod gesture payload: %s", gesture)
                            continue

                        if forced_lod not in (1, 2, 3):
                            logger.warning("force_lod gesture out of range: %s", gesture)
                            continue

                        old_lod = session_ctx.current_lod
                        if forced_lod == old_lod:
                            await _safe_send_json({
                                "type": MessageType.LOD_UPDATE,
                                "lod": forced_lod,
                                "reason": "force_lod_no_change",
                            })
                            continue

                        reason = f"manual_force_lod_{forced_lod}"
                        logger.info("Force LOD gesture received: %d -> %d", old_lod, forced_lod)
                        resume_prompt = on_lod_change(session_ctx, old_lod, forced_lod)
                        session_ctx.current_lod = forced_lod
                        session_meta.record_lod_time(forced_lod)
                        telemetry_agg.update_lod(forced_lod)

                        vad_update = await _sync_runtime_vad_update(forced_lod)
                        await _send_lod_update(
                            forced_lod,
                            session_manager.get_ephemeral_context(session_id),
                            reason,
                        )
                        await _notify_ios_lod_change(
                            forced_lod,
                            reason,
                            {
                                "lod": forced_lod,
                                "prev": old_lod,
                                "reason": reason,
                                "rules": [f"manual:{gesture}"],
                                "forced": True,
                            },
                            vad_update=vad_update,
                        )

                        if resume_prompt:
                            resume_content = types.Content(
                                parts=[types.Part(text=resume_prompt)],
                                role="user",
                            )
                            ctx_queue.inject_immediate(resume_content)

                    elif gesture == "interrupt":
                        logger.info("User interrupt gesture received")
                        ctx_queue.set_ios_playback_drained(True)
                        content = types.Content(
                            parts=[types.Part(text="[USER INTERRUPT] The user has interrupted. Stop current output immediately and wait for their next input.")],
                            role="user",
                        )
                        ctx_queue.inject_immediate(content)

                    elif gesture == "repeat_last":
                        _allow_agent_repeat_until = time.monotonic() + 12.0
                        last_agent = None
                        for entry in reversed(transcript_history):
                            if entry.get("role") == "agent":
                                last_agent = entry.get("text", "")
                                break
                        if last_agent:
                            logger.info("Repeat last gesture: replaying last agent utterance")
                            content = types.Content(
                                parts=[types.Part(text=f'[REPEAT REQUEST] The user wants you to repeat your last response. Please repeat: "{last_agent}"')],
                                role="user",
                            )
                            ctx_queue.inject_immediate(content)
                        else:
                            logger.info("Repeat last gesture: no previous agent utterance found")
                            content = types.Content(
                                parts=[types.Part(text="[REPEAT REQUEST] The user wants you to repeat your last response, but no previous response was found. Let the user know.")],
                                role="user",
                            )
                            ctx_queue.inject_immediate(content)

                    elif gesture == "mute_toggle":
                        # Mute is client-side only (iOS stops sending audio).
                        # Server just acknowledges for logging/debug purposes.
                        muted = _coerce_bool(message.get("muted"), default=True)
                        logger.info("Mute toggle acknowledged: muted=%s", muted)

                    elif gesture == "pause":
                        paused = _coerce_bool(message.get("paused", True), default=True)
                        ctx_queue.set_ios_playback_drained(True)
                        if paused:
                            logger.info("Pause activated for session %s", session_id)
                            content = types.Content(
                                parts=[types.Part(text="[PAUSE] The user has paused the session. Go silent until the user resumes.")],
                                role="user",
                            )
                            ctx_queue.inject_immediate(content)
                            await _safe_send_json({
                                "type": MessageType.LOD_UPDATE,
                                "lod": session_ctx.current_lod,
                                "reason": "paused",
                            })
                        else:
                            logger.info("Session resumed for session %s", session_id)
                            content = types.Content(
                                parts=[types.Part(text="[RESUME] The user has resumed the session. You may speak again.")],
                                role="user",
                            )
                            ctx_queue.inject_immediate(content)
                            await _safe_send_json({
                                "type": MessageType.LOD_UPDATE,
                                "lod": session_ctx.current_lod,
                                "reason": "resumed",
                            })

                    elif gesture == "camera_toggle":
                        nonlocal _client_camera_active, _camera_activated_at, _first_vision_after_camera
                        _client_camera_active = _coerce_bool(
                            message.get("active"), default=not _client_camera_active,
                        )
                        logger.info("Camera toggle: active=%s", _client_camera_active)
                        if _client_camera_active:
                            _camera_activated_at = time.monotonic()
                            _first_vision_after_camera = True
                            # Camera toggle is informational, not safety-critical → use queue
                            # to avoid interrupting current generation via turn_complete=True
                            ctx_queue.enqueue(
                                category="camera_toggle",
                                text=(
                                    "[CAMERA ACTIVATED] The user has turned on the rear camera. "
                                    "Visual context is now available via periodic [VISION ANALYSIS] injections. "
                                    "Do NOT describe every frame. Only speak about safety hazards or when the user asks. "
                                    "Acknowledge camera activation in one brief sentence, then observe silently."
                                ),
                                priority=4,
                                speak=True,
                            )
                        else:
                            ctx_queue.enqueue(
                                category="camera_toggle",
                                text=(
                                    "[CAMERA DEACTIVATED] The user has turned off the camera. "
                                    "You are now in audio-only mode. Do not reference visual information "
                                    "unless recalling something previously seen."
                                ),
                                priority=4,
                                speak=True,
                            )

                    else:
                        logger.debug("Unhandled gesture type: %s", gesture)

                elif message.get("type") == "reload_face_library":
                    logger.info("Reload face library requested for user=%s", user_id)
                    if _face_available:
                        try:
                            face_library.clear()
                            face_library.extend(await asyncio.to_thread(load_face_library, user_id))
                            await _safe_send_json({
                                "type": MessageType.FACE_LIBRARY_RELOADED,
                                "count": len(face_library),
                            })
                            logger.info("Reloaded %d face(s) for user %s", len(face_library), user_id)
                        except Exception:
                            logger.exception("Failed to reload face library")
                            await _safe_send_json({
                                "type": MessageType.ERROR,
                                "error": "Failed to reload face library",
                            })
                    else:
                        await _safe_send_json({
                            "type": MessageType.ERROR,
                            "error": "Face recognition not available",
                        })

                elif message.get("type") == "clear_face_library":
                    logger.info("Clear face library requested for user=%s", user_id)
                    if _face_available:
                        try:
                            from tools.face_tools import clear_face_library
                            count = await asyncio.to_thread(clear_face_library, user_id)
                            face_library.clear()
                            await _safe_send_json({
                                "type": MessageType.FACE_LIBRARY_CLEARED,
                                "deleted_count": count,
                            })
                            logger.info("Cleared %d face(s) for user %s", count, user_id)
                        except Exception:
                            logger.exception("Failed to clear face library")
                            await _safe_send_json({
                                "type": MessageType.ERROR,
                                "error": "Failed to clear face library",
                            })
                    else:
                        await _safe_send_json({
                            "type": MessageType.ERROR,
                            "error": "Face recognition not available",
                        })

                elif message.get("type") == "profile_updated":
                    logger.info("profile_updated received for user=%s", user_id)
                    try:
                        # 1. Invalidate cache and reload from Firestore
                        session_manager.invalidate_user_profile(user_id)
                        fresh_profile = await session_manager.load_user_profile(user_id)

                        # 2. Update the local variable in-place so all closures see new data
                        user_profile.update_from_dict({
                            f.name: getattr(fresh_profile, f.name)
                            for f in __import__("dataclasses").fields(fresh_profile)
                            if f.name != "user_id"
                        })

                        # 3. Inject updated persona context into Gemini session
                        from lod.prompt_builder import _build_persona_block
                        persona_block = _build_persona_block(user_profile)
                        profile_ctx = (
                            "[PROFILE UPDATE]\n"
                            "The user just updated their profile. Use the new settings below "
                            "for all subsequent interactions.\n"
                            f"{persona_block}\n"
                            "Do not narrate this block to the user."
                        )
                        ctx_queue.enqueue(
                            category="profile_update",
                            text=profile_ctx,
                            priority=3,
                            speak=False,
                        )
                        logger.info("Queued profile update context for user %s", user_id)

                        # 4. Ack to client
                        await _safe_send_json({"type": MessageType.PROFILE_UPDATED_ACK})
                    except Exception:
                        logger.exception("Failed to handle profile_updated for user %s", user_id)
                        await _safe_send_json({
                            "type": MessageType.ERROR,
                            "error": "Failed to apply profile update",
                        })

                elif message.get("type") == "playback_drained":
                    ctx_queue.set_ios_playback_drained(True)
                    logger.debug("iOS playback drained — flush gate opened")

                elif message.get("type") == "client_barge_in":
                    # Client-initiated barge-in (under NO_INTERRUPTION mode).
                    # Gemini VAD won't set interrupted=True, so we simulate it
                    # server-side: suppress audio forwarding until turn_complete.
                    # Only suppress if model was actually speaking recently (2.0s
                    # window aligns with iOS 1.5s timeout + network latency).
                    now_mono = time.monotonic()
                    if (now_mono - _last_interrupt_at) < _INTERRUPT_DEBOUNCE_SEC:
                        logger.debug("Client barge-in debounced (%.0fms since last)",
                                     (now_mono - _last_interrupt_at) * 1000)
                    else:
                        model_was_speaking = (
                            _model_audio_last_seen_at > 0
                            and (now_mono - _model_audio_last_seen_at) < 2.0
                        )
                        if model_was_speaking:
                            _is_interrupted = True
                            _last_interrupt_at = now_mono
                            _model_audio_last_seen_at = 0.0
                            ctx_queue._transition_to(ModelState.IDLE)
                            logger.info("Client barge-in — suppressing audio forwarding")
                        else:
                            logger.info("Client barge-in ignored — model not speaking (last audio %.1fs ago)",
                                        now_mono - _model_audio_last_seen_at if _model_audio_last_seen_at > 0 else -1)

                else:
                    logger.warning("Unknown upstream message type: %s", message.get("type"))

        except WebSocketDisconnect:
            stop_downstream.set()
            logger.info("Client disconnected (upstream): user=%s session=%s", user_id, session_id)
        except Exception:
            stop_downstream.set()
            logger.exception("Error in upstream handler: user=%s session=%s", user_id, session_id)

    # -- Telemetry processing ------------------------------------------------

    async def _process_telemetry(telemetry_data: dict) -> None:
        """Process a telemetry tick: semantic text + LOD decision."""
        import time as _time
        nonlocal _last_telemetry_signature, _last_telemetry_context_sent_at
        nonlocal _current_location_ctx

        ephemeral_ctx = parse_telemetry_to_ephemeral(telemetry_data)
        session_manager.update_ephemeral_context(session_id, ephemeral_ctx)

        # Semantic text injection (LOD-aware throttle)
        now = _time.monotonic()
        if telemetry_agg.should_send(now):
            signature = _build_telemetry_signature(ephemeral_ctx)
            should_inject, reasons = _should_inject_telemetry_context(
                previous_signature=_last_telemetry_signature,
                current_signature=signature,
                last_injected_ts=_last_telemetry_context_sent_at,
                now_ts=now,
            )
            if should_inject:
                semantic_text = parse_telemetry(telemetry_data)
                telemetry_text = (
                    "<<<SENSOR_DATA>>>\n"
                    f"{semantic_text}\n"
                    "<<<END_SENSOR_DATA>>>\n"
                    "INSTRUCTION: Do not vocalize any part of the above sensor data."
                )
                ctx_queue.enqueue(
                    category="telemetry",
                    text=telemetry_text,
                    priority=8,
                    speak=False,
                )
                _last_telemetry_context_sent_at = now
                logger.debug("Telemetry context injected: reasons=%s", ",".join(reasons))
            _last_telemetry_signature = signature
            telemetry_agg.mark_sent(now)

        # Phase 6B: Location context evaluation from GPS
        if _location_ctx_service and ephemeral_ctx.gps:
            try:
                _current_location_ctx = await _location_ctx_service.evaluate(
                    ephemeral_ctx.gps.lat, ephemeral_ctx.gps.lng,
                )
                session_ctx.familiarity_score = _current_location_ctx.familiarity_score
                # Track unique locations visited this session
                place = _current_location_ctx.place_name
                if place and place not in session_meta.locations_visited:
                    session_meta.locations_visited.append(place)
            except Exception:
                logger.debug("Location context evaluation failed", exc_info=True)

        await _process_lod_decision(ephemeral_ctx)

    async def _process_lod_decision(ephemeral_ctx) -> None:
        """Run the LOD decision engine and handle transitions."""
        new_lod, decision_log = decide_lod(
            ephemeral=ephemeral_ctx,
            session=session_ctx,
            profile=user_profile,
        )

        # Phase 6D: LLM micro-adjustment for non-safety LOD decisions
        if _lod_evaluator and new_lod >= 2:
            try:
                adjustment = await _lod_evaluator.evaluate(
                    baseline_lod=new_lod,
                    location_ctx=_current_location_ctx,
                    user_profile=user_profile,
                )
                if adjustment.delta != 0:
                    adjusted = max(1, min(3, new_lod + adjustment.delta))
                    if adjusted != new_lod:
                        logger.info(
                            "LOD micro-adjust: %d -> %d (%s)",
                            new_lod, adjusted, adjustment.reason,
                        )
                        decision_log.triggered_rules.append(
                            f"LLM:{adjustment.reason[:40]}"
                        )
                        new_lod = adjusted
            except Exception:
                logger.debug("LOD evaluator failed", exc_info=True)

        old_lod = session_ctx.current_lod

        if new_lod != old_lod:
            logger.info(
                "LOD transition: %d -> %d (%s) session=%s",
                old_lod, new_lod, decision_log.reason, session_id,
            )

            resume_prompt = on_lod_change(session_ctx, old_lod, new_lod)
            session_ctx.current_lod = new_lod
            session_meta.record_lod_time(new_lod)
            # Clear one-shot voice intent flags after LOD decision consumed them
            session_ctx.user_requested_detail = False
            session_ctx.user_said_stop = False
            telemetry_agg.update_lod(new_lod)
            vad_update = await _sync_runtime_vad_update(new_lod)

            await _send_lod_update(new_lod, ephemeral_ctx, decision_log.reason)

            if resume_prompt:
                ctx_queue.enqueue(
                    category="lod_resume",
                    text=resume_prompt,
                    priority=3,
                    speak=True,
                )
                logger.info("Injected [RESUME] prompt for session %s", session_id)

            await _notify_ios_lod_change(
                new_lod,
                decision_log.reason,
                decision_log.to_debug_dict(),
                vad_update=vad_update,
            )

    # -- Downstream handler --------------------------------------------------

    async def _downstream() -> None:
        """Read events from the Live API and forward to the iOS client.

        Processes session_resumption_update events, transcriptions,
        function calls, and content parts (audio binary / text JSON).
        """
        nonlocal _transcript_buffer, _transcript_buffer_last_update, _turn_had_vision_content, _model_audio_last_seen_at
        nonlocal _last_user_activity_at, _last_interrupt_at, _is_interrupted

        def _start_live_events():
            return runner.run_live(
                session_id=session_id,
                user_id=user_id,
                live_request_queue=live_request_queue,
                run_config=run_config,
            )

        live_events = await asyncio.to_thread(_start_live_events)
        _is_interrupted = False
        try:
            async for event in live_events:
                if stop_downstream.is_set():
                    break

                # --- Session resumption update ---
                if event.live_session_resumption_update:
                    update = event.live_session_resumption_update
                    if update.newHandle:
                        session_manager.update_handle(session_id, update.newHandle)
                    if not await _safe_send_json({
                        "type": MessageType.SESSION_RESUMPTION,
                        "handle": update.newHandle,
                    }):
                        break

                # --- GoAway / connection lifecycle signals (SL-76) ---
                if hasattr(event, "go_away") and event.go_away:
                    retry_ms = 500
                    if hasattr(event.go_away, "time_left"):
                        retry_ms = int(event.go_away.time_left.total_seconds() * 1000) if event.go_away.time_left else 500
                    await _safe_send_json({
                        "type": MessageType.GO_AWAY,
                        "retry_ms": retry_ms,
                        "message": "Server requested reconnection.",
                    })
                    logger.warning("GoAway received, retry_ms=%d", retry_ms)

                # --- Interrupt detection (barge-in) ---
                _event_interrupted = (
                    (hasattr(event, "server_content") and event.server_content
                     and getattr(event.server_content, "interrupted", False))
                    or getattr(event, "interrupted", False)
                )
                if _event_interrupted and not _is_interrupted:
                    now_mono = time.monotonic()
                    if (now_mono - _last_interrupt_at) < _INTERRUPT_DEBOUNCE_SEC:
                        logger.debug("Interrupt debounced (%.0fms since last)",
                                     (now_mono - _last_interrupt_at) * 1000)
                    else:
                        _is_interrupted = True
                        _last_interrupt_at = now_mono
                        _model_audio_last_seen_at = 0.0
                        # Interrupt → force state machine to IDLE (no flush)
                        ctx_queue._transition_to(ModelState.IDLE)
                        # Note: do NOT flush queue on interrupt — user is speaking
                        await _safe_send_json({
                            "type": MessageType.INTERRUPTED,
                        })
                        logger.info("Interrupt detected — suppressing audio forwarding")

                # --- Turn complete: resume audio forwarding + flush buffer ---
                # Step 3: Quiet period confirmation — wait 500ms after
                # turn_complete to confirm Gemini isn't just pausing for a
                # function call.  If new audio arrives within 500ms, stay
                # in DRAINING state instead of transitioning to IDLE.
                if event.turn_complete:
                    if _is_interrupted:
                        logger.info("Turn complete — resuming audio forwarding")
                    _is_interrupted = False
                    # Track vision spoken cooldown
                    if _turn_had_vision_content:
                        ctx_queue.record_vision_spoken()
                        _turn_had_vision_content = False
                    # Flush any remaining transcript buffer
                    if _transcript_buffer:
                        if not await _flush_transcript_buffer():
                            break
                    # Quiet period: wait 500ms before confirming turn is done
                    _tc_audio_before = _model_audio_last_seen_at
                    await asyncio.sleep(0.5)
                    if _model_audio_last_seen_at > _tc_audio_before:
                        # New audio arrived during quiet period — model resumed
                        logger.info("Turn complete revoked — audio arrived during quiet period")
                    else:
                        # Confirmed: model is truly done
                        ctx_queue.on_turn_complete()
                        # Flush queued context injections
                        ctx_queue.flush_or_defer_first_turn(camera_active=_client_camera_active)

                # --- Content parts (audio / text) — BEFORE function calls ---
                # Process audio/text first so that any subsequent
                # inject_immediate (function responses) does not interrupt
                # audio chunks from the same event.
                if event.content and event.content.parts and not _is_interrupted:
                    for part in event.content.parts:
                        if stop_downstream.is_set():
                            break

                        if part.inline_data and part.inline_data.data:
                            _model_audio_last_seen_at = time.monotonic()
                            # Audio chunk → GENERATING→DRAINING or stay DRAINING
                            ctx_queue.set_model_audio_timestamp(_model_audio_last_seen_at)
                            ctx_queue.set_model_speaking(True)
                            audio_data = part.inline_data.data
                            if isinstance(audio_data, str):
                                audio_data = base64.b64decode(audio_data)
                            if not await _safe_send_bytes(audio_data):
                                break

                        elif part.text:
                            # Skip if output_transcription already forwarded
                            # the same text in this event (avoids double-send)
                            if _output_transcription_forwarded:
                                logger.debug(
                                    "Skipped content.parts text (already sent via output_transcription): %s",
                                    part.text[:80],
                                )
                                continue
                            if not await _forward_agent_transcript(part.text):
                                break

                    if stop_downstream.is_set():
                        break

                # --- Function calls from Gemini (Phase 3) ---
                function_calls = _extract_function_calls(event)
                if function_calls:
                    for fc in function_calls:
                        # Guard: intercept hallucinated tool calls
                        if fc.name not in ALL_FUNCTIONS:
                            logger.warning(
                                "Model called non-existent tool %r — returning no-op",
                                fc.name,
                            )
                            await _safe_send_json({
                                "type": MessageType.DEBUG_ACTIVITY,
                                "data": {
                                    "event": "hallucinated_tool_call",
                                    "tool": fc.name,
                                    "args": _json_safe(dict(fc.args) if fc.args else {}),
                                },
                            })
                            from google.genai.types import FunctionResponse as _FR
                            noop_response = _FR(
                                name=fc.name,
                                response={
                                    "status": "unavailable",
                                    "message": (
                                        f"'{fc.name}' does not exist. "
                                        "Use only the tools listed in your instructions. "
                                        "OCR/vision results are injected automatically as context."
                                    ),
                                },
                            )
                            ctx_queue.inject_immediate(types.Content(
                                parts=[types.Part(function_response=noop_response)],
                                role="user",
                            ))
                            continue

                        user_speaking = session_ctx.current_activity_state == "user_speaking"
                        behavior = resolve_tool_behavior(
                            tool_name=fc.name,
                            lod=session_ctx.current_lod,
                            is_user_speaking=user_speaking,
                        )
                        await _emit_tool_event(
                            fc.name,
                            behavior,
                            status="invoked",
                            data={"args": _json_safe(dict(fc.args) if fc.args else {})},
                        )
                        result = await _dispatch_function_call(
                            fc.name,
                            dict(fc.args) if fc.args else {},
                            session_id,
                            user_id,
                        )
                        await _safe_send_json({
                            "type": MessageType.TOOL_RESULT,
                            "tool": fc.name,
                            "behavior": behavior_to_text(behavior),
                            "data": _json_safe(result),
                        })

                        if fc.name in NAVIGATION_FUNCTIONS:
                            await _safe_send_json({
                                "type": MessageType.NAVIGATION_RESULT,
                                "summary": str(result.get("destination_direction") or result.get("destination") or ""),
                                "behavior": behavior_to_text(behavior),
                                "data": _json_safe(result),
                            })
                        elif fc.name == "google_search":
                            await _safe_send_json({
                                "type": MessageType.SEARCH_RESULT,
                                "summary": str(result.get("answer") or ""),
                                "behavior": behavior_to_text(behavior),
                                "data": _json_safe(result),
                            })
                        elif fc.name == "identify_person":
                            await _emit_identity_event(
                                person_name=str(result.get("person_name", "unknown")),
                                matched=bool(result.get("matched", False)),
                                similarity=float(result.get("similarity", 0.0)),
                                source="tool_call",
                            )

                        # Send function response back to the model.
                        # For INTERRUPT-behavior tools (safety-critical), prepend a
                        # prompt prefix instructing Gemini to complete delivery without
                        # stopping mid-sentence, since we cannot switch VAD to
                        # NO_INTERRUPTION mid-session.
                        from google.genai.types import FunctionResponse
                        fr = FunctionResponse(
                            name=fc.name,
                            response=result,
                        )
                        parts = [types.Part(function_response=fr)]
                        if behavior == ToolBehavior.INTERRUPT:
                            parts.insert(0, types.Part(
                                text="[SAFETY ALERT — COMPLETE DELIVERY] "
                                     "Do not stop mid-sentence even if you detect user activity. "
                                     "Deliver the full safety information before yielding."
                            ))
                        content = types.Content(parts=parts, role="user")
                        ctx_queue.inject_immediate(content)
                        logger.info("Sent function response for %s (behavior=%s)", fc.name, behavior_to_text(behavior))

                # --- Output transcription (agent speech-to-text) — buffered ---
                _output_transcription_forwarded = False
                if event.output_transcription and event.output_transcription.text:
                    now_mono = time.monotonic()
                    transcript_history.append({
                        "role": "agent",
                        "text": event.output_transcription.text,
                    })
                    # Detect if model is speaking about vision/scene
                    _out_lower = event.output_transcription.text.lower()
                    if any(kw in _out_lower for kw in (
                        "i see", "i can see", "looking at", "in front of",
                        "ahead of", "around you", "your surroundings",
                        "to your left", "to your right", "on the left", "on the right",
                    )):
                        _turn_had_vision_content = True
                    # Buffer fragments instead of forwarding immediately
                    _transcript_buffer += event.output_transcription.text
                    _transcript_buffer_last_update = now_mono
                    logger.debug("Buffered transcript fragment: %s", event.output_transcription.text[:80])
                    # Flush on sentence boundary
                    if _has_sentence_boundary(_transcript_buffer):
                        if not await _flush_transcript_buffer():
                            break
                        _output_transcription_forwarded = True
                    # Flush on timeout (stale buffer)
                    elif (_transcript_buffer_last_update > 0
                          and (now_mono - _transcript_buffer_last_update) > _TRANSCRIPT_FLUSH_TIMEOUT_SEC):
                        if not await _flush_transcript_buffer():
                            break
                        _output_transcription_forwarded = True

                # --- Input transcription (user speech-to-text) with echo detection ---
                if event.input_transcription and event.input_transcription.text:
                    now_mono = time.monotonic()
                    input_text = event.input_transcription.text
                    if _is_likely_echo(input_text, now_mono):
                        logger.debug("Echo detected, reclassifying: %s", input_text[:120])
                        transcript_history.append({
                            "role": "echo",
                            "text": input_text,
                        })
                        await _safe_send_json({
                            "type": MessageType.TRANSCRIPT,
                            "text": input_text,
                            "role": "echo",
                        })
                        # Tell Gemini to disregard the echo input (queued to avoid audio overlap)
                        ctx_queue.enqueue(
                            category="echo_cancel",
                            text="[SYSTEM: The previous audio input was an echo of your own speech. Do not respond to it.]",
                            priority=1,
                            speak=False,
                        )
                        logger.info("Queued echo cancellation for Gemini")
                    else:
                        transcript_history.append({
                            "role": "user",
                            "text": input_text,
                        })
                        _last_user_activity_at = time.monotonic()
                        session_meta.record_interaction()
                        if not await _safe_send_json({
                            "type": MessageType.TRANSCRIPT,
                            "text": input_text,
                            "role": "user",
                        }):
                            break

                        # Voice intent → LOD session flags
                        intent = _detect_voice_intent(input_text)
                        if intent == "detail":
                            session_ctx.user_requested_detail = True
                            session_ctx.user_said_stop = False
                        elif intent == "stop":
                            session_ctx.user_said_stop = True
                            session_ctx.user_requested_detail = False

                # (Content parts already processed above, before function calls.)

        except WebSocketDisconnect:
            stop_downstream.set()
            logger.info("Client disconnected (downstream): user=%s session=%s", user_id, session_id)
        except Exception as exc:
            stop_downstream.set()
            logger.exception("Error in downstream handler: user=%s session=%s", user_id, session_id)

            # Classify: programming bugs should not trigger client reconnect loops
            is_fatal = isinstance(exc, (UnboundLocalError, NameError, AttributeError, TypeError))
            close_code = 1008 if is_fatal else 1011

            await _safe_send_json({
                "type": MessageType.ERROR,
                "error": "Fatal server error." if is_fatal else "Live session failed. Reconnecting...",
                "fatal": is_fatal,
            })
            try:
                await websocket.close(code=close_code, reason="fatal_error" if is_fatal else "downstream_error")
            except Exception:
                pass

    try:
        await asyncio.wait_for(
            asyncio.gather(_upstream(), _downstream()),
            timeout=SESSION_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        logger.info(
            "Session timeout (%ds): user=%s session=%s",
            SESSION_TIMEOUT_SEC, user_id, session_id,
        )
    except Exception:
        logger.exception("Session error: user=%s session=%s", user_id, session_id)
    finally:
        # Phase 5: Auto-extract memories from session transcript
        if _memory_extractor_available and _memory_available and transcript_history:
            try:
                extractor = MemoryExtractor()
                bank = MemoryBankService(user_id)
                budget = memory_budget or MemoryBudgetTracker()
                count = await asyncio.to_thread(
                    extractor.extract_and_store,
                    user_id=user_id,
                    session_id=session_id,
                    transcript_history=transcript_history,
                    memory_bank=bank,
                    budget=budget,
                )
                logger.info(
                    "Auto-extracted %d memories for user=%s session=%s",
                    count, user_id, session_id,
                )
            except Exception:
                logger.exception(
                    "Memory auto-extraction failed for user=%s session=%s",
                    user_id, session_id,
                )

        # Phase 6: Record ACE data to session metadata (best-effort)
        try:
            if _current_location_ctx:
                place = getattr(_current_location_ctx, "place_name", "")
                if place:
                    session_meta.locations_visited.append(place)
        except Exception:
            logger.debug("ACE session end data failed", exc_info=True)

        # Write session metadata before cleanup
        session_meta.space_transitions = list(session_ctx.space_transitions)
        session_meta.set_trip_purpose(session_ctx.trip_purpose or "")
        await session_meta.write_session_end()

        ctx_queue.stop()
        live_request_queue.close()
        _ocr_clear_session(session_id)
        session_manager.remove_session(session_id)
        if _memory_available:
            evict_stale_banks(max_age_sec=SESSION_TIMEOUT_SEC)
        logger.info("Session cleaned up: user=%s session=%s", user_id, session_id)


# ---------------------------------------------------------------------------
# Sub-agent result formatters
# ---------------------------------------------------------------------------


def _format_vision_result(result: dict, lod: int) -> str:
    """Format vision analysis result for Gemini context injection."""
    parts = ["[VISION ANALYSIS]"]

    nav = result.get("navigation_info", {})
    if lod >= 2:
        entrances = nav.get("entrances", [])
        if entrances:
            parts.append("Entrances: " + ", ".join(entrances))
        paths = nav.get("paths", [])
        if paths:
            parts.append("Paths: " + ", ".join(paths))
        landmarks = nav.get("landmarks", [])
        if landmarks:
            parts.append("Landmarks: " + ", ".join(landmarks))

    desc = result.get("scene_description", "")
    if desc:
        parts.append(f"Scene: {desc}")

    text = result.get("detected_text")
    if text and lod >= 2:
        parts.append(f"Visible text: {text}")

    count = result.get("people_count", 0)
    if count > 0 and lod >= 2:
        parts.append(f"People visible: {count}")

    return "\n".join(parts)


def _format_face_results(known_faces: list[dict]) -> str:
    """Format face recognition results for SILENT context injection."""
    parts = ["[FACE ID]"]
    for face in known_faces:
        name = face["person_name"]
        rel = face.get("relationship", "")
        sim = face.get("similarity", 0)
        position = face.get("bbox", [])
        desc = f"{name}"
        if rel:
            desc += f" ({rel})"
        desc += f" (confidence: {sim:.0%})"
        parts.append(desc)
    return "\n".join(parts)


def _format_ocr_result(result: dict) -> str:
    """Format OCR result for Gemini context injection."""
    parts = ["[OCR RESULT]"]

    text_type = result.get("text_type", "unknown")
    parts.append(f"Type: {text_type}")

    items = result.get("items", [])
    if items:
        parts.append("Items:")
        for item in items:
            parts.append(f"  - {item}")
    else:
        text = result.get("text", "")
        if text:
            parts.append(f"Text: {text}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
