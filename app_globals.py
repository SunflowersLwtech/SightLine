"""Application-wide singletons, enums, constants, and capability flags.

Import order contract: server.py must call load_dotenv() and configure
logging BEFORE importing this module.
"""

from __future__ import annotations

import logging
import os
from enum import Enum

from google.adk.runners import Runner

from agents.orchestrator import create_orchestrator_agent
from live_api.session_manager import SessionManager, create_session_service

logger = logging.getLogger("sightline.server")


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


LIVE_MODEL = os.getenv("GEMINI_LIVE_MODEL", "gemini-live-2.5-flash-native-audio")
PORT = int(os.getenv("PORT", "8100"))
SESSION_TIMEOUT_SEC = int(os.getenv("SESSION_TIMEOUT", "3600"))
WS_INACTIVITY_TIMEOUT_SEC = int(os.getenv("WS_INACTIVITY_TIMEOUT", str(SESSION_TIMEOUT_SEC)))

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
    "VertexAiSessionService",
    "VertexAISessionService",
)

AGENT_TEXT_REPEAT_SUPPRESS_SEC = 6.0
VISION_REPEAT_SUPPRESS_SEC = 8.0
OCR_REPEAT_SUPPRESS_SEC = 8.0
VISION_PREFEEDBACK_COOLDOWN_SEC = 12.0
OCR_PREFEEDBACK_COOLDOWN_SEC = 15.0
PASSIVE_SPEECH_GUARD_SEC = 8.0
STARTUP_HABIT_DETECT_TIMEOUT_SEC = float(os.getenv("STARTUP_HABIT_DETECT_TIMEOUT_SEC", "2.0"))

_vision_available = False
_ocr_available = False
_face_available = False

try:
    from agents.vision_agent import analyze_scene

    _ = analyze_scene
    _vision_available = True
except ImportError:
    logger.warning("Vision agent not available (missing dependencies)")

try:
    from agents.ocr_agent import extract_text

    _ = extract_text
    _ocr_available = True
except ImportError:
    logger.warning("OCR agent not available (missing dependencies)")

load_face_library = None
try:
    from agents.face_agent import identify_persons_in_frame
    from tools.face_tools import load_face_library

    _ = (identify_persons_in_frame, load_face_library)
    _face_available = True
except ImportError:
    logger.warning("Face agent not available (missing dependencies)")

FACE_LIBRARY_REFRESH_SEC: float = 60.0

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
    "remember_entity": ("memory", "WHEN_IDLE"),
    "what_do_you_remember": ("memory", "WHEN_IDLE"),
    "forget_entity": ("memory", "SILENT"),
    "forget_recent_memory": ("memory", "SILENT"),
    "extract_text_from_camera": ("ocr", "WHEN_IDLE"),
}

_memory_available = False
_memory_extractor_available = False
MemoryBankService = None
MemoryBudgetTracker = None
MEMORY_WRITE_BUDGET = None

try:
    from memory.memory_bank import (
        MemoryBankService,
        evict_stale_banks,
        load_relevant_memories,
    )
    from memory.memory_budget import MEMORY_WRITE_BUDGET, MemoryBudgetTracker

    _ = (MemoryBankService, evict_stale_banks, MEMORY_WRITE_BUDGET, MemoryBudgetTracker)
    _memory_available = True
except ImportError:
    logger.warning("Memory module not available")

    def load_relevant_memories(user_id: str, context: str, top_k: int = 3) -> list[str]:
        return []

MemoryExtractor = None
try:
    from memory.memory_extractor import MemoryExtractor

    _ = MemoryExtractor
    _memory_extractor_available = True
except ImportError:
    logger.warning("Memory extractor not available")
