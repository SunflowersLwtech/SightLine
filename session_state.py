"""Per-session mutable state for websocket_endpoint().

Replaces the 70+ nonlocal variables and shared locals inside the
WebSocket handler with a single typed dataclass.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from context.location_context import LocationContext


@dataclass
class SessionState:
    """All mutable per-session state that was previously held as nonlocal
    variables inside ``websocket_endpoint()``.

    Fields are grouped by concern and carry the same defaults that
    the original locals used.
    """

    # -- Vision analysis -----------------------------------------------------
    vision_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    vision_in_progress: bool = False
    last_vision_time: float = 0.0
    last_vision_context_text: str = ""
    last_vision_context_sent_at: float = 0.0
    last_vision_prefeedback_at: float = 0.0
    first_vision_after_camera: bool = True
    frame_seq: int = 0

    # -- Face recognition ----------------------------------------------------
    face_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    face_in_progress: bool = False
    face_runtime_available: bool = False  # caller sets from _face_available
    face_unavailable_notified: bool = False
    face_library: list[dict] = field(default_factory=list)
    face_library_loaded_at: float = 0.0
    face_library_refresh_task: asyncio.Task | None = None
    face_library_task: asyncio.Task | None = None
    face_consecutive_misses: int = 0
    face_skip_counter: int = 0

    # -- OCR -----------------------------------------------------------------
    last_ocr_context_text: str = ""
    last_ocr_context_sent_at: float = 0.0
    last_ocr_prefeedback_at: float = 0.0

    # -- Audio / interrupt ---------------------------------------------------
    is_interrupted: bool = False
    last_interrupt_at: float = 0.0
    model_audio_last_seen_at: float = 0.0
    downstream_retry_count: int = 0

    # -- Transcript buffering ------------------------------------------------
    transcript_buffer: str = ""
    transcript_buffer_started_at: float = 0.0
    last_agent_text: str = ""
    last_agent_text_sent_at: float = 0.0
    recent_agent_texts: list[tuple[float, str]] = field(default_factory=list)
    turn_had_vision_content: bool = False

    # -- Telemetry / LOD -----------------------------------------------------
    last_telemetry_signature: dict[str, object] | None = None
    last_telemetry_context_sent_at: float = 0.0
    current_location_ctx: LocationContext | None = None
    proactive_hints: list[Any] | None = None

    # -- Camera state --------------------------------------------------------
    client_camera_active: bool = False
    camera_activated_at: float = 0.0
    last_frame_to_gemini_at: float = 0.0

    # -- Agent repeat --------------------------------------------------------
    allow_agent_repeat_until: float = 0.0

    # -- User activity -------------------------------------------------------
    last_user_activity_at: float = field(default_factory=time.monotonic)

    # -- Memory (Phase 4) ----------------------------------------------------
    memory_top3: list[str] = field(default_factory=list)
    memory_top3_detailed: list[dict] = field(default_factory=list)

    # -- WebSocket write lock ------------------------------------------------
    ws_write_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # -- Constants (per-session, not global) ----------------------------------
    INTERRUPT_DEBOUNCE_SEC: float = field(default=1.0, repr=False)
    DOWNSTREAM_MAX_RETRIES: int = field(default=5, repr=False)
    FACE_BACKOFF_THRESHOLD: int = field(default=3, repr=False)
    FACE_BACKOFF_SKIP_CYCLES: int = field(default=2, repr=False)
    TRANSCRIPT_FLUSH_TIMEOUT_SEC: float = field(default=1.5, repr=False)
    CAMERA_GRACE_PERIOD_SEC: float = field(default=12.0, repr=False)
    FRAME_TO_GEMINI_INTERVAL: float = field(default=2.0, repr=False)
    SENTENCE_BOUNDARY_RE: re.Pattern = field(
        default_factory=lambda: re.compile(r"[。！？.!?\n]"), repr=False,
    )
    TRANSCRIPT_HISTORY_MAX_ENTRIES: int = field(default=1500, repr=False)
    MAGIC_AUDIO: int = field(default=0x01, repr=False)
    MAGIC_IMAGE: int = field(default=0x02, repr=False)

    # -- Transcript history (deque, created post-init) -----------------------
    transcript_history: deque = field(default=None, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.transcript_history is None:
            self.transcript_history = deque(maxlen=self.TRANSCRIPT_HISTORY_MAX_ENTRIES)
