"""WebSocket router extracted from ``server.py``."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket
from google.adk.agents.live_request_queue import LiveRequestQueue

from app_globals import (
    STARTUP_HABIT_DETECT_TIMEOUT_SEC,
    MemoryBudgetTracker,
    _face_available,
    _memory_available,
    load_face_library,
    load_relevant_memories,
    runner,
    session_manager,
)
from context_injection import ContextInjectionQueue, TokenBudgetMonitor
from lod.telemetry_aggregator import TelemetryAggregator
from session_state import SessionState
from telemetry.session_meta_tracker import SessionMetaTracker

logger = logging.getLogger("sightline.server")

router = APIRouter()


@router.websocket("/ws/{user_id}/{session_id}")
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
    cached_resume_handle = session_manager.get_handle(session_id)
    resume_requested = bool(
        resume_handle or cached_resume_handle or session_manager.has_resumable_state(session_id)
    )
    if resume_handle:
        session_manager.update_handle(session_id, resume_handle)
        logger.info("Received resume handle from client for session %s", session_id)

    # -- Per-session LOD state -----------------------------------------------
    telemetry_agg = TelemetryAggregator()
    session_ctx = session_manager.get_session_context(session_id)
    user_profile_task = asyncio.create_task(session_manager.load_user_profile(user_id))
    initial_memories_task = None
    if _memory_available:
        initial_memories_task = asyncio.create_task(
            asyncio.to_thread(
                load_relevant_memories,
                user_id,
                session_ctx.active_task or session_ctx.trip_purpose or "",
                3,
            )
        )
    _initial_face_library_task = None
    if _face_available and load_face_library is not None:
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
    from tools.dedup import AudioGate, MutualExclusionFilter, ToolCallDeduplicator

    tool_dedup = ToolCallDeduplicator()
    tool_mutex = MutualExclusionFilter()
    audio_gate = AudioGate()

    run_config = session_manager.get_run_config(
        session_id,
        lod=session_ctx.current_lod,
        language_code=user_profile.language,
    )

    # -- Phase 6: Context Engine initialisation ---------------------------------
    _location_ctx_service = None
    _lod_evaluator = None
    _assembled_profile = None
    try:
        from context.habit_detector import HabitDetector
        from context.location_context import LocationContextService
        from context.lod_evaluator import LODEvaluator
        from context.profile_assembler import ProfileAssembler

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
                        len(state.proactive_hints),
                        user_id,
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
    memory_budget = MemoryBudgetTracker() if _memory_available and MemoryBudgetTracker is not None else None

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
        resume_requested=resume_requested,
    )
    await handler.run()
