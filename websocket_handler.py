"""WebSocket handler — encapsulates the per-session async logic.

Extracted from ``server.websocket_endpoint()`` to reduce the monolithic
closure.  All former nested functions are now async methods on
:class:`WebSocketHandler`; shared local state lives in ``SessionState``
(created by Phase 1).
"""

import asyncio
import base64
import json
import logging
import time
from collections import deque
from datetime import datetime, timezone

from fastapi import WebSocket, WebSocketDisconnect
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.runners import Runner
from google.genai import types
from starlette.websockets import WebSocketState

from session_state import SessionState
from server import (
    MessageType,
    ModelState,
    ContextInjectionQueue,
    TokenBudgetMonitor,
    _INTERNAL_TAG_RE,
    _coerce_bool,
    _is_repeated_text,
    _should_reset_interrupted_on_activity_start,
    _build_telemetry_signature,
    _changed_signature_fields,
    _should_inject_telemetry_context,
    _detect_voice_intent,
    _allow_navigation_tool_call,
    _json_safe,
    _extract_function_calls,
    _dispatch_function_call,
    _format_vision_result,
    _format_face_results,
    _format_ocr_result,
    build_full_dynamic_prompt,
    TELEMETRY_FORCE_REFRESH_SEC,
    AGENT_TEXT_REPEAT_SUPPRESS_SEC,
    VISION_REPEAT_SUPPRESS_SEC,
    OCR_REPEAT_SUPPRESS_SEC,
    VISION_PREFEEDBACK_COOLDOWN_SEC,
    OCR_PREFEEDBACK_COOLDOWN_SEC,
    PASSIVE_SPEECH_GUARD_SEC,
    FACE_LIBRARY_REFRESH_SEC,
    WS_INACTIVITY_TIMEOUT_SEC,
    SESSION_TIMEOUT_SEC,
    STARTUP_HABIT_DETECT_TIMEOUT_SEC,
    _NEEDS_SESSION_ID_MAPPING,
    session_manager,
    _vision_available,
    _ocr_available,
    _face_available,
    _memory_available,
    _memory_extractor_available,
    _TOOL_CATEGORY_MAP,
    _ocr_set_latest_frame,
    _ocr_clear_session,
)
from lod import (
    build_lod_update_message,
    decide_lod,
    on_lod_change,
)
from lod.lod_engine import should_speak
from telemetry.telemetry_parser import parse_telemetry, parse_telemetry_to_ephemeral
from tools import ALL_FUNCTIONS, ALL_TOOL_DECLARATIONS
from tools.navigation import NAVIGATION_FUNCTIONS
from tools.tool_behavior import ToolBehavior, behavior_to_text, resolve_tool_behavior

logger = logging.getLogger("sightline.server")


class WebSocketHandler:
    """Manages a single WebSocket session's upstream/downstream lifecycle.

    All former nested functions from ``websocket_endpoint()`` live here as
    async methods.  Per-session mutable state is held in a ``SessionState``
    dataclass instance (``self.state``).
    """

    def __init__(
        self,
        *,
        websocket: WebSocket,
        user_id: str,
        session_id: str,
        state: SessionState,
        live_request_queue: LiveRequestQueue,
        runner: Runner,
        ctx_queue: ContextInjectionQueue,
        token_monitor: TokenBudgetMonitor,
        session_ctx,
        session_meta,
        user_profile,
        telemetry_agg,
        stop_downstream: asyncio.Event,
        tool_dedup,
        tool_mutex,
        audio_gate,
        run_config,
        location_ctx_service,
        lod_evaluator,
        assembled_profile,
        memory_budget,
        initial_memories: list | None = None,
    ) -> None:
        self.websocket = websocket
        self.user_id = user_id
        self.session_id = session_id
        self.state = state
        self.live_request_queue = live_request_queue
        self.runner = runner
        self.ctx_queue = ctx_queue
        self.token_monitor = token_monitor
        self.session_ctx = session_ctx
        self.session_meta = session_meta
        self.user_profile = user_profile
        self.telemetry_agg = telemetry_agg
        self.stop_downstream = stop_downstream
        self.tool_dedup = tool_dedup
        self.tool_mutex = tool_mutex
        self.audio_gate = audio_gate
        self.run_config = run_config
        self._location_ctx_service = location_ctx_service
        self._lod_evaluator = lod_evaluator
        self._assembled_profile = assembled_profile
        self.memory_budget = memory_budget
        self._initial_memories = initial_memories or []

    # ── Main entry point ───────────────────────────────────────────────

    async def run(self) -> None:
        """Run the upstream/downstream loop until the session ends.

        Sends session_ready, tools_manifest, and the initial greeting/context
        before starting the upstream/downstream tasks.
        """
        # Notify client the WebSocket is live
        if not await self._safe_send_json({"type": MessageType.SESSION_READY}):
            logger.info(
                "WebSocket closed before session_ready: user=%s session=%s",
                self.user_id, self.session_id,
            )
            return

        asyncio.create_task(self.session_meta.write_session_start())

        # Send tools manifest so iOS Dev Console shows tool/context status
        await self._safe_send_json(self._build_tools_manifest())

        # Inject combined context (LOD + greeting) so model speaks once
        _initial_prompt = build_full_dynamic_prompt(
            lod=self.session_ctx.current_lod,
            profile=self.user_profile,
            ephemeral_semantic="",
            session=self.session_ctx,
            memories=self._initial_memories if self._initial_memories else None,
            assembled_profile=self._assembled_profile,
        )
        _greeting_parts: list[str] = [
            "[SESSION START] Greet the user briefly (1-2 sentences).",
            "Let them know you're ready to help.",
        ]
        if self.user_profile and self.user_profile.preferred_name:
            _greeting_parts.append(
                f"Address them as '{self.user_profile.preferred_name}'."
            )
        _greeting_parts.append("Keep it natural and concise — no instructions or tutorials.")
        _combined_content = types.Content(
            parts=[
                types.Part(text="[CONTEXT UPDATE - DO NOT SPEAK]\n" + _initial_prompt),
                types.Part(text=" ".join(_greeting_parts)),
            ],
            role="user",
        )
        self.ctx_queue.inject_immediate(_combined_content)
        logger.info(
            "Injected combined context (LOD %d) + greeting for session %s",
            self.session_ctx.current_lod, self.session_id,
        )

        if self.state.face_library_task is not None:
            self.state.face_library_refresh_task = asyncio.create_task(
                self._finish_initial_face_library_load()
            )

        try:
            upstream_task = asyncio.create_task(self._upstream())
            downstream_task = asyncio.create_task(self._downstream())
            done, pending = await asyncio.wait(
                [upstream_task, downstream_task],
                timeout=SESSION_TIMEOUT_SEC,
                return_when=asyncio.FIRST_COMPLETED,
            )
            self.stop_downstream.set()
            for task in pending:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            for task in done:
                if task.exception() and not isinstance(
                    task.exception(), asyncio.CancelledError
                ):
                    logger.exception(
                        "Session task failed: user=%s session=%s",
                        self.user_id, self.session_id,
                        exc_info=task.exception(),
                    )
        except asyncio.TimeoutError:
            logger.info(
                "Session timeout (%ds): user=%s session=%s",
                SESSION_TIMEOUT_SEC, self.user_id, self.session_id,
            )
        except Exception:
            logger.exception(
                "Session error: user=%s session=%s",
                self.user_id, self.session_id,
            )
        finally:
            await self._cleanup()

    async def _cleanup(self) -> None:
        """Post-session cleanup: extract memories, write metadata, release resources."""
        if _memory_extractor_available and _memory_available and self.state.transcript_history:
            try:
                from memory.memory_extractor import MemoryExtractor
                from memory.memory_bank import MemoryBankService
                from memory.memory_budget import MemoryBudgetTracker

                extractor = MemoryExtractor()
                bank = MemoryBankService(self.user_id)
                budget = self.memory_budget or MemoryBudgetTracker()
                count = await asyncio.to_thread(
                    extractor.extract_and_store,
                    user_id=self.user_id,
                    session_id=self.session_id,
                    transcript_history=list(self.state.transcript_history),
                    memory_bank=bank,
                    budget=budget,
                )
                logger.info(
                    "Auto-extracted %d memories for user=%s session=%s",
                    count, self.user_id, self.session_id,
                )
            except Exception:
                logger.exception(
                    "Memory auto-extraction failed for user=%s session=%s",
                    self.user_id, self.session_id,
                )

        try:
            if self.state.current_location_ctx:
                place = getattr(self.state.current_location_ctx, "place_name", "")
                if place and place not in self.session_meta.locations_visited:
                    self.session_meta.locations_visited.append(place)
        except Exception:
            logger.debug("ACE session end data failed", exc_info=True)

        self.session_meta.space_transitions = list(self.session_ctx.space_transitions)
        self.session_meta.set_trip_purpose(self.session_ctx.trip_purpose or "")
        await self.session_meta.write_session_end()

        self.ctx_queue.stop()
        self.live_request_queue.close()
        _ocr_clear_session(self.session_id)
        session_manager.remove_session(self.session_id)
        if _memory_available:
            from memory.memory_bank import evict_stale_banks
            evict_stale_banks(max_age_sec=SESSION_TIMEOUT_SEC)
        logger.info("Session cleaned up: user=%s session=%s", self.user_id, self.session_id)

    # ── Transcript helpers ─────────────────────────────────────────────

    def _has_sentence_boundary(self, text: str) -> bool:
        return bool(self.state.SENTENCE_BOUNDARY_RE.search(text))

    async def _flush_transcript_buffer(self) -> bool:
        text = self.state.transcript_buffer.strip()
        self.state.transcript_buffer = ""
        self.state.transcript_buffer_started_at = 0.0
        if not text:
            return True
        now_mono = time.monotonic()
        self.state.recent_agent_texts.append((now_mono, text))
        cutoff = now_mono - 10.0
        while self.state.recent_agent_texts and self.state.recent_agent_texts[0][0] < cutoff:
            self.state.recent_agent_texts.pop(0)
        return await self._forward_agent_transcript(text)

    def _is_websocket_open(self) -> bool:
        return (
            self.websocket.client_state == WebSocketState.CONNECTED
            and self.websocket.application_state == WebSocketState.CONNECTED
        )

    async def _safe_send_json(self, payload: dict) -> bool:
        if not self._is_websocket_open():
            self.stop_downstream.set()
            return False
        try:
            async with self.state.ws_write_lock:
                await self.websocket.send_json(payload)
            return True
        except (WebSocketDisconnect, RuntimeError):
            self.stop_downstream.set()
            return False

    async def _safe_send_bytes(self, payload: bytes) -> bool:
        if not self._is_websocket_open():
            self.stop_downstream.set()
            return False
        try:
            async with self.state.ws_write_lock:
                await self.websocket.send_bytes(payload)
            return True
        except (WebSocketDisconnect, RuntimeError):
            self.stop_downstream.set()
            return False

    async def _forward_agent_transcript(self, text: str) -> bool:
        now_mono = time.monotonic()
        can_repeat = now_mono <= self.state.allow_agent_repeat_until
        is_repeat = _is_repeated_text(
            text,
            previous_text=self.state.last_agent_text,
            now_ts=now_mono,
            previous_ts=self.state.last_agent_text_sent_at,
            cooldown_sec=AGENT_TEXT_REPEAT_SUPPRESS_SEC,
        )
        if is_repeat and not can_repeat:
            logger.debug("Suppressed repeated downstream transcript: %s", text[:120])
            return True
        clean_text = _INTERNAL_TAG_RE.sub("", text).strip()
        if not clean_text:
            return True
        sent = await self._safe_send_json({
            "type": MessageType.TRANSCRIPT,
            "text": clean_text,
            "role": "agent",
        })
        if sent:
            self.state.last_agent_text = clean_text
            self.state.last_agent_text_sent_at = now_mono
        return sent

    def _is_likely_echo(self, candidate: str, now_ts: float) -> bool:
        words_candidate = set(candidate.lower().split())
        model_speaking = (now_ts - self.state.model_audio_last_seen_at) < 3.0
        min_words = 1 if model_speaking else 3
        jaccard_threshold = 0.35 if model_speaking else 0.6
        window_sec = 8.0 if model_speaking else 5.0
        if len(words_candidate) < min_words:
            return False
        cutoff = now_ts - window_sec
        for ts, agent_text in reversed(self.state.recent_agent_texts):
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

    # ── Event emitters ─────────────────────────────────────────────────

    async def _emit_tool_event(self, tool: str, behavior, *, status: str, data: dict | None = None) -> None:
        payload: dict = {
            "type": MessageType.TOOL_EVENT,
            "tool": tool,
            "behavior": behavior_to_text(behavior),
            "status": status,
        }
        if data:
            payload["data"] = _json_safe(data)
        await self._safe_send_json(payload)

    async def _emit_capability_degraded(self, capability: str, reason: str, recoverable: bool = True) -> None:
        await self._safe_send_json({
            "type": MessageType.CAPABILITY_DEGRADED,
            "capability": capability,
            "reason": reason,
            "recoverable": recoverable,
        })

    async def _emit_identity_event(self, *, person_name: str, matched: bool, similarity: float = 0.0, source: str = "face_pipeline") -> None:
        payload = {
            "type": MessageType.IDENTITY_UPDATE,
            "person_name": person_name,
            "matched": matched,
            "similarity": similarity,
            "source": source,
            "behavior": behavior_to_text(ToolBehavior.SILENT),
        }
        await self._safe_send_json(payload)
        if matched:
            await self._safe_send_json({
                "type": MessageType.PERSON_IDENTIFIED,
                "person_name": person_name,
                "similarity": similarity,
                "source": source,
                "behavior": behavior_to_text(ToolBehavior.SILENT),
            })

    # ── Face library helpers ───────────────────────────────────────────

    async def _finish_initial_face_library_load(self) -> None:
        if self.state.face_library_task is None:
            return
        try:
            loaded = await self.state.face_library_task
            self.state.face_library = loaded
            self.state.face_library_loaded_at = time.monotonic()
            logger.info("Loaded %d face(s) for user %s", len(self.state.face_library), self.user_id)
        except Exception:
            logger.exception("Failed to load face library for user %s", self.user_id)
        finally:
            self.state.face_library_task = None
            self.state.face_library_refresh_task = None

    async def _refresh_face_library_background(self) -> None:
        try:
            from tools.face_tools import load_face_library
            refreshed = await asyncio.to_thread(load_face_library, self.user_id)
            self.state.face_library = refreshed
            self.state.face_library_loaded_at = time.monotonic()
            logger.info("Refreshed face library (%d faces) for user %s", len(self.state.face_library), self.user_id)
        except Exception:
            logger.exception("Failed to refresh face library for user %s", self.user_id)
        finally:
            self.state.face_library_refresh_task = None

    async def _inject_face_memories(self, known_faces: list[dict]) -> None:
        if not _memory_available:
            return
        from memory.memory_bank import load_relevant_memories

        names = [
            str(face.get("person_name", "")).strip()
            for face in known_faces
            if str(face.get("person_name", "")).strip() and face.get("person_name") != "unknown"
        ]
        if not names:
            return
        memory_tasks = [
            asyncio.to_thread(load_relevant_memories, self.user_id, f"person {name}", 2)
            for name in names
        ]
        results = await asyncio.gather(*memory_tasks, return_exceptions=True)
        memory_lines: list[str] = []
        for name, result in zip(names, results, strict=False):
            if isinstance(result, Exception):
                logger.debug("Failed to load memories for person %s", name, exc_info=True)
                continue
            if not result:
                continue
            memory_lines.append(f"{name}:")
            memory_lines.extend(f"- {memory}" for memory in result)
        if not memory_lines:
            return
        self.ctx_queue.enqueue(
            category="face_memory",
            text="[FACE MEMORY]\n" + "\n".join(memory_lines),
            priority=3,
            speak=False,
        )

    # ── Tools manifest ─────────────────────────────────────────────────

    def _build_tools_manifest(self) -> dict:
        tools_list = [
            {
                "name": decl["name"],
                "category": _TOOL_CATEGORY_MAP.get(decl["name"], ("unknown", "WHEN_IDLE"))[0],
                "behavior": _TOOL_CATEGORY_MAP.get(decl["name"], ("unknown", "WHEN_IDLE"))[1],
                "description": decl.get("description", ""),
            }
            for decl in ALL_TOOL_DECLARATIONS
        ]
        for mem_name in ("preload_memory", "remember_entity", "what_do_you_remember", "forget_entity", "forget_recent_memory"):
            if mem_name in _TOOL_CATEGORY_MAP:
                cat, beh = _TOOL_CATEGORY_MAP[mem_name]
                tools_list.append({"name": mem_name, "category": cat, "behavior": beh, "description": ""})

        _entity_graph_available = False
        try:
            from context.entity_graph import EntityGraphService  # noqa: F811
            _entity_graph_available = True
        except ImportError:
            pass

        context_modules = [
            {"name": "LocationContext", "status": "ready" if self._location_ctx_service is not None else "unavailable"},
            {"name": "LODEvaluator", "status": "ready" if self._lod_evaluator is not None else "unavailable"},
            {"name": "ProfileAssembler", "status": "ready" if self._assembled_profile is not None else "unavailable"},
            {"name": "HabitDetector", "status": "ready" if self.state.proactive_hints is not None else "unavailable"},
            {"name": "SceneMatcher", "status": "ready" if self._location_ctx_service is not None else "unavailable"},
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

    # ── LOD helpers ────────────────────────────────────────────────────

    async def _send_lod_update(self, new_lod: int, ephemeral_ctx, reason: str) -> None:
        memories = await self._load_session_memories(
            context_hint=self.session_ctx.active_task or self.session_ctx.trip_purpose or ""
        )
        lod_message = build_lod_update_message(
            lod=new_lod,
            ephemeral=ephemeral_ctx,
            session=self.session_ctx,
            profile=self.user_profile,
            reason=reason,
            memories=memories,
            assembled_profile=self._assembled_profile,
            location_ctx=self.state.current_location_ctx,
        )
        self.ctx_queue.enqueue(category="lod", text=lod_message, priority=3, speak=False)
        logger.info("Injected [LOD UPDATE] -> LOD %d (%s)", new_lod, reason)

    async def _load_session_memories(self, context_hint: str = "") -> list[str]:
        try:
            if _memory_available:
                from memory.memory_bank import MemoryBankService

                def _retrieve_memories_sync(hint: str) -> list[dict]:
                    bank = MemoryBankService(self.user_id)
                    return bank.retrieve_memories(hint, top_k=3)

                raw_results = await asyncio.to_thread(_retrieve_memories_sync, context_hint)
                self.state.memory_top3 = [m["content"] for m in raw_results][:3]
                self.state.memory_top3_detailed = [{
                    "content": m.get("content", "")[:120],
                    "category": m.get("category", "general"),
                    "importance": round(float(m.get("importance", 0.5)), 2),
                    "score": round(float(m.get("_composite_score", 0)), 3),
                } for m in raw_results][:3]
                return self.state.memory_top3
            return []
        except Exception:
            logger.exception("Failed to load memories for user %s", self.user_id)
            return []

    async def _sync_runtime_vad_update(self, new_lod: int) -> dict:
        from live_api.session_manager import (
            build_vad_runtime_update_message,
            build_vad_runtime_update_payload,
            supports_runtime_vad_reconfiguration,
        )
        supported, reason = supports_runtime_vad_reconfiguration()
        payload = build_vad_runtime_update_payload(new_lod)
        payload["runtime_hot_reconfig_supported"] = supported
        payload["runtime_note"] = "transport_hot_update_applied" if supported else reason
        self.ctx_queue.enqueue(category="vad", text=build_vad_runtime_update_message(new_lod), priority=7, speak=False)
        if supported:
            logger.info("Injected runtime VAD update payload for LOD %d: %s", new_lod, payload)
        else:
            logger.warning("Runtime VAD transport hot-update unavailable (%s); injected sync marker only: %s", reason, payload)
        return payload

    async def _notify_ios_lod_change(self, new_lod: int, reason: str, debug_dict: dict, vad_update: dict | None = None) -> None:
        await self._safe_send_json({"type": MessageType.LOD_UPDATE, "lod": new_lod, "reason": reason})
        debug_dict["memory_top3"] = self.state.memory_top3
        debug_dict["memory_top3_detailed"] = self.state.memory_top3_detailed
        if vad_update:
            debug_dict["vad_update"] = vad_update
        await self._safe_send_json({"type": MessageType.DEBUG_LOD, "data": debug_dict})

    async def _emit_activity_debug_event(self, *, event_name: str, queue_status: str, queue_note: str = "", source: str = "ios_client") -> None:
        ts = datetime.now(timezone.utc)
        is_activity_start = event_name == "activity_start"
        self.session_ctx.current_activity_state = "user_speaking" if is_activity_start else "idle"
        self.session_ctx.last_activity_event = event_name
        self.session_ctx.last_activity_event_ts = ts
        self.session_ctx.last_activity_source = source
        self.session_ctx.activity_event_count += 1
        await self._safe_send_json({
            "type": MessageType.DEBUG_ACTIVITY,
            "data": {
                "event": event_name,
                "state": self.session_ctx.current_activity_state,
                "source": source,
                "queue_status": queue_status,
                "queue_note": queue_note,
                "timestamp": ts.isoformat(),
                "event_count": self.session_ctx.activity_event_count,
            },
        })

    # ── Sub-agent runners ──────────────────────────────────────────────

    async def _run_vision_analysis(self, image_base64: str) -> None:
        if not _vision_available:
            await self._emit_tool_event("analyze_scene", ToolBehavior.WHEN_IDLE, status="unavailable", data={"reason": "vision_agent_unavailable"})
            return
        async with self.state.vision_lock:
            if self.state.vision_in_progress:
                return
            self.state.vision_in_progress = True

        now_mono = time.monotonic()
        if self.state.camera_activated_at > 0 and (now_mono - self.state.camera_activated_at) < self.state.CAMERA_GRACE_PERIOD_SEC:
            logger.info("Suppressed vision: camera activation grace period (%.1fs)", now_mono - self.state.camera_activated_at)
            async with self.state.vision_lock:
                self.state.vision_in_progress = False
            return

        if (now_mono - self.state.last_vision_prefeedback_at >= VISION_PREFEEDBACK_COOLDOWN_SEC
                and (self.state.camera_activated_at <= 0 or now_mono - self.state.camera_activated_at >= self.state.CAMERA_GRACE_PERIOD_SEC)
                and not self.ctx_queue.vision_spoken_cooldown_active
                and not self.state.first_vision_after_camera):
            await self._forward_agent_transcript("Let me look at that for you...")
            self.state.last_vision_prefeedback_at = now_mono

        try:
            from agents.vision_agent import analyze_scene
            ctx_dict = {
                "space_type": self.session_ctx.space_type,
                "trip_purpose": self.session_ctx.trip_purpose,
                "active_task": self.session_ctx.active_task,
                "motion_state": session_manager.get_ephemeral_context(self.session_id).motion_state,
                "has_guide_dog": self.user_profile.has_guide_dog,
            }
            result = await analyze_scene(image_base64, self.session_ctx.current_lod, ctx_dict)
            result_status = str(result.get("status", "ok")).lower()
            if result_status in {"unavailable", "timeout"}:
                await self._emit_tool_event("analyze_scene", ToolBehavior.WHEN_IDLE, status="unavailable" if result_status == "unavailable" else "error", data={"reason": f"vision_{result_status}"})
                await self._emit_capability_degraded("vision", f"vision_{result_status}")
                return
            vision_text = _format_vision_result(result, self.session_ctx.current_lod)
            now_mono = time.monotonic()
            repeated = _is_repeated_text(vision_text, previous_text=self.state.last_vision_context_text, now_ts=now_mono, previous_ts=self.state.last_vision_context_sent_at, cooldown_sec=VISION_REPEAT_SUPPRESS_SEC)
            if not repeated:
                await self._safe_send_json({"type": MessageType.VISION_RESULT, "summary": result.get("scene_description", ""), "behavior": behavior_to_text(ToolBehavior.WHEN_IDLE), "data": _json_safe(result)})
                self.state.last_vision_context_text = vision_text
                self.state.last_vision_context_sent_at = now_mono
            else:
                logger.debug("Suppressed repeated vision summary within %.1fs window", VISION_REPEAT_SUPPRESS_SEC)
            await self._safe_send_json({"type": MessageType.VISION_DEBUG, "data": {"bounding_boxes": _json_safe(result.get("bounding_boxes", [])), "confidence": float(result.get("confidence", 0.0)), "lod": self.session_ctx.current_lod}})
            await self._emit_tool_event("analyze_scene", ToolBehavior.WHEN_IDLE, status="completed", data={"confidence": float(result.get("confidence", 0.0)), "repeat_suppressed": repeated})

            if result.get("confidence", 0) > 0 and not repeated:
                if self.ctx_queue.vision_spoken_cooldown_active:
                    logger.info("Suppressed vision: vision_spoken_cooldown active")
                else:
                    ephemeral = session_manager.get_ephemeral_context(self.session_id)
                    speak = should_speak(info_type="spatial_description", current_lod=self.session_ctx.current_lod, step_cadence=getattr(ephemeral, "step_cadence", 0.0) or 0.0, ambient_noise_db=getattr(ephemeral, "ambient_noise_db", 50.0) or 50.0)
                    if self.state.first_vision_after_camera:
                        speak = False
                        self.state.first_vision_after_camera = False
                        logger.info("First vision post-camera: forced silent injection")
                    idle_for = time.monotonic() - self.state.last_user_activity_at
                    if idle_for > PASSIVE_SPEECH_GUARD_SEC:
                        speak = False
                        logger.info("Vision passive-guard: forcing silent injection after %.1fs user inactivity", idle_for)
                    if not speak:
                        vision_text = "<<<SILENT_SENSOR_DATA>>>\n" + vision_text + "\n<<<END_SILENT_SENSOR_DATA>>>"
                    self.ctx_queue.enqueue(category="vision", text=vision_text, priority=5, speak=speak)
                    if speak:
                        self.ctx_queue.record_vision_spoken()
                    logger.info("Injected [VISION ANALYSIS] (LOD %d, confidence %.2f, speak=%s)", self.session_ctx.current_lod, result.get("confidence", 0), speak)
        except Exception as exc:
            logger.exception("Vision analysis failed")
            await self._emit_tool_event("analyze_scene", ToolBehavior.WHEN_IDLE, status="error", data={"reason": "vision_analysis_failed"})
            await self._emit_capability_degraded("vision", str(exc)[:200])
        finally:
            async with self.state.vision_lock:
                self.state.vision_in_progress = False

    async def _run_face_recognition(self, image_base64: str) -> None:
        if not _face_available or not self.state.face_runtime_available:
            await self._emit_tool_event("identify_person", ToolBehavior.SILENT, status="unavailable", data={"reason": "face_runtime_unavailable"})
            return
        async with self.state.face_lock:
            if self.state.face_in_progress:
                return
            self.state.face_in_progress = True
        if self.state.face_library_task is not None:
            logger.debug("Face recognition skipped: face library still loading")
            async with self.state.face_lock:
                self.state.face_in_progress = False
            return
        if self.state.face_consecutive_misses >= self.state.FACE_BACKOFF_THRESHOLD:
            self.state.face_skip_counter += 1
            if self.state.face_skip_counter <= self.state.FACE_BACKOFF_SKIP_CYCLES:
                logger.debug("Face detection skipped (backoff: %d misses)", self.state.face_consecutive_misses)
                async with self.state.face_lock:
                    self.state.face_in_progress = False
                return
            self.state.face_skip_counter = 0
        now_mono = time.monotonic()
        if now_mono - self.state.face_library_loaded_at >= FACE_LIBRARY_REFRESH_SEC:
            if self.state.face_library_refresh_task is None or self.state.face_library_refresh_task.done():
                self.state.face_library_refresh_task = asyncio.create_task(self._refresh_face_library_background())
        try:
            from agents.face_agent import identify_persons_in_frame
            results = await asyncio.to_thread(identify_persons_in_frame, image_base64, self.user_id, self.state.face_library, ToolBehavior.SILENT)
            await self._safe_send_json({"type": MessageType.FACE_DEBUG, "data": {"face_boxes": _json_safe([{"bbox": item.get("bbox", []), "label": item.get("person_name", "unknown"), "score": float(item.get("score", 0.0)), "similarity": float(item.get("similarity", 0.0))} for item in results])}})
            if len(results) == 0:
                self.state.face_consecutive_misses += 1
            else:
                self.state.face_consecutive_misses = 0
                self.state.face_skip_counter = 0
            known = [r for r in results if r["person_name"] != "unknown"]
            await self._emit_tool_event("identify_person", ToolBehavior.SILENT, status="completed", data={"detections": len(results), "known": len(known)})
            if known:
                ephemeral = session_manager.get_ephemeral_context(self.session_id)
                speak = should_speak(info_type="face_recognition", current_lod=self.session_ctx.current_lod, step_cadence=getattr(ephemeral, "step_cadence", 0.0) or 0.0, ambient_noise_db=getattr(ephemeral, "ambient_noise_db", 50.0) or 50.0)
                face_text = _format_face_results(known)
                if not speak:
                    face_text = "[SILENT - context only, do not speak aloud]\n" + face_text
                self.ctx_queue.enqueue(category="face", text=face_text, priority=4, speak=speak)
                logger.info("Injected [FACE ID] (speak=%s): %s", speak, ", ".join(r["person_name"] for r in known))
                best = max(known, key=lambda item: float(item.get("similarity", 0.0)))
                await self._emit_identity_event(person_name=str(best.get("person_name", "unknown")), matched=True, similarity=float(best.get("similarity", 0.0)), source="face_match")
                if _memory_available:
                    asyncio.create_task(self._inject_face_memories(known))
            elif results:
                await self._emit_identity_event(person_name="unknown", matched=False, similarity=0.0, source="face_detected_no_match")
        except RuntimeError as exc:
            err_text = str(exc)
            if "InsightFace is unavailable" in err_text:
                self.state.face_runtime_available = False
                if not self.state.face_unavailable_notified:
                    logger.warning("Face runtime unavailable; disabling face recognition for session %s", self.session_id)
                    self.state.face_unavailable_notified = True
                await self._emit_tool_event("identify_person", ToolBehavior.SILENT, status="unavailable", data={"reason": "insightface_unavailable"})
                await self._emit_capability_degraded("face", "InsightFace runtime unavailable")
            else:
                logger.exception("Face recognition runtime error")
                await self._emit_tool_event("identify_person", ToolBehavior.SILENT, status="error", data={"reason": "face_recognition_runtime_error"})
                await self._emit_capability_degraded("face", err_text[:200])
        except Exception as exc:
            logger.exception("Face recognition failed")
            await self._emit_tool_event("identify_person", ToolBehavior.SILENT, status="error", data={"reason": "face_recognition_failed"})
            await self._emit_capability_degraded("face", str(exc)[:200])
        finally:
            async with self.state.face_lock:
                self.state.face_in_progress = False

    async def _run_ocr_analysis(self, image_base64: str, safety_only: bool = False) -> None:
        if not _ocr_available:
            await self._emit_tool_event("extract_text", ToolBehavior.WHEN_IDLE, status="unavailable", data={"reason": "ocr_agent_unavailable"})
            return
        if not safety_only:
            now_mono = time.monotonic()
            if now_mono - self.state.last_ocr_prefeedback_at >= OCR_PREFEEDBACK_COOLDOWN_SEC:
                await self._forward_agent_transcript("Reading the text for you...")
                self.state.last_ocr_prefeedback_at = now_mono
        try:
            from agents.ocr_agent import extract_text
            hint = ""
            if self.session_ctx.space_type:
                hint = f"User is in a {self.session_ctx.space_type} environment."
            if self.session_ctx.active_task:
                hint += f" Currently: {self.session_ctx.active_task}."
            result = await extract_text(image_base64, context_hint=hint, safety_only=safety_only)
            ocr_text = _format_ocr_result(result)
            now_mono = time.monotonic()
            repeated = _is_repeated_text(ocr_text, previous_text=self.state.last_ocr_context_text, now_ts=now_mono, previous_ts=self.state.last_ocr_context_sent_at, cooldown_sec=OCR_REPEAT_SUPPRESS_SEC)
            if not repeated:
                await self._safe_send_json({"type": MessageType.OCR_RESULT, "summary": result.get("text", ""), "behavior": behavior_to_text(ToolBehavior.WHEN_IDLE), "data": _json_safe(result)})
                self.state.last_ocr_context_text = ocr_text
                self.state.last_ocr_context_sent_at = now_mono
            else:
                logger.debug("Suppressed repeated OCR summary within %.1fs window", OCR_REPEAT_SUPPRESS_SEC)
            await self._safe_send_json({"type": MessageType.OCR_DEBUG, "data": {"text_regions": _json_safe(result.get("text_regions", [])), "text_type": result.get("text_type", "unknown"), "confidence": float(result.get("confidence", 0.0))}})
            await self._emit_tool_event("extract_text", ToolBehavior.WHEN_IDLE, status="completed", data={"confidence": float(result.get("confidence", 0.0)), "repeat_suppressed": repeated})
            if result.get("confidence", 0) > 0.3 and result.get("text") and not repeated:
                ephemeral = session_manager.get_ephemeral_context(self.session_id)
                speak = should_speak(info_type="object_enumeration", current_lod=self.session_ctx.current_lod, step_cadence=getattr(ephemeral, "step_cadence", 0.0) or 0.0, ambient_noise_db=getattr(ephemeral, "ambient_noise_db", 50.0) or 50.0)
                if not speak:
                    ocr_text = "<<<SILENT_SENSOR_DATA>>>\n" + ocr_text + "\n<<<END_SILENT_SENSOR_DATA>>>"
                self.ctx_queue.enqueue(category="ocr", text=ocr_text, priority=5, speak=speak)
                logger.info("Injected [OCR RESULT] (%s, confidence %.2f, speak=%s)", result.get("text_type", "unknown"), result.get("confidence", 0), speak)
        except Exception as exc:
            logger.exception("OCR analysis failed")
            await self._emit_tool_event("extract_text", ToolBehavior.WHEN_IDLE, status="error", data={"reason": "ocr_analysis_failed"})
            await self._emit_capability_degraded("ocr", str(exc)[:200])

    # ── Telemetry processing ───────────────────────────────────────────

    async def _process_telemetry(self, telemetry_data: dict) -> None:
        ephemeral_ctx = parse_telemetry_to_ephemeral(telemetry_data)
        session_manager.update_ephemeral_context(self.session_id, ephemeral_ctx)
        now = time.monotonic()
        _signature_changed = False
        if self.telemetry_agg.should_send(now):
            signature = _build_telemetry_signature(ephemeral_ctx)
            should_inject, reasons = _should_inject_telemetry_context(
                previous_signature=self.state.last_telemetry_signature,
                current_signature=signature,
                last_injected_ts=self.state.last_telemetry_context_sent_at,
                now_ts=now,
            )
            _signature_changed = bool(self.state.last_telemetry_signature is None or _changed_signature_fields(self.state.last_telemetry_signature, signature))
            if should_inject:
                semantic_text = parse_telemetry(telemetry_data)
                telemetry_text = "<<<SENSOR_DATA>>>\n" + semantic_text + "\n<<<END_SENSOR_DATA>>>\nINSTRUCTION: Do not vocalize any part of the above sensor data."
                self.ctx_queue.enqueue(category="telemetry", text=telemetry_text, priority=8, speak=False)
                self.state.last_telemetry_context_sent_at = now
                logger.debug("Telemetry context injected: reasons=%s", ",".join(reasons))
            self.state.last_telemetry_signature = signature
            self.telemetry_agg.mark_sent(now)
        if self._location_ctx_service and ephemeral_ctx.gps:
            try:
                self.state.current_location_ctx = await self._location_ctx_service.evaluate(ephemeral_ctx.gps.lat, ephemeral_ctx.gps.lng)
                self.session_ctx.familiarity_score = self.state.current_location_ctx.familiarity_score
                place = self.state.current_location_ctx.place_name
                if place and place not in self.session_meta.locations_visited:
                    self.session_meta.locations_visited.append(place)
            except Exception:
                logger.debug("Location context evaluation failed", exc_info=True)
        if _signature_changed or (now - self.state.last_telemetry_context_sent_at) >= TELEMETRY_FORCE_REFRESH_SEC:
            await self._process_lod_decision(ephemeral_ctx)

    async def _process_lod_decision(self, ephemeral_ctx) -> None:
        new_lod, decision_log = decide_lod(ephemeral=ephemeral_ctx, session=self.session_ctx, profile=self.user_profile)
        if self._lod_evaluator and new_lod >= 2:
            try:
                adjustment = await self._lod_evaluator.evaluate(baseline_lod=new_lod, location_ctx=self.state.current_location_ctx, user_profile=self.user_profile)
                if adjustment.delta != 0:
                    adjusted = max(1, min(3, new_lod + adjustment.delta))
                    if adjusted != new_lod:
                        logger.info("LOD micro-adjust: %d -> %d (%s)", new_lod, adjusted, adjustment.reason)
                        decision_log.triggered_rules.append(f"LLM:{adjustment.reason[:40]}")
                        new_lod = adjusted
            except Exception:
                logger.debug("LOD evaluator failed", exc_info=True)
        old_lod = self.session_ctx.current_lod
        if new_lod != old_lod:
            logger.info("LOD transition: %d -> %d (%s) session=%s", old_lod, new_lod, decision_log.reason, self.session_id)
            resume_prompt = on_lod_change(self.session_ctx, old_lod, new_lod)
            self.session_ctx.current_lod = new_lod
            self.session_meta.record_lod_time(new_lod)
            self.session_ctx.user_requested_detail = False
            self.session_ctx.user_said_stop = False
            self.telemetry_agg.update_lod(new_lod)
            vad_update = await self._sync_runtime_vad_update(new_lod)
            await self._send_lod_update(new_lod, ephemeral_ctx, decision_log.reason)
            if resume_prompt:
                self.ctx_queue.enqueue(category="lod_resume", text=resume_prompt, priority=3, speak=True)
                logger.info("Injected [RESUME] prompt for session %s", self.session_id)
            await self._notify_ios_lod_change(new_lod, decision_log.reason, decision_log.to_debug_dict(), vad_update=vad_update)

    # ── Upstream handler ───────────────────────────────────────────────

    async def _upstream(self) -> None:
        """Read messages from the iOS client and forward to the Live API."""
        try:
            while True:
                try:
                    ws_message = await asyncio.wait_for(self.websocket.receive(), timeout=WS_INACTIVITY_TIMEOUT_SEC)
                except asyncio.TimeoutError:
                    logger.info("WebSocket inactivity timeout (%ds): user=%s session=%s", WS_INACTIVITY_TIMEOUT_SEC, self.user_id, self.session_id)
                    self.stop_downstream.set()
                    try:
                        await self.websocket.close(code=1000, reason="inactivity_timeout")
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

                    if magic == self.state.MAGIC_AUDIO:
                        self.state.last_user_activity_at = time.monotonic()
                        blob = types.Blob(data=payload, mime_type="audio/pcm;rate=16000")
                        self.live_request_queue.send_realtime(blob)
                        continue

                    elif magic == self.state.MAGIC_IMAGE:
                        self.state.frame_seq += 1
                        _ocr_set_latest_frame(self.session_id, base64.b64encode(payload).decode("ascii"))
                        now_frame = time.monotonic()
                        if now_frame - self.state.last_frame_to_gemini_at >= self.state.FRAME_TO_GEMINI_INTERVAL:
                            blob = types.Blob(data=payload, mime_type="image/jpeg")
                            self.live_request_queue.send_realtime(blob)
                            self.state.last_frame_to_gemini_at = now_frame
                        await self._process_image_frame(base64.b64encode(payload).decode("ascii"))
                        continue
                    else:
                        logger.warning("Unknown binary magic byte: 0x%02x", magic)
                        continue

                # --- Text frame (JSON) ---
                raw_text = ws_message.get("text")
                if not raw_text:
                    if ws_message.get("type") == "websocket.disconnect":
                        break
                    continue
                try:
                    message = json.loads(raw_text)
                except json.JSONDecodeError:
                    logger.warning("Received non-JSON text message, ignoring")
                    continue

                msg_type = message.get("type")
                if msg_type == "audio":
                    audio_bytes = base64.b64decode(message["data"])
                    blob = types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000")
                    self.live_request_queue.send_realtime(blob)

                elif msg_type == "image":
                    image_bytes = base64.b64decode(message["data"])
                    mime_type = message.get("mimeType", "image/jpeg")
                    self.state.frame_seq += 1
                    _ocr_set_latest_frame(self.session_id, message["data"])
                    now_frame = time.monotonic()
                    if now_frame - self.state.last_frame_to_gemini_at >= self.state.FRAME_TO_GEMINI_INTERVAL:
                        blob = types.Blob(data=image_bytes, mime_type=mime_type)
                        self.live_request_queue.send_realtime(blob)
                        self.state.last_frame_to_gemini_at = now_frame
                    await self._process_image_frame(message["data"])

                elif msg_type == "camera_failure":
                    camera_error = message.get("error") or message.get("reason") or "camera_unavailable"
                    logger.warning("Camera failure reported: %s", camera_error)
                    await self._emit_capability_degraded("camera", camera_error, recoverable=message.get("recoverable", True))

                elif msg_type == "telemetry":
                    await self._process_telemetry(message.get("data", {}))

                elif msg_type in {"activity_start", "activity_end"}:
                    event_name = str(msg_type)
                    self.state.last_user_activity_at = time.monotonic()
                    self.session_meta.record_interaction()
                    if _should_reset_interrupted_on_activity_start(event_name=event_name, interrupted=self.state.is_interrupted):
                        self.state.is_interrupted = False
                        logger.info("Reset stale interrupted state on activity_start: user=%s session=%s", self.user_id, self.session_id)
                    if event_name == "activity_start":
                        self.tool_dedup.reset()
                    queue_status = "forwarded"
                    queue_note = ""
                    try:
                        if event_name == "activity_start":
                            self.live_request_queue.send_activity_start()
                        else:
                            self.live_request_queue.send_activity_end()
                    except Exception as exc:
                        queue_status = "failed"
                        queue_note = str(exc)[:160]
                        logger.warning("Failed to forward %s to LiveRequestQueue: %s", event_name, queue_note)
                    await self._emit_activity_debug_event(event_name=event_name, queue_status=queue_status, queue_note=queue_note)

                elif msg_type == "gesture":
                    await self._handle_gesture(message)
                elif msg_type == "reload_face_library":
                    await self._handle_reload_face_library()
                elif msg_type == "clear_face_library":
                    await self._handle_clear_face_library()
                elif msg_type == "profile_updated":
                    await self._handle_profile_updated()
                elif msg_type == "playback_drained":
                    self.ctx_queue.set_ios_playback_drained(True)
                    logger.debug("iOS playback drained — flush gate opened")
                elif msg_type == "client_barge_in":
                    await self._handle_client_barge_in()
                else:
                    logger.warning("Unknown upstream message type: %s", msg_type)

        except WebSocketDisconnect:
            self.stop_downstream.set()
            logger.info("Client disconnected (upstream): user=%s session=%s", self.user_id, self.session_id)
        except Exception:
            self.stop_downstream.set()
            logger.exception("Error in upstream handler: user=%s session=%s", self.user_id, self.session_id)

    async def _process_image_frame(self, image_b64: str) -> None:
        """Trigger sub-agents for an image frame (shared between binary and JSON paths)."""
        now = time.monotonic()
        lod = self.session_ctx.current_lod
        vision_interval = {1: 10.0, 2: 8.0, 3: 5.0}.get(lod, 8.0)
        queued_agents: list[str] = []
        if now - self.state.last_vision_time >= vision_interval:
            self.state.last_vision_time = now
            await self._emit_tool_event("analyze_scene", ToolBehavior.WHEN_IDLE, status="queued")
            queued_agents.append("vision")
            asyncio.create_task(self._run_vision_analysis(image_b64))
            if lod >= 2 and self.state.face_runtime_available:
                await self._emit_tool_event("identify_person", ToolBehavior.SILENT, status="queued")
                queued_agents.append("face")
                await self._emit_identity_event(person_name="unknown", matched=False, similarity=0.0, source="queued")
                asyncio.create_task(self._run_face_recognition(image_b64))
            if lod == 1 and now - getattr(self, '_last_safety_ocr_at', 0) >= 15.0:
                self._last_safety_ocr_at = now
                await self._emit_tool_event("extract_text", ToolBehavior.WHEN_IDLE, status="queued")
                queued_agents.append("ocr")
                asyncio.create_task(self._run_ocr_analysis(image_b64, safety_only=True))
        await self._safe_send_json({"type": MessageType.FRAME_ACK, "frame_id": self.state.frame_seq, "queued_agents": queued_agents})

    # ── Upstream sub-handlers ──────────────────────────────────────────

    async def _handle_gesture(self, message: dict) -> None:
        gesture = message.get("gesture")
        self.state.last_user_activity_at = time.monotonic()
        self.session_meta.record_interaction()

        if gesture in ("lod_up", "lod_down"):
            ephemeral_ctx = session_manager.get_ephemeral_context(self.session_id)
            ephemeral_ctx.user_gesture = gesture
            await self._process_lod_decision(ephemeral_ctx)
            ephemeral_ctx.user_gesture = None
            session_manager.update_ephemeral_context(self.session_id, ephemeral_ctx)

        elif isinstance(gesture, str) and gesture.startswith("force_lod_"):
            try:
                forced_lod = int(gesture.rsplit("_", 1)[-1])
            except (TypeError, ValueError):
                logger.warning("Invalid force_lod gesture payload: %s", gesture)
                return
            if forced_lod not in (1, 2, 3):
                logger.warning("force_lod gesture out of range: %s", gesture)
                return
            old_lod = self.session_ctx.current_lod
            if forced_lod == old_lod:
                await self._safe_send_json({"type": MessageType.LOD_UPDATE, "lod": forced_lod, "reason": "force_lod_no_change"})
                return
            reason = f"manual_force_lod_{forced_lod}"
            logger.info("Force LOD gesture received: %d -> %d", old_lod, forced_lod)
            resume_prompt = on_lod_change(self.session_ctx, old_lod, forced_lod)
            self.session_ctx.current_lod = forced_lod
            self.session_ctx.user_requested_detail = False
            self.session_ctx.user_said_stop = False
            self.session_meta.record_lod_time(forced_lod)
            self.telemetry_agg.update_lod(forced_lod)
            vad_update = await self._sync_runtime_vad_update(forced_lod)
            await self._send_lod_update(forced_lod, session_manager.get_ephemeral_context(self.session_id), reason)
            await self._notify_ios_lod_change(forced_lod, reason, {"lod": forced_lod, "prev": old_lod, "reason": reason, "rules": [f"manual:{gesture}"], "forced": True}, vad_update=vad_update)
            if resume_prompt:
                self.ctx_queue.inject_immediate(types.Content(parts=[types.Part(text=resume_prompt)], role="user"))

        elif gesture == "interrupt":
            logger.info("User interrupt gesture received")
            self.ctx_queue.set_ios_playback_drained(True)
            self.ctx_queue.inject_immediate(types.Content(parts=[types.Part(text="[USER INTERRUPT] The user has interrupted. Stop current output immediately and wait for their next input.")], role="user"))

        elif gesture == "repeat_last":
            self.state.allow_agent_repeat_until = time.monotonic() + 12.0
            last_agent = None
            for entry in reversed(self.state.transcript_history):
                if entry.get("role") == "agent":
                    last_agent = entry.get("text", "")
                    break
            if last_agent:
                logger.info("Repeat last gesture: replaying last agent utterance")
                self.ctx_queue.inject_immediate(types.Content(parts=[types.Part(text=f'[REPEAT REQUEST] The user wants you to repeat your last response. Please repeat: "{last_agent}"')], role="user"))
            else:
                logger.info("Repeat last gesture: no previous agent utterance found")
                self.ctx_queue.inject_immediate(types.Content(parts=[types.Part(text="[REPEAT REQUEST] The user wants you to repeat your last response, but no previous response was found. Let the user know.")], role="user"))

        elif gesture == "mute_toggle":
            muted = _coerce_bool(message.get("muted"), default=True)
            logger.info("Mute toggle acknowledged: muted=%s", muted)

        elif gesture == "pause":
            paused = _coerce_bool(message.get("paused", True), default=True)
            self.ctx_queue.set_ios_playback_drained(True)
            if paused:
                logger.info("Pause activated for session %s", self.session_id)
                self.ctx_queue.inject_immediate(types.Content(parts=[types.Part(text="[PAUSE] The user has paused the session. Go silent until the user resumes.")], role="user"))
                await self._safe_send_json({"type": MessageType.LOD_UPDATE, "lod": self.session_ctx.current_lod, "reason": "paused"})
            else:
                logger.info("Session resumed for session %s", self.session_id)
                self.ctx_queue.inject_immediate(types.Content(parts=[types.Part(text="[RESUME] The user has resumed the session. You may speak again.")], role="user"))
                await self._safe_send_json({"type": MessageType.LOD_UPDATE, "lod": self.session_ctx.current_lod, "reason": "resumed"})

        elif gesture == "camera_toggle":
            self.state.client_camera_active = _coerce_bool(message.get("active"), default=not self.state.client_camera_active)
            logger.info("Camera toggle: active=%s", self.state.client_camera_active)
            if self.state.client_camera_active:
                self.state.camera_activated_at = time.monotonic()
                self.state.first_vision_after_camera = True
                self.ctx_queue.enqueue(category="camera_toggle", text="[CAMERA ACTIVATED] The user has turned on the rear camera. Visual context is now available via periodic [VISION ANALYSIS] injections. Do NOT describe every frame. Only speak about safety hazards or when the user asks. Acknowledge camera activation in one brief sentence, then observe silently.", priority=4, speak=True)
            else:
                self.ctx_queue.enqueue(category="camera_toggle", text="[CAMERA DEACTIVATED] The user has turned off the camera. You are now in audio-only mode. Do not reference visual information unless recalling something previously seen.", priority=4, speak=True)
        else:
            logger.debug("Unhandled gesture type: %s", gesture)

    async def _handle_reload_face_library(self) -> None:
        logger.info("Reload face library requested for user=%s", self.user_id)
        if _face_available:
            try:
                from tools.face_tools import load_face_library
                self.state.face_library.clear()
                self.state.face_library.extend(await asyncio.to_thread(load_face_library, self.user_id))
                await self._safe_send_json({"type": MessageType.FACE_LIBRARY_RELOADED, "count": len(self.state.face_library)})
                logger.info("Reloaded %d face(s) for user %s", len(self.state.face_library), self.user_id)
            except Exception:
                logger.exception("Failed to reload face library")
                await self._safe_send_json({"type": MessageType.ERROR, "error": "Failed to reload face library"})
        else:
            await self._safe_send_json({"type": MessageType.ERROR, "error": "Face recognition not available"})

    async def _handle_clear_face_library(self) -> None:
        logger.info("Clear face library requested for user=%s", self.user_id)
        if _face_available:
            try:
                from tools.face_tools import clear_face_library
                count = await asyncio.to_thread(clear_face_library, self.user_id)
                self.state.face_library.clear()
                await self._safe_send_json({"type": MessageType.FACE_LIBRARY_CLEARED, "deleted_count": count})
                logger.info("Cleared %d face(s) for user %s", count, self.user_id)
            except Exception:
                logger.exception("Failed to clear face library")
                await self._safe_send_json({"type": MessageType.ERROR, "error": "Failed to clear face library"})
        else:
            await self._safe_send_json({"type": MessageType.ERROR, "error": "Face recognition not available"})

    async def _handle_profile_updated(self) -> None:
        logger.info("profile_updated received for user=%s", self.user_id)
        try:
            session_manager.invalidate_user_profile(self.user_id)
            fresh_profile = await session_manager.load_user_profile(self.user_id)
            self.user_profile.update_from_dict({
                f.name: getattr(fresh_profile, f.name)
                for f in __import__("dataclasses").fields(fresh_profile)
                if f.name != "user_id"
            })
            from lod.prompt_builder import _build_persona_block
            persona_block = _build_persona_block(self.user_profile)
            profile_ctx = "[PROFILE UPDATE]\nThe user just updated their profile. Use the new settings below for all subsequent interactions.\n" + persona_block + "\nDo not narrate this block to the user."
            self.ctx_queue.enqueue(category="profile_update", text=profile_ctx, priority=3, speak=False)
            logger.info("Queued profile update context for user %s", self.user_id)
            await self._safe_send_json({"type": MessageType.PROFILE_UPDATED_ACK})
        except Exception:
            logger.exception("Failed to handle profile_updated for user %s", self.user_id)
            await self._safe_send_json({"type": MessageType.ERROR, "error": "Failed to apply profile update"})

    async def _handle_client_barge_in(self) -> None:
        now_mono = time.monotonic()
        if (now_mono - self.state.last_interrupt_at) < self.state.INTERRUPT_DEBOUNCE_SEC:
            logger.debug("Client barge-in debounced (%.0fms since last)", (now_mono - self.state.last_interrupt_at) * 1000)
        else:
            model_was_speaking = (self.state.model_audio_last_seen_at > 0 and (now_mono - self.state.model_audio_last_seen_at) < 2.0) or self.ctx_queue.model_speaking
            if model_was_speaking:
                self.state.is_interrupted = True
                self.state.last_interrupt_at = now_mono
                self.state.model_audio_last_seen_at = 0.0
                self.ctx_queue._transition_to(ModelState.IDLE)
            sent_interrupted = await self._safe_send_json({"type": MessageType.INTERRUPTED, "source": "client_barge_in", "accepted": model_was_speaking})
            if model_was_speaking:
                if sent_interrupted:
                    logger.info("Client barge-in — suppressing audio forwarding: user=%s session=%s state=%s", self.user_id, self.session_id, self.ctx_queue.state.value)
                else:
                    logger.warning("Client barge-in accepted but interrupted event send failed: user=%s session=%s", self.user_id, self.session_id)
            else:
                if sent_interrupted:
                    logger.info("Client barge-in ignored — model not speaking (last audio %.1fs ago): user=%s session=%s state=%s", now_mono - self.state.model_audio_last_seen_at if self.state.model_audio_last_seen_at > 0 else -1, self.user_id, self.session_id, self.ctx_queue.state.value)
                else:
                    logger.warning("Client barge-in ignored and interrupted status event send failed: user=%s session=%s", self.user_id, self.session_id)

    # ── Downstream handler ─────────────────────────────────────────────

    async def _downstream(self) -> None:
        """Read events from the Live API and forward to the iOS client."""
        adk_session_id = self.session_id
        if _NEEDS_SESSION_ID_MAPPING:
            cached_adk_id = session_manager.get_adk_session_id(self.session_id)
            if cached_adk_id:
                adk_session_id = cached_adk_id
            else:
                adk_session = await self.runner.session_service.create_session(app_name="sightline", user_id=self.user_id)
                adk_session_id = adk_session.id
                session_manager.set_adk_session_id(self.session_id, adk_session_id)
                logger.info("Created ADK session %s for logical session %s", adk_session_id, self.session_id)

        def _start_live_events():
            return self.runner.run_live(session_id=adk_session_id, user_id=self.user_id, live_request_queue=self.live_request_queue, run_config=self.run_config)

        live_events = await asyncio.to_thread(_start_live_events)
        self.state.is_interrupted = False
        try:
            async for event in live_events:
                if self.stop_downstream.is_set():
                    break
                _output_transcription_forwarded = False

                _usage = getattr(event, "usage_metadata", None)
                if _usage is not None:
                    self.token_monitor.update(_usage)

                if event.live_session_resumption_update:
                    update = event.live_session_resumption_update
                    if update.newHandle:
                        session_manager.update_handle(self.session_id, update.newHandle)
                    if not await self._safe_send_json({"type": MessageType.SESSION_RESUMPTION, "handle": update.newHandle}):
                        break

                if hasattr(event, "go_away") and event.go_away:
                    retry_ms = 500
                    if hasattr(event.go_away, "time_left"):
                        retry_ms = int(event.go_away.time_left.total_seconds() * 1000) if event.go_away.time_left else 500
                    await self._safe_send_json({"type": MessageType.GO_AWAY, "retry_ms": retry_ms, "message": "Server requested reconnection."})
                    logger.warning("GoAway received, retry_ms=%d", retry_ms)

                _event_interrupted = (hasattr(event, "server_content") and event.server_content and getattr(event.server_content, "interrupted", False)) or getattr(event, "interrupted", False)
                if _event_interrupted and not self.state.is_interrupted:
                    now_mono = time.monotonic()
                    if (now_mono - self.state.last_interrupt_at) < self.state.INTERRUPT_DEBOUNCE_SEC:
                        logger.debug("Interrupt debounced (%.0fms since last)", (now_mono - self.state.last_interrupt_at) * 1000)
                    else:
                        self.state.is_interrupted = True
                        self.state.last_interrupt_at = now_mono
                        self.state.model_audio_last_seen_at = 0.0
                        self.ctx_queue._transition_to(ModelState.IDLE)
                        await self._safe_send_json({"type": MessageType.INTERRUPTED})
                        logger.info("Interrupt detected — suppressing audio forwarding")

                if event.turn_complete:
                    if self.state.is_interrupted:
                        logger.info("Turn complete — resuming audio forwarding")
                    self.state.is_interrupted = False
                    if self.state.turn_had_vision_content:
                        self.ctx_queue.record_vision_spoken()
                        self.state.turn_had_vision_content = False
                    if self.state.transcript_buffer:
                        if not await self._flush_transcript_buffer():
                            break
                    _tc_audio_before = self.state.model_audio_last_seen_at
                    await asyncio.sleep(0.5)
                    if self.state.model_audio_last_seen_at > _tc_audio_before:
                        logger.info("Turn complete revoked — audio arrived during quiet period")
                    else:
                        self.ctx_queue.on_turn_complete()
                        self.ctx_queue.enqueue(category="turn_boundary", text="<<<INTERNAL_CONTEXT>>>\n[TURN BOUNDARY] Previous request complete. Await user's next request. Do not carry forward tool-calling intent from previous turn.\n<<<END_INTERNAL_CONTEXT>>>", priority=1, speak=False)
                        self.ctx_queue.flush_or_defer_first_turn(camera_active=self.state.client_camera_active)

                if event.content and event.content.parts and not self.state.is_interrupted:
                    for part in event.content.parts:
                        if self.stop_downstream.is_set():
                            break
                        if part.inline_data and part.inline_data.data:
                            self.state.model_audio_last_seen_at = time.monotonic()
                            self.ctx_queue.set_model_audio_timestamp(self.state.model_audio_last_seen_at)
                            self.ctx_queue.set_model_speaking(True)
                            audio_data = part.inline_data.data
                            if isinstance(audio_data, str):
                                audio_data = base64.b64decode(audio_data)
                            if not await self._safe_send_bytes(audio_data):
                                break
                        elif part.text:
                            if _output_transcription_forwarded:
                                logger.debug("Skipped content.parts text (already sent via output_transcription): %s", part.text[:80])
                                continue
                            if not await self._forward_agent_transcript(part.text):
                                break
                    if self.stop_downstream.is_set():
                        break

                if event.input_transcription and event.input_transcription.text:
                    now_mono = time.monotonic()
                    input_text = event.input_transcription.text
                    if self._is_likely_echo(input_text, now_mono):
                        logger.debug("Echo detected, reclassifying: %s", input_text[:120])
                        self.state.transcript_history.append({"role": "echo", "text": input_text})
                        await self._safe_send_json({"type": MessageType.TRANSCRIPT, "text": input_text, "role": "echo"})
                        self.ctx_queue.enqueue(category="echo_cancel", text="[SYSTEM: The previous audio input was an echo of your own speech. Do not respond to it.]", priority=1, speak=False)
                        logger.info("Queued echo cancellation for Gemini")
                    else:
                        self.state.transcript_history.append({"role": "user", "text": input_text})
                        self.state.last_user_activity_at = time.monotonic()
                        self.session_meta.record_interaction()
                        if not await self._safe_send_json({"type": MessageType.TRANSCRIPT, "text": input_text, "role": "user"}):
                            break
                        intent = _detect_voice_intent(input_text)
                        if intent == "detail":
                            self.session_ctx.user_requested_detail = True
                            self.session_ctx.user_said_stop = False
                        elif intent == "stop":
                            self.session_ctx.user_said_stop = True
                            self.session_ctx.user_requested_detail = False

                function_calls = _extract_function_calls(event)
                if function_calls:
                    await self._handle_function_calls(function_calls)

                if event.output_transcription and event.output_transcription.text:
                    _ot_text = event.output_transcription.text
                    if _ot_text.replace("[Silence]", "").strip() == "":
                        continue
                    now_mono = time.monotonic()
                    self.state.transcript_history.append({"role": "agent", "text": _ot_text})
                    _out_lower = _ot_text.lower()
                    if any(kw in _out_lower for kw in ("i see", "i can see", "looking at", "in front of", "ahead of", "around you", "your surroundings", "to your left", "to your right", "on the left", "on the right")):
                        self.state.turn_had_vision_content = True
                    self.state.transcript_buffer += _ot_text
                    if self.state.transcript_buffer_started_at == 0.0:
                        self.state.transcript_buffer_started_at = now_mono
                    logger.debug("Buffered transcript fragment: %s", _ot_text[:80])
                    if self._has_sentence_boundary(self.state.transcript_buffer):
                        if not await self._flush_transcript_buffer():
                            break
                        _output_transcription_forwarded = True
                    elif self.state.transcript_buffer_started_at > 0 and (now_mono - self.state.transcript_buffer_started_at) > self.state.TRANSCRIPT_FLUSH_TIMEOUT_SEC:
                        if not await self._flush_transcript_buffer():
                            break
                        _output_transcription_forwarded = True

        except WebSocketDisconnect:
            self.stop_downstream.set()
            logger.info("Client disconnected (downstream): user=%s session=%s", self.user_id, self.session_id)
        except Exception as exc:
            exc_text = str(exc).lower()
            is_keepalive_timeout = "keepalive ping timeout" in exc_text
            if is_keepalive_timeout and self.state.downstream_retry_count < self.state.DOWNSTREAM_MAX_RETRIES and self.websocket.client_state == WebSocketState.CONNECTED:
                self.state.downstream_retry_count += 1
                backoff_sec = 0.8 * self.state.downstream_retry_count
                logger.warning("Transient Live API keepalive timeout; retrying downstream (%d/%d) after %.1fs: user=%s session=%s", self.state.downstream_retry_count, self.state.DOWNSTREAM_MAX_RETRIES, backoff_sec, self.user_id, self.session_id)
                await self._safe_send_json({"type": MessageType.GO_AWAY, "retry_ms": int(backoff_sec * 1000), "message": "Live stream transient timeout, retrying."})
                await asyncio.sleep(backoff_sec)
                return await self._downstream()

            if is_keepalive_timeout:
                logger.warning("Live API keepalive timeout persisted after %d retries; requesting client reconnect: user=%s session=%s", self.state.downstream_retry_count, self.user_id, self.session_id)
                await self._safe_send_json({"type": MessageType.GO_AWAY, "retry_ms": 2000, "message": "Live stream timeout. Please reconnect."})
                self.stop_downstream.set()
                try:
                    await self.websocket.close(code=1012, reason="live_keepalive_timeout")
                except Exception:
                    pass
                return

            self.stop_downstream.set()
            logger.exception("Error in downstream handler: user=%s session=%s", self.user_id, self.session_id)
            is_fatal = isinstance(exc, (UnboundLocalError, NameError, AttributeError, TypeError))
            close_code = 1008 if is_fatal else 1011
            await self._safe_send_json({"type": MessageType.ERROR, "error": "Fatal server error." if is_fatal else "Live session failed. Reconnecting...", "fatal": is_fatal})
            try:
                await self.websocket.close(code=close_code, reason="fatal_error" if is_fatal else "downstream_error")
            except Exception:
                pass

    # ── Function call handler ──────────────────────────────────────────

    async def _handle_function_calls(self, function_calls: list) -> None:
        self.tool_mutex.reset()
        for fc in function_calls:
            if fc.name not in ALL_FUNCTIONS:
                logger.warning("Model called non-existent tool %r — returning no-op", fc.name)
                await self._safe_send_json({"type": MessageType.DEBUG_ACTIVITY, "data": {"event": "hallucinated_tool_call", "tool": fc.name, "args": _json_safe(dict(fc.args) if fc.args else {})}})
                from google.genai.types import FunctionResponse as _FR
                noop_response = _FR(name=fc.name, response={"status": "unavailable", "message": f"'{fc.name}' does not exist. Use only the tools listed in your instructions. OCR/vision results are injected automatically as context."})
                self.ctx_queue.inject_immediate(types.Content(parts=[types.Part(function_response=noop_response)], role="user"), is_function_response=True)
                continue

            call_args = dict(fc.args) if fc.args else {}
            dedup_ok, dedup_reason = self.tool_dedup.should_execute(fc.name, call_args)
            if not dedup_ok:
                from google.genai.types import FunctionResponse as _FR
                skip_fr = _FR(name=fc.name, response={"status": "skipped", "reason": dedup_reason, "message": "Duplicate call skipped. Use the result from the previous call."})
                self.ctx_queue.inject_immediate(types.Content(parts=[types.Part(function_response=skip_fr)], role="user"), is_function_response=True)
                logger.info("Dedup skipped %s: %s", fc.name, dedup_reason)
                continue

            mutex_ok, mutex_reason = self.tool_mutex.should_execute(fc.name)
            if not mutex_ok:
                from google.genai.types import FunctionResponse as _FR
                skip_fr = _FR(name=fc.name, response={"status": "skipped", "reason": mutex_reason, "message": "Mutually exclusive tool already called in this batch."})
                self.ctx_queue.inject_immediate(types.Content(parts=[types.Part(function_response=skip_fr)], role="user"), is_function_response=True)
                logger.info("Mutex skipped %s: %s", fc.name, mutex_reason)
                continue

            user_speaking = self.session_ctx.current_activity_state == "user_speaking"
            behavior = resolve_tool_behavior(tool_name=fc.name, lod=self.session_ctx.current_lod, is_user_speaking=user_speaking)

            allow_call, block_reason = _allow_navigation_tool_call(func_name=fc.name, func_args=call_args, transcript_history=self.state.transcript_history)
            if not allow_call:
                blocked_result = {"status": "blocked", "reason": block_reason, "message": "Navigation tools require an explicit navigation request from the user in this session."}
                await self._emit_tool_event(fc.name, behavior, status="blocked", data={"reason": block_reason, "args": _json_safe(call_args)})
                await self._safe_send_json({"type": MessageType.TOOL_RESULT, "tool": fc.name, "behavior": behavior_to_text(behavior), "data": blocked_result})
                if fc.name in NAVIGATION_FUNCTIONS:
                    await self._safe_send_json({"type": MessageType.NAVIGATION_RESULT, "summary": "", "behavior": behavior_to_text(behavior), "data": blocked_result})
                from google.genai.types import FunctionResponse as _FR
                blocked_fr = _FR(name=fc.name, response=blocked_result)
                self.ctx_queue.inject_immediate(types.Content(parts=[types.Part(function_response=blocked_fr)], role="user"), is_function_response=True)
                logger.warning("Blocked %s tool call: %s", fc.name, block_reason)
                continue

            await self._emit_tool_event(fc.name, behavior, status="invoked", data={"args": _json_safe(call_args)})
            self.audio_gate.enter()
            try:
                result = await _dispatch_function_call(fc.name, call_args, self.session_id, self.user_id)
            finally:
                self.audio_gate.exit()
            await self._safe_send_json({"type": MessageType.TOOL_RESULT, "tool": fc.name, "behavior": behavior_to_text(behavior), "data": _json_safe(result)})

            if fc.name in NAVIGATION_FUNCTIONS:
                await self._safe_send_json({"type": MessageType.NAVIGATION_RESULT, "summary": str(result.get("destination_direction") or result.get("destination") or ""), "behavior": behavior_to_text(behavior), "data": _json_safe(result)})
            elif fc.name == "google_search":
                await self._safe_send_json({"type": MessageType.SEARCH_RESULT, "summary": str(result.get("answer") or ""), "behavior": behavior_to_text(behavior), "data": _json_safe(result)})
            elif fc.name == "identify_person":
                await self._emit_identity_event(person_name=str(result.get("person_name", "unknown")), matched=bool(result.get("matched", False)), similarity=float(result.get("similarity", 0.0)), source="tool_call")

            from google.genai.types import FunctionResponse
            fr = FunctionResponse(name=fc.name, response=result)
            parts = [types.Part(function_response=fr)]
            if behavior == ToolBehavior.INTERRUPT:
                parts.insert(0, types.Part(text="[SAFETY ALERT — COMPLETE DELIVERY] Do not stop mid-sentence even if you detect user activity. Deliver the full safety information before yielding."))
            content = types.Content(parts=parts, role="user")
            self.ctx_queue.inject_immediate(content, is_function_response=True)
            logger.info("Sent function response for %s (behavior=%s)", fc.name, behavior_to_text(behavior))
