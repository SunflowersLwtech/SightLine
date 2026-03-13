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
import re
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
    _has_navigation_intent,
    _has_location_query_intent,
    _recent_user_utterances,
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
from live_api.tts_fallback import (
    synthesize_local_pcm as synthesize_local_fallback_pcm,
    synthesize_pcm as synthesize_fallback_pcm,
)

logger = logging.getLogger("sightline.server")

_STALE_QUEUE_CATEGORIES = {
    "echo_cancel",
    "face",
    "lod",
    "lod_resume",
    "ocr",
    "telemetry",
    "turn_boundary",
    "vad",
    "vision",
}

_SILENT_TURN_WATCHDOG_SEC = 18.0
_LIVE_SESSION_READY_TIMEOUT_SEC = 60.0


def _exception_chain_text(exc: BaseException) -> str:
    """Flatten an exception chain into lowercase text for coarse classification."""
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = str(current).strip()
        if text:
            parts.append(text)
        for arg in getattr(current, "args", ()):
            if isinstance(arg, str) and arg.strip():
                parts.append(arg.strip())
        current = current.__cause__ or current.__context__
    return " | ".join(parts).lower()


def _tool_preference_hint(text: str) -> str:
    lowered = (text or "").strip().lower()
    if not lowered:
        return ""

    hints: list[str] = []
    if any(token in lowered for token in (*_VISION_QUERY_HINTS, *_AREA_CONTEXT_HINTS)):
        hints.append(
            "This is a live scene-understanding request. Prefer the latest [VISION ANALYSIS] "
            "and camera context. Do NOT call google_search for scene description, danger checks, "
            "or 'what's ahead/around me' style questions. For 'tell me more about this area/place', "
            "prefer current location context plus vision context."
        )
    if any(token in lowered for token in ("read", "sign", "menu", "label", "text")):
        hints.append(
            "This is an explicit text-reading request. Prefer extract_text_from_camera "
            "using the latest camera frame."
        )
    if any(token in lowered for token in ("navigate", "directions", "route me", "take me", "guide me")):
        hints.append(
            "This is an explicit active navigation request. Prefer navigate_to or "
            "get_walking_directions. Do not substitute maps_query or nearby_search "
            "unless the user asked for informational browsing only."
        )
    elif any(token in lowered for token in ("around me", "nearby", "near me", "where am i", "what's around")):
        hints.append(
            "This is a nearby/location-awareness request. Prefer get_location_info "
            "or nearby_search."
        )
    if any(token in lowered for token in ("weather", "today", "news", "search", "look up")):
        hints.append(
            "This is an external knowledge request. Prefer google_search."
        )
    if any(token in lowered for token in _MEMORY_STORE_HINTS):
        hints.append(
            "This is a memory-store request. Prefer remember_entity."
        )
    if any(token in lowered for token in _MEMORY_RECALL_HINTS):
        hints.append(
            "This is a memory-recall request. Prefer what_do_you_remember."
        )
    return "\n".join(hints)


_NAVIGATION_PLACE_TYPE_HINTS: dict[str, list[str]] = {
    "pharmacy": ["pharmacy"],
    "drugstore": ["pharmacy"],
    "cafe": ["cafe"],
    "coffee": ["cafe"],
    "restaurant": ["restaurant"],
    "bus stop": ["bus_stop"],
    "atm": ["atm"],
    "bank": ["bank"],
    "hospital": ["hospital"],
    "hotel": ["lodging"],
}

_VISION_QUERY_HINTS = (
    "what's ahead",
    "what is ahead",
    "what do you see",
    "what can you see",
    "what's around me",
    "what is around me",
    "describe what you see",
    "describe everything you see",
    "how many people",
    "is there any danger ahead",
)

_MEMORY_RECALL_HINTS = (
    "what pharmacy did i mention",
    "what did i mention",
    "what destination did i just tell you",
    "what intersection did i like",
    "what do you remember",
    "remember earlier",
)

_MEMORY_STORE_HINTS = (
    "remember that",
    "please remember",
    "remember this",
    "i want you to remember",
)

_FAREWELL_HINTS = (
    "thank you, that's all",
    "thanks, that's all",
    "that's all for now",
    "goodbye",
    "bye for now",
    "thank you goodbye",
)

_AREA_CONTEXT_HINTS = (
    "tell me more about this area",
    "tell me more about this place",
    "what else can you tell me about this place",
    "what else can you tell me about this area",
)


def _tool_result_fallback_text(tool_name: str, result: dict[str, object]) -> str | None:
    name = (tool_name or "").strip().lower()
    if not isinstance(result, dict):
        return None

    if name == "extract_text_from_camera":
        text = str(result.get("text") or "").strip()
        if text:
            return text
        message = str(result.get("message") or "").strip()
        return message or "I couldn't read the text clearly. Please pan the camera a little and try again."

    if name == "get_location_info":
        address = str(result.get("address") or "").strip()
        nearby = result.get("nearby_places") or []
        if isinstance(nearby, list) and nearby:
            first = nearby[0]
            place_name = str(first.get("name") or "a place nearby")
            distance = first.get("distance_meters")
            distance_text = f", about {distance} meters away" if distance is not None else ""
            return f"You're at {address or 'your current location'}. The nearest point of interest is {place_name}{distance_text}."
        if address:
            return f"You're at {address}."
        return None

    if name == "navigate_to":
        direction = str(result.get("destination_direction") or "").strip()
        destination = str(result.get("destination") or "").strip()
        if direction and destination:
            return f"{destination} is {direction}. I'll guide you there."
        if destination:
            return f"I found a route to {destination}."
        error = str(result.get("error") or "").strip()
        return error or None

    if name == "google_search":
        answer = str(result.get("answer") or "").strip()
        return answer or str(result.get("message") or "").strip() or None

    if name == "remember_entity":
        message = str(result.get("message") or "").strip()
        if message:
            return message
        content = str(result.get("content") or "").strip()
        return f"I'll remember that{': ' + content if content else '.'}"

    if name == "what_do_you_remember":
        answer = str(result.get("answer") or result.get("summary") or result.get("message") or "").strip()
        return answer or None

    if name == "nearby_search":
        places = result.get("places") or []
        if isinstance(places, list) and places:
            first = places[0]
            name_text = str(first.get("name") or "a nearby place").strip()
            distance = first.get("distance_meters")
            distance_text = f", about {distance} meters away" if distance is not None else ""
            return f"The nearest match is {name_text}{distance_text}."

    if name == "maps_query":
        answer = str(result.get("answer") or result.get("message") or "").strip()
        return answer or None

    if name == "remember_entity":
        answer = str(result.get("message") or "").strip()
        return answer or None

    return None


_NAVIGATION_DESTINATION_PATTERNS = (
    "navigate me to",
    "navigate to",
    "take me to",
    "guide me to",
    "route me to",
    "how do i get to",
    "go to",
    "walk to",
    "head to",
)


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
        resume_requested: bool = False,
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
        self._response_watchdog_task: asyncio.Task | None = None
        self._resume_requested = resume_requested
        self._pending_resume_context: str | None = None
        self._downstream_ready = asyncio.Event()
        self._downstream_init_error: Exception | None = None

    def _current_run_config(self):
        language_code = getattr(self.user_profile, "language", "") or ""
        self.run_config = session_manager.get_run_config(
            self.session_id,
            lod=self.session_ctx.current_lod,
            language_code=language_code,
        )
        return self.run_config

    def _is_stale_turn(self, origin_turn_seq: int | None) -> bool:
        return origin_turn_seq is not None and origin_turn_seq < self.state.user_turn_seq

    def _register_user_activity(
        self,
        *,
        explicit_turn_start: bool = False,
        source: str = "generic",
    ) -> bool:
        """Mark fresh user activity and discard queued context from older turns."""
        now_mono = time.monotonic()
        idle_gap = now_mono - self.state.last_user_activity_at
        hinted_turn_continues = (
            explicit_turn_start
            and source == "activity_start"
            and self.state.last_text_hint_at > 0
            and (now_mono - self.state.last_text_hint_at) <= self.state.USER_TURN_GAP_SEC
        )
        is_new_turn = (explicit_turn_start or idle_gap >= self.state.USER_TURN_GAP_SEC) and not hinted_turn_continues
        self.state.last_user_activity_at = now_mono
        if not is_new_turn:
            return False

        self.state.user_turn_seq += 1
        self.state.turn_output_seen = False
        self.state.turn_audio_output_seen = False
        self.state.latest_agent_transcript_for_turn = ""
        self.state.pending_fallback_text = None
        self.state.pending_fallback_turn_seq = self.state.user_turn_seq
        self._schedule_response_watchdog(self.state.user_turn_seq)
        discard_stale = getattr(self.ctx_queue, "discard_stale", None)
        if callable(discard_stale):
            discard_stale(
                min_turn_seq=self.state.user_turn_seq,
                categories=_STALE_QUEUE_CATEGORIES,
            )
        return True

    def _should_reconnect_silent_turn(self) -> bool:
        return self.state.user_turn_seq > 0 and not self.state.turn_output_seen

    def _cancel_response_watchdog(self) -> None:
        if self._response_watchdog_task is not None:
            self._response_watchdog_task.cancel()
            self._response_watchdog_task = None

    def _schedule_response_watchdog(
        self,
        turn_seq: int,
        *,
        delay_sec: float = _SILENT_TURN_WATCHDOG_SEC,
    ) -> None:
        self._cancel_response_watchdog()
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        self._response_watchdog_task = asyncio.create_task(
            self._response_watchdog(turn_seq, delay_sec)
        )

    async def _response_watchdog(self, turn_seq: int, delay_sec: float) -> None:
        try:
            await asyncio.sleep(delay_sec)
            if self.stop_downstream.is_set():
                return
            if turn_seq != self.state.user_turn_seq:
                return
            if self.state.turn_output_seen:
                return
            if await self._emit_pending_fallback_output(turn_seq):
                return
            if self._is_current_turn_farewell():
                await self._emit_local_agent_response(
                    "You're welcome. Goodbye for now.",
                    source="farewell_fallback",
                )
                return
            fallback_text = self._silent_turn_fallback_text()
            if fallback_text:
                logger.warning(
                    "No client-visible output for turn %d within %.1fs; emitting local fallback instead of reconnect",
                    turn_seq,
                    delay_sec,
                )
                await self._emit_local_agent_response(
                    fallback_text,
                    source="silent_turn_fallback",
                )
                return
            logger.warning(
                "No client-visible output for turn %d within %.1fs; requesting reconnect",
                turn_seq,
                delay_sec,
            )
            await self._safe_send_json({
                "type": MessageType.GO_AWAY,
                "retry_ms": 750,
                "message": "No response generated for the last turn. Reconnecting.",
            })
            self.stop_downstream.set()
            try:
                await self.websocket.close(code=1012, reason="silent_turn_watchdog")
            except Exception:
                pass
        except asyncio.CancelledError:
            return

    async def _emit_pending_fallback_output(self, turn_seq: int) -> bool:
        if not (
            self.state.pending_fallback_text
            and self.state.pending_fallback_turn_seq == turn_seq
        ):
            return False

        fallback_text = self.state.pending_fallback_text
        pcm = b""
        try:
            pcm = await synthesize_fallback_pcm(fallback_text)
        except Exception:
            logger.exception("Fallback TTS synthesis failed")
        sent = await self._safe_send_json({
            "type": MessageType.TRANSCRIPT,
            "text": fallback_text,
            "role": "agent",
            "source": "tool_fallback",
        })
        if not sent:
            return False
        if pcm:
            await self._safe_send_bytes(pcm)
        self.state.turn_output_seen = True
        self.state.pending_fallback_text = None
        self._cancel_response_watchdog()
        logger.info("Emitted tool-result fallback output for turn %d", turn_seq)
        return True

    async def _emit_local_agent_response(self, text: str, *, source: str) -> bool:
        clean = (text or "").strip()
        if not clean:
            return False
        sent = await self._safe_send_json({
            "type": MessageType.TRANSCRIPT,
            "text": clean,
            "role": "agent",
            "source": source,
        })
        if not sent:
            return False
        self.state.latest_agent_transcript_for_turn = clean
        self.state.turn_output_seen = True
        try:
            pcm = await synthesize_local_fallback_pcm(clean)
        except Exception:
            logger.exception("Local agent-response TTS synthesis failed")
            return sent
        if pcm:
            await self._safe_send_bytes(pcm)
        return sent

    async def _emit_prefeedback_output(self, text: str) -> bool:
        return await self._emit_local_agent_response(text, source="prefeedback")

    def _is_farewell_text(self, text: str) -> bool:
        lowered = (text or "").strip().lower()
        if not lowered:
            return False
        return any(token in lowered for token in _FAREWELL_HINTS)

    def _is_current_turn_farewell(self) -> bool:
        recent = _recent_user_utterances(self.state.transcript_history, max_items=1)
        if not recent:
            return False
        return self._is_farewell_text(recent[0])

    def _silent_turn_fallback_text(self) -> str | None:
        recent = _recent_user_utterances(self.state.transcript_history, max_items=1)
        if not recent:
            return "I'm having trouble responding right now. Please try again."

        text = recent[0]
        lowered = text.strip().lower()
        ephemeral = session_manager.get_ephemeral_context(self.session_id)
        gps = getattr(ephemeral, "gps", None)
        has_gps = isinstance(getattr(gps, "lat", None), (int, float)) and isinstance(getattr(gps, "lng", None), (int, float))

        if self._is_farewell_text(text):
            return "You're welcome. Goodbye for now."
        if any(token in lowered for token in _MEMORY_STORE_HINTS):
            return "I heard that memory request, but I need you to repeat the key detail one more time."
        if any(token in lowered for token in _MEMORY_RECALL_HINTS):
            return "I'm having trouble recalling that right now. Please ask me again in a moment."
        if _has_location_query_intent(text) and not has_gps:
            return "I need your location to answer that. Tell me your city, or enable location and ask again."
        if any(token in lowered for token in ("read", "sign", "menu", "label", "text")):
            return "I couldn't read that clearly. Please pan the camera a little and try again."
        if any(token in lowered for token in (*_VISION_QUERY_HINTS, *_AREA_CONTEXT_HINTS)):
            return "I need a steadier camera view to describe that. Please hold the camera steady and ask again."
        if _has_navigation_intent(text):
            return "I'm still working on directions. Please ask me again in a moment."
        if _has_location_query_intent(text):
            return "I'm still checking that for you. Please ask again in a moment."
        return "I'm having trouble responding right now. Please try again."

    def _infer_navigation_redirect_types(self, question: str) -> list[str] | None:
        lowered = (question or "").lower()
        for token, place_types in _NAVIGATION_PLACE_TYPE_HINTS.items():
            if token in lowered:
                return place_types
        return None

    async def _maybe_redirect_maps_query(
        self,
        *,
        question: str,
        user_speaking: bool,
    ) -> tuple[bool, dict[str, object]]:
        lowered = (question or "").strip().lower()
        if any(token in lowered for token in _AREA_CONTEXT_HINTS):
            summary_parts: list[str] = []
            ephemeral = session_manager.get_ephemeral_context(self.session_id)
            if ephemeral.gps:
                location_result = await _dispatch_function_call(
                    "get_location_info",
                    {},
                    self.session_id,
                    self.user_id,
                )
                fallback_text = _tool_result_fallback_text("get_location_info", location_result)
                if fallback_text:
                    summary_parts.append(fallback_text)
            if self.state.last_vision_context_text:
                summary_parts.append(self.state.last_vision_context_text.replace("[VISION ANALYSIS]\n", "").strip())
            answer = " ".join(part for part in summary_parts if part).strip()
            if not answer:
                answer = "I'm still checking the area. Please hold the camera steady for a moment."
            return True, {
                "success": True,
                "redirected_to": "vision_context",
                "answer": answer,
            }

        search_types = self._infer_navigation_redirect_types(question)
        if _has_location_query_intent(question) and search_types:
            search_result = await _dispatch_function_call(
                "nearby_search",
                {"types": search_types, "radius": 500},
                self.session_id,
                self.user_id,
            )
            fallback_text = _tool_result_fallback_text("nearby_search", search_result)
            answer = fallback_text or str(search_result.get("message") or "").strip()
            return True, {
                "success": bool(search_result.get("success", True)),
                "redirected_to": "nearby_search",
                "answer": answer,
                "nearby_result": search_result,
            }

        recent_user = _recent_user_utterances(self.state.transcript_history, max_items=3)
        if not any(_has_navigation_intent(text) for text in recent_user):
            return False, {}

        search_args: dict[str, object] = {}
        if search_types:
            search_args["types"] = search_types

        search_result = await _dispatch_function_call(
            "nearby_search",
            search_args,
            self.session_id,
            self.user_id,
        )
        places = list(search_result.get("places", [])) if isinstance(search_result, dict) else []
        if not places:
            return False, {}

        destination = str(places[0].get("address") or places[0].get("name") or question).strip()
        if not destination:
            return False, {}

        behavior = resolve_tool_behavior(
            tool_name="navigate_to",
            lod=self.session_ctx.current_lod,
            is_user_speaking=user_speaking,
        )
        await self._emit_tool_event(
            "navigate_to",
            behavior,
            status="invoked",
            data={"redirected_from": "maps_query", "destination": destination},
        )
        result = await _dispatch_function_call(
            "navigate_to",
            {"destination": destination},
            self.session_id,
            self.user_id,
        )
        await self._safe_send_json({
            "type": MessageType.TOOL_RESULT,
            "tool": "navigate_to",
            "behavior": behavior_to_text(behavior),
            "data": _json_safe(result),
        })
        await self._safe_send_json({
            "type": MessageType.NAVIGATION_RESULT,
            "summary": str(result.get("destination_direction") or result.get("destination") or destination),
            "behavior": behavior_to_text(behavior),
            "data": _json_safe(result),
        })
        fallback_text = _tool_result_fallback_text("navigate_to", result)
        if fallback_text:
            self.state.pending_fallback_text = fallback_text
            self.state.pending_fallback_turn_seq = self.state.user_turn_seq

        redirected_result = {
            "success": True,
            "redirected_to": "navigate_to",
            "selected_place": places[0],
            "navigation_result": result,
        }
        return True, redirected_result

    async def _maybe_redirect_google_search(
        self,
        *,
        query: str,
    ) -> tuple[bool, dict[str, object]]:
        lowered = (query or "").strip().lower()
        if not lowered:
            return False, {}

        if any(token in lowered for token in _VISION_QUERY_HINTS):
            ephemeral = session_manager.get_ephemeral_context(self.session_id)
            summary_parts: list[str] = []
            if ephemeral.gps:
                location_result = await _dispatch_function_call(
                    "get_location_info",
                    {},
                    self.session_id,
                    self.user_id,
                )
                fallback_text = _tool_result_fallback_text("get_location_info", location_result)
                if fallback_text:
                    summary_parts.append(fallback_text)
            if self.state.last_vision_context_text:
                summary_parts.append(self.state.last_vision_context_text.replace("[VISION ANALYSIS]\n", "").strip())
            answer = " ".join(part for part in summary_parts if part).strip()
            if not answer:
                answer = "I'm still checking the scene. Please hold the camera steady for a moment."
            return True, {
                "success": True,
                "redirected_to": "vision_context",
                "answer": answer,
            }

        if any(token in lowered for token in _MEMORY_STORE_HINTS):
            name = "important note"
            entity_type = "event"
            attributes = query.strip()

            match = re.search(
                r"pharmacy(?:\s+is\s+called)?\s+([A-Za-z0-9'& -]+)",
                query,
                re.IGNORECASE,
            )
            if match:
                name = match.group(1).strip(" .,!?:;")
                entity_type = "place"
                attributes = f"type=pharmacy,description={query.strip()}"

            memory_result = await _dispatch_function_call(
                "remember_entity",
                {
                    "name": name,
                    "entity_type": entity_type,
                    "attributes": attributes,
                },
                self.session_id,
                self.user_id,
            )
            return True, {
                "success": bool(memory_result.get("status") not in {"failed", "error"}),
                "redirected_to": "remember_entity",
                **memory_result,
            }

        if any(token in lowered for token in _MEMORY_RECALL_HINTS):
            memory_result = await _dispatch_function_call(
                "what_do_you_remember",
                {"query": query},
                self.session_id,
                self.user_id,
            )
            return True, {
                "success": bool(memory_result.get("success", True)),
                "redirected_to": "what_do_you_remember",
                **memory_result,
            }

        return False, {}

    def _extract_navigation_destination(self, text: str) -> tuple[str | None, list[str] | None]:
        lowered = (text or "").strip().lower()
        if not lowered:
            return None, None

        search_types = self._infer_navigation_redirect_types(lowered)
        if search_types and any(token in lowered for token in ("nearest", "closest")):
            return None, search_types

        destination = None
        for marker in _NAVIGATION_DESTINATION_PATTERNS:
            if marker in lowered:
                start = lowered.index(marker) + len(marker)
                destination = text[start:].strip()
                break

        if destination is None:
            destination = text.strip()

        destination = re.sub(r"^(the|a|an)\s+", "", destination, flags=re.IGNORECASE)
        destination = re.sub(r"\b(please|thanks?|now)\b", "", destination, flags=re.IGNORECASE)
        destination = destination.strip(" ,.?!")
        return (destination or None), search_types

    async def _maybe_handle_direct_navigation_intent(self, input_text: str) -> bool:
        current_turn = self.state.user_turn_seq
        if current_turn <= 0 or self.state.direct_tool_handled_turn_seq == current_turn:
            return False
        if not _has_navigation_intent(input_text):
            return False
        ephemeral = session_manager.get_ephemeral_context(self.session_id)
        gps = getattr(ephemeral, "gps", None)
        lat = getattr(gps, "lat", None)
        lng = getattr(gps, "lng", None)
        if not isinstance(lat, (int, float)) or not isinstance(lng, (int, float)):
            return False

        destination, search_types = self._extract_navigation_destination(input_text)
        user_speaking = self.session_ctx.current_activity_state == "user_speaking"
        behavior = resolve_tool_behavior(
            tool_name="navigate_to",
            lod=self.session_ctx.current_lod,
            is_user_speaking=user_speaking,
        )

        result: dict[str, object]
        selected_place: dict[str, object] | None = None
        if search_types:
            await self._emit_tool_event(
                "nearby_search",
                ToolBehavior.WHEN_IDLE,
                status="invoked",
                data={"source": "server_shortcut", "types": search_types},
            )
            search_result = await _dispatch_function_call(
                "nearby_search",
                {"types": search_types, "radius": 500},
                self.session_id,
                self.user_id,
            )
            places = list(search_result.get("places", [])) if isinstance(search_result, dict) else []
            if not places:
                result = {
                    "success": False,
                    "error": "No nearby destination found.",
                }
            else:
                selected_place = places[0]
                destination = str(
                    selected_place.get("address") or selected_place.get("name") or ""
                ).strip()
                result = await _dispatch_function_call(
                    "navigate_to",
                    {"destination": destination},
                    self.session_id,
                    self.user_id,
                )
        elif destination:
            result = await _dispatch_function_call(
                "navigate_to",
                {"destination": destination},
                self.session_id,
                self.user_id,
            )
        else:
            return False

        await self._emit_tool_event(
            "navigate_to",
            behavior,
            status="invoked",
            data={"source": "server_shortcut", "destination": destination or input_text},
        )
        await self._safe_send_json({
            "type": MessageType.TOOL_RESULT,
            "tool": "navigate_to",
            "behavior": behavior_to_text(behavior),
            "data": _json_safe(result),
        })
        await self._safe_send_json({
            "type": MessageType.NAVIGATION_RESULT,
            "summary": str(result.get("destination_direction") or result.get("destination") or destination or ""),
            "behavior": behavior_to_text(behavior),
            "data": _json_safe(result),
        })

        fallback_text = _tool_result_fallback_text("navigate_to", result)
        if fallback_text:
            self.state.pending_fallback_text = fallback_text
            self.state.pending_fallback_turn_seq = current_turn
            await self._emit_pending_fallback_output(current_turn)

        from google.genai.types import FunctionResponse
        shortcut_response = {
            "success": bool(result.get("success", True)),
            "handled_by": "server_shortcut",
            "selected_place": selected_place,
            "navigation_result": result,
        }
        fr = FunctionResponse(name="navigate_to", response=shortcut_response)
        self.ctx_queue.inject_immediate(
            types.Content(
                parts=[
                    types.Part(
                        text=(
                            "[TOOL RESULT READY] The navigation request has already been "
                            "handled. Continue from this result without greeting again."
                        )
                    ),
                    types.Part(function_response=fr),
                ],
                role="user",
            ),
            is_function_response=True,
        )

        self.state.direct_tool_handled_turn_seq = current_turn
        self.state.is_interrupted = True
        self.state.last_interrupt_at = time.monotonic()
        self.state.model_audio_last_seen_at = 0.0
        self.ctx_queue._transition_to(ModelState.IDLE)
        logger.info("Handled direct navigation shortcut for turn %d", current_turn)
        return True

    async def _maybe_handle_direct_nearby_search_intent(self, input_text: str) -> bool:
        current_turn = self.state.user_turn_seq
        if current_turn <= 0 or self.state.direct_tool_handled_turn_seq == current_turn:
            return False
        if _has_navigation_intent(input_text):
            return False
        if not _has_location_query_intent(input_text):
            return False

        search_types = self._infer_navigation_redirect_types(input_text)
        if not search_types:
            return False

        ephemeral = session_manager.get_ephemeral_context(self.session_id)
        gps = getattr(ephemeral, "gps", None)
        lat = getattr(gps, "lat", None)
        lng = getattr(gps, "lng", None)
        if not isinstance(lat, (int, float)) or not isinstance(lng, (int, float)):
            self.state.direct_tool_handled_turn_seq = current_turn
            await self._emit_local_agent_response(
                "I need your location to search nearby places. Tell me your city, or enable location and ask again.",
                source="nearby_search_location_required",
            )
            logger.info("Handled direct nearby_search location-missing shortcut for turn %d", current_turn)
            return True

        behavior = resolve_tool_behavior(
            tool_name="nearby_search",
            lod=self.session_ctx.current_lod,
            is_user_speaking=self.session_ctx.current_activity_state == "user_speaking",
        )
        await self._emit_tool_event(
            "nearby_search",
            behavior,
            status="invoked",
            data={"source": "server_shortcut", "types": search_types},
        )
        result = await _dispatch_function_call(
            "nearby_search",
            {"types": search_types, "radius": 500},
            self.session_id,
            self.user_id,
        )
        await self._safe_send_json({
            "type": MessageType.TOOL_RESULT,
            "tool": "nearby_search",
            "behavior": behavior_to_text(behavior),
            "data": _json_safe(result),
        })
        fallback_text = _tool_result_fallback_text("nearby_search", result)
        if fallback_text:
            self.state.pending_fallback_text = fallback_text
            self.state.pending_fallback_turn_seq = current_turn
            await self._emit_pending_fallback_output(current_turn)

        from google.genai.types import FunctionResponse
        fr = FunctionResponse(name="nearby_search", response=result)
        self.ctx_queue.inject_immediate(
            types.Content(
                parts=[
                    types.Part(
                        text=(
                            "[TOOL RESULT READY] The nearby search request has already been "
                            "handled. Use the result to answer the user now."
                        )
                    ),
                    types.Part(function_response=fr),
                ],
                role="user",
            ),
            is_function_response=True,
        )

        self.state.direct_tool_handled_turn_seq = current_turn
        logger.info("Handled direct nearby_search shortcut for turn %d", current_turn)
        return True

    # ── Main entry point ───────────────────────────────────────────────

    async def run(self) -> None:
        """Run the upstream/downstream loop until the session ends.

        Sends session_ready, tools_manifest, and the initial greeting/context
        before starting the upstream/downstream tasks.
        """
        downstream_task = asyncio.create_task(self._downstream())
        try:
            await asyncio.wait_for(
                self._downstream_ready.wait(),
                timeout=_LIVE_SESSION_READY_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            logger.error(
                "Timed out waiting for Live session init: user=%s session=%s",
                self.user_id, self.session_id,
            )
            downstream_task.cancel()
            try:
                await downstream_task
            except (asyncio.CancelledError, Exception):
                pass
            await self._safe_send_json({
                "type": MessageType.ERROR,
                "error": "Live session initialization timed out. Please retry.",
            })
            await self._cleanup()
            return

        if self._downstream_init_error is not None:
            logger.exception(
                "Live session init failed before session_ready: user=%s session=%s",
                self.user_id, self.session_id,
                exc_info=self._downstream_init_error,
            )
            await self._safe_send_json({
                "type": MessageType.ERROR,
                "error": "Failed to initialize live session. Please retry.",
            })
            await self._cleanup()
            return

        # Notify client the WebSocket is live
        if not await self._safe_send_json({"type": MessageType.SESSION_READY}):
            logger.info(
                "WebSocket closed before session_ready: user=%s session=%s",
                self.user_id, self.session_id,
            )
            downstream_task.cancel()
            try:
                await downstream_task
            except (asyncio.CancelledError, Exception):
                pass
            return

        asyncio.create_task(self.session_meta.write_session_start())

        # Send tools manifest so iOS Dev Console shows tool/context status
        await self._safe_send_json(self._build_tools_manifest())

        _initial_prompt = build_full_dynamic_prompt(
            lod=self.session_ctx.current_lod,
            profile=self.user_profile,
            ephemeral_semantic="",
            session=self.session_ctx,
            memories=self._initial_memories if self._initial_memories else None,
            assembled_profile=self._assembled_profile,
        )

        if self._resume_requested:
            self._pending_resume_context = (
                "[CONTEXT UPDATE - DO NOT SPEAK]\n"
                + _initial_prompt
                + "\n\n[SESSION RESUME] This session has resumed after an interruption. "
                  "Do not greet again. Continue helping with the user's current request "
                  "or wait silently for their next input."
            )
            logger.info(
                "Resume requested for session %s; queued silent resume context for next user turn",
                self.session_id,
            )
        else:
            # Inject combined context (LOD + greeting) so model speaks once
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
        self._cancel_response_watchdog()
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
            if payload and self.state.user_turn_seq > 0:
                self.state.turn_output_seen = True
                self.state.turn_audio_output_seen = True
                self.state.pending_fallback_text = None
                self._cancel_response_watchdog()
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
            self.state.latest_agent_transcript_for_turn = clean_text
            if self.state.user_turn_seq > 0:
                self.state.turn_output_seen = True
                self.state.pending_fallback_text = None
                self._cancel_response_watchdog()
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
        self.ctx_queue.enqueue(
            category="lod",
            text=lod_message,
            priority=3,
            speak=False,
            turn_seq=self.state.user_turn_seq,
        )
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
        self.ctx_queue.enqueue(
            category="vad",
            text=build_vad_runtime_update_message(new_lod),
            priority=7,
            speak=False,
            turn_seq=self.state.user_turn_seq,
        )
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

    async def _run_vision_analysis(
        self,
        image_base64: str,
        origin_turn_seq: int | None = None,
    ) -> None:
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
            await self._emit_prefeedback_output("Let me look at that for you...")
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
            if self._is_stale_turn(origin_turn_seq):
                logger.info(
                    "Dropping stale vision result from turn %s (current=%d)",
                    origin_turn_seq,
                    self.state.user_turn_seq,
                )
                await self._emit_tool_event(
                    "analyze_scene",
                    ToolBehavior.WHEN_IDLE,
                    status="stale",
                    data={
                        "origin_turn_seq": origin_turn_seq,
                        "current_turn_seq": self.state.user_turn_seq,
                    },
                )
                return
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
                    self.ctx_queue.enqueue(
                        category="vision",
                        text=vision_text,
                        priority=5,
                        speak=speak,
                        turn_seq=origin_turn_seq or self.state.user_turn_seq,
                    )
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

    async def _run_face_recognition(
        self,
        image_base64: str,
        origin_turn_seq: int | None = None,
    ) -> None:
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
            if self._is_stale_turn(origin_turn_seq):
                logger.info(
                    "Dropping stale face result from turn %s (current=%d)",
                    origin_turn_seq,
                    self.state.user_turn_seq,
                )
                await self._emit_tool_event(
                    "identify_person",
                    ToolBehavior.SILENT,
                    status="stale",
                    data={
                        "origin_turn_seq": origin_turn_seq,
                        "current_turn_seq": self.state.user_turn_seq,
                    },
                )
                return
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
                self.ctx_queue.enqueue(
                    category="face",
                    text=face_text,
                    priority=4,
                    speak=speak,
                    turn_seq=origin_turn_seq or self.state.user_turn_seq,
                )
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

    async def _run_ocr_analysis(
        self,
        image_base64: str,
        safety_only: bool = False,
        origin_turn_seq: int | None = None,
    ) -> None:
        if not _ocr_available:
            await self._emit_tool_event("extract_text", ToolBehavior.WHEN_IDLE, status="unavailable", data={"reason": "ocr_agent_unavailable"})
            return
        if not safety_only:
            now_mono = time.monotonic()
            if now_mono - self.state.last_ocr_prefeedback_at >= OCR_PREFEEDBACK_COOLDOWN_SEC:
                await self._emit_prefeedback_output("Reading the text for you...")
                self.state.last_ocr_prefeedback_at = now_mono
        try:
            from agents.ocr_agent import extract_text
            hint = ""
            if self.session_ctx.space_type:
                hint = f"User is in a {self.session_ctx.space_type} environment."
            if self.session_ctx.active_task:
                hint += f" Currently: {self.session_ctx.active_task}."
            result = await extract_text(image_base64, context_hint=hint, safety_only=safety_only)
            if self._is_stale_turn(origin_turn_seq):
                logger.info(
                    "Dropping stale OCR result from turn %s (current=%d)",
                    origin_turn_seq,
                    self.state.user_turn_seq,
                )
                await self._emit_tool_event(
                    "extract_text",
                    ToolBehavior.WHEN_IDLE,
                    status="stale",
                    data={
                        "origin_turn_seq": origin_turn_seq,
                        "current_turn_seq": self.state.user_turn_seq,
                    },
                )
                return
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
                self.ctx_queue.enqueue(
                    category="ocr",
                    text=ocr_text,
                    priority=5,
                    speak=speak,
                    turn_seq=origin_turn_seq or self.state.user_turn_seq,
                )
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
                self.ctx_queue.enqueue(
                    category="telemetry",
                    text=telemetry_text,
                    priority=8,
                    speak=False,
                    turn_seq=self.state.user_turn_seq,
                )
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
                self.ctx_queue.enqueue(
                    category="lod_resume",
                    text=resume_prompt,
                    priority=3,
                    speak=True,
                    turn_seq=self.state.user_turn_seq,
                )
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
                        self._register_user_activity()
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
                    self._register_user_activity()
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

                elif msg_type == "text_hint":
                    input_text = str(message.get("text", "")).strip()
                    if not input_text:
                        continue
                    self.state.last_text_hint_at = time.monotonic()
                    self._register_user_activity(explicit_turn_start=True, source="text_hint")
                    self.session_ctx.current_activity_state = "user_speaking"
                    self.state.transcript_history.append({"role": "user", "text": input_text})
                    self.session_meta.record_interaction()
                    await self._safe_send_json({
                        "type": MessageType.TRANSCRIPT,
                        "text": input_text,
                        "role": "user",
                        "source": "client_hint",
                    })
                    if await self._maybe_handle_direct_navigation_intent(input_text):
                        continue
                    hint_text = (
                        "[TRANSCRIPT HINT - DO NOT SPEAK]\n"
                        f"The user's current utterance is: {input_text}\n"
                        "Use this semantic hint to interpret their live audio and "
                        "respond to the same request. Do not mention the hint."
                    )
                    preference_hint = _tool_preference_hint(input_text)
                    if preference_hint:
                        hint_text += "\n" + preference_hint
                    if self._pending_resume_context:
                        hint_text = self._pending_resume_context + "\n\n" + hint_text
                        self._pending_resume_context = None
                    hint_content = types.Content(
                        parts=[types.Part(
                            text=hint_text
                        )],
                        role="user",
                    )
                    self.ctx_queue.inject_immediate(hint_content, is_function_response=True)

                elif msg_type in {"activity_start", "activity_end"}:
                    event_name = str(msg_type)
                    if event_name == "activity_start":
                        self._register_user_activity(
                            explicit_turn_start=True,
                            source="activity_start",
                        )
                    else:
                        self.state.last_user_activity_at = time.monotonic()
                        self.state.last_text_hint_at = 0.0
                    self.session_meta.record_interaction()
                    had_interrupt = self.state.is_interrupted
                    if _should_reset_interrupted_on_activity_start(event_name=event_name, interrupted=had_interrupt):
                        self.state.is_interrupted = False
                        logger.info("Reset stale interrupted state on activity_start: user=%s session=%s", self.user_id, self.session_id)
                    if event_name == "activity_start":
                        now_mono = time.monotonic()
                        model_was_speaking = (
                            self.state.model_audio_last_seen_at > 0
                            and (now_mono - self.state.model_audio_last_seen_at) < 2.0
                        ) or self.ctx_queue.model_speaking
                        if model_was_speaking:
                            self.state.is_interrupted = True
                            self.state.last_interrupt_at = now_mono
                            self.state.model_audio_last_seen_at = 0.0
                            self.ctx_queue._transition_to(ModelState.IDLE)
                            logger.info(
                                "Activity-start barge-in accepted: user=%s session=%s",
                                self.user_id,
                                self.session_id,
                            )
                    if event_name == "activity_start":
                        if self._pending_resume_context:
                            resume_content = types.Content(
                                parts=[types.Part(text=self._pending_resume_context)],
                                role="user",
                            )
                            self.ctx_queue.inject_immediate(resume_content)
                            self._pending_resume_context = None
                            logger.info("Injected resume context on activity_start for session %s", self.session_id)
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
        current_turn_seq = self.state.user_turn_seq
        if now - self.state.last_vision_time >= vision_interval:
            self.state.last_vision_time = now
            await self._emit_tool_event("analyze_scene", ToolBehavior.WHEN_IDLE, status="queued")
            queued_agents.append("vision")
            asyncio.create_task(
                self._run_vision_analysis(image_b64, origin_turn_seq=current_turn_seq)
            )
            if lod >= 2 and self.state.face_runtime_available:
                await self._emit_tool_event("identify_person", ToolBehavior.SILENT, status="queued")
                queued_agents.append("face")
                await self._emit_identity_event(person_name="unknown", matched=False, similarity=0.0, source="queued")
                asyncio.create_task(
                    self._run_face_recognition(image_b64, origin_turn_seq=current_turn_seq)
                )
            if lod == 1 and now - self.state.last_safety_ocr_at >= 15.0:
                self.state.last_safety_ocr_at = now
                await self._emit_tool_event("extract_text", ToolBehavior.WHEN_IDLE, status="queued")
                queued_agents.append("ocr")
                asyncio.create_task(
                    self._run_ocr_analysis(
                        image_b64,
                        safety_only=True,
                        origin_turn_seq=current_turn_seq,
                    )
                )
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
                self.ctx_queue.enqueue(category="camera_toggle", text="[CAMERA ACTIVATED] The user has turned on the rear camera. Visual context is now available via periodic [VISION ANALYSIS] injections. Do NOT describe every frame. Only speak about safety hazards or when the user asks. Acknowledge camera activation in one brief sentence, then observe silently.", priority=4, speak=True, turn_seq=self.state.user_turn_seq)
            else:
                self.ctx_queue.enqueue(category="camera_toggle", text="[CAMERA DEACTIVATED] The user has turned off the camera. You are now in audio-only mode. Do not reference visual information unless recalling something previously seen.", priority=4, speak=True, turn_seq=self.state.user_turn_seq)
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
            self.ctx_queue.enqueue(category="profile_update", text=profile_ctx, priority=3, speak=False, turn_seq=self.state.user_turn_seq)
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
        try:
            adk_session_id = self.session_id
            if _NEEDS_SESSION_ID_MAPPING:
                cached_adk_id = session_manager.get_adk_session_id(self.session_id)
                if cached_adk_id:
                    adk_session_id = cached_adk_id
                else:
                    adk_session = await self.runner.session_service.create_session(
                        app_name="sightline",
                        user_id=self.user_id,
                    )
                    adk_session_id = adk_session.id
                    session_manager.set_adk_session_id(self.session_id, adk_session_id)
                    logger.info(
                        "Created ADK session %s for logical session %s",
                        adk_session_id, self.session_id,
                    )

            def _start_live_events():
                return self.runner.run_live(
                    session_id=adk_session_id,
                    user_id=self.user_id,
                    live_request_queue=self.live_request_queue,
                    run_config=self._current_run_config(),
                )

            live_events = await asyncio.to_thread(_start_live_events)
            self.state.is_interrupted = False
            self._downstream_ready.set()

            async for event in live_events:
                if self.stop_downstream.is_set():
                    break
                _output_transcription_forwarded = False
                if self.state.downstream_retry_count:
                    logger.info(
                        "Live stream recovered after %d retry attempt(s): user=%s session=%s",
                        self.state.downstream_retry_count,
                        self.user_id,
                        self.session_id,
                    )
                    self.state.downstream_retry_count = 0

                _usage = getattr(event, "usage_metadata", None)
                if _usage is not None:
                    self.token_monitor.update(_usage)

                if event.live_session_resumption_update:
                    update = event.live_session_resumption_update
                    new_handle = getattr(update, "new_handle", None) or getattr(update, "newHandle", None)
                    last_consumed_idx = getattr(
                        update,
                        "last_consumed_client_message_index",
                        None,
                    )
                    if last_consumed_idx is None:
                        last_consumed_idx = getattr(
                            update,
                            "lastConsumedClientMessageIndex",
                            None,
                        )
                    if isinstance(last_consumed_idx, int):
                        self.state.last_consumed_client_message_index = last_consumed_idx
                    if new_handle:
                        session_manager.update_handle(self.session_id, new_handle)
                    if not await self._safe_send_json({
                        "type": MessageType.SESSION_RESUMPTION,
                        "handle": new_handle or "",
                    }):
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
                        if (
                            self.state.turn_output_seen
                            and not self.state.turn_audio_output_seen
                            and self.state.latest_agent_transcript_for_turn
                        ):
                            await self._emit_local_agent_response(
                                self.state.latest_agent_transcript_for_turn,
                                source="audio_backfill",
                            )
                        self.ctx_queue.on_turn_complete()
                        if await self._emit_pending_fallback_output(self.state.user_turn_seq):
                            self.ctx_queue.enqueue(category="turn_boundary", text="<<<INTERNAL_CONTEXT>>>\n[TURN BOUNDARY] Previous request complete. Await user's next request. Do not carry forward tool-calling intent from previous turn.\n<<<END_INTERNAL_CONTEXT>>>", priority=1, speak=False, turn_seq=self.state.user_turn_seq)
                            self.ctx_queue.flush_or_defer_first_turn(camera_active=self.state.client_camera_active)
                            continue
                        if self._should_reconnect_silent_turn():
                            logger.info(
                                "Turn %d completed without client-visible output; awaiting watchdog before reconnect",
                                self.state.user_turn_seq,
                            )
                            continue
                        self.ctx_queue.enqueue(category="turn_boundary", text="<<<INTERNAL_CONTEXT>>>\n[TURN BOUNDARY] Previous request complete. Await user's next request. Do not carry forward tool-calling intent from previous turn.\n<<<END_INTERNAL_CONTEXT>>>", priority=1, speak=False, turn_seq=self.state.user_turn_seq)
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
                        self.ctx_queue.enqueue(category="echo_cancel", text="[SYSTEM: The previous audio input was an echo of your own speech. Do not respond to it.]", priority=1, speak=False, turn_seq=self.state.user_turn_seq)
                        logger.info("Queued echo cancellation for Gemini")
                    else:
                        self.state.transcript_history.append({"role": "user", "text": input_text})
                        self.state.last_user_activity_at = time.monotonic()
                        self.session_meta.record_interaction()
                        if not await self._safe_send_json({"type": MessageType.TRANSCRIPT, "text": input_text, "role": "user"}):
                            break
                        if self._is_farewell_text(input_text):
                            self.state.is_interrupted = True
                            self.state.last_interrupt_at = time.monotonic()
                            self.state.model_audio_last_seen_at = 0.0
                            self.state.direct_tool_handled_turn_seq = self.state.user_turn_seq
                            self.ctx_queue._transition_to(ModelState.IDLE)
                            await self._emit_local_agent_response(
                                "You're welcome. Goodbye for now.",
                                source="farewell_direct",
                            )
                            continue
                        if await self._maybe_handle_direct_nearby_search_intent(input_text):
                            continue
                        if await self._maybe_handle_direct_navigation_intent(input_text):
                            continue
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
            if not self._downstream_ready.is_set():
                self._downstream_init_error = exc
                self._downstream_ready.set()
            exc_text = _exception_chain_text(exc)
            is_keepalive_timeout = any(
                token in exc_text for token in (
                    "keepalive ping timeout",
                    "abnormal closure [internal]",
                    "1011 (internal error)",
                )
            )
            if is_keepalive_timeout and self.state.downstream_retry_count < self.state.DOWNSTREAM_MAX_RETRIES and self.websocket.client_state == WebSocketState.CONNECTED:
                self.state.downstream_retry_count += 1
                backoff_sec = 0.8 * self.state.downstream_retry_count
                self.state.is_interrupted = False
                self.state.model_audio_last_seen_at = 0.0
                self.state.transcript_buffer = ""
                self.state.transcript_buffer_started_at = 0.0
                self.ctx_queue._transition_to(ModelState.IDLE)
                logger.warning(
                    "Transient Live API keepalive timeout; retrying downstream internally (%d/%d) after %.1fs: user=%s session=%s",
                    self.state.downstream_retry_count,
                    self.state.DOWNSTREAM_MAX_RETRIES,
                    backoff_sec,
                    self.user_id,
                    self.session_id,
                )
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

            if fc.name == "maps_query":
                redirected, redirected_result = await self._maybe_redirect_maps_query(
                    question=str(call_args.get("question", "")),
                    user_speaking=user_speaking,
                )
                if redirected:
                    await self._emit_tool_event(
                        fc.name,
                        behavior,
                        status="redirected",
                        data={"redirected_to": "navigate_to", "args": _json_safe(call_args)},
                    )
                    await self._safe_send_json({
                        "type": MessageType.TOOL_RESULT,
                        "tool": fc.name,
                        "behavior": behavior_to_text(behavior),
                        "data": _json_safe(redirected_result),
                    })
                    from google.genai.types import FunctionResponse
                    fr = FunctionResponse(name=fc.name, response=redirected_result)
                    parts = [
                        types.Part(
                            text=(
                                "[TOOL RESULT READY] The navigation request was resolved. "
                                "Use the redirected navigation result to answer the user now."
                            )
                        ),
                        types.Part(function_response=fr),
                    ]
                    self.ctx_queue.inject_immediate(
                        types.Content(parts=parts, role="user"),
                        is_function_response=True,
                    )
                    logger.info("Redirected maps_query to navigate_to for explicit navigation request")
                    continue

            if fc.name == "google_search":
                redirected, redirected_result = await self._maybe_redirect_google_search(
                    query=str(call_args.get("query", "")),
                )
                if redirected:
                    await self._emit_tool_event(
                        fc.name,
                        behavior,
                        status="redirected",
                        data={"redirected_result": _json_safe(redirected_result)},
                    )
                    await self._safe_send_json({
                        "type": MessageType.TOOL_RESULT,
                        "tool": fc.name,
                        "behavior": behavior_to_text(behavior),
                        "data": _json_safe(redirected_result),
                    })
                    fallback_text = _tool_result_fallback_text(
                        redirected_result.get("redirected_to", "google_search"),
                        redirected_result,
                    ) or str(redirected_result.get("answer") or "").strip()
                    if fallback_text:
                        self.state.pending_fallback_text = fallback_text
                        self.state.pending_fallback_turn_seq = self.state.user_turn_seq
                    from google.genai.types import FunctionResponse as _FR
                    redirected_fr = _FR(name=fc.name, response=redirected_result)
                    self.ctx_queue.inject_immediate(
                        types.Content(parts=[types.Part(function_response=redirected_fr)], role="user"),
                        is_function_response=True,
                    )
                    logger.info("Redirected google_search for query intent: %s", query[:120] if (query := str(call_args.get('query', ''))) else "")
                    continue

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
            if behavior == ToolBehavior.WHEN_IDLE:
                fallback_text = _tool_result_fallback_text(fc.name, result)
                if fallback_text:
                    self.state.pending_fallback_text = fallback_text
                    self.state.pending_fallback_turn_seq = self.state.user_turn_seq
                    self._schedule_response_watchdog(
                        self.state.user_turn_seq,
                        delay_sec=3.0,
                    )

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
            elif behavior == ToolBehavior.WHEN_IDLE:
                parts.insert(
                    0,
                    types.Part(
                        text=(
                            "[TOOL RESULT READY] Use the tool result to answer the user's "
                            "current request now in one concise spoken response. Do not "
                            "greet again. If the tool failed, briefly explain that and "
                            "offer the next best step."
                        )
                    ),
                )
            content = types.Content(parts=parts, role="user")
            self.ctx_queue.inject_immediate(content, is_function_response=True)
            logger.info("Sent function response for %s (behavior=%s)", fc.name, behavior_to_text(behavior))
