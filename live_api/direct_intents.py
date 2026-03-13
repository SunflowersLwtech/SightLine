"""Direct-intent helpers extracted from ``websocket_handler``.

Keeps shortcut/redirect logic in a focused module so the main handler can
stay closer to orchestration concerns.
"""

from __future__ import annotations

import logging
import re
import time

from google.genai import types

logger = logging.getLogger("sightline.server")

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


def _handler_module():
    import websocket_handler as handler_module

    return handler_module


def tool_preference_hint(text: str) -> str:
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
        hints.append("This is an external knowledge request. Prefer google_search.")
    if any(token in lowered for token in _MEMORY_STORE_HINTS):
        hints.append("This is a memory-store request. Prefer remember_entity.")
    if any(token in lowered for token in _MEMORY_RECALL_HINTS):
        hints.append("This is a memory-recall request. Prefer what_do_you_remember.")
    return "\n".join(hints)


def tool_result_fallback_text(tool_name: str, result: dict[str, object]) -> str | None:
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


class DirectIntentMixin:
    """Shortcut and redirect logic for text/navigation/location requests."""

    def _is_farewell_text(self, text: str) -> bool:
        lowered = (text or "").strip().lower()
        if not lowered:
            return False
        return any(token in lowered for token in _FAREWELL_HINTS)

    def _is_current_turn_farewell(self) -> bool:
        handler_module = _handler_module()
        recent = handler_module._recent_user_utterances(self.state.transcript_history, max_items=1)
        if not recent:
            return False
        return self._is_farewell_text(recent[0])

    def _silent_turn_fallback_text(self) -> str | None:
        handler_module = _handler_module()
        recent = handler_module._recent_user_utterances(self.state.transcript_history, max_items=1)
        if not recent:
            return "I'm having trouble responding right now. Please try again."

        text = recent[0]
        lowered = text.strip().lower()
        ephemeral = handler_module.session_manager.get_ephemeral_context(self.session_id)
        gps = getattr(ephemeral, "gps", None)
        has_gps = isinstance(getattr(gps, "lat", None), (int, float)) and isinstance(getattr(gps, "lng", None), (int, float))

        if self._is_farewell_text(text):
            return "You're welcome. Goodbye for now."
        if any(token in lowered for token in _MEMORY_STORE_HINTS):
            return "I heard that memory request, but I need you to repeat the key detail one more time."
        if any(token in lowered for token in _MEMORY_RECALL_HINTS):
            return "I'm having trouble recalling that right now. Please ask me again in a moment."
        if handler_module._has_location_query_intent(text) and not has_gps:
            return "I need your location to answer that. Tell me your city, or enable location and ask again."
        if any(token in lowered for token in ("read", "sign", "menu", "label", "text")):
            return "I couldn't read that clearly. Please pan the camera a little and try again."
        if any(token in lowered for token in (*_VISION_QUERY_HINTS, *_AREA_CONTEXT_HINTS)):
            return "I need a steadier camera view to describe that. Please hold the camera steady and ask again."
        if handler_module._has_navigation_intent(text):
            return "I'm still working on directions. Please ask me again in a moment."
        if handler_module._has_location_query_intent(text):
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
        handler_module = _handler_module()
        lowered = (question or "").strip().lower()
        if any(token in lowered for token in _AREA_CONTEXT_HINTS):
            summary_parts: list[str] = []
            ephemeral = handler_module.session_manager.get_ephemeral_context(self.session_id)
            if ephemeral.gps:
                location_result = await handler_module._dispatch_function_call(
                    "get_location_info",
                    {},
                    self.session_id,
                    self.user_id,
                )
                fallback_text = tool_result_fallback_text("get_location_info", location_result)
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
        if handler_module._has_location_query_intent(question) and search_types:
            search_result = await handler_module._dispatch_function_call(
                "nearby_search",
                {"types": search_types, "radius": 500},
                self.session_id,
                self.user_id,
            )
            fallback_text = tool_result_fallback_text("nearby_search", search_result)
            answer = fallback_text or str(search_result.get("message") or "").strip()
            return True, {
                "success": bool(search_result.get("success", True)),
                "redirected_to": "nearby_search",
                "answer": answer,
                "nearby_result": search_result,
            }

        recent_user = handler_module._recent_user_utterances(self.state.transcript_history, max_items=3)
        if not any(handler_module._has_navigation_intent(text) for text in recent_user):
            return False, {}

        search_args: dict[str, object] = {}
        if search_types:
            search_args["types"] = search_types

        search_result = await handler_module._dispatch_function_call(
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

        behavior = handler_module.resolve_tool_behavior(
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
        result = await handler_module._dispatch_function_call(
            "navigate_to",
            {"destination": destination},
            self.session_id,
            self.user_id,
        )
        await self._safe_send_json({
            "type": handler_module.MessageType.TOOL_RESULT,
            "tool": "navigate_to",
            "behavior": handler_module.behavior_to_text(behavior),
            "data": handler_module._json_safe(result),
        })
        await self._safe_send_json({
            "type": handler_module.MessageType.NAVIGATION_RESULT,
            "summary": str(result.get("destination_direction") or result.get("destination") or destination),
            "behavior": handler_module.behavior_to_text(behavior),
            "data": handler_module._json_safe(result),
        })
        fallback_text = tool_result_fallback_text("navigate_to", result)
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
        handler_module = _handler_module()
        lowered = (query or "").strip().lower()
        if not lowered:
            return False, {}

        if any(token in lowered for token in _VISION_QUERY_HINTS):
            ephemeral = handler_module.session_manager.get_ephemeral_context(self.session_id)
            summary_parts: list[str] = []
            if ephemeral.gps:
                location_result = await handler_module._dispatch_function_call(
                    "get_location_info",
                    {},
                    self.session_id,
                    self.user_id,
                )
                fallback_text = tool_result_fallback_text("get_location_info", location_result)
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

            memory_result = await handler_module._dispatch_function_call(
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
            memory_result = await handler_module._dispatch_function_call(
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
        handler_module = _handler_module()
        current_turn = self.state.user_turn_seq
        if current_turn <= 0 or self.state.direct_tool_handled_turn_seq == current_turn:
            return False
        if not handler_module._has_navigation_intent(input_text):
            return False
        ephemeral = handler_module.session_manager.get_ephemeral_context(self.session_id)
        gps = getattr(ephemeral, "gps", None)
        lat = getattr(gps, "lat", None)
        lng = getattr(gps, "lng", None)
        if not isinstance(lat, (int, float)) or not isinstance(lng, (int, float)):
            return False

        destination, search_types = self._extract_navigation_destination(input_text)
        user_speaking = self.session_ctx.current_activity_state == "user_speaking"
        behavior = handler_module.resolve_tool_behavior(
            tool_name="navigate_to",
            lod=self.session_ctx.current_lod,
            is_user_speaking=user_speaking,
        )

        result: dict[str, object]
        selected_place: dict[str, object] | None = None
        if search_types:
            await self._emit_tool_event(
                "nearby_search",
                handler_module.ToolBehavior.WHEN_IDLE,
                status="invoked",
                data={"source": "server_shortcut", "types": search_types},
            )
            search_result = await handler_module._dispatch_function_call(
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
                result = await handler_module._dispatch_function_call(
                    "navigate_to",
                    {"destination": destination},
                    self.session_id,
                    self.user_id,
                )
        elif destination:
            result = await handler_module._dispatch_function_call(
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
            "type": handler_module.MessageType.TOOL_RESULT,
            "tool": "navigate_to",
            "behavior": handler_module.behavior_to_text(behavior),
            "data": handler_module._json_safe(result),
        })
        await self._safe_send_json({
            "type": handler_module.MessageType.NAVIGATION_RESULT,
            "summary": str(result.get("destination_direction") or result.get("destination") or destination or ""),
            "behavior": handler_module.behavior_to_text(behavior),
            "data": handler_module._json_safe(result),
        })

        fallback_text = tool_result_fallback_text("navigate_to", result)
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
        self.ctx_queue._transition_to(handler_module.ModelState.IDLE)
        logger.info("Handled direct navigation shortcut for turn %d", current_turn)
        return True

    async def _maybe_handle_direct_nearby_search_intent(self, input_text: str) -> bool:
        handler_module = _handler_module()
        current_turn = self.state.user_turn_seq
        if current_turn <= 0 or self.state.direct_tool_handled_turn_seq == current_turn:
            return False
        if handler_module._has_navigation_intent(input_text):
            return False
        if not handler_module._has_location_query_intent(input_text):
            return False

        search_types = self._infer_navigation_redirect_types(input_text)
        if not search_types:
            return False

        ephemeral = handler_module.session_manager.get_ephemeral_context(self.session_id)
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

        behavior = handler_module.resolve_tool_behavior(
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
        result = await handler_module._dispatch_function_call(
            "nearby_search",
            {"types": search_types, "radius": 500},
            self.session_id,
            self.user_id,
        )
        await self._safe_send_json({
            "type": handler_module.MessageType.TOOL_RESULT,
            "tool": "nearby_search",
            "behavior": handler_module.behavior_to_text(behavior),
            "data": handler_module._json_safe(result),
        })
        fallback_text = tool_result_fallback_text("nearby_search", result)
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
