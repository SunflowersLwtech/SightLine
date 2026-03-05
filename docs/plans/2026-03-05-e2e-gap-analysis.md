# E2E Test Gap Analysis — Consolidated Report

> Generated 2026-03-05 by 4-agent parallel analysis team

## Executive Summary

Analysis of 3 E2E test files, 19 backend modules, and the iOS client against 35+ features reveals **significant coverage gaps**. Current E2E tests heavily cover the happy path for vision/OCR/navigation/search but miss error handling, face recognition, real-world accessibility workflows, and state management edge cases.

| Severity | Count | Key Areas |
|----------|-------|-----------|
| **P0 Critical** | 16 | 4 tools with zero coverage, 12/26 WS messages unvalidated, zero error paths tested, face pipeline untested, 3 potential bugs found |
| **P1 Important** | 20 | Behavior modes unverified, LOD gaps, stale context, real-world scenarios missing |
| **P2 Valuable** | 12 | Race conditions, resource cleanup, profile edge cases |

---

## 1. UNTESTED TOOLS (P0)

### Zero Coverage (never invoked in any E2E test)

| Tool | Behavior | Risk |
|------|----------|------|
| `identify_person` | SILENT | Core differentiator — face recognition never tested |
| `preload_memory` | SILENT | Auto-triggered at session start, never validated |
| `what_do_you_remember` | WHEN_IDLE | Memory recall never assertion-tested |
| `forget_recent_memory` | SILENT | Time-based memory deletion never tested |

### Lacking Assertions (invoked but not validated)

| Tool | Issue |
|------|-------|
| `validate_address` | Conv G05 sends speech but no `expect_tool` assertion |
| `get_location_info` | Only in `expect_not_blocked` checks, never expected to fire |
| `forget_entity` | Conv E04 tests implicitly but no `expect_tool` assertion |

---

## 2. UNVALIDATED WEBSOCKET MESSAGES (P0)

**12 of 26 downstream message types have zero validation:**

| Message Type | Category | Notes |
|-------------|----------|-------|
| `session_resumption` | Protocol | No reconnect test |
| `capability_degraded` | Error | No degradation scenario |
| `identity_update` | Face | Requires face pipeline |
| `person_identified` | Face | Requires face pipeline |
| `vision_debug` | Debug | Never checked |
| `ocr_debug` | Debug | Never checked |
| `face_debug` | Debug | Never checked |
| `face_library_reloaded` | Face | No upstream trigger |
| `face_library_cleared` | Face | No upstream trigger |
| `debug_lod` | Debug | LOD debug never validated |
| `profile_updated_ack` | Profile | No profile update test |
| `tools_manifest` | Protocol | Sent at start, completely ignored |

**5 upstream message types never sent by tests:**
`camera_failure`, `reload_face_library`, `clear_face_library`, `profile_updated`, `playback_drained`

---

## 3. ZERO ERROR/FAILURE PATH TESTING (P0)

No negative testing exists anywhere:

1. **API quota/rate limit during tool execution** — no graceful degradation test
2. **Vision/OCR timeout** — no timeout handling test
3. **Malformed telemetry** — invalid JSON, wrong types, extreme values
4. **Unknown gesture type** — unhandled gesture string
5. **Protocol violations** — double `activity_start`, `activity_end` without start
6. **Oversized payloads** — >10MB image, 0-byte audio
7. **Concurrent same-session WebSocket** — two connections, one user
8. **Tool returning exception** — `tool_execution_failed` path
9. **Hallucinated tool call** — model calls non-existent function
10. **Memory operation on non-existent entity** — forget unknown entity
11. **Malformed binary frames** — 0-byte, 1-byte, unknown magic byte
12. **WebSocket inactivity timeout** — timeout-based cleanup

---

## 4. POTENTIAL BUGS FOUND (P0)

| ID | Issue | Location | Risk |
|----|-------|----------|------|
| BUG-1 | **Ephemeral memory not queryable**: When Firestore is down, `store_memory` falls back to ephemeral cache. But `retrieve_memories` only queries Firestore vector search, not `_memories_cache`. Stored memories become invisible. | `memory/memory_bank.py` | Silent data loss |
| BUG-2 | **`remember_entity` bypasses memory budget**: `MemoryBudgetTracker` limits auto-extracted memories to 5/session, but explicit `remember_entity` tool calls never check the budget. | `memory/memory_tools.py` vs `memory/memory_extractor.py` | Design gap |
| BUG-3 | **Token budget CRITICAL with no remediation**: `TokenBudgetMonitor` logs at 85% utilization but takes NO action — no context trimming, no session refresh, no client notification. Session silently fails at 100%. | `server.py:236-274` | Silent session death |

---

## 5. TOOL BEHAVIOR MODES NOT VERIFIED (P1)

| Mode | Status | Gap |
|------|--------|-----|
| **INTERRUPT** | Field-checked only | No test proves `navigate_to` actually interrupts agent speech |
| **WHEN_IDLE** | Field-checked only | No test proves tools wait until model finishes speaking |
| **SILENT** | Never verified | No test proves `remember_entity`, `identify_person` etc. don't produce agent speech about the tool call |

---

## 6. MISSING REAL-WORLD SCENARIOS (P0-P1)

### P0 — Critical Daily Scenarios

| Scenario | What's Missing | Tools/Features Exercised |
|----------|---------------|--------------------------|
| **Intersection Crossing** | No crosswalk/traffic signal scenario | depth + `get_accessibility_info` + heading + motion |
| **Face Recognition Social** | Zero face pipeline coverage | `[FACE ID]` injection + `remember_entity` + natural speech |
| **Indoor Navigation (Store/Hospital)** | No indoor wayfinding | OCR (signs/labels) + vision + space transitions |
| **Proactive Safety (no user prompt)** | All hazard tests require user asking | Vision auto-detect → agent speaks unprompted |
| **Full Document Reading** | Only restaurant menus tested | Mail, prescriptions, price tags, business cards |

### P1 — Important Scenarios

| Scenario | What's Missing |
|----------|---------------|
| **Emergency / Fall Detection** | watch stability=0 + HR spike + SpO2 drop combined |
| **Night Walking / Low Light** | `currentTimeContext()` generates "late_night" but never tested |
| **Public Transit Full Journey** | Bus stop → GPS degradation → transfer → exit |
| **Multi-Language** | System prompt says "respond in user's language" — zero non-English tests |
| **Cooking / Kitchen Assistance** | OCR labels + memory + safety (hot surfaces) |
| **Cycling Mode** | `MotionManager` detects cycling but never tested |

---

## 7. LOD TRANSITION GAPS (P1)

### Never Tested Transitions

| Transition | Description |
|-----------|-------------|
| stationary → walking | No test starts from stationary |
| LOD hysteresis/debounce | Rapid alternating telemetry to verify no oscillation |
| `force_lod_1` / `force_lod_2` gestures | Only `force_lod_3` is tested standalone |
| LOD during active navigation | Does LOD behavior change when `navigate_to` is active? |
| GPS accuracy degradation → LOD | Degrading accuracy effect on LOD |
| Time context → LOD | morning_commute/work_hours sent but LOD not validated |

---

## 8. UNTESTED BACKEND CODE PATHS (P1-P2)

### Error Handling

| Path | Location | Description |
|------|----------|-------------|
| Vision 503 cooldown | `vision_agent.py:258-321` | 30s suppression on Gemini 503, never tested |
| OCR generic failure | `ocr_agent.py:198-204` | Returns empty result silently |
| Face runtime unavailable | `face_agent.py:23-26` | Graceful degradation flag, one-time warning |
| Firestore client failure | `session_manager.py:215-226` | Profile fallback to defaults |
| Memory bank Firestore failure | `memory_bank.py:63-80` | Ephemeral cache fallback (but retrieval broken — BUG-1) |
| Session service 3-tier fallback | `session_manager.py:41-122` | VertexAi → Database → InMemory |
| Tool execution exception | `server.py:1336-1345` | Returns `tool_execution_failed` |

### Race Conditions

| Path | Location | Risk |
|------|----------|------|
| OCR has no lock | `server.py` | Multiple `_run_ocr_analysis` tasks can run concurrently |
| Vision tasks not cancelled on disconnect | `server.py:3613-3658` | Fire-and-forget tasks access stale session state |
| Face library refresh during recognition | `server.py:2115-2123` | Shared list mutated during read |
| ModelState concurrent transitions | `server.py:294-310` | Multiple handlers call `_transition_to()` |
| Context injection queue flush race | `server.py:324-329` | `check_max_age()` vs `_deferred_flush_callback` |

### Resource Cleanup

| Path | Location | Issue |
|------|----------|-------|
| Fire-and-forget tasks on session end | `server.py:3613-3658` | Not cancelled, may access stale state |
| Firestore connections never closed | `face_tools.py`, `session_manager.py` | Connection accumulation |
| `_latest_frames` dict on abnormal disconnect | `tools/ocr_tool.py` | No eviction policy |
| genai.Client singletons | `vision_agent.py`, `ocr_agent.py`, `search.py` | Stale if API key rotates |

---

## 9. TOOL INTERACTION EDGE CASES (P1-P2)

### Tool Chaining

| Case | Risk |
|------|------|
| Search → Navigate chain timing | `navigate_to` blocked by intent gate if original request scrolled past 3-utterance window |
| `validate_address` → `navigate_to` | No server-side enforcement of confirmation before navigation |
| `preview_destination` → `navigate_to` | Nested event loop risk (ThreadPoolExecutor + asyncio.run) |
| Reverse geocode → nearby_search GPS consistency | Coordinates may shift between sequential calls |

### State Management

| Case | Risk |
|------|------|
| Entity attribute merge inconsistency | Different keys (`work` vs `workplace`) both persist |
| Forget entity misses entity_ref-linked memories | Semantic search may miss or over-match |
| Face library mid-session registration | 60s refresh delay before new face is recognized |
| Language change mid-session | Delayed until next LOD transition |
| `_latest_frames` leak on crash | Module-global dict, no eviction |

### Rapid Sequences

| Case | Risk |
|------|------|
| 3 rapid LOD gestures | Intermediate injections may confuse model |
| Camera on/off/on toggle | All 3 messages queued, model gets conflicting state |
| Pause → immediate resume | ModelState IDLE→GENERATING oscillation |
| Repeat gesture with empty history | Model generates confused output |

---

## 10. CONTEXT TAG LEAK COVERAGE (P1)

### Currently Checked (run_multiturn_e2e.py only)
```
[CONTEXT UPDATE, [SILENT, [DO NOT SPEAK, [TELEMETRY,
[LOD UPDATE, [VISION ANALYSIS, <<<INTERNAL_CONTEXT>>>,
<<<SILENT_SENSOR_DATA>>>, <<<END_
```

### Missing Checks
- Raw JSON in agent speech (`{"type":`, `"tool":`, `"status":`)
- Function call syntax (`navigate_to(`, `google_search(`)
- System prompt fragments (`You are a vision-first`, `LOD Level`)
- Raw GPS coordinates echoed verbatim
- Only 1 of 3 test files checks for leaks

---

## 11. RECOMMENDED NEW E2E CONVERSATIONS (Priority Order)

### Tier 1 — Must Have

| Conv | Name | Turns | Primary Gaps Covered |
|------|------|-------|---------------------|
| L | **Face Recognition Social** | 8 | identify_person, [FACE ID], identity_update, person_identified, remember_entity chaining |
| M | **Intersection Crossing** | 6 | get_accessibility_info + depth + heading + proactive safety |
| N | **Error Resilience** | 8 | Tool failures, malformed input, capability_degraded, hallucinated tools |
| O | **Proactive Safety (No Prompt)** | 6 | Auto-hazard detection, INTERRUPT mode verification, context tag leak under stress |

### Tier 2 — Should Have

| Conv | Name | Turns | Primary Gaps Covered |
|------|------|-------|---------------------|
| P | **Indoor Store Navigation** | 8 | OCR labels/prices, space transitions, indoor GPS degradation |
| Q | **Session Recovery** | 6 | session_resumption, resume_handle, reconnect mid-tool-execution |
| R | **Full Tool Chain** | 8 | validate_address→navigate_to→get_walking_directions, preview_destination chain |
| S | **Rapid Gesture Stress** | 6 | LOD oscillation, camera toggle flood, repeat with empty history |

### Tier 3 — Nice to Have

| Conv | Name | Turns | Primary Gaps Covered |
|------|------|-------|---------------------|
| T | **Night / Weather Extremes** | 6 | Time context, severe weather variants, low-light vision |
| U | **Emergency Fall** | 6 | SpO2 + HR spike + stability=0, urgent response |
| V | **Long Session Stability** | 12 | Token budget monitoring, memory budget, transcript overflow |

---

## 12. METRICS AFTER CLOSING GAPS

| Metric | Current | Target |
|--------|---------|--------|
| Tools with E2E coverage | 12/19 (63%) | 19/19 (100%) |
| WS message types validated | 14/26 (54%) | 22/26 (85%) |
| Error paths tested | 0/12 | 8/12 (67%) |
| Tool behavior modes verified | 0/3 | 3/3 (100%) |
| Real-world scenarios | 11 conv | 22 conv |
| LOD transitions tested | 8/14 | 12/14 |
| Face pipeline coverage | 0% | 80%+ |
