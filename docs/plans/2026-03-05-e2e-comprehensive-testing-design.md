# Comprehensive E2E Testing Design

**Date**: 2026-03-05
**Goal**: Full backend validation via multi-turn, stateful, cross-session E2E tests using real video/audio against real Firestore.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Test Orchestrator                             │
│  1. Seed Firestore (past memories for cross-session test)       │
│  2. Generate TTS audio + Imagen images (cached)                 │
│  3. Run Session 1 (Conv A-E, G-K) — stores memories/entities   │
│  4. Run Session 2 (Conv F) — recalls Session 1 memories         │
│  5. Validate cross-session persistence                          │
│  6. Cleanup test data                                           │
│  7. Generate report                                             │
└──────────────────────────┬──────────────────────────────────────┘
                           │ WebSocket (binary audio + image frames)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SightLine Server (server.py:8100)             │
│  Live API ↔ Vision ↔ OCR ↔ Face ↔ Tools ↔ Memory ↔ Firestore  │
└─────────────────────────────────────────────────────────────────┘
```

## New Conversations (E through K)

### Conversation E: Memory Lifecycle (8 turns)

| Turn | User Says | Image | Telemetry | Expected Tool | Validation |
|------|-----------|-------|-----------|---------------|------------|
| E01 | "Hi, I'm at a new coffee shop called Blue Bottle on Market Street." | building_scene | stationary, GPS: SF | `remember_entity` | Entity stored with name+type |
| E02 | "The barista's name is Sarah and she's very friendly." | None | None | `remember_entity` | Person entity stored |
| E03 | "What do you remember about this place?" | None | None | `what_do_you_remember` | Returns Blue Bottle + Sarah |
| E04 | "Actually, forget what I just told you about Sarah." | None | None | `forget_entity` | Sarah entity deleted |
| E05 | "What do you remember now?" | None | None | `what_do_you_remember` | Blue Bottle yes, Sarah no |
| E06 | "Remember that I prefer oat milk lattes here." | None | None | `remember_entity` | Preference stored |
| E07 | "Remember this is my favorite morning spot." | None | stationary, time: morning | `remember_entity` | Place preference stored |
| E08 | "Goodbye, thanks for helping!" | None | None | — | Session ends cleanly |

### Conversation F: Cross-Session Memory Recall (8 turns)

**Pre-condition**: Conv E already stored memories. F runs in a NEW WebSocket session (same user_id).

| Turn | User Says | Image | Telemetry | Expected Tool | Validation |
|------|-----------|-------|-----------|---------------|------------|
| F01 | "Good morning! I'm heading to get coffee." | street_scene | walking, GPS: SF, time: morning | `preload_memory` | Preloads morning coffee memories from E |
| F02 | "Do you remember my favorite coffee place?" | None | None | `what_do_you_remember` | Returns "Blue Bottle on Market Street" from Conv E |
| F03 | "What do I usually order there?" | None | None | `what_do_you_remember` | Returns "oat milk latte" from E06 |
| F04 | "Navigate me to Blue Bottle." | None | walking, GPS: SF | `navigate_to` | Navigation uses remembered location |
| F05 | "I'm arriving now. What can you tell me about this place?" | building_scene | stationary, GPS: near Blue Bottle | `preload_memory` | Context-aware: loads Blue Bottle memories specifically |
| F06 | "Remember that today they have a new Ethiopian blend." | None | None | `remember_entity` | New memory added to existing entity |
| F07 | "What's the full history of my visits here?" | None | None | `what_do_you_remember` | Returns all Blue Bottle memories (Conv E + F06) |
| F08 | "Thanks, see you tomorrow!" | None | None | — | Session closure |

### Conversation G: Accessibility & Location Tools (8 turns)

| Turn | User Says | Image | Telemetry | Expected Tool | Validation |
|------|-----------|-------|-----------|---------------|------------|
| G01 | "Are there any accessible features near me?" | street_scene | stationary, GPS: NYC | `get_accessibility_info` | Returns tactile paving, ramps, audio signals |
| G02 | "What's my exact address right now?" | None | GPS: 40.7580, -73.9855 | `reverse_geocode` | Returns street address |
| G03 | "What's my location code I can share with someone?" | None | GPS: same | `convert_to_plus_code` | Returns Plus Code |
| G04 | "My friend says they're at 87G8Q2JM+VW. Where is that?" | None | None | `resolve_plus_code` | Decodes to lat/lng + address |
| G05 | "I need to go to one-two-three West Forty-Second Street." | None | None | `validate_address` | Corrects to "123 W 42nd St" |
| G06 | "Is that area wheelchair accessible?" | None | None | `get_accessibility_info` | Accessibility info for destination |
| G07 | "Search for accessible restaurants near there." | None | None | `nearby_search` | Nearby search with context |
| G08 | "Thanks, that's helpful!" | None | None | — | Clean end |

### Conversation H: LOD Transitions & Commute Simulation (8 turns)

| Turn | User Says | Image | Telemetry | Expected Tool | Validation |
|------|-----------|-------|-----------|---------------|------------|
| H01 | "Good morning, I'm leaving home for work." | street_scene | walking, cadence:100, time:morning_commute, GPS:home | — | LOD adjusted for morning_commute |
| H02 | "I'm going to start running to catch the bus." | street_scene | running, cadence:150, noise:75 | — | LOD → 1 (running baseline) |
| H03 | "I made it on the bus." | building_scene | in_vehicle, cadence:0, noise:60 | — | LOD → 3 (vehicle mode) |
| H04 | *gesture: lod_down* "Less detail please." | None | gesture: lod_down | — | LOD decreases by 1 |
| H05 | "I'm getting off now." | street_scene | walking, cadence:90, noise:80 | — | LOD capped by noise (>80dB → LOD 1) |
| H06 | *gesture: lod_up* "More detail please." | park_scene | gesture: lod_up, walking, cadence:80, noise:55 | — | LOD increases by 1 |
| H07 | "I just entered a building I've never been to." | building_scene | stationary, cadence:0, space_transition: outdoor→indoor, familiarity:0.0 | — | LOD boost for space transition + unfamiliar |
| H08 | "OK I'm settled at my desk now." | None | stationary, cadence:0, noise:40, time:work_hours | — | LOD → 3 (stationary + quiet) |

### Conversation I: Safety & Emergency (6 turns)

| Turn | User Says | Image | Telemetry | Expected Tool | Validation |
|------|-----------|-------|-----------|---------------|------------|
| I01 | "I'm walking in heavy rain, help me stay safe." | hazard_scene | walking, cadence:80, weather:rain, visibility:2000, noise:72 | — | Weather context affects guidance |
| I02 | "Is there anything dangerous ahead?" | hazard_scene | walking, depth_center:3.0, depth_min:1.2, depth_min_region:left | — | Depth data used for obstacle warning |
| I03 | "My watch says my heart rate is elevated." | None | walking, hr:120, watch_stability:0.3 | — | HR + instability trigger safety mode |
| I04 | "What's that sign say? Is it a warning?" | text_sign | stationary | `extract_text_from_camera` | Safety sign detection mode |
| I05 | "I hear construction, describe everything carefully." | hazard_scene | stationary, noise:85 | — | LOD 1 due to extreme noise |
| I06 | "OK the area seems clear now, thanks." | park_scene | walking, cadence:90, noise:50 | — | LOD recovers from noise |

### Conversation J: Destination Preview & Advanced Navigation (6 turns)

| Turn | User Says | Image | Telemetry | Expected Tool | Validation |
|------|-----------|-------|-----------|---------------|------------|
| J01 | "What's a good place for breakfast nearby?" | None | stationary, GPS: NYC | `maps_query` | Conversational search (not structured nearby_search) |
| J02 | "What does the entrance look like before I go?" | None | None | `preview_destination` | Street View preview description |
| J03 | "Give me step-by-step walking directions there." | None | GPS: NYC | `get_walking_directions` | Turn-by-turn with clock positions |
| J04 | "Are there any steep hills on this route?" | None | None | — | Slope warnings (>8% grade) |
| J05 | "I'm walking now, what's around me?" | street_scene | walking, cadence:95, heading:45 | — | Vision with navigation context |
| J06 | "I think I've arrived." | building_scene | stationary | — | Arrival confirmation |

### Conversation K: Narrative Resume & Familiarity (6 turns)

| Turn | User Says | Image | Telemetry | Expected Tool | Validation |
|------|-----------|-------|-----------|---------------|------------|
| K01 | "Read this menu for me, start from the top." | text_sign | stationary, noise:45, LOD:3 | `extract_text_from_camera` | Full menu reading begins |
| K02 | *noise spike* | None | stationary, noise:82 | — | LOD drops to 1, narrative snapshot saved |
| K03 | *noise clears* "Continue reading the menu." | text_sign | stationary, noise:45 | — | LOD restores to 3, narrative snapshot resumed |
| K04 | "I come here every morning, you should know that." | None | time:morning_commute | `remember_entity` | Habit/familiarity stored |
| K05 | "Describe everything around me in detail." | park_scene | stationary, cadence:0 | — | LOD 3 detailed description |
| K06 | "Thanks, see you tomorrow morning!" | None | None | — | Session end |

## Cross-Session Test Flow

```
Phase 1: Seed Firestore
  → Create test user profile (e2e-test-user)
  → Pre-seed 3 memories for Conv F cross-session test

Phase 2: Session 1 (single WebSocket connection)
  → Run Conversations A through E, G through K
  → Memories stored during E are persisted to Firestore

Phase 3: Session 2 (NEW WebSocket connection, same user_id)
  → Run Conversation F
  → Verify F01-F03 recall memories from Conv E

Phase 4: Validation
  → Query Firestore directly to verify stored memories/entities
  → Compare expected vs actual memory retrieval results

Phase 5: Cleanup
  → Delete test user data from Firestore
  → Generate comprehensive report
```

## New Test Images Needed

| Scene ID | Prompt | Purpose |
|----------|--------|---------|
| existing 5 | (reuse) | street, text, hazard, park, building |

No new images needed — existing 5 scenes cover all scenarios.

## New Audio Turns Needed

50 new TTS utterances (E01-E08, F01-F08, G01-G08, H01-H08, I01-I06, J01-J06, K01-K06).

## Validation Framework

Each turn validates:
1. **Agent responded** (audio + transcript)
2. **Expected tool called** (if specified)
3. **No context tag leaks** (LEAKED_MARKERS check)
4. **No duplicate tool calls** (dedup validation)
5. **No mutex violations** (tool group mutual exclusion)
6. **LOD compliance** (if LOD change expected)
7. **Memory correctness** (if recall expected, content matches)
8. **Latency** (first agent response < 10s)

## Success Criteria

- All 88 turns produce agent responses
- Tool coverage reaches ≥85% (15+ of 18 tools called)
- Cross-session memory recall succeeds (Conv F retrieves Conv E data)
- LOD transitions follow rule-based logic
- Zero context tag leaks
- Zero mutex violations
