# Comprehensive E2E Testing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 7 new multi-turn conversations (E-K, 50 turns) + cross-session orchestration to `scripts/run_multiturn_e2e.py`, then run the full suite against the real server with real Firestore.

**Architecture:** Extend existing `run_multiturn_e2e.py` with new conversation definitions and a cross-session test flow. Add a Firestore seeder for cross-session memory pre-population. The test runner generates TTS audio via Gemini, streams it + images over WebSocket, and validates tool calls, agent responses, LOD transitions, and memory persistence.

**Tech Stack:** Python 3.12, websockets, google-genai (TTS + Imagen), google-cloud-firestore, numpy, pytest-asyncio

---

### Task 1: Add Conversation E — Memory Lifecycle

**Files:**
- Modify: `scripts/run_multiturn_e2e.py` (after line 612, before `ALL_CONVERSATIONS`)

**Step 1: Add CONVERSATION_E definition**

Insert after CONVERSATION_D (line 610), before `ALL_CONVERSATIONS`:

```python
# Conversation E: Memory lifecycle — store, recall, forget, re-store
CONVERSATION_E = {
    "id": "conv_memory_lifecycle",
    "description": "Memory lifecycle: store entities → recall → forget → re-store → verify",
    "turns": [
        {
            "id": "E01_store_place",
            "text": "I'm at a new coffee shop called Blue Bottle on Market Street.",
            "send_image": "building_scene",
            "send_telemetry": {
                "motion_state": "stationary",
                "step_cadence": 0,
                "ambient_noise_db": 50,
                "heading": 90,
                "gps": {"latitude": 37.7903, "longitude": -122.3971, "accuracy": 5.0},
            },
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_tool": "remember_entity",
            "collect_sec": 30.0,
            "notes": "Store place entity — Blue Bottle coffee shop",
        },
        {
            "id": "E02_store_person",
            "text": "The barista's name is Sarah and she's very friendly.",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_tool": "remember_entity",
            "collect_sec": 25.0,
            "notes": "Store person entity — Sarah the barista",
        },
        {
            "id": "E03_recall_all",
            "text": "What do you remember about this place?",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 30.0,
            "notes": "Memory recall — should mention Blue Bottle and Sarah",
        },
        {
            "id": "E04_forget_person",
            "text": "Actually, forget what I just told you about Sarah.",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 25.0,
            "notes": "Forget entity — Sarah should be removed",
        },
        {
            "id": "E05_verify_forget",
            "text": "What do you remember now?",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 25.0,
            "notes": "Verify forget — Blue Bottle yes, Sarah should not appear",
        },
        {
            "id": "E06_store_preference",
            "text": "Remember that I prefer oat milk lattes here.",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_tool": "remember_entity",
            "collect_sec": 25.0,
            "notes": "Store preference — oat milk latte preference",
        },
        {
            "id": "E07_store_habit",
            "text": "Remember this is my favorite morning spot.",
            "send_image": None,
            "send_telemetry": {
                "motion_state": "stationary",
                "step_cadence": 0,
                "ambient_noise_db": 50,
                "heading": 90,
                "time_context": "morning_commute",
            },
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_tool": "remember_entity",
            "collect_sec": 25.0,
            "notes": "Store habit — morning spot preference",
        },
        {
            "id": "E08_goodbye",
            "text": "Goodbye, thanks for helping today!",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 15.0,
            "notes": "Session end — clean closure",
        },
    ],
}
```

**Step 2: Run to verify definition parses**

Run: `/opt/anaconda3/envs/sightline/bin/python -c "exec(open('scripts/run_multiturn_e2e.py').read().split('def _create_client')[0]); print(f'E turns: {len(CONVERSATION_E[\"turns\"])}')"` from the SightLine directory.
Expected: `E turns: 8`

**Step 3: Commit**

```bash
git add scripts/run_multiturn_e2e.py
git commit -m "feat(e2e): add Conversation E — memory lifecycle"
```

---

### Task 2: Add Conversation F — Cross-Session Memory Recall

**Files:**
- Modify: `scripts/run_multiturn_e2e.py`

**Step 1: Add CONVERSATION_F definition**

Insert after CONVERSATION_E:

```python
# Conversation F: Cross-session memory recall (requires Conv E to have run first)
CONVERSATION_F = {
    "id": "conv_cross_session_recall",
    "description": "Cross-session: recall memories stored in Conv E from a new WebSocket session",
    "turns": [
        {
            "id": "F01_morning_coffee",
            "text": "Good morning! I'm heading out to get coffee.",
            "send_image": "street_scene",
            "send_telemetry": {
                "motion_state": "walking",
                "step_cadence": 95,
                "ambient_noise_db": 60,
                "heading": 180,
                "gps": {"latitude": 37.7899, "longitude": -122.3975, "accuracy": 5.0},
                "time_context": "morning_commute",
            },
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 25.0,
            "notes": "Morning greeting — context should trigger memory preload",
        },
        {
            "id": "F02_recall_place",
            "text": "Do you remember my favorite coffee place?",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 30.0,
            "notes": "Cross-session recall — should return Blue Bottle from Conv E",
        },
        {
            "id": "F03_recall_preference",
            "text": "What do I usually order there?",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 30.0,
            "notes": "Cross-session recall — should return oat milk latte from E06",
        },
        {
            "id": "F04_navigate_recalled",
            "text": "Navigate me to Blue Bottle coffee.",
            "send_image": None,
            "send_telemetry": {
                "motion_state": "walking",
                "step_cadence": 95,
                "ambient_noise_db": 60,
                "heading": 180,
                "gps": {"latitude": 37.7899, "longitude": -122.3975, "accuracy": 5.0},
            },
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_tool": "navigate_to",
            "collect_sec": 55.0,
            "notes": "Navigation using remembered place — tool should fire",
        },
        {
            "id": "F05_arriving_preload",
            "text": "I'm arriving at the coffee shop now. What can you tell me about this place?",
            "send_image": "building_scene",
            "send_telemetry": {
                "motion_state": "stationary",
                "step_cadence": 0,
                "ambient_noise_db": 55,
                "heading": 90,
                "gps": {"latitude": 37.7903, "longitude": -122.3971, "accuracy": 5.0},
            },
            "send_gesture": None,
            "send_video_frames": 3,
            "expect_agent_response": True,
            "collect_sec": 35.0,
            "notes": "Context-aware preload — should load Blue Bottle memories",
        },
        {
            "id": "F06_new_memory",
            "text": "Remember that today they have a new Ethiopian blend on special.",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_tool": "remember_entity",
            "collect_sec": 25.0,
            "notes": "Add new memory to existing place entity",
        },
        {
            "id": "F07_full_history",
            "text": "What's the full history you know about my visits here?",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 30.0,
            "notes": "Full memory retrieval — should include E + F06 memories",
        },
        {
            "id": "F08_goodbye",
            "text": "Thanks, see you tomorrow!",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 15.0,
            "notes": "Session closure",
        },
    ],
}
```

**Step 2: Commit**

```bash
git add scripts/run_multiturn_e2e.py
git commit -m "feat(e2e): add Conversation F — cross-session memory recall"
```

---

### Task 3: Add Conversation G — Accessibility & Location Tools

**Files:**
- Modify: `scripts/run_multiturn_e2e.py`

**Step 1: Add CONVERSATION_G definition**

```python
# Conversation G: Accessibility tools, plus codes, address validation, reverse geocode
CONVERSATION_G = {
    "id": "conv_accessibility_location",
    "description": "Accessibility features, plus codes, address validation, reverse geocode",
    "turns": [
        {
            "id": "G01_accessibility",
            "text": "Are there any accessible features near me? Like wheelchair ramps or tactile paving?",
            "send_image": "street_scene",
            "send_telemetry": {
                "motion_state": "stationary",
                "step_cadence": 0,
                "ambient_noise_db": 55,
                "heading": 180,
                "gps": {"latitude": 40.7580, "longitude": -73.9855, "accuracy": 5.0},
            },
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_tool": "get_accessibility_info",
            "collect_sec": 40.0,
            "notes": "Accessibility query — should trigger get_accessibility_info",
        },
        {
            "id": "G02_reverse_geocode",
            "text": "What's my exact address right now?",
            "send_image": None,
            "send_telemetry": {
                "motion_state": "stationary",
                "step_cadence": 0,
                "ambient_noise_db": 55,
                "heading": 180,
                "gps": {"latitude": 40.7580, "longitude": -73.9855, "accuracy": 5.0},
            },
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_tool": "reverse_geocode",
            "collect_sec": 30.0,
            "notes": "Reverse geocode — should return street address from GPS",
        },
        {
            "id": "G03_to_plus_code",
            "text": "What's my location code that I can share with someone?",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_tool": "convert_to_plus_code",
            "collect_sec": 25.0,
            "notes": "Convert GPS to Plus Code for sharing",
        },
        {
            "id": "G04_from_plus_code",
            "text": "My friend says they're at 87G8Q2JM plus VW. Where is that?",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_tool": "resolve_plus_code",
            "collect_sec": 25.0,
            "notes": "Resolve Plus Code to address",
        },
        {
            "id": "G05_validate_address",
            "text": "I need to go to one two three West Forty Second Street in New York.",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 30.0,
            "notes": "Address validation — speech-to-text correction",
        },
        {
            "id": "G06_accessibility_dest",
            "text": "Is that area wheelchair accessible?",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 30.0,
            "notes": "Accessibility info for destination area",
        },
        {
            "id": "G07_nearby_accessible",
            "text": "Find me accessible restaurants near there.",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_tool": "nearby_search",
            "collect_sec": 35.0,
            "notes": "Nearby search with accessibility context",
        },
        {
            "id": "G08_goodbye",
            "text": "Thanks, that's helpful!",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 15.0,
            "notes": "Clean end",
        },
    ],
}
```

**Step 2: Commit**

```bash
git add scripts/run_multiturn_e2e.py
git commit -m "feat(e2e): add Conversation G — accessibility & location tools"
```

---

### Task 4: Add Conversation H — LOD Transitions & Commute

**Files:**
- Modify: `scripts/run_multiturn_e2e.py`

**Step 1: Add CONVERSATION_H definition**

```python
# Conversation H: LOD transitions — commute simulation with all motion states
CONVERSATION_H = {
    "id": "conv_lod_transitions_commute",
    "description": "Morning commute: walking → running → vehicle → walking (noise) → gesture control → space transition",
    "turns": [
        {
            "id": "H01_morning_leave",
            "text": "Good morning, I'm leaving home for work.",
            "send_image": "street_scene",
            "send_telemetry": {
                "motion_state": "walking",
                "step_cadence": 100,
                "ambient_noise_db": 50,
                "heading": 0,
                "gps": {"latitude": 40.7484, "longitude": -73.9856, "accuracy": 5.0},
                "time_context": "morning_commute",
            },
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 25.0,
            "notes": "Morning commute start — LOD adjusted for morning_commute time",
        },
        {
            "id": "H02_start_running",
            "text": "I'm going to start running to catch the bus!",
            "send_image": "street_scene",
            "send_telemetry": {
                "motion_state": "running",
                "step_cadence": 155,
                "ambient_noise_db": 75,
                "heading": 45,
                "gps": {"latitude": 40.7486, "longitude": -73.9852, "accuracy": 8.0},
            },
            "send_gesture": None,
            "send_video_frames": 5,
            "expect_agent_response": True,
            "collect_sec": 25.0,
            "notes": "Running mode — LOD should drop to 1 (safety priority)",
        },
        {
            "id": "H03_on_bus",
            "text": "I made it on the bus, I can relax now.",
            "send_image": "building_scene",
            "send_telemetry": {
                "motion_state": "in_vehicle",
                "step_cadence": 0,
                "ambient_noise_db": 60,
                "heading": 90,
                "gps": {"latitude": 40.7500, "longitude": -73.9800, "accuracy": 15.0},
            },
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 25.0,
            "notes": "Vehicle mode — LOD should go to 3 (passenger, low urgency)",
        },
        {
            "id": "H04_lod_down_gesture",
            "text": "Less detail please.",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": "lod_down",
            "expect_agent_response": True,
            "collect_sec": 20.0,
            "notes": "Gesture: lod_down — LOD decreases by 1",
        },
        {
            "id": "H05_exit_loud",
            "text": "I'm getting off the bus now, it's very loud out here.",
            "send_image": "street_scene",
            "send_telemetry": {
                "motion_state": "walking",
                "step_cadence": 90,
                "ambient_noise_db": 82,
                "heading": 270,
                "gps": {"latitude": 40.7520, "longitude": -73.9780, "accuracy": 5.0},
            },
            "send_gesture": None,
            "send_video_frames": 3,
            "expect_agent_response": True,
            "collect_sec": 25.0,
            "notes": "High noise (>80dB) — LOD capped at 1",
        },
        {
            "id": "H06_lod_up_gesture",
            "text": "More detail please, I want to know what's around me.",
            "send_image": "park_scene",
            "send_telemetry": {
                "motion_state": "walking",
                "step_cadence": 80,
                "ambient_noise_db": 50,
                "heading": 180,
                "gps": {"latitude": 40.7525, "longitude": -73.9775, "accuracy": 5.0},
            },
            "send_gesture": "lod_up",
            "send_video_frames": 3,
            "expect_agent_response": True,
            "collect_sec": 30.0,
            "notes": "Gesture: lod_up + quieter — LOD increases by 1",
        },
        {
            "id": "H07_enter_building",
            "text": "I just walked into a building I've never been to before.",
            "send_image": "building_scene",
            "send_telemetry": {
                "motion_state": "walking",
                "step_cadence": 60,
                "ambient_noise_db": 40,
                "heading": 0,
                "gps": {"latitude": 40.7530, "longitude": -73.9770, "accuracy": 10.0},
            },
            "send_gesture": None,
            "send_video_frames": 3,
            "expect_agent_response": True,
            "collect_sec": 30.0,
            "notes": "Space transition (outdoor→indoor) + unfamiliar — LOD boost expected",
        },
        {
            "id": "H08_settled_desk",
            "text": "I'm settled at my desk now. Thanks for the help today.",
            "send_image": None,
            "send_telemetry": {
                "motion_state": "stationary",
                "step_cadence": 0,
                "ambient_noise_db": 35,
                "heading": 90,
                "time_context": "work_hours",
            },
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 20.0,
            "notes": "Stationary + quiet + work hours — LOD 3",
        },
    ],
}
```

**Step 2: Commit**

```bash
git add scripts/run_multiturn_e2e.py
git commit -m "feat(e2e): add Conversation H — LOD transitions & commute"
```

---

### Task 5: Add Conversation I — Safety & Emergency

**Files:**
- Modify: `scripts/run_multiturn_e2e.py`

**Step 1: Add CONVERSATION_I definition**

```python
# Conversation I: Safety & emergency — weather, depth, heart rate, watch stability
CONVERSATION_I = {
    "id": "conv_safety_emergency",
    "description": "Safety: rain + depth sensing + elevated HR + watch instability + construction + recovery",
    "turns": [
        {
            "id": "I01_rain_safety",
            "text": "I'm walking in heavy rain, help me stay safe.",
            "send_image": "hazard_scene",
            "send_telemetry": {
                "motion_state": "walking",
                "step_cadence": 80,
                "ambient_noise_db": 72,
                "heading": 90,
                "gps": {"latitude": 40.7580, "longitude": -73.9855, "accuracy": 8.0},
                "weather_condition": "rain",
                "weather_visibility": 2000,
                "weather_precipitation": "rain",
            },
            "send_gesture": None,
            "send_video_frames": 4,
            "expect_agent_response": True,
            "collect_sec": 30.0,
            "notes": "Weather context (rain, low visibility) affects safety guidance",
        },
        {
            "id": "I02_depth_obstacle",
            "text": "Is there anything dangerous ahead of me?",
            "send_image": "hazard_scene",
            "send_telemetry": {
                "motion_state": "walking",
                "step_cadence": 75,
                "ambient_noise_db": 70,
                "heading": 90,
                "depth_center": 3.0,
                "depth_min": 1.2,
                "depth_min_region": "left",
            },
            "send_gesture": None,
            "send_video_frames": 4,
            "expect_agent_response": True,
            "collect_sec": 30.0,
            "notes": "Depth data — close obstacle at 1.2m on left side",
        },
        {
            "id": "I03_hr_unstable",
            "text": "I'm feeling a bit dizzy. My watch says my heart rate is high.",
            "send_image": None,
            "send_telemetry": {
                "motion_state": "walking",
                "step_cadence": 70,
                "ambient_noise_db": 65,
                "heading": 90,
                "heart_rate": 125,
                "watch_stability_score": 0.3,
                "watch_pitch": 15.0,
                "watch_roll": 20.0,
                "watch_yaw": 5.0,
                "sp_o2": 95.0,
                "device_type": "phone_and_watch",
            },
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 25.0,
            "notes": "HR elevated + low stability score — safety concern mode",
        },
        {
            "id": "I04_read_warning_sign",
            "text": "What's that sign say? Is it a warning?",
            "send_image": "text_sign",
            "send_telemetry": {
                "motion_state": "stationary",
                "step_cadence": 0,
                "ambient_noise_db": 65,
                "heading": 90,
            },
            "send_gesture": None,
            "send_video_frames": 2,
            "expect_agent_response": True,
            "expect_tool": "extract_text_from_camera",
            "collect_sec": 30.0,
            "notes": "OCR for safety sign detection",
        },
        {
            "id": "I05_extreme_noise",
            "text": "I hear very loud construction noise, describe everything carefully.",
            "send_image": "hazard_scene",
            "send_telemetry": {
                "motion_state": "stationary",
                "step_cadence": 0,
                "ambient_noise_db": 88,
                "heading": 0,
                "watch_noise_exposure": 85.0,
            },
            "send_gesture": None,
            "send_video_frames": 3,
            "expect_agent_response": True,
            "collect_sec": 30.0,
            "notes": "Extreme noise (>85dB) — LOD forced to 1, safety only",
        },
        {
            "id": "I06_recovery",
            "text": "The area seems clear now, I feel better. Thanks for watching out.",
            "send_image": "park_scene",
            "send_telemetry": {
                "motion_state": "walking",
                "step_cadence": 90,
                "ambient_noise_db": 48,
                "heading": 180,
                "heart_rate": 80,
                "watch_stability_score": 0.9,
            },
            "send_gesture": None,
            "send_video_frames": 3,
            "expect_agent_response": True,
            "collect_sec": 25.0,
            "notes": "Recovery — HR normal, stability good, noise low → LOD can increase",
        },
    ],
}
```

**Step 2: Commit**

```bash
git add scripts/run_multiturn_e2e.py
git commit -m "feat(e2e): add Conversation I — safety & emergency scenarios"
```

---

### Task 6: Add Conversation J — Destination Preview & Advanced Navigation

**Files:**
- Modify: `scripts/run_multiturn_e2e.py`

**Step 1: Add CONVERSATION_J definition**

```python
# Conversation J: Destination preview, walking directions, maps_query, slope warnings
CONVERSATION_J = {
    "id": "conv_advanced_navigation",
    "description": "Advanced nav: maps_query → preview_destination → walking_directions → slopes → arrival",
    "turns": [
        {
            "id": "J01_maps_query",
            "text": "What's a good place for breakfast nearby?",
            "send_image": None,
            "send_telemetry": {
                "motion_state": "stationary",
                "step_cadence": 0,
                "ambient_noise_db": 50,
                "heading": 0,
                "gps": {"latitude": 40.7580, "longitude": -73.9855, "accuracy": 5.0},
            },
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 40.0,
            "notes": "Conversational place query — may trigger maps_query or nearby_search",
        },
        {
            "id": "J02_preview_destination",
            "text": "What does the entrance look like before I go there?",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_tool": "preview_destination",
            "collect_sec": 35.0,
            "notes": "Destination preview — Street View description before arrival",
        },
        {
            "id": "J03_walking_directions",
            "text": "Give me step by step walking directions there.",
            "send_image": None,
            "send_telemetry": {
                "motion_state": "stationary",
                "step_cadence": 0,
                "ambient_noise_db": 50,
                "heading": 0,
                "gps": {"latitude": 40.7580, "longitude": -73.9855, "accuracy": 5.0},
            },
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_tool": "get_walking_directions",
            "collect_sec": 45.0,
            "notes": "Turn-by-turn walking directions with clock positions",
        },
        {
            "id": "J04_slope_question",
            "text": "Are there any steep hills on this route?",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 25.0,
            "notes": "Slope/elevation concern — agent should reference any grade warnings",
        },
        {
            "id": "J05_walking_vision",
            "text": "I'm walking now, what's around me?",
            "send_image": "street_scene",
            "send_telemetry": {
                "motion_state": "walking",
                "step_cadence": 95,
                "ambient_noise_db": 60,
                "heading": 45,
                "gps": {"latitude": 40.7582, "longitude": -73.9850, "accuracy": 5.0},
            },
            "send_gesture": None,
            "send_video_frames": 4,
            "expect_agent_response": True,
            "collect_sec": 30.0,
            "notes": "Vision during navigation — context awareness",
        },
        {
            "id": "J06_arrival",
            "text": "I think I've arrived. Is this the right place?",
            "send_image": "building_scene",
            "send_telemetry": {
                "motion_state": "stationary",
                "step_cadence": 0,
                "ambient_noise_db": 55,
                "heading": 90,
            },
            "send_gesture": None,
            "send_video_frames": 3,
            "expect_agent_response": True,
            "collect_sec": 30.0,
            "notes": "Arrival confirmation with vision",
        },
    ],
}
```

**Step 2: Commit**

```bash
git add scripts/run_multiturn_e2e.py
git commit -m "feat(e2e): add Conversation J — destination preview & advanced navigation"
```

---

### Task 7: Add Conversation K — Narrative Resume & Familiarity

**Files:**
- Modify: `scripts/run_multiturn_e2e.py`

**Step 1: Add CONVERSATION_K definition**

```python
# Conversation K: Narrative resume (LOD downgrade/upgrade) + familiarity scoring
CONVERSATION_K = {
    "id": "conv_narrative_familiarity",
    "description": "Narrative snapshot: menu reading interrupted by noise → resume + familiarity scoring",
    "turns": [
        {
            "id": "K01_read_menu",
            "text": "Read this menu for me please, start from the top.",
            "send_image": "text_sign",
            "send_telemetry": {
                "motion_state": "stationary",
                "step_cadence": 0,
                "ambient_noise_db": 42,
                "heading": 0,
            },
            "send_gesture": "force_lod_3",
            "send_video_frames": 3,
            "expect_agent_response": True,
            "expect_tool": "extract_text_from_camera",
            "collect_sec": 35.0,
            "notes": "LOD3 menu reading — full detail, sets up narrative snapshot",
        },
        {
            "id": "K02_noise_spike",
            "text": "Oh no it suddenly got very loud in here!",
            "send_image": None,
            "send_telemetry": {
                "motion_state": "stationary",
                "step_cadence": 0,
                "ambient_noise_db": 85,
                "heading": 0,
            },
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 20.0,
            "notes": "Noise spike (85dB) — LOD drops, narrative snapshot should be saved",
        },
        {
            "id": "K03_noise_clears",
            "text": "OK it's quiet again. Can you continue reading the menu where you left off?",
            "send_image": "text_sign",
            "send_telemetry": {
                "motion_state": "stationary",
                "step_cadence": 0,
                "ambient_noise_db": 42,
                "heading": 0,
            },
            "send_gesture": "force_lod_3",
            "send_video_frames": 2,
            "expect_agent_response": True,
            "collect_sec": 35.0,
            "notes": "Noise clears — LOD restores, narrative snapshot should resume",
        },
        {
            "id": "K04_store_habit",
            "text": "I come here every morning for breakfast. Remember that.",
            "send_image": None,
            "send_telemetry": {
                "motion_state": "stationary",
                "step_cadence": 0,
                "ambient_noise_db": 45,
                "time_context": "morning_commute",
            },
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_tool": "remember_entity",
            "collect_sec": 25.0,
            "notes": "Habit + familiarity store — morning breakfast habit",
        },
        {
            "id": "K05_detailed_scene",
            "text": "Describe everything around me in detail.",
            "send_image": "park_scene",
            "send_telemetry": {
                "motion_state": "stationary",
                "step_cadence": 0,
                "ambient_noise_db": 40,
                "heading": 270,
            },
            "send_gesture": "force_lod_3",
            "send_video_frames": 4,
            "expect_agent_response": True,
            "collect_sec": 35.0,
            "notes": "LOD3 full description — tests maximum detail output",
        },
        {
            "id": "K06_goodbye",
            "text": "Thanks, see you tomorrow morning!",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 15.0,
            "notes": "Session end — reinforces morning habit",
        },
    ],
}
```

**Step 2: Commit**

```bash
git add scripts/run_multiturn_e2e.py
git commit -m "feat(e2e): add Conversation K — narrative resume & familiarity"
```

---

### Task 8: Update ALL_CONVERSATIONS and Cross-Session Orchestration

**Files:**
- Modify: `scripts/run_multiturn_e2e.py` (lines ~612, ~1337, ~1373-1380, ~1271-1467)

**Step 1: Update ALL_CONVERSATIONS list**

Replace the existing `ALL_CONVERSATIONS` line:

```python
# Session 1 conversations (single WebSocket, stores memories)
SESSION_1_CONVERSATIONS = [
    CONVERSATION_A, CONVERSATION_B, CONVERSATION_C, CONVERSATION_D,
    CONVERSATION_E, CONVERSATION_G, CONVERSATION_H, CONVERSATION_I,
    CONVERSATION_J, CONVERSATION_K,
]

# Session 2 conversations (NEW WebSocket, recalls Session 1 memories)
SESSION_2_CONVERSATIONS = [CONVERSATION_F]

ALL_CONVERSATIONS = SESSION_1_CONVERSATIONS + SESSION_2_CONVERSATIONS
```

**Step 2: Update required_chain_tools**

Expand the tool chain coverage set (around line 1373):

```python
required_chain_tools = {
    "analyze_scene",
    "extract_text_from_camera",
    "google_search",
    "navigate_to",
    "remember_entity",
    "nearby_search",
    "get_accessibility_info",
    "reverse_geocode",
    "convert_to_plus_code",
    "resolve_plus_code",
    "preview_destination",
    "get_walking_directions",
}
```

**Step 3: Update async_main for cross-session flow**

Replace the conversation running loop in `async_main` (around line 1337) with cross-session logic:

```python
    # Step 3: Run Session 1 conversations (single WS per conversation)
    log.info("--- Session 1: Running %d conversations ---", len(SESSION_1_CONVERSATIONS))
    results: list[ConversationResult] = []

    for conv in SESSION_1_CONVERSATIONS:
        result = await run_conversation(
            ws_base_url=args.server,
            conversation=conv,
            pcm_cache=pcm_cache,
            image_dir=image_dir,
            timeout_mult=args.timeout,
            ws_ping_interval=ws_ping_interval,
            ws_ping_timeout=ws_ping_timeout,
        )
        results.append(result)
        log.info(
            "Conversation %s: %d/%d passed, %d issues, %.1fs total",
            result.conversation_id,
            result.passed_turns,
            result.total_turns,
            len(result.issues),
            result.total_duration_sec,
        )
        await asyncio.sleep(3.0)

    # Step 4: Run Session 2 (cross-session memory recall)
    log.info("--- Session 2: Cross-session memory recall ---")
    log.info("(Conv E stored memories → Conv F should recall them in new WS session)")

    for conv in SESSION_2_CONVERSATIONS:
        result = await run_conversation(
            ws_base_url=args.server,
            conversation=conv,
            pcm_cache=pcm_cache,
            image_dir=image_dir,
            timeout_mult=args.timeout,
            ws_ping_interval=ws_ping_interval,
            ws_ping_timeout=ws_ping_timeout,
        )
        results.append(result)
        log.info(
            "[CROSS-SESSION] %s: %d/%d passed, %d issues, %.1fs total",
            result.conversation_id,
            result.passed_turns,
            result.total_turns,
            len(result.issues),
            result.total_duration_sec,
        )
        await asyncio.sleep(3.0)
```

**Step 4: Commit**

```bash
git add scripts/run_multiturn_e2e.py
git commit -m "feat(e2e): wire up Session 1/2 orchestration + expanded tool chain"
```

---

### Task 9: Seed E2E Test User Profile in Firestore

**Files:**
- Modify: `scripts/seed_firestore.py`

**Step 1: Add E2E test user to DEMO_USERS**

Add after the existing users (around line 102):

```python
    {
        "doc_id": "e2e-user",
        "data": {
            "vision_status": "totally_blind",
            "blindness_onset": "congenital",
            "onset_age": None,
            "has_guide_dog": False,
            "has_white_cane": True,
            "tts_speed": 1.5,
            "verbosity_preference": "detailed",
            "language": "en-US",
            "description_priority": "spatial",
            "color_description": False,
            "om_level": "intermediate",
            "travel_frequency": "weekly",
            "preferred_name": "TestUser",
            "created_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP,
        },
    },
```

**Step 2: Run seeder**

```bash
cd /Users/sunfl/Documents/study/glac/SightLine
/opt/anaconda3/envs/sightline/bin/python scripts/seed_firestore.py
```

Expected: `[OK] user_profiles/e2e-user`

**Step 3: Commit**

```bash
git add scripts/seed_firestore.py
git commit -m "feat(e2e): add e2e-user test profile to Firestore seeder"
```

---

### Task 10: Start Server and Run Full E2E Suite

**Step 1: Start the server in background**

```bash
cd /Users/sunfl/Documents/study/glac/SightLine
/opt/anaconda3/envs/sightline/bin/python server.py &
```

Wait for `Uvicorn running on http://0.0.0.0:8100`.

**Step 2: Run the full E2E test suite**

```bash
cd /Users/sunfl/Documents/study/glac/SightLine
/opt/anaconda3/envs/sightline/bin/python scripts/run_multiturn_e2e.py \
    --server ws://127.0.0.1:8100 \
    --timeout 1.5 \
    --ws-ping-interval 20 \
    --ws-ping-timeout 60
```

Expected output:
- TTS audio generation for 50 new turns (E01-K06)
- 11 conversations run sequentially
- Report saved to `artifacts/e2e_multiturn/report.json`

**Step 3: Analyze results**

Check the report for:
1. Pass rate per conversation
2. Tool chain coverage (12+ tools observed)
3. Cross-session memory recall success (Conv F turns)
4. LOD transition compliance (Conv H)
5. Safety context handling (Conv I)
6. Zero context tag leaks across all 88 turns

**Step 4: Commit report**

```bash
git add artifacts/e2e_multiturn/report.json
git commit -m "test(e2e): comprehensive 88-turn multi-session test results"
```

---

### Task 11: Fix Issues and Re-run

Based on the report, fix any failing turns:

1. If tools aren't being called — check orchestrator system prompt and tool descriptions
2. If memory recall fails — verify Firestore vector index is active
3. If LOD transitions don't match — check telemetry parsing in `server.py`
4. If cross-session recall fails — verify MemoryBankService user_id matches

Re-run after fixes until pass rate reaches acceptable level (target: ≥80%).

---

## Execution Notes

- **TTS rate limit**: Gemini TTS has 10 RPM limit. The script waits 7s between requests. 50 new turns × 7s = ~6 minutes for audio generation (cached after first run).
- **Total runtime**: ~15-20 minutes for all 11 conversations (each turn has 20-55s collect window).
- **Firestore costs**: Minimal — each test creates <50 documents total.
- **Server must be running**: The test connects via WebSocket to `ws://127.0.0.1:8100`.
