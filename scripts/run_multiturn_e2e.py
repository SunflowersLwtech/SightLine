#!/usr/bin/env python3
"""Multi-turn E2E test for SightLine.

Generates TTS audio + synthetic test images, then runs 12+ turn conversations
against the real server to validate full pipeline behavior.

Usage:
    python scripts/run_multiturn_e2e.py
    python scripts/run_multiturn_e2e.py --server ws://127.0.0.1:8100
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import fnmatch
import io
import json
import logging
import os
import re
import subprocess
import sys
import time
import uuid
import wave
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import websockets
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("multiturn_e2e")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAGIC_AUDIO = 0x01
MAGIC_IMAGE = 0x02
TTS_MODEL = "gemini-2.5-flash-preview-tts"
TARGET_SAMPLE_RATE = 16000
IMAGEN_MODEL = "imagen-4.0-fast-generate-001"

# Scene prompts for Imagen image generation
SCENE_PROMPTS = {
    "street_scene": (
        "A busy city sidewalk during daytime with pedestrians walking, storefronts visible, "
        "crosswalk ahead, and a street sign. Realistic first-person pedestrian perspective photo."
    ),
    "text_sign": (
        "A close-up photo of a restaurant menu board on a wall with clearly readable text: "
        "'Today Special: Grilled Salmon $15, Caesar Salad $9, Coffee $4'. Realistic photo."
    ),
    "hazard_scene": (
        "A wet sidewalk with a large puddle, construction barrier with orange cones, "
        "and a yellow CAUTION WET FLOOR sign. First-person perspective photo."
    ),
    "park_scene": (
        "A sunny park with a paved walking path, green grass, two benches with people sitting, "
        "a small fountain, and trees. First-person perspective daytime photo."
    ),
    "building_scene": (
        "A modern building entrance with glass doors, a wheelchair accessible ramp, "
        "and a directory sign showing floor numbers. First-person perspective photo."
    ),
}

# Markers that should never appear in agent speech (context tag leak detection)
_LEAKED_MARKERS = [
    "[CONTEXT UPDATE", "[SILENT", "[DO NOT SPEAK", "[TELEMETRY",
    "[LOD UPDATE", "[VISION ANALYSIS", "<<<INTERNAL_CONTEXT>>>",
    "<<<SILENT_SENSOR_DATA>>>", "<<<END_",
]

# Mutex tool groups — at most one tool from each group per turn
_MUTEX_TOOL_GROUPS = [
    {"nearby_search", "maps_query"},
    {"navigate_to", "get_walking_directions"},
]

# ---------------------------------------------------------------------------
# Multi-turn conversation definitions
# ---------------------------------------------------------------------------

# Conversation A: Mixed scenario — greeting, vision, OCR, navigation, memory
CONVERSATION_A = {
    "id": "conv_mixed_full",
    "description": "Full mixed conversation: greeting → vision → OCR → nav → search → memory → follow-up",
    "turns": [
        {
            "id": "A01_greeting",
            "text": "Hello, I just stepped outside. Can you help me today?",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 20.0,
            "notes": "Initial greeting, expect warm response",
        },
        {
            "id": "A02_whats_around",
            "text": "What's around me right now?",
            "send_image": "street_scene",
            "send_telemetry": {
                "motion_state": "stationary",
                "step_cadence": 0,
                "ambient_noise_db": 55,
                "heading": 180,
                "gps": {"latitude": 40.7580, "longitude": -73.9855, "accuracy": 5.0},
            },
            "send_gesture": None,
            "send_video_frames": 3,
            "expect_agent_response": True,
            "expect_not_blocked": ["get_location_info", "nearby_search"],
            "collect_sec": 30.0,
            "notes": "Vision query with image + telemetry context",
        },
        {
            "id": "A03_read_sign",
            "text": "Can you read that sign for me?",
            "send_image": "text_sign",
            "send_telemetry": None,
            "send_gesture": None,
            "send_video_frames": 2,
            "expect_agent_response": True,
            "expect_tool": "extract_text_from_camera",
            "collect_sec": 30.0,
            "notes": "Explicit OCR request — should trigger extract_text_from_camera",
        },
        {
            "id": "A04_navigate",
            "text": "Navigate me to the nearest pharmacy please.",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_tool": "navigate_to",
            "collect_sec": 55.0,
            "notes": "Navigation request — should trigger navigate_to tool",
        },
        {
            "id": "A05_search",
            "text": "What's the weather like in New York today?",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_tool": "google_search",
            "collect_sec": 45.0,
            "notes": "Search request — should trigger google_search tool",
        },
        {
            "id": "A06_remember",
            "text": "Please remember that the pharmacy is called Walgreens on 5th Avenue.",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_tool": "remember_entity",
            "collect_sec": 25.0,
            "notes": "Memory store request",
        },
        {
            "id": "A07_whats_ahead",
            "text": "What's ahead of me now?",
            "send_image": "street_scene",
            "send_telemetry": {
                "motion_state": "walking",
                "step_cadence": 100,
                "ambient_noise_db": 65,
                "heading": 90,
                "gps": {"latitude": 40.7582, "longitude": -73.9850, "accuracy": 5.0},
            },
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 30.0,
            "notes": "Second vision query — different telemetry context, check context awareness",
        },
        {
            "id": "A08_danger_check",
            "text": "Is there any danger ahead?",
            "send_image": "hazard_scene",
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 30.0,
            "notes": "Safety-focused query with hazard image",
        },
        {
            "id": "A09_recall_memory",
            "text": "What pharmacy did I mention earlier?",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 25.0,
            "notes": "Memory recall — should remember Walgreens on 5th Avenue",
        },
        {
            "id": "A10_describe_detail",
            "text": "Describe everything you see in detail.",
            "send_image": "park_scene",
            "send_telemetry": None,
            "send_gesture": "force_lod_3",
            "expect_agent_response": True,
            "collect_sec": 35.0,
            "notes": "LOD3 detailed description request with gesture",
        },
        {
            "id": "A11_how_many_people",
            "text": "How many people can you see?",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 20.0,
            "notes": "Follow-up question about previous image — tests context retention",
        },
        {
            "id": "A12_goodbye",
            "text": "Thank you, that's all for now. Goodbye.",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 15.0,
            "notes": "Session closure — polite ending",
        },
    ],
}

# Conversation B: Rapid context switching stress test
CONVERSATION_B = {
    "id": "conv_rapid_context",
    "description": "Rapid context switching: alternating vision/OCR/nav/search to test model coherence",
    "turns": [
        {
            "id": "B01_greeting",
            "text": "Hi, I need your help navigating the city.",
            "send_image": None,
            "send_telemetry": {
                "motion_state": "walking",
                "step_cadence": 110,
                "ambient_noise_db": 70,
                "heading": 45,
                "gps": {"latitude": 37.7749, "longitude": -122.4194, "accuracy": 8.0},
            },
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 20.0,
            "notes": "Greeting with urban walking context",
        },
        {
            "id": "B02_read_menu",
            "text": "Read the menu for me please.",
            "send_image": "text_sign",
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_tool": "extract_text_from_camera",
            "collect_sec": 30.0,
            "notes": "OCR request",
        },
        {
            "id": "B03_search_restaurant",
            "text": "Search for the best Italian restaurant nearby.",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_any_tool": ["google_search", "maps_query", "nearby_search"],
            "collect_sec": 45.0,
            "notes": "Immediate switch to search",
        },
        {
            "id": "B04_whats_this",
            "text": "What is this building?",
            "send_image": "building_scene",
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 30.0,
            "notes": "Vision query — context switch from search to vision",
        },
        {
            "id": "B05_navigate",
            "text": "Take me to Union Square.",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_tool": "navigate_to",
            "collect_sec": 55.0,
            "notes": "Navigation — context switch from vision to nav",
        },
        {
            "id": "B06_danger",
            "text": "Wait, is that area safe?",
            "send_image": "hazard_scene",
            "send_telemetry": {
                "motion_state": "running",
                "step_cadence": 150,
                "ambient_noise_db": 80,
                "heading": 270,
            },
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 25.0,
            "notes": "Safety concern mid-navigation with high-motion telemetry",
        },
        {
            "id": "B07_read_that",
            "text": "What does that say?",
            "send_image": "text_sign",
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_tool": "extract_text_from_camera",
            "collect_sec": 30.0,
            "notes": "Another OCR request — switching back to text reading",
        },
        {
            "id": "B08_where_am_i",
            "text": "Where am I right now?",
            "send_image": None,
            "send_telemetry": {
                "motion_state": "stationary",
                "step_cadence": 0,
                "ambient_noise_db": 50,
                "heading": 0,
                "gps": {"latitude": 37.7880, "longitude": -122.4075, "accuracy": 3.0},
            },
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 30.0,
            "notes": "Location query — tests GPS context integration",
        },
        {
            "id": "B09_remember",
            "text": "Remember that I like this intersection.",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_tool": "remember_entity",
            "collect_sec": 20.0,
            "notes": "Memory store",
        },
        {
            "id": "B10_lod_change",
            "text": "Give me more detail about my surroundings.",
            "send_image": "park_scene",
            "send_telemetry": None,
            "send_gesture": "force_lod_3",
            "expect_agent_response": True,
            "collect_sec": 35.0,
            "notes": "LOD3 detailed view with gesture",
        },
        {
            "id": "B11_followup",
            "text": "What else can you tell me about this place?",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 25.0,
            "notes": "Follow-up without new image — tests context retention",
        },
        {
            "id": "B12_recall",
            "text": "What intersection did I like?",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 25.0,
            "notes": "Memory recall test",
        },
    ],
}

# Conversation C: Defect regression suite (E2E-001 through E2E-007)
CONVERSATION_C = {
    "id": "conv_defect_regression",
    "description": "Defect regression: dedup, tag leak, nav blocking, token stability",
    "turns": [
        {
            "id": "C01_location_query",
            "text": "Where am I right now?",
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
            "expect_not_blocked": ["get_location_info", "nearby_search"],
            "collect_sec": 25.0,
            "notes": "E2E-007: location query should NOT be blocked",
        },
        {
            "id": "C02_whats_nearby",
            "text": "What's nearby? Any restaurants?",
            "send_image": "street_scene",
            "send_telemetry": {
                "motion_state": "stationary",
                "step_cadence": 0,
                "ambient_noise_db": 55,
                "heading": 180,
                "gps": {"latitude": 40.7580, "longitude": -73.9855, "accuracy": 5.0},
            },
            "send_gesture": None,
            "send_video_frames": 3,
            "expect_agent_response": True,
            "expect_not_blocked": ["nearby_search", "maps_query"],
            "collect_sec": 30.0,
            "notes": "E2E-002/003: should call ONE of nearby_search/maps_query, not both",
        },
        {
            "id": "C03_read_menu",
            "text": "Read that menu for me.",
            "send_image": "text_sign",
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_tool": "extract_text_from_camera",
            "collect_sec": 30.0,
            "notes": "E2E-005: tool result should be spoken in this turn",
        },
        {
            "id": "C04_navigate_cafe",
            "text": "Take me to the closest cafe.",
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
            "expect_tool": "navigate_to",
            "collect_sec": 35.0,
            "notes": "E2E-002: navigate_to called at most once",
        },
        {
            "id": "C05_whats_ahead_video",
            "text": "What's ahead of me?",
            "send_image": "hazard_scene",
            "send_telemetry": None,
            "send_gesture": None,
            "send_video_frames": 4,
            "expect_agent_response": True,
            "collect_sec": 30.0,
            "notes": "E2E-001/004: context tags must not leak in agent speech",
        },
        {
            "id": "C06_context_silence",
            "text": "Tell me more about this area.",
            "send_image": "park_scene",
            "send_telemetry": {
                "motion_state": "walking",
                "step_cadence": 90,
                "ambient_noise_db": 45,
                "heading": 90,
                "gps": {"latitude": 40.7585, "longitude": -73.9850, "accuracy": 5.0},
            },
            "send_gesture": None,
            "send_video_frames": 3,
            "expect_agent_response": True,
            "collect_sec": 30.0,
            "notes": "E2E-004: context injection should not be echoed",
        },
        {
            "id": "C07_search_then_nav",
            "text": "Find me a pharmacy nearby and then navigate there.",
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
            "collect_sec": 40.0,
            "notes": "E2E-002/003: mutex should prevent nearby_search + maps_query both firing",
        },
        {
            "id": "C08_final_stability",
            "text": "Thanks, that's all for now. Goodbye.",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "collect_sec": 15.0,
            "notes": "E2E-006: session should remain stable through turn 8",
        },
    ],
}

# Conversation D: Real-time camera + audio multi-turn stress (mobile-like loop)
CONVERSATION_D = {
    "id": "conv_realtime_stream_stress",
    "description": "Real-time stream stress: continuous camera frames + multi-turn audio with tool chain coverage",
    "turns": [
        {
            "id": "D01_live_scene_boot",
            "text": "I just raised my phone. Tell me what you can see right now.",
            "send_image": "street_scene",
            "send_telemetry": {
                "motion_state": "walking",
                "step_cadence": 96,
                "ambient_noise_db": 68,
                "heading": 120,
                "gps": {"latitude": 37.7752, "longitude": -122.4186, "accuracy": 4.0},
            },
            "send_gesture": None,
            "send_video_frames": 10,
            "expect_agent_response": True,
            "collect_sec": 35.0,
            "notes": "Real-time style startup: camera frames + first spoken answer",
        },
        {
            "id": "D02_live_ocr",
            "text": "Please read that sign while I keep moving.",
            "send_image": "text_sign",
            "send_telemetry": {
                "motion_state": "walking",
                "step_cadence": 105,
                "ambient_noise_db": 70,
                "heading": 130,
            },
            "send_gesture": None,
            "send_video_frames": 8,
            "expect_agent_response": True,
            "expect_tool": "extract_text_from_camera",
            "collect_sec": 35.0,
            "notes": "Live OCR under motion",
        },
        {
            "id": "D03_live_search",
            "text": "Find a nearby coffee shop with good reviews.",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_any_tool": ["google_search", "maps_query", "nearby_search"],
            "collect_sec": 40.0,
            "notes": "Search API chain validation in a live flow",
        },
        {
            "id": "D04_live_nav",
            "text": "Great, now navigate me there.",
            "send_image": None,
            "send_telemetry": {
                "motion_state": "stationary",
                "step_cadence": 0,
                "ambient_noise_db": 58,
                "heading": 110,
                "gps": {"latitude": 37.7751, "longitude": -122.4183, "accuracy": 4.0},
            },
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_tool": "navigate_to",
            "collect_sec": 45.0,
            "notes": "Navigation API chain validation",
        },
        {
            "id": "D05_live_memory_store",
            "text": "Remember this as my favorite route.",
            "send_image": None,
            "send_telemetry": None,
            "send_gesture": None,
            "expect_agent_response": True,
            "expect_tool": "remember_entity",
            "collect_sec": 25.0,
            "notes": "Memory write in a real-time navigation context",
        },
        {
            "id": "D06_live_memory_recall",
            "text": "What route did I ask you to remember?",
            "send_image": "building_scene",
            "send_telemetry": None,
            "send_gesture": None,
            "send_video_frames": 6,
            "expect_agent_response": True,
            "collect_sec": 30.0,
            "notes": "Memory recall after live stream rounds",
        },
    ],
}

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
            "expect_agent_response": False,
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

# Session 1 conversations (single WebSocket, stores memories)
SESSION_1_CONVERSATIONS = [
    CONVERSATION_A, CONVERSATION_B, CONVERSATION_C, CONVERSATION_D,
    CONVERSATION_E, CONVERSATION_G, CONVERSATION_H, CONVERSATION_I,
    CONVERSATION_J, CONVERSATION_K,
]

# Session 2 conversations (NEW WebSocket, recalls Session 1 memories)
SESSION_2_CONVERSATIONS = [CONVERSATION_F]

ALL_CONVERSATIONS = SESSION_1_CONVERSATIONS + SESSION_2_CONVERSATIONS


# ---------------------------------------------------------------------------
# Turn result
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    turn_id: str
    text: str
    passed: bool = False
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    user_transcripts: list[str] = field(default_factory=list)
    agent_transcripts: list[str] = field(default_factory=list)
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    lod_updates: list[dict[str, Any]] = field(default_factory=list)
    frame_acks: list[dict[str, Any]] = field(default_factory=list)
    message_counts: dict[str, int] = field(default_factory=dict)
    audio_bytes_received: int = 0
    first_user_transcript_latency: float | None = None
    first_agent_transcript_latency: float | None = None
    collect_duration_sec: float = 0.0
    notes: str = ""
    request_reconnect: bool = False


@dataclass
class ConversationResult:
    conversation_id: str
    description: str
    total_turns: int = 0
    passed_turns: int = 0
    failed_turns: int = 0
    turns: list[TurnResult] = field(default_factory=list)
    issues: list[dict[str, Any]] = field(default_factory=list)
    total_duration_sec: float = 0.0


# ---------------------------------------------------------------------------
# Audio helpers (from gemini_tts_multiturn_test.py patterns)
# ---------------------------------------------------------------------------

def _create_client() -> genai.Client:
    """Create a client for TTS/Imagen, preferring Vertex credentials when available."""
    use_vertex_raw = (os.getenv("GOOGLE_GENAI_USE_VERTEXAI") or "").strip().lower()
    vertex_explicit_true = use_vertex_raw in {"1", "true", "yes", "on"}
    vertex_explicit_false = use_vertex_raw in {"0", "false", "no", "off"}
    project = (os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip()
    location = (
        os.getenv("GOOGLE_CLOUD_LOCATION")
        or os.getenv("GOOGLE_CLOUD_REGION")
        or "us-central1"
    ).strip()
    auto_prefer_vertex = not use_vertex_raw and bool(project)
    use_vertex = vertex_explicit_true or auto_prefer_vertex

    if use_vertex and project:
        try:
            return genai.Client(vertexai=True, project=project, location=location)
        except Exception as exc:
            if vertex_explicit_true:
                log.warning("Vertex client init failed (explicitly requested), falling back to API key: %s", exc)
            else:
                log.info("Vertex auto-detect init failed, falling back to API key: %s", exc)
    elif vertex_explicit_true and not project:
        log.warning("GOOGLE_GENAI_USE_VERTEXAI=true but GOOGLE_CLOUD_PROJECT is empty; falling back to API key.")
    elif vertex_explicit_false:
        log.info("GOOGLE_GENAI_USE_VERTEXAI explicitly disabled; using API key path.")

    api_key = (
        os.getenv("_GOOGLE_AI_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or ""
    ).strip()
    if not api_key:
        raise RuntimeError("No API key and Vertex init unavailable. Set GOOGLE_GENAI_USE_VERTEXAI or GEMINI_API_KEY.")
    return genai.Client(api_key=api_key, vertexai=False)


def _signal_smoothness(samples: np.ndarray) -> float:
    if samples.size < 2:
        return float("inf")
    return float(np.abs(np.diff(samples.astype(np.int32))).mean())


def _wav_to_mono_int16(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if sw == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.int16)
    elif sw == 4:
        samples = (np.frombuffer(raw, dtype="<i4") >> 16).astype(np.int16)
    else:
        raise ValueError(f"unsupported sample width {sw}")
    if ch > 1:
        samples = samples.reshape(-1, ch).astype(np.int32).mean(axis=1).astype(np.int16)
    return samples, sr


def _resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return samples.astype(np.int16)
    src_len = len(samples)
    dst_len = max(1, int(round(src_len * (dst_rate / float(src_rate)))))
    src_x = np.linspace(0, 1, src_len, endpoint=False)
    dst_x = np.linspace(0, 1, dst_len, endpoint=False)
    dst = np.interp(dst_x, src_x, samples.astype(np.float64))
    return np.clip(np.round(dst), -32768, 32767).astype(np.int16)


def _extract_audio(response: types.GenerateContentResponse) -> tuple[bytes, str]:
    for part in (response.parts or []):
        if part.inline_data and part.inline_data.data:
            data = part.inline_data.data
            if isinstance(data, str):
                data = base64.b64decode(data, validate=True)
            return data, part.inline_data.mime_type or "application/octet-stream"
    raise RuntimeError("No audio in TTS response")


def _parse_rate_from_mime(mime: str) -> int:
    m = re.search(r"rate=(\d+)", mime, re.IGNORECASE)
    return int(m.group(1)) if m else 24000


def _synthesize_pcm_local_fallback(text: str) -> bytes:
    """Fallback TTS via local macOS `say` + ffmpeg when Gemini quota is exhausted."""
    tmp_root = Path("/tmp/sightline_runs/tts_local_fallback")
    tmp_root.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:10]
    aiff_path = tmp_root / f"{token}.aiff"
    pcm_path = tmp_root / f"{token}.pcm16k.raw"

    say_cmd = ["say", "-v", "Samantha", "-o", str(aiff_path), text]
    say_result = subprocess.run(say_cmd, capture_output=True, text=True)
    if say_result.returncode != 0:
        raise RuntimeError(f"local say fallback failed: {say_result.stderr[:200]}")

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", str(aiff_path),
        "-ac", "1",
        "-ar", str(TARGET_SAMPLE_RATE),
        "-f", "s16le",
        str(pcm_path),
    ]
    ffmpeg_result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if ffmpeg_result.returncode != 0:
        raise RuntimeError(f"local ffmpeg fallback failed: {ffmpeg_result.stderr[:200]}")
    return pcm_path.read_bytes()


def synthesize_pcm(client: genai.Client, text: str, voice: str = "Aoede") -> bytes:
    """Synthesize text to 16kHz mono int16 PCM bytes."""
    try:
        response = client.models.generate_content(
            model=TTS_MODEL,
            contents=(
                "Read the following text naturally. Do not add extra words.\n\n"
                + text
            ),
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                    ),
                    language_code="en-US",
                ),
            ),
        )
    except Exception as exc:
        exc_text = str(exc)
        if "429" in exc_text or "RESOURCE_EXHAUSTED" in exc_text:
            log.warning("Gemini TTS quota exhausted; using local fallback for this turn.")
            return _synthesize_pcm_local_fallback(text)
        raise
    audio_data, mime = _extract_audio(response)
    mime_lower = mime.lower()

    if "audio/wav" in mime_lower or "audio/x-wav" in mime_lower:
        samples, src_rate = _wav_to_mono_int16(audio_data)
    elif "audio/pcm" in mime_lower or "audio/l16" in mime_lower:
        src_rate = _parse_rate_from_mime(mime)
        usable = len(audio_data) - (len(audio_data) % 2)
        if "audio/l16" in mime_lower:
            le = np.frombuffer(audio_data[:usable], dtype="<i2").astype(np.int16)
            be = np.frombuffer(audio_data[:usable], dtype=">i2").astype(np.int16)
            samples = le if _signal_smoothness(le) <= _signal_smoothness(be) else be
        else:
            samples = np.frombuffer(audio_data[:usable], dtype="<i2").astype(np.int16)
    else:
        raise RuntimeError(f"Unsupported mime: {mime}")

    resampled = _resample(samples, src_rate, TARGET_SAMPLE_RATE)
    return resampled.tobytes()


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

_IMG_MAX_DIM = 640
_IMG_JPEG_QUALITY = 80


def _compress_image(path: Path) -> None:
    """Resize + recompress an image to ≤640px and JPEG quality 80.

    Keeps images small enough for WebSocket streaming without overwhelming
    the server or Live API.  Uses opencv-python-headless (already a dep).
    """
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        return
    h, w = img.shape[:2]
    if max(h, w) > _IMG_MAX_DIM:
        scale = _IMG_MAX_DIM / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, _IMG_JPEG_QUALITY])


def _generate_synthetic_image(
    scene_id: str,
    output_dir: Path,
    *,
    client: genai.Client | None = None,
) -> Path:
    """Generate a test image: try Imagen first, fall back to ffmpeg.

    Images are cached — if the file already exists it is returned immediately.
    """
    out_path = output_dir / f"{scene_id}.jpg"

    if out_path.exists():
        return out_path

    # --- Try Imagen ---
    prompt = SCENE_PROMPTS.get(scene_id)
    if client and prompt:
        try:
            log.info("[imagen] generating %s ...", scene_id)
            response = client.models.generate_images(
                model=IMAGEN_MODEL,
                prompt=prompt,
                config={"number_of_images": 1},
            )
            if response.generated_images:
                image_bytes = response.generated_images[0].image.image_bytes
                if image_bytes:
                    out_path.write_bytes(image_bytes)
                    _compress_image(out_path)
                    log.info("[imagen] saved %s (%d bytes)", out_path.name, out_path.stat().st_size)
                    return out_path
            log.warning("[imagen] empty response for %s, falling back to ffmpeg", scene_id)
        except Exception as exc:
            log.warning("[imagen] failed for %s (%s), falling back to ffmpeg", scene_id, exc)

    # --- Fallback: ffmpeg synthetic image ---
    colors = {
        "street_scene": "0x4488BB",
        "text_sign": "0x228844",
        "hazard_scene": "0xCC4422",
        "park_scene": "0x44AA44",
        "building_scene": "0x666699",
    }
    color = colors.get(scene_id, "0x888888")

    cmd = [
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"color=c={color}:s=640x480:d=1",
        "-vf", f"drawtext=text='{scene_id}':fontsize=36:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2",
        "-vframes", "1", "-q:v", "3",
        str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        return out_path

    # Fallback: plain color (no text overlay)
    cmd2 = [
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"color=c={color}:s=640x480:d=1",
        "-vframes", "1", "-q:v", "3",
        str(out_path),
    ]
    r2 = subprocess.run(cmd2, capture_output=True, text=True)
    if r2.returncode == 0:
        return out_path

    raise RuntimeError(f"Failed to generate image for {scene_id}")


# ---------------------------------------------------------------------------
# WebSocket multi-turn runner
# ---------------------------------------------------------------------------

async def run_conversation(
    *,
    ws_base_url: str,
    conversation: dict[str, Any],
    pcm_cache: dict[str, bytes],
    image_dir: Path,
    timeout_mult: float = 1.0,
    ws_ping_interval: float | None = None,
    ws_ping_timeout: float | None = None,
    connect_retries: int = 4,
    connect_backoff_sec: float = 4.0,
    session_ready_timeout_sec: float = 35.0,
    turn_retry_on_reconnect: int = 1,
    inter_turn_delay_sec: float = 1.5,
) -> ConversationResult:
    """Run a full multi-turn conversation against the server."""
    conv_id = conversation["id"]
    turns_def = conversation["turns"]
    log.info("=== Starting conversation: %s (%d turns) ===", conv_id, len(turns_def))

    # Vertex AI session service requires alphanumeric + hyphen IDs (no underscores)
    safe_session_id = conv_id.replace("_", "-")
    ws_url = f"{ws_base_url.rstrip('/')}/ws/e2e-user/{safe_session_id}"
    turn_results: list[TurnResult] = []
    t0_total = time.monotonic()

    async def _open_ws(*, drain_initial_greeting: bool) -> websockets.ClientConnection:
        last_exc: Exception | None = None
        attempts = max(1, int(connect_retries))
        for attempt in range(1, attempts + 1):
            ws: websockets.ClientConnection | None = None
            try:
                ws = await websockets.connect(
                    ws_url,
                    max_size=None,
                    ping_interval=ws_ping_interval,
                    ping_timeout=ws_ping_timeout,
                    close_timeout=10.0,
                    open_timeout=max(5.0, float(session_ready_timeout_sec)),
                )

                deadline = time.monotonic() + float(session_ready_timeout_sec)
                while True:
                    remain = deadline - time.monotonic()
                    if remain <= 0:
                        raise TimeoutError("session_ready timeout")
                    raw = await asyncio.wait_for(ws.recv(), timeout=remain)
                    if isinstance(raw, bytes):
                        continue
                    msg = json.loads(raw)
                    if msg.get("type") == "session_ready":
                        log.info("[%s] session_ready received (attempt %d/%d)", conv_id, attempt, attempts)
                        break

                if drain_initial_greeting:
                    await _drain_initial_greeting(ws, timeout_sec=15.0)
                return ws
            except Exception as exc:
                last_exc = exc
                if ws is not None:
                    try:
                        await ws.close()
                    except Exception:
                        pass
                if attempt >= attempts:
                    break
                backoff = max(0.0, float(connect_backoff_sec)) * attempt
                log.warning(
                    "[%s] WS connect/session_ready failed (attempt %d/%d): %s; retrying in %.1fs",
                    conv_id,
                    attempt,
                    attempts,
                    exc,
                    backoff,
                )
                if backoff > 0:
                    await asyncio.sleep(backoff)
        raise RuntimeError(f"failed to connect conversation websocket: {last_exc}")

    ws: websockets.ClientConnection | None = None
    try:
        ws = await _open_ws(drain_initial_greeting=True)

        # Run each turn with optional reconnect+retry
        for i, turn_def in enumerate(turns_def):
            turn_id = turn_def["id"]
            text = turn_def["text"]
            collect_sec = turn_def.get("collect_sec", 25.0) * timeout_mult
            log.info("[Turn %d/%d] %s: \"%s\"", i + 1, len(turns_def), turn_id, text)

            retry_budget = max(0, int(turn_retry_on_reconnect))
            tr: TurnResult | None = None
            while True:
                try:
                    tr = await _run_single_turn(
                        ws=ws,
                        turn_def=turn_def,
                        pcm_cache=pcm_cache,
                        image_dir=image_dir,
                        collect_sec=collect_sec,
                    )
                except (websockets.exceptions.ConnectionClosed, ConnectionError, OSError) as exc:
                    if retry_budget > 0:
                        retry_budget -= 1
                        log.warning(
                            "[%s] Connection lost mid-turn (%s). Reconnecting and retrying turn (%d retries left).",
                            turn_id,
                            exc,
                            retry_budget,
                        )
                        if ws is not None:
                            try:
                                await ws.close()
                            except Exception:
                                pass
                        ws = await _open_ws(drain_initial_greeting=False)
                        continue
                    tr = TurnResult(
                        turn_id=turn_id,
                        text=text,
                        failures=[f"connection_lost: {exc}"],
                    )
                    break

                if tr.request_reconnect and retry_budget > 0:
                    retry_budget -= 1
                    log.warning(
                        "[%s] Server requested reconnect (go_away/error). Retrying turn (%d retries left).",
                        turn_id,
                        retry_budget,
                    )
                    if ws is not None:
                        try:
                            await ws.close()
                        except Exception:
                            pass
                    ws = await _open_ws(drain_initial_greeting=False)
                    continue
                break

            if tr is None:
                tr = TurnResult(
                    turn_id=turn_id,
                    text=text,
                    failures=["turn_result_missing"],
                )
            turn_results.append(tr)

            status = "PASS" if tr.passed else "FAIL"
            agent_text = " ".join(tr.agent_transcripts)[:120]
            log.info(
                "[%s] %s | user_lat=%.2fs agent_lat=%.2fs audio=%d | agent: %s",
                status, turn_id,
                tr.first_user_transcript_latency or -1,
                tr.first_agent_transcript_latency or -1,
                tr.audio_bytes_received,
                agent_text or "(no transcript)",
            )
            if tr.failures:
                for f in tr.failures:
                    log.warning("  FAILURE: %s", f)
            if tr.warnings:
                for w in tr.warnings:
                    if w.startswith("go_away_received"):
                        log.warning("  WARNING: %s", w)

            # Brief pause between turns to let model settle
            if inter_turn_delay_sec > 0:
                await asyncio.sleep(inter_turn_delay_sec)

    except Exception as exc:
        log.error("Conversation %s failed: %s", conv_id, exc)
        # Add a failed turn for the connection error
        turn_results.append(TurnResult(
            turn_id="CONNECTION_ERROR",
            text=str(exc),
            failures=[f"connection_error: {exc}"],
        ))
    finally:
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass

    total_duration = time.monotonic() - t0_total
    passed = sum(1 for t in turn_results if t.passed)
    failed = sum(1 for t in turn_results if not t.passed)

    result = ConversationResult(
        conversation_id=conv_id,
        description=conversation["description"],
        total_turns=len(turn_results),
        passed_turns=passed,
        failed_turns=failed,
        turns=turn_results,
        total_duration_sec=total_duration,
    )

    # Build issues from failures
    result.issues = _build_issues(turn_results)
    return result


async def _drain_initial_greeting(
    ws: websockets.ClientConnection,
    timeout_sec: float = 15.0,
) -> None:
    """Drain any initial greeting audio/transcripts from the model."""
    deadline = time.monotonic() + timeout_sec
    last_event = time.monotonic()
    while True:
        now = time.monotonic()
        if now >= deadline:
            break
        # If 3s of silence after last event, model has settled
        if now - last_event >= 3.0:
            break
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            last_event = time.monotonic()
        except asyncio.TimeoutError:
            continue


async def _run_single_turn(
    *,
    ws: websockets.ClientConnection,
    turn_def: dict[str, Any],
    pcm_cache: dict[str, bytes],
    image_dir: Path,
    collect_sec: float,
) -> TurnResult:
    """Execute a single conversation turn and collect the response."""
    turn_id = turn_def["id"]
    text = turn_def["text"]
    pcm = pcm_cache.get(turn_id, b"")

    if not pcm:
        log.warning("No PCM audio for turn %s — skipping audio send", turn_id)
        return TurnResult(
            turn_id=turn_id,
            text=text,
            failures=["no_pcm_audio"],
            notes=turn_def.get("notes", ""),
        )

    # Containers
    counts: dict[str, int] = {}
    tool_events: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    lod_updates: list[dict[str, Any]] = []
    user_transcripts: list[str] = []
    agent_transcripts: list[str] = []
    frame_acks: list[dict[str, Any]] = []
    audio_bytes_count = 0

    t0 = time.monotonic()
    first_user_lat: float | None = None
    first_agent_lat: float | None = None

    # --- Send stimuli ---

    # 1. Telemetry (context setup)
    if turn_def.get("send_telemetry"):
        await ws.send(json.dumps({"type": "telemetry", "data": turn_def["send_telemetry"]}))

    # 2. Gesture
    if turn_def.get("send_gesture"):
        await ws.send(json.dumps({"type": "gesture", "gesture": turn_def["send_gesture"]}))

    # 3. Image (before audio so vision pipeline starts)
    img_bytes_for_frames: bytes | None = None
    if turn_def.get("send_image"):
        img_id = turn_def["send_image"]
        img_path = image_dir / f"{img_id}.jpg"
        if img_path.exists():
            img_bytes_for_frames = img_path.read_bytes()
            await ws.send(bytes([MAGIC_IMAGE]) + img_bytes_for_frames)
        else:
            log.warning("Image not found: %s", img_path)

    # 4. Audio + optional video frame streaming
    send_video_frames = turn_def.get("send_video_frames", 0)
    await ws.send(json.dumps({"type": "activity_start"}))

    audio_start = time.monotonic()
    audio_duration = len(pcm) / 2.0 / 16000.0
    if img_bytes_for_frames and send_video_frames:
        frame_interval = audio_duration / max(send_video_frames, 1)
    else:
        frame_interval = float("inf")
    frames_sent = 0

    for offset in range(0, len(pcm), 1280):
        chunk = pcm[offset: offset + 1280]
        await ws.send(bytes([MAGIC_AUDIO]) + chunk)
        await asyncio.sleep((len(chunk) / 2.0) / 16000.0)

        # Interleave video frames at ~1 FPS during audio streaming
        if img_bytes_for_frames and frames_sent < send_video_frames:
            elapsed = time.monotonic() - audio_start
            if elapsed >= frame_interval * (frames_sent + 1):
                await ws.send(bytes([MAGIC_IMAGE]) + img_bytes_for_frames)
                frames_sent += 1

    await ws.send(json.dumps({"type": "activity_end"}))

    # --- Collect responses ---
    _collect_warnings: list[str] = []
    _collect_failures: list[str] = []
    request_reconnect = False
    deadline = time.monotonic() + collect_sec
    last_agent_ts: float | None = None

    while time.monotonic() < deadline:
        # Early exit: if we've received agent response and 4s of quiet
        if last_agent_ts and (time.monotonic() - last_agent_ts) >= 4.0:
            break

        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
        except asyncio.TimeoutError:
            continue

        now = time.monotonic()

        if isinstance(raw, bytes):
            audio_bytes_count += 1
            last_agent_ts = now
            continue

        payload = json.loads(raw)
        msg_type = str(payload.get("type", "unknown"))
        counts[msg_type] = counts.get(msg_type, 0) + 1

        if msg_type == "transcript":
            role = payload.get("role", "")
            txt = str(payload.get("text", "")).strip()
            if not txt:
                continue
            if role == "user":
                user_transcripts.append(txt)
                if first_user_lat is None:
                    first_user_lat = now - t0
            elif role == "agent":
                agent_transcripts.append(txt)
                last_agent_ts = now
                if first_agent_lat is None:
                    first_agent_lat = now - t0

        elif msg_type == "tool_event":
            tool_events.append({
                "tool": payload.get("tool"),
                "status": payload.get("status"),
                "behavior": payload.get("behavior"),
            })
        elif msg_type == "tool_result":
            tool_results.append({
                "tool": payload.get("tool"),
                "behavior": payload.get("behavior"),
                "data": payload.get("data"),
            })
        elif msg_type == "lod_update":
            lod_updates.append({
                "lod": payload.get("lod"),
                "reason": payload.get("reason"),
            })
        elif msg_type == "frame_ack":
            frame_acks.append({
                "queued_agents": payload.get("queued_agents"),
            })
        elif msg_type == "vision_result":
            tool_results.append({"_type": "vision_result", **payload})
        elif msg_type == "ocr_result":
            tool_results.append({"_type": "ocr_result", **payload})
        elif msg_type in ("search_result", "navigation_result"):
            tool_results.append({"_type": msg_type, **payload})
        elif msg_type == "go_away":
            reason = payload.get("reason", "unknown")
            log.warning("Received go_away: %s", reason)
            _collect_warnings.append(f"go_away_received: {reason}")
            request_reconnect = True
            break  # Stop collecting, server is about to close
        elif msg_type == "error":
            error_msg = payload.get("message", "unknown error")
            log.warning("Received error: %s", error_msg)
            _collect_failures.append(f"server_error: {error_msg}")
            request_reconnect = True
            break  # Stop collecting on error

    duration = time.monotonic() - t0

    # --- Validate ---
    failures: list[str] = list(_collect_failures)
    warnings: list[str] = list(_collect_warnings)

    if turn_def.get("expect_agent_response", True):
        if not agent_transcripts:
            failures.append("no_agent_transcript")
        if audio_bytes_count == 0:
            failures.append("no_agent_audio")

    if not user_transcripts:
        warnings.append("no_user_transcript_echoed")

    # Check expected tool
    expected_tool = turn_def.get("expect_tool")
    if expected_tool:
        tool_names = {e.get("tool") for e in tool_events}
        if expected_tool not in tool_names:
            failures.append(f"expected_tool_{expected_tool}_not_called")

    # Check flexible tool expectation (any of the listed tools)
    expect_any = turn_def.get("expect_any_tool")
    if expect_any:
        tool_names = {e.get("tool") for e in tool_events}
        if not tool_names & set(expect_any):
            failures.append(f"expected_one_of_{expect_any}_none_called")

    # Check for unwanted OCR on vision queries
    if "describe" in text.lower() or "what's around" in text.lower() or "what is this" in text.lower():
        ocr_tools = [e for e in tool_events if e.get("tool") == "extract_text_from_camera"]
        if ocr_tools and not turn_def.get("expect_tool") == "extract_text_from_camera":
            warnings.append("unwanted_ocr_triggered_on_vision_query")

    # --- Defect-specific assertions ---

    # E2E-001 / E2E-004: Context tag leak detection
    for marker in _LEAKED_MARKERS:
        if any(marker.lower() in t.lower() for t in agent_transcripts):
            failures.append(f"context_tag_leaked: {marker}")

    # E2E-002 / E2E-003: Duplicate tool call detection
    tool_call_counts: dict[str, int] = {}
    for e in tool_events:
        if e.get("status") == "invoked":
            name = e.get("tool", "")
            tool_call_counts[name] = tool_call_counts.get(name, 0) + 1
    for name, count in tool_call_counts.items():
        if count > 1:
            warnings.append(f"duplicate_tool_call: {name} called {count}x")

    # E2E-002 / E2E-003: Mutex tool group violation detection
    invoked_tools = {e["tool"] for e in tool_events if e.get("status") == "invoked" and e.get("tool")}
    for group in _MUTEX_TOOL_GROUPS:
        overlap = invoked_tools & group
        if len(overlap) > 1:
            failures.append(f"mutex_violation: {sorted(overlap)} both called")

    # E2E-005: Tool result delivery — if tool expected and result returned, agent must speak
    if expected_tool and tool_results:
        if not agent_transcripts:
            failures.append("tool_result_not_spoken")

    # E2E-007: Navigation over-blocking detection
    expect_not_blocked = turn_def.get("expect_not_blocked")
    if expect_not_blocked:
        blocked_tools = [
            e for e in tool_events
            if e.get("status") == "blocked" and e.get("tool") in expect_not_blocked
        ]
        if blocked_tools:
            failures.append(f"over_blocked: {[e['tool'] for e in blocked_tools]}")

    return TurnResult(
        turn_id=turn_id,
        text=text,
        passed=not failures,
        failures=failures,
        warnings=warnings,
        user_transcripts=user_transcripts,
        agent_transcripts=agent_transcripts,
        tool_events=tool_events,
        tool_results=tool_results,
        lod_updates=lod_updates,
        frame_acks=frame_acks,
        message_counts=counts,
        audio_bytes_received=audio_bytes_count,
        first_user_transcript_latency=first_user_lat,
        first_agent_transcript_latency=first_agent_lat,
        collect_duration_sec=duration,
        notes=turn_def.get("notes", ""),
        request_reconnect=request_reconnect,
    )


def _build_issues(turns: list[TurnResult]) -> list[dict[str, Any]]:
    """Build issue definitions from turn results."""
    issues: list[dict[str, Any]] = []
    counter = 1

    for t in turns:
        # Failed turns
        for f in t.failures:
            issues.append({
                "id": f"MT-{counter:03d}",
                "turn": t.turn_id,
                "title": f,
                "severity": "high" if "no_agent" in f else "medium",
                "evidence": {
                    "user_transcripts": t.user_transcripts,
                    "agent_transcripts": t.agent_transcripts[:5],
                    "tool_events": t.tool_events,
                    "message_counts": t.message_counts,
                },
            })
            counter += 1

        # Warnings (observation issues)
        for w in t.warnings:
            issues.append({
                "id": f"MT-{counter:03d}",
                "turn": t.turn_id,
                "title": w,
                "severity": "observation",
                "evidence": {
                    "tool_events": t.tool_events,
                    "agent_transcripts": t.agent_transcripts[:3],
                },
            })
            counter += 1

    return issues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main(args: argparse.Namespace) -> int:
    def _filter_conversations(
        conversations: list[dict[str, Any]],
        patterns_csv: str,
    ) -> list[dict[str, Any]]:
        raw_patterns = [p.strip() for p in patterns_csv.split(",") if p.strip()]
        if not raw_patterns:
            return list(conversations)
        selected: list[dict[str, Any]] = []
        for conv in conversations:
            conv_id = str(conv.get("id", ""))
            if any(fnmatch.fnmatch(conv_id, p) for p in raw_patterns):
                selected.append(conv)
        return selected

    session_1_conversations = list(SESSION_1_CONVERSATIONS)
    session_2_conversations = list(SESSION_2_CONVERSATIONS)
    if args.conversations:
        session_1_conversations = _filter_conversations(session_1_conversations, args.conversations)
        session_2_conversations = _filter_conversations(session_2_conversations, args.conversations)
        log.info(
            "Conversation filter '%s' => session1=%d session2=%d",
            args.conversations,
            len(session_1_conversations),
            len(session_2_conversations),
        )

    selected_conversations = session_1_conversations + session_2_conversations
    if not selected_conversations:
        log.error("No conversations selected. Use --conversations with valid patterns.")
        return 1

    out_dir = Path(args.output_dir).resolve()
    image_dir = out_dir / "images"
    audio_dir = out_dir / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(exist_ok=True)
    audio_dir.mkdir(exist_ok=True)

    # Create API client (used for Imagen + TTS)
    client = _create_client()

    # Step 1: Generate test images (Imagen with ffmpeg fallback)
    log.info("--- Generating test images ---")
    needed_images = set()
    for conv in selected_conversations:
        for turn in conv["turns"]:
            img = turn.get("send_image")
            if img:
                needed_images.add(img)

    for img_id in sorted(needed_images):
        path = _generate_synthetic_image(img_id, image_dir, client=client)
        log.info("Image ready: %s (%d bytes)", path.name, path.stat().st_size)

    # Step 2: Generate TTS audio for all turns
    log.info("--- Generating TTS audio for all turns ---")
    pcm_cache: dict[str, bytes] = {}

    all_turns = []
    for conv in selected_conversations:
        all_turns.extend(conv["turns"])

    for i, turn in enumerate(all_turns):
        turn_id = turn["id"]
        text = turn["text"]
        cache_path = audio_dir / f"{turn_id}.pcm16k.raw"

        if cache_path.exists() and cache_path.stat().st_size > 0:
            pcm_cache[turn_id] = cache_path.read_bytes()
            log.info("[%d/%d] Loaded cached: %s", i + 1, len(all_turns), turn_id)
            continue

        log.info("[%d/%d] Synthesizing: %s → \"%s\"", i + 1, len(all_turns), turn_id, text)
        try:
            pcm = synthesize_pcm(client, text, voice=args.voice)
            cache_path.write_bytes(pcm)
            pcm_cache[turn_id] = pcm
            dur = len(pcm) / 2.0 / TARGET_SAMPLE_RATE
            log.info("  → %d bytes, %.1fs", len(pcm), dur)
            # TTS rate limit: 10 RPM. Sleep 7s between requests to stay safe.
            if i + 1 < len(all_turns):
                await asyncio.sleep(7.0)
        except Exception as exc:
            log.error("  → TTS FAILED: %s", exc)
            pcm_cache[turn_id] = b""
            # On rate limit, wait longer before next attempt
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                log.info("  → Rate limited, waiting 30s...")
                await asyncio.sleep(30.0)

    # Step 3: Run Session 1 conversations (single WS per conversation)
    log.info("--- Session 1: Running %d conversations ---", len(session_1_conversations))
    results: list[ConversationResult] = []
    ws_ping_interval = args.ws_ping_interval if args.ws_ping_interval and args.ws_ping_interval > 0 else None
    ws_ping_timeout = args.ws_ping_timeout if args.ws_ping_timeout and args.ws_ping_timeout > 0 else None

    for conv in session_1_conversations:
        result = await run_conversation(
            ws_base_url=args.server,
            conversation=conv,
            pcm_cache=pcm_cache,
            image_dir=image_dir,
            timeout_mult=args.timeout,
            ws_ping_interval=ws_ping_interval,
            ws_ping_timeout=ws_ping_timeout,
            connect_retries=args.connect_retries,
            connect_backoff_sec=args.connect_backoff_sec,
            session_ready_timeout_sec=args.session_ready_timeout_sec,
            turn_retry_on_reconnect=args.turn_retry_on_reconnect,
            inter_turn_delay_sec=args.inter_turn_delay_sec,
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
        if args.conversation_cooldown_sec > 0:
            await asyncio.sleep(args.conversation_cooldown_sec)

    # Step 4: Run Session 2 (cross-session memory recall)
    if session_2_conversations:
        log.info("--- Session 2: Cross-session memory recall ---")
        log.info("(Conv E stored memories → Conv F should recall them in new WS session)")

        for conv in session_2_conversations:
            result = await run_conversation(
                ws_base_url=args.server,
                conversation=conv,
                pcm_cache=pcm_cache,
                image_dir=image_dir,
                timeout_mult=args.timeout,
                ws_ping_interval=ws_ping_interval,
                ws_ping_timeout=ws_ping_timeout,
                connect_retries=args.connect_retries,
                connect_backoff_sec=args.connect_backoff_sec,
                session_ready_timeout_sec=args.session_ready_timeout_sec,
                turn_retry_on_reconnect=args.turn_retry_on_reconnect,
                inter_turn_delay_sec=args.inter_turn_delay_sec,
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
            if args.conversation_cooldown_sec > 0:
                await asyncio.sleep(args.conversation_cooldown_sec)

    # Step 5: Write report
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "suite_id": f"multiturn_e2e_{stamp}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "server_url": args.server,
        "conversations": [],
    }

    total_turns = 0
    total_passed = 0
    total_failed = 0
    all_issues: list[dict[str, Any]] = []
    observed_tools: set[str] = set()
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

    for r in results:
        total_turns += r.total_turns
        total_passed += r.passed_turns
        total_failed += r.failed_turns
        all_issues.extend(r.issues)
        for t in r.turns:
            for ev in t.tool_events:
                name = ev.get("tool")
                if name:
                    observed_tools.add(str(name))

        conv_data = {
            "id": r.conversation_id,
            "description": r.description,
            "total_turns": r.total_turns,
            "passed": r.passed_turns,
            "failed": r.failed_turns,
            "duration_sec": round(r.total_duration_sec, 1),
            "turns": [asdict(t) for t in r.turns],
            "issues": r.issues,
        }
        report["conversations"].append(conv_data)

    enforce_chain_coverage = not bool(args.conversations)
    missing_chain_tools = sorted(required_chain_tools - observed_tools)
    if enforce_chain_coverage and missing_chain_tools:
        all_issues.append({
            "id": "MT-CHAIN-001",
            "turn": "SUITE",
            "title": f"api_chain_missing_tools: {missing_chain_tools}",
            "severity": "high",
            "evidence": {
                "required_tools": sorted(required_chain_tools),
                "observed_tools": sorted(observed_tools),
            },
        })

    report["summary"] = {
        "total_turns": total_turns,
        "passed": total_passed,
        "failed": total_failed,
        "pass_rate": f"{total_passed / total_turns * 100:.1f}%" if total_turns > 0 else "N/A",
        "total_issues": len(all_issues),
        "tool_chain_coverage": {
            "required": sorted(required_chain_tools),
            "observed": sorted(observed_tools),
            "missing": missing_chain_tools,
        },
    }
    report["all_issues"] = all_issues

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Report: %s", report_path)

    # Print summary
    print("\n" + "=" * 70)
    print("  SightLine Multi-Turn E2E Test Results")
    print("=" * 70)
    print(f"  Total turns: {total_turns}")
    print(f"  Passed:      {total_passed}")
    print(f"  Failed:      {total_failed}")
    print(f"  Pass rate:   {report['summary']['pass_rate']}")
    print()

    for r in results:
        status = "OK" if r.failed_turns == 0 else "ISSUES"
        print(f"  [{status}] {r.conversation_id}: {r.passed_turns}/{r.total_turns} passed ({r.total_duration_sec:.0f}s)")
        for t in r.turns:
            mark = "+" if t.passed else "X"
            agent_preview = " ".join(t.agent_transcripts)[:80]
            tools = ", ".join(e.get("tool", "?") for e in t.tool_events) or "-"
            print(f"    [{mark}] {t.turn_id}: tools=[{tools}] agent=\"{agent_preview}\"")
            for f in t.failures:
                print(f"        FAIL: {f}")
            for w in t.warnings:
                print(f"        WARN: {w}")

    if all_issues:
        print(f"\n  Issues ({len(all_issues)}):")
        for iss in all_issues:
            print(f"    {iss['id']} [{iss['severity']}] {iss['turn']}: {iss['title']}")

    print("=" * 70 + "\n")

    return 0 if total_failed == 0 else 2


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    p = argparse.ArgumentParser(description="Multi-turn E2E test for SightLine.")
    p.add_argument("--server", default="ws://127.0.0.1:8100", help="Server WebSocket URL")
    p.add_argument("--output-dir", default="artifacts/e2e_multiturn", help="Output directory")
    p.add_argument("--voice", default="Aoede", help="TTS voice")
    p.add_argument("--timeout", type=float, default=1.0, help="Timeout multiplier")
    p.add_argument("--ws-ping-interval", type=float, default=0.0, help="WebSocket ping interval seconds (<=0 disables)")
    p.add_argument("--ws-ping-timeout", type=float, default=0.0, help="WebSocket ping timeout seconds (<=0 disables)")
    p.add_argument("--conversations", default="", help="Comma-separated fnmatch patterns for conversation IDs (e.g. 'conv_mixed_*,conv_narrative_*').")
    p.add_argument("--connect-retries", type=int, default=4, help="Retries for websocket connect/session_ready handshake.")
    p.add_argument("--connect-backoff-sec", type=float, default=4.0, help="Base backoff seconds between connect retries.")
    p.add_argument("--session-ready-timeout-sec", type=float, default=35.0, help="Timeout waiting for session_ready.")
    p.add_argument("--turn-retry-on-reconnect", type=int, default=1, help="How many times to retry a turn after reconnect/go_away.")
    p.add_argument("--inter-turn-delay-sec", type=float, default=1.5, help="Delay between turns.")
    p.add_argument("--conversation-cooldown-sec", type=float, default=8.0, help="Delay between conversations to avoid local capacity overlap.")
    args = p.parse_args()
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
