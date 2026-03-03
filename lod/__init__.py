"""SightLine LOD (Level of Detail) decision engine.

Implements the rule-based LOD decision system that determines
information density based on user physical state and context.
"""

from lod.lod_engine import LODDecisionLog, decide_lod, should_speak
from lod.models import (
    EphemeralContext,
    GPSData,
    NarrativeSnapshot,
    SessionContext,
    UserProfile,
)
from lod.narrative_snapshot import on_lod_change
from lod.prompt_builder import (
    build_dynamic_prompt,
    build_full_dynamic_prompt,
    build_lod_update_message,
)
from lod.telemetry_aggregator import LOD_TELEMETRY_INTERVAL, TelemetryAggregator

__all__ = [
    "LODDecisionLog",
    "EphemeralContext",
    "GPSData",
    "NarrativeSnapshot",
    "SessionContext",
    "UserProfile",
    "LOD_TELEMETRY_INTERVAL",
    "TelemetryAggregator",
    "decide_lod",
    "should_speak",
    "on_lod_change",
    "build_dynamic_prompt",
    "build_full_dynamic_prompt",
    "build_lod_update_message",
]
