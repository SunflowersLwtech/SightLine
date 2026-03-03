"""SightLine telemetry parsing, semantic conversion, and session tracking.

Converts raw sensor JSON from iOS into semantic text
suitable for injection into Gemini Live API context.

Phase 2: Also converts raw telemetry into ``EphemeralContext``
for the LOD decision engine.

Phase 5: SessionMetaTracker for runtime session lifecycle telemetry.
"""

from telemetry.telemetry_parser import parse_telemetry, parse_telemetry_to_ephemeral
from telemetry.session_meta_tracker import SessionMetaTracker

__all__ = ["parse_telemetry", "parse_telemetry_to_ephemeral", "SessionMetaTracker"]
