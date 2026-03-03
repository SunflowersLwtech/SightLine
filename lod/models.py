"""SightLine LOD data models.

Core data structures for the Level-of-Detail decision engine,
ephemeral context, session context, and user profile.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Ephemeral Context — real-time sensor snapshot (ms~s lifetime)
# ---------------------------------------------------------------------------


@dataclass
class GPSData:
    """GPS location data from CoreLocation."""

    lat: float = 0.0
    lng: float = 0.0
    accuracy: float = 0.0  # metres
    speed: float = 0.0  # m/s
    altitude: float = 0.0


@dataclass
class EphemeralContext:
    """Real-time sensor snapshot consumed by the LOD engine.

    Fields map 1:1 to the iOS ``TelemetryData`` struct and the
    ``[TELEMETRY UPDATE]`` JSON payload.
    """

    motion_state: str = "stationary"  # stationary | walking | running | in_vehicle | cycling
    step_cadence: float = 0.0  # steps/minute (CMPedometer)
    ambient_noise_db: float = 70.0  # dB SPL (conservative default when sensor unavailable)
    gps: Optional[GPSData] = None
    heading: float = 0.0  # degrees, 0=N
    time_context: str = "unknown"  # morning_commute | work_hours | evening | late_night
    heart_rate: Optional[float] = None  # BPM (None when no watch)
    user_gesture: Optional[str] = None  # lod_up | lod_down | tap | shake
    device_type: str = "phone_only"  # phone_only | phone_and_watch
    weather_condition: str = "unknown"  # clear | rain | snow | fog | cloudy etc.
    weather_visibility: float = 10000.0  # meters
    weather_precipitation: str = "none"  # none | rain | snow | sleet | hail | mixed
    depth_center: Optional[float] = None  # center distance in meters (None = no depth data)
    depth_min: Optional[float] = None  # closest object distance in meters
    depth_min_region: Optional[str] = None  # region of closest object
    # Watch extended context
    watch_pitch: float = 0.0
    watch_roll: float = 0.0
    watch_yaw: float = 0.0
    watch_stability_score: float = 1.0  # 1.0 = stable, 0.0 = unstable
    watch_heading: Optional[float] = None  # magnetic heading from watch compass
    watch_heading_accuracy: Optional[float] = None
    sp_o2: Optional[float] = None  # blood oxygen %
    watch_noise_exposure: Optional[float] = None  # dB from watch HealthKit
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Session Context — current trip/conversation state (min~hr lifetime)
# ---------------------------------------------------------------------------


@dataclass
class NarrativeSnapshot:
    """Checkpoint saved when LOD downgrades while a task is active."""

    task_type: str = ""  # e.g. "menu_reading", "document_reading"
    progress: str = ""  # e.g. "Read items 1-3"
    remaining: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SessionContext:
    """State accumulated during a single session.

    Maintained in ADK Session State / Gemini context window.
    """

    trip_purpose: Optional[str] = None  # "going to interview", "daily commute"
    space_type: str = "unknown"  # indoor | outdoor | vehicle | unknown
    space_transitions: list[str] = field(default_factory=list)
    avg_cadence_30min: float = 0.0
    conversation_topics: list[str] = field(default_factory=list)
    active_task: Optional[str] = None  # "reading menu" / None
    narrative_snapshot: Optional[NarrativeSnapshot] = None
    recent_space_transition: bool = False
    user_requested_detail: bool = False  # user said "tell me more"
    user_said_stop: bool = False  # user said "stop"
    familiarity_score: float = 0.5  # 0.0 (never been) to 1.0 (daily visit)
    current_lod: int = 2  # tracks the active LOD level
    current_activity_state: str = "idle"  # idle | user_speaking
    last_activity_event: Optional[str] = None  # activity_start | activity_end
    last_activity_event_ts: Optional[datetime] = None
    last_activity_source: str = "none"  # ios_client | system
    activity_event_count: int = 0


# ---------------------------------------------------------------------------
# User Profile — long-term preferences (cross-session, from Firestore)
# ---------------------------------------------------------------------------


@dataclass
class UserProfile:
    """User profile loaded from Firestore ``user_profiles/{user_id}``.

    Design based on *Beyond the Cane* (ACM TACCESS 2022).
    """

    user_id: str = "default"
    vision_status: str = "totally_blind"  # totally_blind | low_vision
    blindness_onset: str = "congenital"  # congenital | acquired
    onset_age: Optional[int] = None
    has_guide_dog: bool = False
    has_white_cane: bool = True
    tts_speed: float = 1.5  # 1.0 ~ 3.0
    verbosity_preference: str = "concise"  # concise | detailed
    language: str = "en-US"
    description_priority: str = "spatial"  # spatial | object | text
    color_description: bool = True
    om_level: str = "intermediate"  # beginner | intermediate | advanced
    travel_frequency: str = "weekly"  # daily | weekly | rarely
    preferred_name: str = ""

    def update_from_dict(self, doc: dict) -> None:
        """Update fields in-place from a Firestore-style dict."""
        for f in dataclasses.fields(self):
            if f.name == "user_id":
                continue
            if f.name in doc:
                setattr(self, f.name, doc[f.name])

    @classmethod
    def default(cls) -> UserProfile:
        """Sensible defaults for first-time / anonymous users."""
        return cls()

    @classmethod
    def from_firestore(cls, doc: dict, user_id: str = "default") -> UserProfile:
        """Construct from a Firestore document dict."""
        return cls(
            user_id=user_id,
            vision_status=doc.get("vision_status", "totally_blind"),
            blindness_onset=doc.get("blindness_onset", "congenital"),
            onset_age=doc.get("onset_age"),
            has_guide_dog=doc.get("has_guide_dog", False),
            has_white_cane=doc.get("has_white_cane", True),
            tts_speed=doc.get("tts_speed", 1.5),
            verbosity_preference=doc.get("verbosity_preference", "concise"),
            language=doc.get("language", "en-US"),
            description_priority=doc.get("description_priority", "spatial"),
            color_description=doc.get("color_description", True),
            om_level=doc.get("om_level", "intermediate"),
            travel_frequency=doc.get("travel_frequency", "weekly"),
            preferred_name=doc.get("preferred_name", ""),
        )
