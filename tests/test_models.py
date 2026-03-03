"""Tests for SightLine LOD data models."""

from lod.models import EphemeralContext, GPSData, NarrativeSnapshot, SessionContext, UserProfile


def test_ephemeral_context_defaults():
    ctx = EphemeralContext()
    assert ctx.motion_state == "stationary"
    assert ctx.step_cadence == 0.0
    assert ctx.ambient_noise_db == 70.0
    assert ctx.heart_rate is None
    assert ctx.gps is None
    assert ctx.user_gesture is None
    assert ctx.device_type == "phone_only"


def test_session_context_defaults():
    s = SessionContext()
    assert s.current_lod == 2
    assert s.active_task is None
    assert s.narrative_snapshot is None
    assert s.recent_space_transition is False
    assert s.user_requested_detail is False
    assert s.user_said_stop is False


def test_user_profile_default_factory():
    p = UserProfile.default()
    assert p.vision_status == "totally_blind"
    assert p.blindness_onset == "congenital"
    assert p.verbosity_preference == "concise"
    assert p.om_level == "intermediate"
    assert p.travel_frequency == "weekly"
    assert p.tts_speed == 1.5


def test_user_profile_from_firestore():
    doc = {
        "vision_status": "low_vision",
        "blindness_onset": "acquired",
        "onset_age": 25,
        "has_guide_dog": True,
        "tts_speed": 2.0,
        "verbosity_preference": "detailed",
        "om_level": "advanced",
        "travel_frequency": "daily",
        "preferred_name": "Alex",
    }
    p = UserProfile.from_firestore(doc, user_id="user123")
    assert p.user_id == "user123"
    assert p.vision_status == "low_vision"
    assert p.blindness_onset == "acquired"
    assert p.onset_age == 25
    assert p.has_guide_dog is True
    assert p.tts_speed == 2.0
    assert p.verbosity_preference == "detailed"
    assert p.om_level == "advanced"
    assert p.travel_frequency == "daily"
    assert p.preferred_name == "Alex"


def test_gps_data_defaults():
    g = GPSData()
    assert g.lat == 0.0
    assert g.lng == 0.0
    assert g.accuracy == 0.0
    assert g.speed == 0.0
    assert g.altitude == 0.0
