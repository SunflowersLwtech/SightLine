"""Tests for SightLine LOD decision engine."""

from lod.lod_engine import LODDecisionLog, decide_lod, should_speak
from lod.models import EphemeralContext, SessionContext, UserProfile


# =====================================================================
# decide_lod() — Rule 1: Motion state baseline (experience-driven)
# =====================================================================


def test_running_is_lod1(running_ephemeral, default_session, default_profile):
    lod, _ = decide_lod(running_ephemeral, default_session, default_profile)
    assert lod == 1


def test_high_cadence_is_lod1(default_session, default_profile):
    ctx = EphemeralContext(motion_state="walking", step_cadence=130)
    lod, _ = decide_lod(ctx, default_session, default_profile)
    assert lod == 1


def test_walking_is_lod2(walking_ephemeral, default_session, default_profile):
    # walking now always maps to LOD 2 (experience-driven)
    lod, _ = decide_lod(walking_ephemeral, default_session, default_profile)
    assert lod == 2


def test_walking_slow_is_lod2(default_session, default_profile):
    ctx = EphemeralContext(motion_state="walking", step_cadence=50)
    lod, _ = decide_lod(ctx, default_session, default_profile)
    assert lod == 2


def test_stationary_is_lod2_with_concise(stationary_ephemeral, default_session, default_profile):
    lod, _ = decide_lod(stationary_ephemeral, default_session, default_profile)
    assert lod == 2  # base LOD 3, concise -1 → LOD 2


def test_vehicle_is_lod2_with_concise(vehicle_ephemeral, default_session, default_profile):
    lod, _ = decide_lod(vehicle_ephemeral, default_session, default_profile)
    assert lod == 2  # base LOD 3, concise -1 → LOD 2


def test_cycling_is_lod1(default_session, default_profile):
    ctx = EphemeralContext(motion_state="cycling")
    lod, _ = decide_lod(ctx, default_session, default_profile)
    assert lod == 1


# =====================================================================
# decide_lod() — Rule 2: Ambient noise adjustment
# =====================================================================


def test_loud_noise_caps_lod1(default_session, default_profile):
    ctx = EphemeralContext(motion_state="stationary", ambient_noise_db=85)
    lod, log = decide_lod(ctx, default_session, default_profile)
    assert lod == 1
    assert any("noise" in r.lower() for r in log.triggered_rules)


def test_moderate_noise_no_effect(default_session, default_profile):
    ctx = EphemeralContext(motion_state="stationary", ambient_noise_db=65)
    lod, _ = decide_lod(ctx, default_session, default_profile)
    assert lod == 2  # base LOD 3, concise -1 → LOD 2; noise 65dB has no effect


# =====================================================================
# decide_lod() — Rule 3: Space transition boost
# =====================================================================


def test_space_transition_boosts_to_lod2(default_profile):
    ctx = EphemeralContext(motion_state="running", step_cadence=130)
    session = SessionContext(recent_space_transition=True)
    lod, log = decide_lod(ctx, session, default_profile)
    assert lod >= 2
    assert any("space_transition" in r for r in log.triggered_rules)


# =====================================================================
# decide_lod() — Rule 4: Verbosity preference
# =====================================================================


def test_concise_pref_decreases_lod_from_3(stationary_ephemeral, default_session, concise_profile):
    lod, log = decide_lod(stationary_ephemeral, default_session, concise_profile)
    assert lod == 2  # 3 - 1 = 2


def test_detailed_pref_increases_lod(default_session, detailed_profile):
    ctx = EphemeralContext(motion_state="walking", step_cadence=50)
    lod, _ = decide_lod(ctx, default_session, detailed_profile)
    assert lod == 3  # 2 + 1 = 3


def test_concise_pref_decreases_lod(stationary_ephemeral, default_session, default_profile):
    profile = UserProfile.default()
    profile.verbosity_preference = "concise"
    lod, log = decide_lod(stationary_ephemeral, default_session, profile)
    assert lod == 2  # 3 - 1 = 2
    assert any("concise_pref" in r for r in log.triggered_rules)


# =====================================================================
# decide_lod() — Rule 4: Concise collapse fix (regression tests)
# =====================================================================


def test_concise_does_not_collapse_slow_walk(default_session, concise_profile):
    """Slow walk (LOD 2) + concise should stay LOD 2, not collapse to LOD 1."""
    ctx = EphemeralContext(motion_state="walking", step_cadence=50)
    lod, log = decide_lod(ctx, default_session, concise_profile)
    assert lod == 2
    assert not any("concise_pref" in r for r in log.triggered_rules)


def test_concise_does_not_collapse_commute(default_session, concise_profile):
    """Stationary + morning_commute (LOD 2 after Rule1b) + concise -> stays LOD 2."""
    ctx = EphemeralContext(motion_state="stationary", time_context="morning_commute")
    lod, log = decide_lod(ctx, default_session, concise_profile)
    assert lod == 2
    assert not any("concise_pref" in r for r in log.triggered_rules)


def test_concise_still_reduces_stationary(default_session, concise_profile):
    """Stationary (LOD 3) + concise -> LOD 2 (correct reduction from LOD 3)."""
    ctx = EphemeralContext(motion_state="stationary")
    lod, log = decide_lod(ctx, default_session, concise_profile)
    assert lod == 2
    assert any("concise_pref" in r for r in log.triggered_rules)


def test_concise_walking_stays_lod2(default_session, concise_profile):
    """Walking (LOD 2) + concise -> stays LOD 2, concise doesn't reduce LOD 2."""
    ctx = EphemeralContext(motion_state="walking", step_cadence=80)
    lod, _ = decide_lod(ctx, default_session, concise_profile)
    assert lod == 2


def test_concise_does_not_reduce_below_3(default_session, concise_profile):
    """Concise only reduces from LOD >= 3, should not reduce LOD 2."""
    ctx = EphemeralContext(motion_state="walking", step_cadence=50)
    lod, log = decide_lod(ctx, default_session, concise_profile)
    assert lod == 2  # base LOD 2, concise doesn't reduce
    assert not any("concise_pref" in r for r in log.triggered_rules)


# =====================================================================
# decide_lod() — Rule 5: Advanced O&M
# =====================================================================


def test_advanced_daily_decreases_lod(stationary_ephemeral, default_session, advanced_daily_profile):
    lod, _ = decide_lod(stationary_ephemeral, default_session, advanced_daily_profile)
    assert lod == 1  # base LOD 3, concise -1 -> 2, advanced_daily -1 -> 1


# =====================================================================
# decide_lod() — Rule 6: User override
# =====================================================================


def test_user_detail_request_forces_lod3(walking_ephemeral, default_profile):
    session = SessionContext(user_requested_detail=True)
    lod, log = decide_lod(walking_ephemeral, session, default_profile)
    assert lod == 3
    assert any("user_requested_detail" in r for r in log.triggered_rules)


def test_user_stop_forces_lod1(stationary_ephemeral, default_profile):
    session = SessionContext(user_said_stop=True)
    lod, log = decide_lod(stationary_ephemeral, session, default_profile)
    assert lod == 1
    assert any("user_said_stop" in r for r in log.triggered_rules)


# =====================================================================
# decide_lod() — Gesture override
# =====================================================================


def test_gesture_lod_up(default_session, default_profile):
    ctx = EphemeralContext(motion_state="walking", step_cadence=50, user_gesture="lod_up")
    lod, log = decide_lod(ctx, default_session, default_profile)
    assert lod == 3  # base LOD 2 + 1 = 3
    assert any("lod_up" in r for r in log.triggered_rules)


def test_gesture_lod_down(default_session, default_profile):
    ctx = EphemeralContext(motion_state="walking", step_cadence=50, user_gesture="lod_down")
    lod, log = decide_lod(ctx, default_session, default_profile)
    assert lod == 1  # base LOD 2 - 1 = 1
    assert any("lod_down" in r for r in log.triggered_rules)


# =====================================================================
# LODDecisionLog
# =====================================================================


def test_decision_log_has_triggered_rules(walking_ephemeral, default_session, default_profile):
    _, log = decide_lod(walking_ephemeral, default_session, default_profile)
    assert len(log.triggered_rules) > 0


def test_decision_log_to_debug_dict(walking_ephemeral, default_session, default_profile):
    _, log = decide_lod(walking_ephemeral, default_session, default_profile)
    d = log.to_debug_dict()
    assert "lod" in d
    assert "prev" in d
    assert "reason" in d
    assert "rules" in d
    assert isinstance(d["rules"], list)


# =====================================================================
# should_speak()
# =====================================================================


def test_navigation_always_speaks_normal():
    assert should_speak("navigation", current_lod=2, step_cadence=50, ambient_noise_db=50) is True


def test_low_value_high_noise_no_speak():
    assert should_speak("atmosphere", current_lod=1, step_cadence=80, ambient_noise_db=85) is False


def test_face_recognition_normal_conditions():
    assert should_speak("face_recognition", current_lod=2, step_cadence=50, ambient_noise_db=50) is True


# =====================================================================
# decide_lod() — Gesture + voice interaction
# =====================================================================


def test_gesture_takes_priority_over_voice_detail(default_profile):
    """Gesture lod_down should NOT be undone by user_requested_detail."""
    ctx = EphemeralContext(motion_state="stationary", user_gesture="lod_down")
    session = SessionContext(user_requested_detail=True)
    lod, log = decide_lod(ctx, session, default_profile)
    # Stationary base LOD 3, concise -> 2, gesture lod_down -> 1 (not overridden)
    assert lod == 1
    assert any("lod_down" in r for r in log.triggered_rules)
    assert not any("user_requested_detail" in r for r in log.triggered_rules)


def test_gesture_takes_priority_over_voice_stop(default_profile):
    """Gesture lod_up should NOT be undone by user_said_stop."""
    ctx = EphemeralContext(motion_state="walking", step_cadence=50, user_gesture="lod_up")
    session = SessionContext(user_said_stop=True)
    lod, log = decide_lod(ctx, session, default_profile)
    # Walking slow base is LOD 2, gesture lod_up -> LOD 3 (not overridden to LOD 1)
    assert lod == 3
    assert any("lod_up" in r for r in log.triggered_rules)
    assert not any("user_said_stop" in r for r in log.triggered_rules)


# =====================================================================
# decide_lod() — Unknown gesture
# =====================================================================


def test_unknown_gesture_ignored(default_session, default_profile):
    """Unknown gesture strings should be silently ignored."""
    ctx = EphemeralContext(motion_state="stationary", user_gesture="triple_tap")
    lod, log = decide_lod(ctx, default_session, default_profile)
    assert lod == 2  # stationary LOD 3, concise -1 -> 2, no gesture effect
    assert not any("Gesture" in r for r in log.triggered_rules)


def test_whitespace_gesture_ignored(default_session, default_profile):
    """Whitespace-only gesture should be treated as None."""
    ctx = EphemeralContext(motion_state="stationary", user_gesture="  ")
    lod, log = decide_lod(ctx, default_session, default_profile)
    assert not any("Gesture" in r for r in log.triggered_rules)


# =====================================================================
# decide_lod() — Rule 1b: Time context
# =====================================================================


def test_morning_commute_reduces_lod(commute_ephemeral, default_session, default_profile):
    """morning_commute should reduce LOD by 1."""
    lod, log = decide_lod(commute_ephemeral, default_session, default_profile)
    # Stationary base LOD 3, morning_commute -> LOD 2
    assert lod == 2
    assert any("Rule1b:morning_commute" in r for r in log.triggered_rules)


def test_late_night_reduces_lod(default_session, default_profile):
    """late_night should reduce LOD by 1."""
    ctx = EphemeralContext(motion_state="stationary", time_context="late_night")
    lod, log = decide_lod(ctx, default_session, default_profile)
    assert lod == 2
    assert any("Rule1b:late_night" in r for r in log.triggered_rules)


def test_time_context_no_effect_at_lod1(default_session, default_profile):
    """Time context should NOT reduce below LOD 1."""
    ctx = EphemeralContext(motion_state="running", time_context="morning_commute")
    lod, _ = decide_lod(ctx, default_session, default_profile)
    assert lod == 1


# =====================================================================
# decide_lod() — step_cadence boundary
# =====================================================================


def test_step_cadence_120_triggers_lod1(default_session, default_profile):
    """step_cadence=120 (boundary) should trigger LOD 1."""
    ctx = EphemeralContext(motion_state="walking", step_cadence=120)
    lod, log = decide_lod(ctx, default_session, default_profile)
    assert lod == 1


# =====================================================================
# decide_lod() — Default ambient noise
# =====================================================================


def test_default_ambient_noise_is_70():
    """EphemeralContext default ambient_noise_db should be 70.0."""
    ctx = EphemeralContext()
    assert ctx.ambient_noise_db == 70.0


# =====================================================================
# decide_lod() — Rule 5b: Familiarity adjustment
# =====================================================================


def test_high_familiarity_reduces_lod(default_profile):
    """High familiarity (>0.8) should reduce LOD by 1 when not at LOD 1."""
    ctx = EphemeralContext(motion_state="stationary")
    session = SessionContext(familiarity_score=0.9)
    lod, log = decide_lod(ctx, session, default_profile)
    # Stationary base LOD 3, concise -1 -> 2, familiar -1 -> 1
    assert lod == 1
    assert any("Rule5b:familiar" in r for r in log.triggered_rules)


def test_low_familiarity_increases_lod(default_profile):
    """Low familiarity (<0.2) should increase LOD by 1 when not at LOD 3."""
    ctx = EphemeralContext(motion_state="walking", step_cadence=70)
    session = SessionContext(familiarity_score=0.1)
    lod, log = decide_lod(ctx, session, default_profile)
    # Walking base LOD 2, unfamiliar +1 -> 3
    assert lod == 3
    assert any("Rule5b:unfamiliar" in r for r in log.triggered_rules)


def test_default_familiarity_no_effect(default_session, default_profile):
    """Default familiarity (0.5) should not trigger Rule 5b."""
    ctx = EphemeralContext(motion_state="stationary")
    lod, log = decide_lod(ctx, default_session, default_profile)
    assert not any("Rule5b" in r for r in log.triggered_rules)


def test_high_familiarity_cannot_reduce_below_lod1(default_profile):
    """High familiarity should not reduce LOD below 1."""
    ctx = EphemeralContext(motion_state="running")  # base LOD 1
    session = SessionContext(familiarity_score=0.95)
    lod, log = decide_lod(ctx, session, default_profile)
    assert lod == 1


# =====================================================================
# decide_lod() — Rule 1c: Watch stability
# =====================================================================


def test_unstable_reduces_lod(default_session, default_profile):
    """Unstable gait (<0.4) should reduce LOD by 1 when not at LOD 1."""
    ctx = EphemeralContext(motion_state="walking", step_cadence=50, watch_stability_score=0.3)
    lod, log = decide_lod(ctx, default_session, default_profile)
    assert lod == 1  # base LOD 2, unstable -1 -> 1
    assert any("unstable" in r for r in log.triggered_rules)


def test_stable_no_effect(default_session, default_profile):
    """Stable gait (>0.4) should not trigger stability rule."""
    ctx = EphemeralContext(motion_state="walking", step_cadence=50, watch_stability_score=0.85)
    lod, log = decide_lod(ctx, default_session, default_profile)
    assert lod == 2
    assert not any("unstable" in r for r in log.triggered_rules)
