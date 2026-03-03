"""Tests for SightLine dynamic prompt builder."""

from lod.models import EphemeralContext, NarrativeSnapshot, SessionContext, UserProfile
from lod.prompt_builder import (
    _language_display,
    _build_persona_block,
    build_full_dynamic_prompt,
    build_lod_update_message,
)


def _default_args(lod: int = 2, **overrides):
    """Helper to build default args for build_lod_update_message."""
    kwargs = dict(
        lod=lod,
        ephemeral=EphemeralContext(),
        session=SessionContext(),
        profile=UserProfile.default(),
        reason="test",
    )
    kwargs.update(overrides)
    return kwargs


def test_lod_update_contains_lod_level():
    msg = build_lod_update_message(**_default_args(lod=1))
    assert "LOD 1" in msg


def test_lod_update_contains_persona():
    msg = build_lod_update_message(**_default_args())
    assert "User Profile" in msg


def test_lod_update_lod1_no_cot():
    msg = build_lod_update_message(**_default_args(lod=1))
    assert "<think>" not in msg


def test_lod_update_lod2_has_cot():
    msg = build_lod_update_message(**_default_args(lod=2))
    assert "internally reason" in msg
    # CoT should NOT contain raw sensor placeholders
    assert "{motion_state}" not in msg
    assert "{cadence" not in msg
    assert "{noise_db" not in msg


def test_lod_update_with_memories():
    msg = build_lod_update_message(**_default_args(memories=["test memory"]))
    assert "Relevant Memories" in msg
    assert "test memory" in msg


def test_lod_update_with_snapshot():
    session = SessionContext(
        narrative_snapshot=NarrativeSnapshot(
            task_type="menu_reading",
            progress="item 3",
            remaining=["item 4"],
        )
    )
    msg = build_lod_update_message(**_default_args(lod=2, session=session))
    assert "Resume Point" in msg


def test_full_prompt_contains_principles():
    msg = build_full_dynamic_prompt(
        lod=2,
        profile=UserProfile.default(),
        ephemeral_semantic="test context",
        session=SessionContext(),
    )
    assert "EXPERIENCE FIRST" in msg


def test_congenital_blind_no_color():
    profile = UserProfile.default()
    profile.blindness_onset = "congenital"
    profile.color_description = False
    msg = build_lod_update_message(**_default_args(profile=profile))
    assert "DISABLED" in msg


# ---------------------------------------------------------------------------
# Multi-language support tests
# ---------------------------------------------------------------------------


def test_language_display_known_codes():
    """Verify mapping for en-US, zh-CN, zh-TW, ja-JP, ko-KR — all English names."""
    assert _language_display("en-US") == "English"
    assert _language_display("zh-CN") == "Simplified Chinese"
    assert _language_display("zh-TW") == "Traditional Chinese"
    assert _language_display("ja-JP") == "Japanese"
    assert _language_display("ko-KR") == "Korean"


def test_language_display_unknown_code_passthrough():
    """Unknown locale codes should be returned as-is."""
    assert _language_display("fr-FR") == "fr-FR"
    assert _language_display("de-DE") == "de-DE"


def test_chinese_language_persona_block():
    """zh-CN profile should produce Google-template language constraint."""
    profile = UserProfile.default()
    profile.language = "zh-CN"
    block = _build_persona_block(profile)
    assert "Simplified Chinese" in block
    assert "RESPOND IN SIMPLIFIED CHINESE" in block
    assert "YOU MUST RESPOND UNMISTAKABLY IN SIMPLIFIED CHINESE" in block
    # Input language hint for native audio model
    assert "The user speaks Simplified Chinese" in block
    assert "Listen for Simplified Chinese" in block


def test_english_language_persona_block():
    """Default en-US profile should NOT inject redundant language constraint."""
    profile = UserProfile.default()
    block = _build_persona_block(profile)
    assert "Language: English" in block
    # Native audio auto-detects English; no explicit constraint needed
    assert "RESPOND IN" not in block


def test_full_prompt_chinese_language():
    """build_full_dynamic_prompt should include language constraint for zh-CN user."""
    profile = UserProfile.default()
    profile.language = "zh-CN"
    msg = build_full_dynamic_prompt(
        lod=2,
        profile=profile,
        ephemeral_semantic="test context",
        session=SessionContext(),
    )
    assert "Simplified Chinese" in msg
    assert "RESPOND IN SIMPLIFIED CHINESE" in msg


def test_persona_block_includes_preferred_name_when_present():
    profile = UserProfile.default()
    profile.preferred_name = "Liu Wei"
    block = _build_persona_block(profile)
    assert "Preferred name: Liu Wei" in block
    assert "address the user with this name when appropriate" in block


def test_persona_block_omits_preferred_name_when_empty():
    profile = UserProfile.default()
    profile.preferred_name = "   "
    block = _build_persona_block(profile)
    assert "Preferred name:" not in block
