"""Tests for SightLine Vision Sub-Agent.

All Gemini API calls are mocked — no network access required.
"""

import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.vision_agent import (
    VISION_MODEL,
    _EMPTY_RESULT,
    _MEDIA_RESOLUTION_BY_LOD,
    _SYSTEM_PROMPTS,
    _build_context_user_message,
    analyze_scene,
)
from google.genai import types


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# 1x1 red JPEG pixel (valid base64-encoded JPEG)
_TINY_JPEG_B64 = base64.b64encode(
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
    b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
    b"\x1f\x1e\x1d\x1a\x1c\x1c $.\x27 \",.+\x1c\x1c(7),01444\x1f\x27"
    b"9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
    b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08"
    b"\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03"
    b"\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12"
    b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd2\x8a(\x03\xff\xd9"
).decode()


@pytest.fixture
def tiny_image() -> str:
    """A minimal valid-ish JPEG encoded as base64."""
    return _TINY_JPEG_B64


@pytest.fixture
def sample_lod1_response() -> dict:
    return {
        "safety_warnings": ["Stairs descending at 12 o'clock, 3 meters"],
        "navigation_info": {"entrances": [], "paths": [], "landmarks": []},
        "scene_description": "Staircase directly ahead.",
        "detected_text": None,
        "people_count": 0,
        "confidence": 0.92,
    }


@pytest.fixture
def sample_lod2_response() -> dict:
    return {
        "safety_warnings": [],
        "navigation_info": {
            "entrances": ["Glass door at 1 o'clock, 5 meters"],
            "paths": ["Clear corridor ahead"],
            "landmarks": ["Reception desk at 10 o'clock"],
        },
        "scene_description": "Office lobby with glass entrance and reception area.",
        "detected_text": "Welcome to TechCorp",
        "people_count": 3,
        "confidence": 0.88,
    }


@pytest.fixture
def sample_lod3_response() -> dict:
    return {
        "safety_warnings": [],
        "navigation_info": {
            "entrances": ["Main entrance at 12 o'clock"],
            "paths": ["Wide sidewalk continuing ahead"],
            "landmarks": ["Bus stop bench at 3 o'clock"],
        },
        "scene_description": (
            "A sunny downtown street with a wide sidewalk. Two people are "
            "walking toward you at 11 o'clock. A coffee shop with outdoor "
            "seating is at 2 o'clock. The traffic light ahead is green."
        ),
        "detected_text": "Cafe Luna\nOpen 7am-9pm",
        "people_count": 5,
        "confidence": 0.95,
    }


# ---------------------------------------------------------------------------
# Helper to build a mock response
# ---------------------------------------------------------------------------


def _make_mock_response(result_dict: dict) -> MagicMock:
    """Create a mock Gemini response with .text returning JSON."""
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(result_dict)
    return mock_resp


# ---------------------------------------------------------------------------
# Tests: _build_context_user_message
# ---------------------------------------------------------------------------


class TestBuildContextMessage:
    def test_minimal(self):
        msg = _build_context_user_message(1, {})
        assert "LOD level 1" in msg

    def test_with_space_type(self):
        msg = _build_context_user_message(2, {"space_type": "indoor"})
        assert "indoor" in msg

    def test_unknown_space_type_excluded(self):
        msg = _build_context_user_message(2, {"space_type": "unknown"})
        assert "unknown" not in msg.split("LOD level")[1]

    def test_with_all_context(self):
        ctx = {
            "space_type": "outdoor",
            "trip_purpose": "going to interview",
            "active_task": "reading menu",
            "motion_state": "walking",
        }
        msg = _build_context_user_message(3, ctx)
        assert "outdoor" in msg
        assert "going to interview" in msg
        assert "reading menu" in msg
        assert "walking" in msg


# ---------------------------------------------------------------------------
# Tests: analyze_scene
# ---------------------------------------------------------------------------


class TestAnalyzeScene:
    @pytest.mark.asyncio
    async def test_lod1_safety_only(self, tiny_image, sample_lod1_response):
        mock_generate = AsyncMock(return_value=_make_mock_response(sample_lod1_response))

        with patch("agents.vision_agent._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = mock_generate
            mock_get_client.return_value = mock_client

            result = await analyze_scene(tiny_image, lod=1)

        assert result["safety_warnings"] == ["Stairs descending at 12 o'clock, 3 meters"]
        assert result["people_count"] == 0
        assert result["confidence"] == 0.92

        # Verify correct model and media resolution used
        call_kwargs = mock_generate.call_args
        assert call_kwargs.kwargs["model"] == VISION_MODEL
        config = call_kwargs.kwargs["config"]
        assert config.media_resolution == types.MediaResolution.MEDIA_RESOLUTION_LOW

    @pytest.mark.asyncio
    async def test_lod2_navigation(self, tiny_image, sample_lod2_response):
        mock_generate = AsyncMock(return_value=_make_mock_response(sample_lod2_response))

        with patch("agents.vision_agent._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = mock_generate
            mock_get_client.return_value = mock_client

            result = await analyze_scene(tiny_image, lod=2, session_context={"space_type": "indoor"})

        assert len(result["navigation_info"]["entrances"]) == 1
        assert result["people_count"] == 3
        assert result["detected_text"] == "Welcome to TechCorp"

        config = mock_generate.call_args.kwargs["config"]
        assert config.media_resolution == types.MediaResolution.MEDIA_RESOLUTION_MEDIUM

    @pytest.mark.asyncio
    async def test_lod3_full_narrative(self, tiny_image, sample_lod3_response):
        mock_generate = AsyncMock(return_value=_make_mock_response(sample_lod3_response))

        with patch("agents.vision_agent._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = mock_generate
            mock_get_client.return_value = mock_client

            result = await analyze_scene(tiny_image, lod=3)

        assert "coffee shop" in result["scene_description"]
        assert result["people_count"] == 5
        assert "Cafe Luna" in result["detected_text"]

        config = mock_generate.call_args.kwargs["config"]
        assert config.media_resolution == types.MediaResolution.MEDIA_RESOLUTION_HIGH

    @pytest.mark.asyncio
    async def test_lod_clamping_below(self, tiny_image, sample_lod1_response):
        """LOD values below 1 should be clamped to 1."""
        mock_generate = AsyncMock(return_value=_make_mock_response(sample_lod1_response))

        with patch("agents.vision_agent._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = mock_generate
            mock_get_client.return_value = mock_client

            result = await analyze_scene(tiny_image, lod=0)

        assert result["confidence"] == 0.92
        config = mock_generate.call_args.kwargs["config"]
        assert config.media_resolution == types.MediaResolution.MEDIA_RESOLUTION_LOW

    @pytest.mark.asyncio
    async def test_lod_clamping_above(self, tiny_image, sample_lod3_response):
        """LOD values above 3 should be clamped to 3."""
        mock_generate = AsyncMock(return_value=_make_mock_response(sample_lod3_response))

        with patch("agents.vision_agent._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = mock_generate
            mock_get_client.return_value = mock_client

            result = await analyze_scene(tiny_image, lod=5)

        config = mock_generate.call_args.kwargs["config"]
        assert config.media_resolution == types.MediaResolution.MEDIA_RESOLUTION_HIGH

    @pytest.mark.asyncio
    async def test_invalid_base64_returns_empty(self):
        """Invalid base64 should return empty result, not raise."""
        result = await analyze_scene("not-valid-base64!!!", lod=2)
        assert result == _EMPTY_RESULT

    @pytest.mark.asyncio
    async def test_api_error_returns_empty(self, tiny_image):
        """Gemini API errors should return empty result, not raise."""
        mock_generate = AsyncMock(side_effect=RuntimeError("API unavailable"))

        with patch("agents.vision_agent._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = mock_generate
            mock_get_client.return_value = mock_client

            result = await analyze_scene(tiny_image, lod=2)

        assert result == _EMPTY_RESULT

    @pytest.mark.asyncio
    async def test_empty_model_response_returns_empty(self, tiny_image):
        """Empty model response should return empty result."""
        mock_resp = MagicMock()
        mock_resp.text = ""
        mock_generate = AsyncMock(return_value=mock_resp)

        with patch("agents.vision_agent._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = mock_generate
            mock_get_client.return_value = mock_client

            result = await analyze_scene(tiny_image, lod=1)

        assert result == _EMPTY_RESULT

    @pytest.mark.asyncio
    async def test_malformed_json_returns_empty(self, tiny_image):
        """Malformed JSON from model should return empty result."""
        mock_resp = MagicMock()
        mock_resp.text = "this is not json {{"
        mock_generate = AsyncMock(return_value=mock_resp)

        with patch("agents.vision_agent._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = mock_generate
            mock_get_client.return_value = mock_client

            result = await analyze_scene(tiny_image, lod=2)

        assert result == _EMPTY_RESULT

    @pytest.mark.asyncio
    async def test_partial_response_gets_defaults(self, tiny_image):
        """Missing keys in the model response should get default values."""
        partial = {"safety_warnings": ["Watch out!"], "confidence": 0.7}
        mock_generate = AsyncMock(return_value=_make_mock_response(partial))

        with patch("agents.vision_agent._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = mock_generate
            mock_get_client.return_value = mock_client

            result = await analyze_scene(tiny_image, lod=1)

        assert result["safety_warnings"] == ["Watch out!"]
        assert result["confidence"] == 0.7
        # Missing keys should get defaults
        assert result["scene_description"] == ""
        assert result["people_count"] == 0
        assert result["detected_text"] is None

    @pytest.mark.asyncio
    async def test_none_session_context(self, tiny_image, sample_lod2_response):
        """None session_context should be handled gracefully."""
        mock_generate = AsyncMock(return_value=_make_mock_response(sample_lod2_response))

        with patch("agents.vision_agent._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = mock_generate
            mock_get_client.return_value = mock_client

            result = await analyze_scene(tiny_image, lod=2, session_context=None)

        assert result["people_count"] == 3

    @pytest.mark.asyncio
    async def test_system_prompt_matches_lod(self, tiny_image, sample_lod1_response):
        """Verify the correct system prompt is sent for each LOD level."""
        for lod in (1, 2, 3):
            mock_generate = AsyncMock(return_value=_make_mock_response(sample_lod1_response))

            with patch("agents.vision_agent._get_client") as mock_get_client:
                mock_client = MagicMock()
                mock_client.aio.models.generate_content = mock_generate
                mock_get_client.return_value = mock_client

                await analyze_scene(tiny_image, lod=lod)

            config = mock_generate.call_args.kwargs["config"]
            assert config.system_instruction == _SYSTEM_PROMPTS[lod]


# ---------------------------------------------------------------------------
# Tests: Module constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_media_resolution_mapping(self):
        assert _MEDIA_RESOLUTION_BY_LOD[1] == types.MediaResolution.MEDIA_RESOLUTION_LOW
        assert _MEDIA_RESOLUTION_BY_LOD[2] == types.MediaResolution.MEDIA_RESOLUTION_MEDIUM
        assert _MEDIA_RESOLUTION_BY_LOD[3] == types.MediaResolution.MEDIA_RESOLUTION_HIGH

    def test_system_prompts_exist_for_all_lods(self):
        for lod in (1, 2, 3):
            assert lod in _SYSTEM_PROMPTS
            assert len(_SYSTEM_PROMPTS[lod]) > 50

    def test_lod1_prompt_safety_focused(self):
        prompt = _SYSTEM_PROMPTS[1]
        assert "hazard" in prompt.lower() or "safety" in prompt.lower()
        assert "atmosphere" not in prompt.lower() or "NOT" in prompt

    def test_lod3_prompt_comprehensive(self):
        prompt = _SYSTEM_PROMPTS[3]
        assert "atmosphere" in prompt.lower() or "comprehensive" in prompt.lower()

    def test_empty_result_has_all_keys(self):
        expected_keys = {
            "safety_warnings",
            "navigation_info",
            "scene_description",
            "detected_text",
            "people_count",
            "confidence",
        }
        assert set(_EMPTY_RESULT.keys()) == expected_keys
