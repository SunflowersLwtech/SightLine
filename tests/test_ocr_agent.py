"""Tests for SightLine OCR Sub-Agent.

All Gemini API calls are mocked — no network access required.
"""

import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.ocr_agent import (
    OCR_MODEL,
    _EMPTY_RESULT,
    _SAFETY_SYSTEM_PROMPT,
    extract_text,
)
from google.genai import types


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
    return _TINY_JPEG_B64


@pytest.fixture
def menu_response() -> dict:
    return {
        "text": "Grilled Salmon - $18.99\nCaesar Salad - $12.50\nIced Tea - $3.99",
        "text_type": "menu",
        "items": [
            "Grilled Salmon - $18.99",
            "Caesar Salad - $12.50",
            "Iced Tea - $3.99",
        ],
        "confidence": 0.95,
    }


@pytest.fixture
def sign_response() -> dict:
    return {
        "text": "EXIT\nEmergency Exit Only\nAlarm Will Sound",
        "text_type": "sign",
        "items": [
            "EXIT",
            "Emergency Exit Only",
            "Alarm Will Sound",
        ],
        "confidence": 0.98,
    }


@pytest.fixture
def document_response() -> dict:
    return {
        "text": "Patient: John Doe\nDate: 2024-01-15\nDiagnosis: Routine checkup",
        "text_type": "document",
        "items": [
            "Patient: John Doe",
            "Date: 2024-01-15",
            "Diagnosis: Routine checkup",
        ],
        "confidence": 0.87,
    }


@pytest.fixture
def label_response() -> dict:
    return {
        "text": "Ibuprofen 200mg\nTake 1-2 tablets every 4-6 hours",
        "text_type": "label",
        "items": [
            "Ibuprofen 200mg",
            "Take 1-2 tablets every 4-6 hours",
        ],
        "confidence": 0.91,
    }


@pytest.fixture
def no_text_response() -> dict:
    return {
        "text": "",
        "text_type": "unknown",
        "items": [],
        "confidence": 0.0,
    }


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_mock_response(result_dict: dict) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(result_dict)
    return mock_resp


# ---------------------------------------------------------------------------
# Tests: extract_text
# ---------------------------------------------------------------------------


class TestExtractText:
    @pytest.mark.asyncio
    async def test_menu_extraction(self, tiny_image, menu_response):
        mock_generate = AsyncMock(return_value=_make_mock_response(menu_response))

        with patch("agents.ocr_agent._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = mock_generate
            mock_get_client.return_value = mock_client

            result = await extract_text(tiny_image, context_hint="user is at a restaurant")

        assert result["text_type"] == "menu"
        assert len(result["items"]) == 3
        assert "$18.99" in result["items"][0]
        assert result["confidence"] == 0.95

        # Verify context hint was included in the user message
        call_kwargs = mock_generate.call_args
        contents = call_kwargs.kwargs["contents"]
        user_text = contents[0].parts[1].text
        assert "restaurant" in user_text

    @pytest.mark.asyncio
    async def test_sign_extraction(self, tiny_image, sign_response):
        mock_generate = AsyncMock(return_value=_make_mock_response(sign_response))

        with patch("agents.ocr_agent._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = mock_generate
            mock_get_client.return_value = mock_client

            result = await extract_text(tiny_image)

        assert result["text_type"] == "sign"
        assert "EXIT" in result["text"]
        assert result["confidence"] == 0.98

    @pytest.mark.asyncio
    async def test_document_extraction(self, tiny_image, document_response):
        mock_generate = AsyncMock(return_value=_make_mock_response(document_response))

        with patch("agents.ocr_agent._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = mock_generate
            mock_get_client.return_value = mock_client

            result = await extract_text(tiny_image)

        assert result["text_type"] == "document"
        assert "John Doe" in result["text"]

    @pytest.mark.asyncio
    async def test_label_extraction(self, tiny_image, label_response):
        mock_generate = AsyncMock(return_value=_make_mock_response(label_response))

        with patch("agents.ocr_agent._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = mock_generate
            mock_get_client.return_value = mock_client

            result = await extract_text(tiny_image, context_hint="reading a medicine bottle")

        assert result["text_type"] == "label"
        assert "Ibuprofen" in result["text"]

    @pytest.mark.asyncio
    async def test_no_text_in_image(self, tiny_image, no_text_response):
        mock_generate = AsyncMock(return_value=_make_mock_response(no_text_response))

        with patch("agents.ocr_agent._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = mock_generate
            mock_get_client.return_value = mock_client

            result = await extract_text(tiny_image)

        assert result["text"] == ""
        assert result["text_type"] == "unknown"
        assert result["items"] == []
        assert result["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_no_context_hint(self, tiny_image, sign_response):
        """Calling without context_hint should work fine."""
        mock_generate = AsyncMock(return_value=_make_mock_response(sign_response))

        with patch("agents.ocr_agent._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = mock_generate
            mock_get_client.return_value = mock_client

            result = await extract_text(tiny_image)

        assert result["text_type"] == "sign"
        # Verify no "Context:" in user message when no hint given
        call_kwargs = mock_generate.call_args
        contents = call_kwargs.kwargs["contents"]
        user_text = contents[0].parts[1].text
        assert "Context:" not in user_text

    @pytest.mark.asyncio
    async def test_uses_correct_model(self, tiny_image, menu_response):
        mock_generate = AsyncMock(return_value=_make_mock_response(menu_response))

        with patch("agents.ocr_agent._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = mock_generate
            mock_get_client.return_value = mock_client

            await extract_text(tiny_image)

        call_kwargs = mock_generate.call_args
        assert call_kwargs.kwargs["model"] == OCR_MODEL

    @pytest.mark.asyncio
    async def test_uses_medium_resolution(self, tiny_image, menu_response):
        mock_generate = AsyncMock(return_value=_make_mock_response(menu_response))

        with patch("agents.ocr_agent._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = mock_generate
            mock_get_client.return_value = mock_client

            await extract_text(tiny_image)

        config = mock_generate.call_args.kwargs["config"]
        assert config.media_resolution == types.MediaResolution.MEDIA_RESOLUTION_MEDIUM

    @pytest.mark.asyncio
    async def test_invalid_base64_returns_empty(self):
        result = await extract_text("not-valid-base64!!!")
        assert result == _EMPTY_RESULT

    @pytest.mark.asyncio
    async def test_api_error_returns_empty(self, tiny_image):
        mock_generate = AsyncMock(side_effect=RuntimeError("API error"))

        with patch("agents.ocr_agent._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = mock_generate
            mock_get_client.return_value = mock_client

            result = await extract_text(tiny_image)

        assert result == _EMPTY_RESULT

    @pytest.mark.asyncio
    async def test_empty_response_returns_empty(self, tiny_image):
        mock_resp = MagicMock()
        mock_resp.text = ""
        mock_generate = AsyncMock(return_value=mock_resp)

        with patch("agents.ocr_agent._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = mock_generate
            mock_get_client.return_value = mock_client

            result = await extract_text(tiny_image)

        assert result == _EMPTY_RESULT

    @pytest.mark.asyncio
    async def test_malformed_json_returns_empty(self, tiny_image):
        mock_resp = MagicMock()
        mock_resp.text = "{bad json"
        mock_generate = AsyncMock(return_value=mock_resp)

        with patch("agents.ocr_agent._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = mock_generate
            mock_get_client.return_value = mock_client

            result = await extract_text(tiny_image)

        assert result == _EMPTY_RESULT

    @pytest.mark.asyncio
    async def test_partial_response_gets_defaults(self, tiny_image):
        partial = {"text": "Hello World", "confidence": 0.8}
        mock_generate = AsyncMock(return_value=_make_mock_response(partial))

        with patch("agents.ocr_agent._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = mock_generate
            mock_get_client.return_value = mock_client

            result = await extract_text(tiny_image)

        assert result["text"] == "Hello World"
        assert result["confidence"] == 0.8
        assert result["text_type"] == "unknown"
        assert result["items"] == []

    @pytest.mark.asyncio
    async def test_safety_only_uses_low_resolution(self, tiny_image, sign_response):
        """When safety_only=True, MEDIA_RESOLUTION_LOW should be used."""
        mock_generate = AsyncMock(return_value=_make_mock_response(sign_response))

        with patch("agents.ocr_agent._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = mock_generate
            mock_get_client.return_value = mock_client

            await extract_text(tiny_image, safety_only=True)

        config = mock_generate.call_args.kwargs["config"]
        assert config.media_resolution == types.MediaResolution.MEDIA_RESOLUTION_LOW

    @pytest.mark.asyncio
    async def test_safety_only_uses_safety_prompt(self, tiny_image, sign_response):
        """When safety_only=True, the safety system prompt should be used."""
        mock_generate = AsyncMock(return_value=_make_mock_response(sign_response))

        with patch("agents.ocr_agent._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = mock_generate
            mock_get_client.return_value = mock_client

            await extract_text(tiny_image, safety_only=True)

        config = mock_generate.call_args.kwargs["config"]
        assert config.system_instruction == _SAFETY_SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_default_safety_only_false(self, tiny_image, menu_response):
        """Default behavior (safety_only=False) should use MEDIUM resolution."""
        mock_generate = AsyncMock(return_value=_make_mock_response(menu_response))

        with patch("agents.ocr_agent._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = mock_generate
            mock_get_client.return_value = mock_client

            await extract_text(tiny_image)

        config = mock_generate.call_args.kwargs["config"]
        assert config.media_resolution == types.MediaResolution.MEDIA_RESOLUTION_MEDIUM


# ---------------------------------------------------------------------------
# Tests: Module constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_ocr_model_is_flash(self):
        assert "flash" in OCR_MODEL.lower()

    def test_empty_result_structure(self):
        expected_keys = {"text", "text_type", "items", "confidence"}
        assert set(_EMPTY_RESULT.keys()) == expected_keys

    def test_empty_result_defaults(self):
        assert _EMPTY_RESULT["text"] == ""
        assert _EMPTY_RESULT["text_type"] == "unknown"
        assert _EMPTY_RESULT["items"] == []
        assert _EMPTY_RESULT["confidence"] == 0.0
