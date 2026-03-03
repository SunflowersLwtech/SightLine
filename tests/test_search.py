"""Tests for SightLine Google Search grounding tool.

All Gemini API calls are mocked — no real API key needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tools.search import (
    SEARCH_FUNCTIONS,
    SEARCH_TOOL_DECLARATIONS,
    _extract_sources,
    google_search,
)


# ---------------------------------------------------------------------------
# Mock response builders
# ---------------------------------------------------------------------------


def _make_mock_response(
    text: str = "The answer is 42.",
    has_grounding: bool = True,
    grounding_chunks: list | None = None,
) -> MagicMock:
    """Build a mock Gemini GenerateContentResponse."""
    response = MagicMock()

    # Content parts
    part = MagicMock()
    part.text = text
    content = MagicMock()
    content.parts = [part]

    candidate = MagicMock()
    candidate.content = content

    # Grounding metadata
    if has_grounding:
        grounding_metadata = MagicMock()
        if grounding_chunks is None:
            web1 = MagicMock()
            web1.title = "Wikipedia"
            web1.uri = "https://en.wikipedia.org/wiki/42"
            chunk1 = MagicMock()
            chunk1.web = web1

            web2 = MagicMock()
            web2.title = "Stack Overflow"
            web2.uri = "https://stackoverflow.com/q/42"
            chunk2 = MagicMock()
            chunk2.web = web2

            grounding_metadata.grounding_chunks = [chunk1, chunk2]
        else:
            grounding_metadata.grounding_chunks = grounding_chunks
        grounding_metadata.search_entry_point = None
        candidate.grounding_metadata = grounding_metadata
    else:
        candidate.grounding_metadata = None

    response.candidates = [candidate]
    return response


def _make_empty_response() -> MagicMock:
    """Build a mock response with no candidates."""
    response = MagicMock()
    response.candidates = []
    return response


def _make_search_entry_response(rendered: str) -> MagicMock:
    """Build a mock response with only search_entry_point (no chunks)."""
    response = MagicMock()

    part = MagicMock()
    part.text = "Some answer"
    content = MagicMock()
    content.parts = [part]

    candidate = MagicMock()
    candidate.content = content

    grounding_metadata = MagicMock()
    grounding_metadata.grounding_chunks = None

    search_entry = MagicMock()
    search_entry.rendered_content = rendered
    grounding_metadata.search_entry_point = search_entry

    candidate.grounding_metadata = grounding_metadata
    response.candidates = [candidate]
    return response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_genai():
    """Provide a mocked Genai client."""
    with patch("tools.search._get_client") as mock_get:
        client = MagicMock()
        mock_get.return_value = client
        yield client


# ---------------------------------------------------------------------------
# google_search tests
# ---------------------------------------------------------------------------


class TestGoogleSearch:
    """Test google_search with mocked Gemini API."""

    def test_success_with_grounding(self, mock_genai):
        mock_genai.models.generate_content.return_value = _make_mock_response()

        result = google_search("What is the meaning of life?")

        assert result["success"] is True
        assert "42" in result["answer"]
        assert len(result["sources"]) == 2
        assert result["sources"][0]["title"] == "Wikipedia"
        assert result["confidence"] == 0.9

    def test_success_without_grounding(self, mock_genai):
        mock_genai.models.generate_content.return_value = _make_mock_response(
            text="I think so.",
            has_grounding=False,
        )

        result = google_search("Is water wet?")

        assert result["success"] is True
        assert result["answer"] == "I think so."
        assert result["sources"] == []
        assert result["confidence"] == 0.5

    def test_empty_response(self, mock_genai):
        mock_genai.models.generate_content.return_value = _make_empty_response()

        result = google_search("Something obscure")

        assert result["success"] is True
        assert result["answer"] == ""
        assert result["sources"] == []

    def test_api_error(self, mock_genai):
        mock_genai.models.generate_content.side_effect = Exception("Rate limited")

        result = google_search("test query")

        assert result["success"] is False
        assert "Rate limited" in result["answer"]
        assert result["sources"] == []
        assert result["confidence"] == 0.0

    def test_multipart_response(self, mock_genai):
        response = MagicMock()
        part1 = MagicMock()
        part1.text = "First part. "
        part2 = MagicMock()
        part2.text = "Second part."
        content = MagicMock()
        content.parts = [part1, part2]
        candidate = MagicMock()
        candidate.content = content
        candidate.grounding_metadata = None
        response.candidates = [candidate]

        mock_genai.models.generate_content.return_value = response

        result = google_search("multi part query")

        assert result["success"] is True
        assert result["answer"] == "First part. Second part."

    def test_search_entry_point_fallback(self, mock_genai):
        mock_genai.models.generate_content.return_value = (
            _make_search_entry_response("Search results for: test")
        )

        result = google_search("test")

        assert result["success"] is True
        assert len(result["sources"]) == 1
        assert result["sources"][0]["title"] == "Google Search"


# ---------------------------------------------------------------------------
# _extract_sources tests
# ---------------------------------------------------------------------------


class TestExtractSources:
    """Test source extraction from various response formats."""

    def test_with_grounding_chunks(self):
        response = _make_mock_response()
        sources = _extract_sources(response)

        assert len(sources) == 2
        assert sources[0]["url"] == "https://en.wikipedia.org/wiki/42"

    def test_without_grounding(self):
        response = _make_mock_response(has_grounding=False)
        sources = _extract_sources(response)

        assert sources == []

    def test_empty_candidates(self):
        response = _make_empty_response()
        sources = _extract_sources(response)

        assert sources == []

    def test_no_web_in_chunk(self):
        chunk = MagicMock()
        chunk.web = None
        response = _make_mock_response(grounding_chunks=[chunk])
        sources = _extract_sources(response)

        assert sources == []

    def test_search_entry_fallback(self):
        response = _make_search_entry_response("Rendered HTML content here")
        sources = _extract_sources(response)

        assert len(sources) == 1
        assert sources[0]["snippet"] == "Rendered HTML content here"


# ---------------------------------------------------------------------------
# Declaration / registration tests
# ---------------------------------------------------------------------------


class TestDeclarations:
    """Verify ADK tool declarations are well-formed."""

    def test_all_functions_have_declarations(self):
        declared_names = {d["name"] for d in SEARCH_TOOL_DECLARATIONS}
        func_names = set(SEARCH_FUNCTIONS.keys())
        assert declared_names == func_names

    def test_declaration_fields(self):
        decl = SEARCH_TOOL_DECLARATIONS[0]
        assert decl["name"] == "google_search"
        assert "description" in decl
        assert "parameters" in decl
        assert "query" in decl["parameters"]["properties"]
        assert "query" in decl["parameters"]["required"]

    def test_function_is_callable(self):
        assert callable(SEARCH_FUNCTIONS["google_search"])
