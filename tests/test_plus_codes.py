"""Tests for SightLine Plus Codes tool.

All operations are offline — no API key or mocking needed.
"""

from __future__ import annotations

import pytest

from tools.plus_codes import (
    PLUS_CODES_FUNCTIONS,
    PLUS_CODES_TOOL_DECLARATIONS,
    convert_to_plus_code,
    resolve_plus_code,
)


class TestConvertToPlusCode:
    """Test GPS → Plus Code encoding."""

    def test_san_francisco(self):
        result = convert_to_plus_code(37.7749, -122.4194)
        assert result["success"] is True
        assert "+" in result["code"]
        assert result["latitude"] == 37.7749
        assert result["longitude"] == -122.4194

    def test_equator_prime_meridian(self):
        result = convert_to_plus_code(0.0, 0.0)
        assert result["success"] is True
        assert "+" in result["code"]

    def test_negative_coords(self):
        # Sydney, Australia
        result = convert_to_plus_code(-33.8688, 151.2093)
        assert result["success"] is True


class TestResolvePlusCode:
    """Test Plus Code → GPS decoding."""

    def test_valid_full_code(self):
        # First encode, then decode
        encoded = convert_to_plus_code(37.7749, -122.4194)
        result = resolve_plus_code(encoded["code"])

        assert result["success"] is True
        assert abs(result["latitude"] - 37.7749) < 0.001
        assert abs(result["longitude"] - (-122.4194)) < 0.001

    def test_invalid_code(self):
        result = resolve_plus_code("INVALID_CODE")
        assert result["success"] is False
        assert "Invalid" in result["error"]

    def test_short_code_rejected(self):
        # Short codes need a reference location which we don't support
        result = resolve_plus_code("QJQ5+JQ")
        assert result["success"] is False
        assert "Short Plus Code" in result["error"] or "Invalid" in result["error"]

    def test_whitespace_stripped(self):
        encoded = convert_to_plus_code(40.7128, -74.0060)
        result = resolve_plus_code(f"  {encoded['code']}  ")
        assert result["success"] is True


class TestRoundtrip:
    """Test encode → decode roundtrip accuracy."""

    @pytest.mark.parametrize(
        "lat,lng",
        [
            (37.7749, -122.4194),   # San Francisco
            (40.7128, -74.0060),    # NYC
            (51.5074, -0.1278),     # London
            (35.6762, 139.6503),    # Tokyo
            (-33.8688, 151.2093),   # Sydney
            (0.0, 0.0),            # Null Island
        ],
    )
    def test_roundtrip_accuracy(self, lat, lng):
        encoded = convert_to_plus_code(lat, lng)
        decoded = resolve_plus_code(encoded["code"])

        assert decoded["success"] is True
        # codeLength=11 gives ~3m precision
        assert abs(decoded["latitude"] - lat) < 0.001
        assert abs(decoded["longitude"] - lng) < 0.001


class TestDeclarations:
    """Verify ADK tool declarations are well-formed."""

    def test_all_functions_have_declarations(self):
        declared_names = {d["name"] for d in PLUS_CODES_TOOL_DECLARATIONS}
        func_names = set(PLUS_CODES_FUNCTIONS.keys())
        assert declared_names == func_names

    def test_declarations_have_required_fields(self):
        for decl in PLUS_CODES_TOOL_DECLARATIONS:
            assert "name" in decl
            assert "description" in decl
            assert "parameters" in decl

    def test_functions_are_callable(self):
        for name, func in PLUS_CODES_FUNCTIONS.items():
            assert callable(func)
