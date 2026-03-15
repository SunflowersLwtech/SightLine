"""Tests for SightLine Emergency Help Tool.

All Google Maps API calls are mocked — no network access required.
"""

from unittest.mock import MagicMock, patch

import pytest

from tools.emergency import (
    _DEFAULT_EMERGENCY,
    EMERGENCY_FUNCTIONS,
    EMERGENCY_TOOL_DECLARATIONS,
    _get_emergency_numbers,
    get_emergency_help,
)

# ---------------------------------------------------------------------------
# Tests: Emergency number mapping
# ---------------------------------------------------------------------------


class TestEmergencyNumbers:
    def test_us_emergency_numbers(self):
        nums = _get_emergency_numbers("US")
        assert nums["general"] == "911"
        assert nums["police"] == "911"

    def test_uk_emergency_numbers(self):
        nums = _get_emergency_numbers("GB")
        assert nums["general"] == "999"

    def test_japan_emergency_numbers(self):
        nums = _get_emergency_numbers("JP")
        assert nums["police"] == "110"
        assert nums["ambulance"] == "119"

    def test_eu_member_uses_112(self):
        """EU member states not in the explicit map should use 112."""
        nums = _get_emergency_numbers("IT")
        assert nums["general"] == "112"

    def test_unknown_country_uses_default(self):
        nums = _get_emergency_numbers("ZZ")
        assert nums == _DEFAULT_EMERGENCY
        assert nums["general"] == "112"

    def test_case_insensitive(self):
        nums = _get_emergency_numbers("us")
        assert nums["general"] == "911"

    def test_germany_has_separate_police(self):
        nums = _get_emergency_numbers("DE")
        assert nums["police"] == "110"
        assert nums["fire"] == "112"


# ---------------------------------------------------------------------------
# Tests: get_emergency_help
# ---------------------------------------------------------------------------


class TestGetEmergencyHelp:
    @patch("tools.emergency._get_client")
    def test_basic_emergency_call(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Mock reverse geocode
        mock_client.reverse_geocode.return_value = [
            {
                "address_components": [
                    {"types": ["country"], "short_name": "US"},
                ],
            }
        ]

        # Mock places nearby
        mock_client.places_nearby.return_value = {
            "results": [
                {
                    "name": "City Hospital",
                    "vicinity": "123 Main St",
                    "geometry": {"location": {"lat": 37.7750, "lng": -122.4194}},
                    "opening_hours": {"open_now": True},
                },
            ]
        }

        result = get_emergency_help("medical", lat=37.7749, lng=-122.4194)

        assert result["success"] is True
        assert result["emergency_type"] == "medical"
        assert result["country_code"] == "US"
        assert result["emergency_numbers"]["general"] == "911"
        assert len(result["nearest_services"]) == 1
        assert result["nearest_services"][0]["name"] == "City Hospital"
        assert "summary" in result
        assert "911" in result["summary"]

    @patch("tools.emergency._get_client")
    def test_general_emergency_type_default(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.reverse_geocode.return_value = []
        mock_client.places_nearby.return_value = {"results": []}

        result = get_emergency_help(lat=0.0, lng=0.0)

        assert result["emergency_type"] == "general"

    @patch("tools.emergency._get_client")
    def test_invalid_type_defaults_to_general(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.reverse_geocode.return_value = []
        mock_client.places_nearby.return_value = {"results": []}

        result = get_emergency_help("earthquake", lat=0.0, lng=0.0)

        assert result["emergency_type"] == "general"

    @patch("tools.emergency._get_client")
    def test_geocode_failure_uses_default_country(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.reverse_geocode.side_effect = Exception("API error")
        mock_client.places_nearby.return_value = {"results": []}

        result = get_emergency_help("police", lat=0.0, lng=0.0)

        assert result["success"] is True
        assert result["country_code"] == "US"

    @patch("tools.emergency._get_client")
    def test_places_failure_returns_empty_services(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.reverse_geocode.return_value = []
        mock_client.places_nearby.side_effect = Exception("Places API error")

        result = get_emergency_help("fire", lat=35.6762, lng=139.6503)

        assert result["success"] is True
        assert result["nearest_services"] == []
        assert "summary" in result

    @patch("tools.emergency._get_client")
    def test_location_code_in_result(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.reverse_geocode.return_value = []
        mock_client.places_nearby.return_value = {"results": []}

        result = get_emergency_help(lat=37.7749, lng=-122.4194)

        assert result["user_location_code"] != ""
        assert result["user_coordinates"]["lat"] == 37.7749

    @patch("tools.emergency._get_client")
    def test_multiple_services_capped_at_3(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.reverse_geocode.return_value = []

        # Return 5 results
        mock_client.places_nearby.return_value = {
            "results": [
                {
                    "name": f"Hospital {i}",
                    "vicinity": f"{i} St",
                    "geometry": {"location": {"lat": 37.77 + i * 0.01, "lng": -122.42}},
                }
                for i in range(5)
            ]
        }

        result = get_emergency_help(lat=37.77, lng=-122.42)

        assert len(result["nearest_services"]) == 3


# ---------------------------------------------------------------------------
# Tests: Tool declarations and registry
# ---------------------------------------------------------------------------


class TestEmergencyRegistry:
    def test_declarations_structure(self):
        assert len(EMERGENCY_TOOL_DECLARATIONS) == 1
        decl = EMERGENCY_TOOL_DECLARATIONS[0]
        assert decl["name"] == "get_emergency_help"
        assert "emergency" in decl["description"].lower()
        params = decl["parameters"]["properties"]
        assert "emergency_type" in params
        assert "lat" in params
        assert "lng" in params

    def test_functions_map(self):
        assert "get_emergency_help" in EMERGENCY_FUNCTIONS
        assert EMERGENCY_FUNCTIONS["get_emergency_help"] is get_emergency_help

    def test_emergency_in_tool_registry(self):
        """Emergency tool should be registered in the main tool registry."""
        from tools import (
            ALL_FUNCTIONS,
            ALL_TOOL_CATEGORIES,
            CALLABLE_TOOL_NAMES,
        )

        assert "get_emergency_help" in CALLABLE_TOOL_NAMES
        assert "get_emergency_help" in ALL_FUNCTIONS
        assert ALL_TOOL_CATEGORIES["get_emergency_help"] == "emergency"

    def test_emergency_interrupt_behavior(self):
        """Emergency tool should always use INTERRUPT behavior."""
        from tools.tool_behavior import ToolBehavior, resolve_tool_behavior

        assert resolve_tool_behavior("get_emergency_help", lod=1) == ToolBehavior.INTERRUPT
        assert resolve_tool_behavior("get_emergency_help", lod=2) == ToolBehavior.INTERRUPT
        assert resolve_tool_behavior("get_emergency_help", lod=3) == ToolBehavior.INTERRUPT

    def test_emergency_gps_injection(self):
        """Emergency tool should have GPS injection configured."""
        from tools import ALL_TOOL_RUNTIME_METADATA

        meta = ALL_TOOL_RUNTIME_METADATA["get_emergency_help"]
        assert meta["gps_injection"] == "lat_lng"
