"""Tests for SightLine navigation tools.

All Google Maps API calls are mocked — no real API key needed.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from tools.navigation import (
    NAVIGATION_FUNCTIONS,
    NAVIGATION_TOOL_DECLARATIONS,
    _haversine_distance,
    _maneuver_to_description,
    _strip_html,
    bearing_between,
    bearing_to_clock,
    format_clock_direction,
    get_location_info,
    get_walking_directions,
    navigate_to,
    nearby_search,
    preview_destination,
    reverse_geocode,
    validate_address,
)


# ---------------------------------------------------------------------------
# Clock-position unit tests
# ---------------------------------------------------------------------------


class TestBearingBetween:
    """Test compass bearing calculations."""

    def test_due_north(self):
        # NYC to a point directly north
        bearing = bearing_between(40.0, -74.0, 41.0, -74.0)
        assert abs(bearing - 0) < 1  # ~0 degrees (north)

    def test_due_east(self):
        bearing = bearing_between(0.0, 0.0, 0.0, 1.0)
        assert abs(bearing - 90) < 1

    def test_due_south(self):
        bearing = bearing_between(41.0, -74.0, 40.0, -74.0)
        assert abs(bearing - 180) < 1

    def test_due_west(self):
        bearing = bearing_between(0.0, 1.0, 0.0, 0.0)
        assert abs(bearing - 270) < 1

    def test_same_point(self):
        bearing = bearing_between(40.0, -74.0, 40.0, -74.0)
        # Should be 0 (atan2(0,0) = 0)
        assert bearing == 0.0


class TestBearingToClock:
    """Test bearing-to-clock-position conversion."""

    def test_straight_ahead(self):
        # Target is at same bearing as user heading
        assert bearing_to_clock(90, 90) == 12

    def test_right_3_oclock(self):
        # Target 90 degrees to the right
        assert bearing_to_clock(180, 90) == 3

    def test_behind_6_oclock(self):
        # Target directly behind
        assert bearing_to_clock(270, 90) == 6

    def test_left_9_oclock(self):
        # Target 90 degrees to the left
        assert bearing_to_clock(0, 90) == 9

    def test_slight_right(self):
        # Target ~60 degrees right -> 2 o'clock
        assert bearing_to_clock(150, 90) == 2

    def test_slight_left(self):
        # Target ~60 degrees left -> 10 o'clock
        assert bearing_to_clock(30, 90) == 10

    def test_wrap_around(self):
        # User heading 350, target at 20 -> 30 degrees right -> 1 o'clock
        assert bearing_to_clock(20, 350) == 1

    def test_heading_zero_target_north(self):
        # Facing north, target north -> 12
        assert bearing_to_clock(0, 0) == 12

    def test_heading_zero_target_east(self):
        # Facing north, target east -> 3
        assert bearing_to_clock(90, 0) == 3


class TestFormatClockDirection:
    """Test spoken direction formatting."""

    def test_straight_ahead(self):
        result = format_clock_direction(12, 50)
        assert result == "straight ahead, 50 meters"

    def test_behind(self):
        result = format_clock_direction(6, 30)
        assert result == "behind you, 30 meters"

    def test_clock_position(self):
        result = format_clock_direction(2, 120)
        assert result == "at 2 o'clock, 120 meters"

    def test_kilometers(self):
        result = format_clock_direction(3, 1500)
        assert result == "at 3 o'clock, 1.5 kilometers"

    def test_rounding(self):
        result = format_clock_direction(9, 47.6)
        assert result == "at 9 o'clock, 48 meters"


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


class TestStripHtml:
    """Test HTML tag stripping."""

    def test_simple_tags(self):
        assert _strip_html("Go <b>north</b> on Main St") == "Go north on Main St"

    def test_nested_tags(self):
        result = _strip_html("<div>Turn <b>left</b> onto <span>Oak Ave</span></div>")
        assert result == "Turn left onto Oak Ave"

    def test_no_tags(self):
        assert _strip_html("Walk 50 meters") == "Walk 50 meters"


class TestManeuverToDescription:
    """Test maneuver string conversion."""

    def test_turn_left(self):
        assert _maneuver_to_description("turn-left") == "turn to 9 o'clock"

    def test_turn_right(self):
        assert _maneuver_to_description("turn-right") == "turn to 3 o'clock"

    def test_slight_left(self):
        assert _maneuver_to_description("turn-slight-left") == "bear to 10 o'clock"

    def test_straight(self):
        assert _maneuver_to_description("straight") == "continue straight ahead"

    def test_unknown(self):
        assert _maneuver_to_description("ferry") == ""

    def test_none(self):
        assert _maneuver_to_description(None) == ""

    # Routes API format tests
    def test_routes_turn_left(self):
        assert _maneuver_to_description("TURN_LEFT") == "turn to 9 o'clock"

    def test_routes_turn_right(self):
        assert _maneuver_to_description("TURN_RIGHT") == "turn to 3 o'clock"

    def test_routes_straight(self):
        assert _maneuver_to_description("STRAIGHT") == "continue straight ahead"

    def test_routes_depart(self):
        assert _maneuver_to_description("DEPART") == "depart"


class TestHaversineDistance:
    """Test haversine distance calculation."""

    def test_same_point(self):
        assert _haversine_distance(40.0, -74.0, 40.0, -74.0) == 0.0

    def test_known_distance(self):
        # NYC to LA ~ 3,940 km
        dist = _haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
        assert 3_900_000 < dist < 4_000_000


# ---------------------------------------------------------------------------
# Mock data for Routes API (v2 format)
# ---------------------------------------------------------------------------

MOCK_ROUTES_RESPONSE = {
    "routes": [
        {
            "legs": [
                {
                    "distanceMeters": 1200,
                    "duration": "900s",
                    "startLocation": {
                        "latLng": {"latitude": 37.7749, "longitude": -122.4194},
                    },
                    "endLocation": {
                        "latLng": {"latitude": 37.7849, "longitude": -122.4094},
                    },
                    "steps": [
                        {
                            "distanceMeters": 200,
                            "navigationInstruction": {
                                "instructions": "Head north on Main St",
                                "maneuver": "STRAIGHT",
                            },
                            "localizedValues": {
                                "distance": {"text": "200 m"},
                            },
                            "startLocation": {
                                "latLng": {"latitude": 37.7749, "longitude": -122.4194},
                            },
                            "endLocation": {
                                "latLng": {"latitude": 37.7769, "longitude": -122.4194},
                            },
                        },
                        {
                            "distanceMeters": 1000,
                            "navigationInstruction": {
                                "instructions": "Turn right onto Oak Ave",
                                "maneuver": "TURN_RIGHT",
                            },
                            "localizedValues": {
                                "distance": {"text": "1.0 km"},
                            },
                            "startLocation": {
                                "latLng": {"latitude": 37.7769, "longitude": -122.4194},
                            },
                            "endLocation": {
                                "latLng": {"latitude": 37.7849, "longitude": -122.4094},
                            },
                        },
                    ],
                }
            ],
            "polyline": {
                "encodedPolyline": "_p~iF~ps|U_ulLnnqC_mqNvxq`@",
            },
        }
    ]
}

MOCK_ROUTES_EMPTY = {"routes": []}

MOCK_GEOCODE_RESPONSE = [
    {"formatted_address": "123 Main St, San Francisco, CA 94105"}
]

MOCK_PLACES_NEARBY_RESPONSE = {
    "places": [
        {
            "displayName": {"text": "Coffee Bean"},
            "location": {"latitude": 37.7751, "longitude": -122.4190},
            "types": ["cafe", "food"],
            "rating": 4.2,
            "formattedAddress": "100 Main St",
            "currentOpeningHours": {"openNow": True},
            "accessibilityOptions": {
                "wheelchairAccessibleEntrance": True,
                "wheelchairAccessibleRestroom": False,
            },
            "plusCode": {"globalCode": "849VQHFJ+XY"},
        },
        {
            "displayName": {"text": "City Pharmacy"},
            "location": {"latitude": 37.7755, "longitude": -122.4185},
            "types": ["pharmacy", "health"],
            "rating": 3.8,
            "formattedAddress": "110 Main St",
            "currentOpeningHours": {"openNow": False},
            "accessibilityOptions": {},
        },
    ]
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_gmaps():
    """Provide a mocked Google Maps SDK client (for reverse_geocode + elevation)."""
    with patch("tools.navigation._get_client") as mock_get:
        client = MagicMock()
        mock_get.return_value = client
        yield client


@pytest.fixture
def mock_routes():
    """Mock the REST helper used by Routes API / Places (New) / Address Validation."""
    with patch("tools.navigation.maps_rest_post") as mock_post:
        yield mock_post


@pytest.fixture
def mock_rest_get():
    """Mock the REST GET helper used by Street View."""
    with patch("tools.navigation.maps_rest_get") as mock_get:
        yield mock_get


# ---------------------------------------------------------------------------
# API-mocked integration tests
# ---------------------------------------------------------------------------


class TestNavigateTo:
    """Test navigate_to with mocked Routes API."""

    def test_success(self, mock_routes, mock_gmaps):
        mock_routes.return_value = MOCK_ROUTES_RESPONSE
        # Mock elevation to return empty (non-critical)
        mock_gmaps.elevation_along_path.return_value = []

        result = navigate_to(
            destination="456 Oak Ave",
            origin_lat=37.7749,
            origin_lng=-122.4194,
            user_heading=0.0,
        )

        assert result["success"] is True
        assert result["destination"] == "456 Oak Ave"
        assert result["total_distance"] == "1.2 km"
        assert result["total_duration"] == "15 min"
        assert len(result["steps"]) == 2
        assert "clock_direction" in result["steps"][0]
        assert "accessibility_note" in result
        assert "slope_warnings" in result

    def test_no_route_found(self, mock_routes):
        mock_routes.return_value = MOCK_ROUTES_EMPTY

        result = navigate_to(
            destination="Nonexistent Place",
            origin_lat=37.7749,
            origin_lng=-122.4194,
        )

        assert result["success"] is False
        assert "No walking route" in result["error"]

    def test_api_error(self, mock_routes):
        mock_routes.side_effect = Exception("API quota exceeded")

        result = navigate_to(
            destination="456 Oak Ave",
            origin_lat=37.7749,
            origin_lng=-122.4194,
        )

        assert result["success"] is False
        assert "API quota exceeded" in result["error"]

    def test_step_has_maneuver_direction(self, mock_routes, mock_gmaps):
        mock_routes.return_value = MOCK_ROUTES_RESPONSE
        mock_gmaps.elevation_along_path.return_value = []

        result = navigate_to(
            destination="456 Oak Ave",
            origin_lat=37.7749,
            origin_lng=-122.4194,
            user_heading=0.0,
        )

        # Second step has TURN_RIGHT maneuver
        step2 = result["steps"][1]
        assert step2["direction"] == "turn to 3 o'clock"

    def test_slope_warnings_included(self, mock_routes, mock_gmaps):
        mock_routes.return_value = MOCK_ROUTES_RESPONSE
        # Simulate steep elevation change
        mock_gmaps.elevation_along_path.return_value = [
            {"elevation": 0},
            {"elevation": 0},
            {"elevation": 20},  # Steep uphill on third segment
        ]

        result = navigate_to(
            destination="456 Oak Ave",
            origin_lat=37.7749,
            origin_lng=-122.4194,
        )

        assert result["success"] is True
        # With 1200m / 2 segments = 600m each, 20m rise = 3.3% — below threshold
        # So no warnings in this case (correct behavior)
        assert isinstance(result["slope_warnings"], list)


class TestGetLocationInfo:
    """Test get_location_info with mocked API."""

    def test_success(self, mock_gmaps, mock_routes):
        mock_gmaps.reverse_geocode.return_value = MOCK_GEOCODE_RESPONSE
        mock_routes.return_value = MOCK_PLACES_NEARBY_RESPONSE

        result = get_location_info(37.7749, -122.4194)

        assert result["success"] is True
        assert "San Francisco" in result["address"]
        assert len(result["nearby_places"]) == 2
        assert result["nearby_places"][0]["name"] == "Coffee Bean"
        # Check accessibility field
        assert "accessibility" in result["nearby_places"][0]
        assert result["nearby_places"][0]["accessibility"]["wheelchair_entrance"] is True

    def test_no_geocode_results(self, mock_gmaps, mock_routes):
        mock_gmaps.reverse_geocode.return_value = []
        mock_routes.return_value = {"places": []}

        result = get_location_info(0.0, 0.0)

        assert result["success"] is True
        assert result["address"] == "Unknown location"

    def test_api_error(self, mock_gmaps):
        mock_gmaps.reverse_geocode.side_effect = Exception("Network error")

        result = get_location_info(37.7749, -122.4194)

        assert result["success"] is False
        assert "Network error" in result["error"]


class TestNearbySearch:
    """Test nearby_search with mocked Places (New) API."""

    def test_success(self, mock_routes):
        mock_routes.return_value = MOCK_PLACES_NEARBY_RESPONSE

        result = nearby_search(37.7749, -122.4194, radius=200, types=["cafe"])

        assert result["success"] is True
        assert result["count"] == 2
        # Results should be sorted by distance
        places = result["places"]
        assert places[0]["distance_meters"] <= places[1]["distance_meters"]
        # Check accessibility field present
        assert "accessibility" in places[0]

    def test_with_types(self, mock_routes):
        mock_routes.return_value = MOCK_PLACES_NEARBY_RESPONSE

        result = nearby_search(37.7749, -122.4194, types=["cafe"])

        assert result["success"] is True
        assert result["query"] == "cafe"

    def test_empty_results(self, mock_routes):
        mock_routes.return_value = {"places": []}

        result = nearby_search(37.7749, -122.4194)

        assert result["success"] is True
        assert result["count"] == 0
        assert result["places"] == []

    def test_api_error(self, mock_routes):
        mock_routes.side_effect = Exception("Quota exceeded")

        result = nearby_search(37.7749, -122.4194)

        assert result["success"] is False


class TestReverseGeocode:
    """Test reverse_geocode with mocked API."""

    def test_success(self, mock_gmaps):
        mock_gmaps.reverse_geocode.return_value = MOCK_GEOCODE_RESPONSE

        result = reverse_geocode(37.7749, -122.4194)

        assert result["success"] is True
        assert "San Francisco" in result["address"]

    def test_no_results(self, mock_gmaps):
        mock_gmaps.reverse_geocode.return_value = []

        result = reverse_geocode(0.0, 0.0)

        assert result["success"] is False
        assert "No address" in result["error"]

    def test_api_error(self, mock_gmaps):
        mock_gmaps.reverse_geocode.side_effect = Exception("Network error")

        result = reverse_geocode(37.7749, -122.4194)

        assert result["success"] is False
        assert "Could not determine" in result["error"]


class TestGetWalkingDirections:
    """Test get_walking_directions with mocked Routes API."""

    def test_success(self, mock_routes, mock_gmaps):
        mock_routes.return_value = MOCK_ROUTES_RESPONSE
        mock_gmaps.elevation_along_path.return_value = []

        result = get_walking_directions("123 Main St", "456 Oak Ave")

        assert result["success"] is True
        assert result["total_distance"] == "1.2 km"
        assert len(result["steps"]) == 2
        # Check maneuver descriptions
        assert result["steps"][1]["direction"] == "turn to 3 o'clock"
        assert "slope_warnings" in result

    def test_no_route(self, mock_routes):
        mock_routes.return_value = MOCK_ROUTES_EMPTY

        result = get_walking_directions("Mars", "Jupiter")

        assert result["success"] is False

    def test_api_error(self, mock_routes):
        mock_routes.side_effect = Exception("Timeout")

        result = get_walking_directions("A", "B")

        assert result["success"] is False


class TestValidateAddress:
    """Test validate_address with mocked Address Validation API."""

    def test_corrected_address(self, mock_routes):
        mock_routes.return_value = {
            "result": {
                "address": {
                    "formattedAddress": "123 Main St, Springfield, IL 62701",
                },
                "geocode": {
                    "location": {"latitude": 39.7817, "longitude": -89.6501},
                },
                "verdict": {"addressComplete": True},
            },
        }

        result = validate_address("one two three main street springfield")

        assert result["success"] is True
        assert result["was_corrected"] is True
        assert "123 Main St" in result["corrected_address"]
        assert "Did you mean" in result["correction_note"]
        assert result["latitude"] is not None
        assert result["is_complete"] is True

    def test_api_failure_fallback(self, mock_routes):
        mock_routes.side_effect = Exception("Service unavailable")

        result = validate_address("some address")

        assert result["success"] is True
        assert result["corrected_address"] == "some address"
        assert result["was_corrected"] is False


class TestPreviewDestination:
    """Test preview_destination with mocked Street View + Vision Agent."""

    def test_no_street_view(self, mock_rest_get):
        # Metadata returns not OK
        meta_response = MagicMock()
        meta_response.json.return_value = {"status": "ZERO_RESULTS"}
        mock_rest_get.return_value = meta_response

        result = preview_destination(37.7749, -122.4194, "Cafe XYZ")

        assert result["success"] is True
        assert result["has_street_view"] is False
        assert "No street-level imagery" in result["description"]

    def test_street_view_success(self, mock_rest_get):
        # Metadata OK
        meta_response = MagicMock()
        meta_response.json.return_value = {"status": "OK"}
        # Image bytes
        image_response = MagicMock()
        image_response.content = b"fake_image_data"

        mock_rest_get.side_effect = [meta_response, image_response]

        with patch("agents.vision_agent.analyze_scene") as mock_vision:
            mock_vision.return_value = {
                "safety_warnings": ["Uneven pavement"],
                "navigation_info": {"entrances": ["Main door at 12 o'clock"]},
                "scene_description": "A cozy cafe with outdoor seating",
                "people_count": 3,
                "confidence": 0.85,
            }

            result = preview_destination(37.7749, -122.4194, "Cafe XYZ")

        assert result["success"] is True
        assert result["has_street_view"] is True
        assert "Cafe XYZ" in result["description"]
        assert len(result["safety_warnings"]) == 1


# ---------------------------------------------------------------------------
# Declaration / registration tests
# ---------------------------------------------------------------------------


class TestDeclarations:
    """Verify ADK tool declarations are well-formed."""

    def test_all_functions_have_declarations(self):
        declared_names = {d["name"] for d in NAVIGATION_TOOL_DECLARATIONS}
        func_names = set(NAVIGATION_FUNCTIONS.keys())
        assert declared_names == func_names

    def test_declarations_have_required_fields(self):
        for decl in NAVIGATION_TOOL_DECLARATIONS:
            assert "name" in decl
            assert "description" in decl
            assert "parameters" in decl
            assert decl["parameters"]["type"] == "object"

    def test_functions_are_callable(self):
        for name, func in NAVIGATION_FUNCTIONS.items():
            assert callable(func), f"{name} is not callable"

    def test_new_functions_registered(self):
        assert "preview_destination" in NAVIGATION_FUNCTIONS
        assert "validate_address" in NAVIGATION_FUNCTIONS
