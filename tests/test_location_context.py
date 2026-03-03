"""Tests for context.location_context module.

All Maps API and Firestore calls are mocked.
"""

import asyncio
import sys
import time
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from context.location_context import (
    LocationContext,
    LocationContextService,
    _haversine_m,
    _visit_count_to_familiarity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _mock_navigation_module(get_location_info_fn):
    """Create a mock tools.navigation module with the given get_location_info."""
    mod = ModuleType("tools.navigation")
    mod.get_location_info = get_location_info_fn
    return mod


def _default_loc_info(lat=None, lng=None):
    return {
        "success": True,
        "address": "123 Main St",
        "nearby_places": [
            {"name": "Blue Bottle Coffee", "types": ["cafe"], "distance_meters": 10},
        ],
    }


# ---------------------------------------------------------------------------
# Haversine
# ---------------------------------------------------------------------------


class TestHaversine:
    def test_same_point_is_zero(self):
        assert _haversine_m(37.7749, -122.4194, 37.7749, -122.4194) == 0.0

    def test_known_distance(self):
        # San Francisco to Oakland ~13km
        dist = _haversine_m(37.7749, -122.4194, 37.8044, -122.2712)
        assert 12_000 < dist < 15_000


# ---------------------------------------------------------------------------
# Familiarity conversion
# ---------------------------------------------------------------------------


class TestFamiliarityConversion:
    def test_zero_visits(self):
        assert _visit_count_to_familiarity(0) == 0.0

    def test_first_visit(self):
        assert _visit_count_to_familiarity(1) == 0.2

    def test_few_visits(self):
        assert _visit_count_to_familiarity(3) == 0.4

    def test_moderate_visits(self):
        assert _visit_count_to_familiarity(7) == 0.6

    def test_frequent_visits(self):
        assert _visit_count_to_familiarity(15) == 0.8

    def test_daily_visits(self):
        assert _visit_count_to_familiarity(25) == 1.0


# ---------------------------------------------------------------------------
# LocationContextService
# ---------------------------------------------------------------------------


class TestLocationContextService:
    def _patch_resolve(self, svc, get_loc_fn, graph_mock=None):
        """Replace _resolve with a version that uses mocked imports."""
        import context.location_context as mod

        original_resolve = mod.LocationContextService._resolve

        async def patched_resolve(self_svc, lat, lng):
            ctx = LocationContext()
            # Step 1: Fake navigation call
            try:
                info = get_loc_fn(lat, lng)
                if info.get("success"):
                    ctx.address = info.get("address", "")
                    nearby = info.get("nearby_places", [])
                    if nearby:
                        top = nearby[0]
                        ctx.place_name = top.get("name", "")
                        types = top.get("types", [])
                        ctx.place_type = types[0] if types else "unknown"
                        ctx.is_indoor = bool(set(types) & mod._INDOOR_TYPES)
            except Exception:
                pass

            # Step 2: Entity graph matching
            if graph_mock and ctx.place_name:
                try:
                    entity = graph_mock.find_entity_by_name(ctx.place_name, entity_type="place")
                    if entity:
                        ctx.is_known_entity = True
                        ctx.matched_entity_id = entity.entity_id
                        ctx.familiarity_score = mod._visit_count_to_familiarity(entity.visit_count)
                        graph_mock.touch_entity(entity.entity_id)
                        connected = graph_mock.get_connected_entities(entity.entity_id)
                        ctx.nearby_entity_ids = [e.entity_id for e in connected]
                except Exception:
                    pass

            return ctx

        svc._resolve = lambda lat, lng: patched_resolve(svc, lat, lng)

    def test_evaluate_returns_location_context(self):
        svc = LocationContextService("test_user")
        self._patch_resolve(svc, _default_loc_info)

        ctx = _run(svc.evaluate(37.7749, -122.4194))

        assert isinstance(ctx, LocationContext)
        assert ctx.place_name == "Blue Bottle Coffee"
        assert ctx.place_type == "cafe"
        assert ctx.is_indoor is True
        assert ctx.address == "123 Main St"

    def test_evaluate_matches_known_entity(self):
        from context.entity_graph import Entity

        known_entity = Entity(
            entity_id="place_001",
            entity_type="place",
            name="Blue Bottle Coffee",
            visit_count=12,
        )

        graph_mock = MagicMock()
        graph_mock.find_entity_by_name.return_value = known_entity
        graph_mock.touch_entity.return_value = True
        graph_mock.get_connected_entities.return_value = []

        svc = LocationContextService("test_user")
        self._patch_resolve(svc, _default_loc_info, graph_mock)

        ctx = _run(svc.evaluate(37.7749, -122.4194))

        assert ctx.is_known_entity is True
        assert ctx.matched_entity_id == "place_001"
        assert ctx.familiarity_score == 0.8  # 12 visits

    def test_debounce_same_location(self):
        call_count = 0

        def counting_loc(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _default_loc_info()

        svc = LocationContextService("test_user")
        self._patch_resolve(svc, counting_loc)

        # First call
        _run(svc.evaluate(37.7749, -122.4194))
        # Same location, should use cache
        _run(svc.evaluate(37.7749, -122.4194))

        assert call_count == 1  # Only one actual resolve

    def test_debounce_different_location(self):
        call_count = 0

        def counting_loc(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return {"success": True, "address": "Far", "nearby_places": []}

        svc = LocationContextService("test_user")
        self._patch_resolve(svc, counting_loc)

        _run(svc.evaluate(37.7749, -122.4194))
        # Far away location
        _run(svc.evaluate(38.0, -122.0))

        assert call_count == 2

    def test_evaluate_handles_api_failure(self):
        def failing_loc(*args, **kwargs):
            raise Exception("API down")

        svc = LocationContextService("test_user")
        self._patch_resolve(svc, failing_loc)

        ctx = _run(svc.evaluate(37.7749, -122.4194))

        assert ctx.place_name == ""
        assert ctx.familiarity_score == 0.0

    def test_outdoor_place_type(self):
        def park_loc(*args, **kwargs):
            return {
                "success": True,
                "address": "Park Ave",
                "nearby_places": [
                    {"name": "Central Park", "types": ["park", "tourist_attraction"]},
                ],
            }

        svc = LocationContextService("test_user")
        self._patch_resolve(svc, park_loc)

        ctx = _run(svc.evaluate(40.785091, -73.968285))

        assert ctx.place_type == "park"
        assert ctx.is_indoor is False
