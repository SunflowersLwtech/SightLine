"""Location intelligence — GPS-to-place resolution with entity matching.

Resolves raw GPS coordinates into a rich LocationContext by combining
Google Maps reverse geocode / nearby POIs (via tools/navigation.py)
with the user's entity graph.

Debounce: re-evaluates only on 50m movement or 60s elapsed.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Debounce thresholds
_DISTANCE_THRESHOLD_M = 50.0  # metres — re-evaluate on this much movement
_TIME_THRESHOLD_S = 60.0  # seconds — re-evaluate even without movement


@dataclass
class LocationContext:
    """Resolved location information for the current GPS position."""

    place_name: str = ""
    place_type: str = "unknown"  # cafe, supermarket, park, intersection, ...
    is_indoor: bool = False
    familiarity_score: float = 0.0  # 0.0 (never been) to 1.0 (daily visit)
    is_known_entity: bool = False
    matched_entity_id: Optional[str] = None
    nearby_entity_ids: list[str] = field(default_factory=list)
    address: str = ""


def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Haversine distance in metres between two GPS coordinates."""
    R = 6_371_000  # Earth radius in metres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# Indoor place types from Google Places API
_INDOOR_TYPES = frozenset({
    "restaurant", "cafe", "bar", "store", "supermarket", "shopping_mall",
    "hospital", "pharmacy", "bank", "library", "museum", "gym",
    "movie_theater", "school", "university", "church", "airport",
    "train_station", "subway_station", "bus_station", "hotel",
    "office", "doctor", "dentist", "post_office",
})

# Familiarity tiers based on visit_count
_FAMILIARITY_TIERS = [
    (20, 1.0),   # 20+ visits → fully familiar
    (10, 0.8),   # 10-19 visits
    (5, 0.6),    # 5-9 visits
    (2, 0.4),    # 2-4 visits
    (1, 0.2),    # first revisit
]


def _visit_count_to_familiarity(visit_count: int) -> float:
    """Convert an entity visit_count to a 0-1 familiarity score."""
    for threshold, score in _FAMILIARITY_TIERS:
        if visit_count >= threshold:
            return score
    return 0.0  # never visited


class LocationContextService:
    """Resolves GPS → place → entity with debounce caching."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self._last_lat: float = 0.0
        self._last_lng: float = 0.0
        self._last_eval_time: float = 0.0
        self._cached: Optional[LocationContext] = None

    async def evaluate(self, lat: float, lng: float) -> LocationContext:
        """Resolve GPS coordinates to a LocationContext.

        Uses debounce: returns cached result if GPS hasn't moved 50m
        and less than 60s have elapsed.
        """
        now = time.time()

        # Debounce check
        if self._cached is not None:
            dist = _haversine_m(lat, lng, self._last_lat, self._last_lng)
            elapsed = now - self._last_eval_time
            if dist < _DISTANCE_THRESHOLD_M and elapsed < _TIME_THRESHOLD_S:
                return self._cached

        # Evaluate fresh
        ctx = await self._resolve(lat, lng)
        self._last_lat = lat
        self._last_lng = lng
        self._last_eval_time = now
        self._cached = ctx
        return ctx

    async def _resolve(self, lat: float, lng: float) -> LocationContext:
        """Core resolution: Maps API + entity graph matching."""
        ctx = LocationContext()

        # Step 1: Reverse geocode + nearby POIs via existing navigation tool
        try:
            from tools.navigation import get_location_info
            import asyncio
            info = await asyncio.to_thread(get_location_info, lat, lng)
            if info.get("success"):
                ctx.address = info.get("address", "")
                nearby = info.get("nearby_places", [])
                if nearby:
                    top = nearby[0]
                    ctx.place_name = top.get("name", "")
                    types = top.get("types", [])
                    ctx.place_type = types[0] if types else "unknown"
                    ctx.is_indoor = bool(set(types) & _INDOOR_TYPES)
        except Exception:
            logger.debug("Location info lookup failed", exc_info=True)

        # Step 2: Match against entity graph
        try:
            from context.entity_graph import EntityGraphService
            graph = EntityGraphService(self.user_id)

            # Try to match place name in entities
            if ctx.place_name:
                entity = graph.find_entity_by_name(ctx.place_name, entity_type="place")
                if entity:
                    ctx.is_known_entity = True
                    ctx.matched_entity_id = entity.entity_id
                    ctx.familiarity_score = _visit_count_to_familiarity(entity.visit_count)
                    # Bump visit count
                    graph.touch_entity(entity.entity_id)

                    # 1-hop: get connected entities (people at this place)
                    connected = graph.get_connected_entities(entity.entity_id)
                    ctx.nearby_entity_ids = [e.entity_id for e in connected]
        except Exception:
            logger.debug("Entity graph matching failed", exc_info=True)

        return ctx
