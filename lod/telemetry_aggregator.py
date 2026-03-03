"""LOD-aware telemetry throttling utilities.

This module provides the Phase 2 SL-31 runtime contract used by tests:
- ``LOD_TELEMETRY_INTERVAL`` interval table
- ``TelemetryAggregator`` with update_lod/send_interval/should_send/mark_sent
"""

from __future__ import annotations

from dataclasses import dataclass

# SL-31 contract: send interval windows per LOD.
LOD_TELEMETRY_INTERVAL: dict[int, tuple[float, float]] = {
    1: (3.0, 4.0),
    2: (2.0, 3.0),
    3: (5.0, 10.0),
}


@dataclass
class TelemetryAggregator:
    """Simple LOD-aware telemetry throttle state holder."""

    current_lod: int = 2
    _last_sent_ts: float | None = None

    def update_lod(self, lod: int) -> None:
        """Update active LOD; falls back to LOD2 for invalid values."""
        self.current_lod = lod if lod in LOD_TELEMETRY_INTERVAL else 2

    @property
    def send_interval(self) -> float:
        """Current send interval in seconds (midpoint of allowed range)."""
        low, high = LOD_TELEMETRY_INTERVAL.get(self.current_lod, LOD_TELEMETRY_INTERVAL[2])
        return (low + high) / 2.0

    def should_send(self, now_ts: float) -> bool:
        """Return True when telemetry should be sent at ``now_ts``."""
        if self._last_sent_ts is None:
            return True
        return (now_ts - self._last_sent_ts) >= self.send_interval

    def mark_sent(self, now_ts: float) -> None:
        """Record the timestamp of the latest telemetry send."""
        self._last_sent_ts = float(now_ts)
