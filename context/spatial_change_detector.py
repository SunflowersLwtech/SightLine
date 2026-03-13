"""Rule-based (non-LLM) spatial change detection.

Compares consecutive vision results to detect meaningful scene changes
that warrant proactive announcements to the user.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("sightline.spatial_change_detector")


@dataclass
class SpatialChange:
    """A detected change between consecutive vision frames."""

    change_type: str  # "new_person_approaching", "layout_change", "hazard_appeared", "person_left"
    severity: str  # "safety", "significant", "minor"
    details: str


class SpatialChangeDetector:
    """Detect meaningful spatial changes between consecutive vision results."""

    def detect(
        self,
        previous: dict,
        current: dict,
        motion_state: str,
    ) -> list[SpatialChange]:
        """Compare previous and current vision results, return changes.

        Args:
            previous: Previous vision result dict (may be empty on first call).
            current: Current vision result dict.
            motion_state: User's current motion state (stationary, walking, running, etc.).

        Returns:
            List of detected spatial changes, ordered by severity.
        """
        if not previous or not current:
            return []

        changes: list[SpatialChange] = []

        # Rule 1: New safety warnings not in previous
        prev_warnings = set(previous.get("safety_warnings", []))
        curr_warnings = set(current.get("safety_warnings", []))
        new_warnings = curr_warnings - prev_warnings
        for warning in new_warnings:
            changes.append(SpatialChange(
                change_type="hazard_appeared",
                severity="safety",
                details=warning,
            ))

        # Rule 2: People count increased AND person at <2m → new_person_approaching
        prev_count = previous.get("people_count", 0) or 0
        curr_count = current.get("people_count", 0) or 0
        if curr_count > prev_count:
            close_person = False
            for obj in current.get("spatial_objects", []):
                if not isinstance(obj, dict):
                    continue
                if obj.get("label") == "person":
                    dist = obj.get("distance_estimate", "")
                    if dist in ("within_reach", "1m"):
                        close_person = True
                        break
            if close_person:
                changes.append(SpatialChange(
                    change_type="new_person_approaching",
                    severity="significant",
                    details=f"People count {prev_count} → {curr_count}, person nearby",
                ))

        # Rule 3: People count decreased by 2+
        if prev_count - curr_count >= 2:
            changes.append(SpatialChange(
                change_type="person_left",
                severity="minor",
                details=f"People count {prev_count} → {curr_count}",
            ))

        # Rule 4: >50% spatial_objects labels changed AND stationary → layout_change
        # Suppress during walking/running (camera bounce causes false positives)
        if motion_state in ("stationary", "in_vehicle"):
            prev_labels = _extract_labels(previous.get("spatial_objects", []))
            curr_labels = _extract_labels(current.get("spatial_objects", []))
            if prev_labels and curr_labels:
                intersection = prev_labels & curr_labels
                union = prev_labels | curr_labels
                overlap = len(intersection) / len(union) if union else 1.0
                if overlap < 0.5:
                    changes.append(SpatialChange(
                        change_type="layout_change",
                        severity="significant",
                        details=f"Scene composition changed ({len(prev_labels)} → {len(curr_labels)} objects, {overlap:.0%} overlap)",
                    ))

        # Sort by severity: safety > significant > minor
        severity_order = {"safety": 0, "significant": 1, "minor": 2}
        changes.sort(key=lambda c: severity_order.get(c.severity, 3))
        return changes


def _extract_labels(spatial_objects: list) -> set[str]:
    """Extract unique labels from spatial_objects list."""
    labels = set()
    for obj in spatial_objects:
        if isinstance(obj, dict) and obj.get("label"):
            labels.add(obj["label"])
    return labels
