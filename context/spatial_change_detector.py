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

    change_type: str  # "new_person_approaching", "layout_change", "hazard_appeared", "person_left", "vehicle_approaching", "sudden_obstacle", "person_very_close"
    severity: str  # "safety", "significant", "minor"
    details: str
    urgency: str = "awareness"  # "immediate" (within_reach), "approaching" (1-2m), "awareness" (3m+)


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

        # Rule 5: Approaching vehicle — distance decreased between frames
        prev_vehicles = _extract_objects_by_label(previous.get("spatial_objects", []), "vehicle")
        curr_vehicles = _extract_objects_by_label(current.get("spatial_objects", []), "vehicle")
        for v in curr_vehicles:
            dist = v.get("distance_estimate", "")
            motion = v.get("motion_direction", "")
            clock = v.get("clock_position", "")
            is_close = dist in ("within_reach", "1m", "2m")
            is_approaching = motion == "approaching"
            # Check if vehicle was previously farther away
            was_farther = not any(
                pv.get("distance_estimate", "") in ("within_reach", "1m", "2m")
                for pv in prev_vehicles
            ) if prev_vehicles else False
            if is_close and (is_approaching or was_farther):
                urgency = "immediate" if dist == "within_reach" else "approaching"
                clock_str = f" from {clock} o'clock" if clock else ""
                changes.append(SpatialChange(
                    change_type="vehicle_approaching",
                    severity="safety",
                    details=f"Vehicle approaching{clock_str}, {dist}",
                    urgency=urgency,
                ))

        # Rule 6: Sudden obstacle in path — new object at 11-1 o'clock within 2m
        prev_obj_keys = {
            (o.get("label", ""), o.get("clock_position"))
            for o in previous.get("spatial_objects", [])
            if isinstance(o, dict)
        }
        for obj in current.get("spatial_objects", []):
            if not isinstance(obj, dict):
                continue
            label = obj.get("label", "")
            clock = obj.get("clock_position")
            dist = obj.get("distance_estimate", "")
            salience = obj.get("salience", "")
            obj_key = (label, clock)
            if (
                obj_key not in prev_obj_keys
                and clock in (11, 12, 1)
                and dist in ("within_reach", "1m", "2m")
                and salience in ("safety", "navigation")
                and label not in ("person",)  # people handled by Rule 2/7
            ):
                urgency = "immediate" if dist == "within_reach" else "approaching"
                changes.append(SpatialChange(
                    change_type="sudden_obstacle",
                    severity="safety",
                    details=f"{label} appeared at {clock} o'clock, {dist}",
                    urgency=urgency,
                ))

        # Rule 7: Person very close — person at within_reach distance
        for obj in current.get("spatial_objects", []):
            if not isinstance(obj, dict):
                continue
            if obj.get("label") == "person" and obj.get("distance_estimate") == "within_reach":
                clock = obj.get("clock_position", "")
                # Only flag if this person wasn't already within_reach in previous frame
                was_close = any(
                    isinstance(po, dict)
                    and po.get("label") == "person"
                    and po.get("distance_estimate") == "within_reach"
                    for po in previous.get("spatial_objects", [])
                )
                if not was_close:
                    clock_str = f" at {clock} o'clock" if clock else ""
                    changes.append(SpatialChange(
                        change_type="person_very_close",
                        severity="safety",
                        details=f"Person very close{clock_str}",
                        urgency="immediate",
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


def _extract_objects_by_label(spatial_objects: list, label: str) -> list[dict]:
    """Extract all spatial objects matching a given label."""
    return [
        obj for obj in spatial_objects
        if isinstance(obj, dict) and obj.get("label") == label
    ]
