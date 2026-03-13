"""Result formatters extracted from ``server.py``."""

from __future__ import annotations

import re

# Regex to strip internal context tags before sending to client
_INTERNAL_TAG_RE = re.compile(
    r"\[(?:VISION ANALYSIS|OCR RESULT|FACE RECOGNITION|NAVIGATION|SEARCH RESULT|"
    r"ACCESSIBILITY|MEMORY CONTEXT|ENTITY UPDATE|LOCATION INFO|DEPTH MAP)\]",
    re.IGNORECASE,
)


def _format_vision_result(result: dict, lod: int) -> str:
    """Format vision analysis result for Gemini context injection."""
    parts = ["[VISION ANALYSIS]"]

    # Safety warnings always first
    warnings = result.get("safety_warnings", [])
    for w in warnings:
        parts.append(f"SAFETY: {w}")

    # Navigation info as compact spatial summary
    nav = result.get("navigation_info", {})
    if lod >= 2:
        nav_items = []
        for key in ("entrances", "paths", "landmarks"):
            items = nav.get(key, [])
            if items:
                nav_items.extend(items)
        if nav_items:
            parts.append("Spatial: " + " | ".join(nav_items))

    desc = result.get("scene_description", "")
    if desc:
        parts.append(f"Scene: {desc}")

    text = result.get("detected_text")
    if text and lod >= 2:
        parts.append(f"Text spotted: {text}")

    count = result.get("people_count", 0)
    if count > 0 and lod >= 2:
        if count == 1:
            parts.append("1 person nearby")
        else:
            parts.append(f"{count} people nearby")

    # Spatial objects grouped by salience
    spatial_objects = result.get("spatial_objects", [])
    if spatial_objects:
        by_salience: dict[str, list[str]] = {}
        for obj in spatial_objects:
            if not isinstance(obj, dict):
                continue
            label = obj.get("label", "object")
            clock = obj.get("clock_position")
            dist = obj.get("distance_estimate", "")
            salience = obj.get("salience", "background")
            desc = label
            if clock:
                desc += f" at {clock} o'clock"
            if dist:
                desc += f", {dist}"
            by_salience.setdefault(salience, []).append(desc)

        if "safety" in by_salience:
            parts.append("Nearby hazards: " + "; ".join(by_salience["safety"]))
        if "navigation" in by_salience:
            parts.append("Key landmarks: " + "; ".join(by_salience["navigation"]))
        if lod >= 2 and "interaction" in by_salience:
            parts.append("Near you: " + "; ".join(by_salience["interaction"]))
        if lod >= 3 and "background" in by_salience:
            parts.append("Also visible: " + "; ".join(by_salience["background"]))

    return "\n".join(parts)


def _format_face_results(known_faces: list[dict]) -> str:
    """Format face recognition results for SILENT context injection."""
    parts = ["[FACE ID]"]
    for face in known_faces:
        name = face["person_name"]
        rel = face.get("relationship", "")
        sim = face.get("similarity", 0)
        desc = f"{name}"
        if rel:
            desc += f" ({rel})"
        if sim >= 0.85:
            desc += " — high confidence"
        elif sim >= 0.70:
            desc += " — moderate confidence, verify if possible"
        else:
            desc += " — low confidence, do not announce unless user asks"
        parts.append(desc)
    return "\n".join(parts)


def _format_ocr_result(result: dict) -> str:
    """Format OCR result for Gemini context injection."""
    parts = ["[OCR RESULT]"]

    text_type = result.get("text_type", "unknown")
    confidence = result.get("confidence", 1.0)

    # Context-aware type labels
    type_hints = {
        "menu": "Menu text detected — read items with prices:",
        "sign": "Sign text detected:",
        "document": "Document text detected:",
        "label": "Label text detected:",
    }
    parts.append(type_hints.get(text_type, f"Text detected ({text_type}):"))

    if confidence < 0.5:
        parts.append("(Note: text quality is poor, some characters may be inaccurate)")

    items = result.get("items", [])
    if items:
        for item in items:
            parts.append(f"  - {item}")
    else:
        text = result.get("text", "")
        if text:
            parts.append(text)

    return "\n".join(parts)

