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

    # Light level (LOD 2+)
    light_level = result.get("light_level")
    if light_level and lod >= 2:
        parts.append(f"Light: {light_level}")

    # Emotions (LOD 2+)
    emotions = result.get("emotions")
    if emotions and lod >= 2:
        emotion_descs = []
        for em in emotions:
            if not isinstance(em, dict):
                continue
            pos = em.get("person_position", "nearby")
            expr = em.get("expression", "")
            if expr:
                emotion_descs.append(f"{expr} ({pos})")
        if emotion_descs:
            parts.append("Expressions: " + "; ".join(emotion_descs))

    # Currency detected
    currency = result.get("currency_detected")
    if currency:
        parts.append("Currency: " + ", ".join(currency))

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
            motion = obj.get("motion_direction", "")
            desc = label
            if clock:
                desc += f" at {clock} o'clock"
            if dist:
                desc += f", {dist}"
            if motion:
                desc += f" ({motion})"
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
        "medicine_label": "Medication label detected:",
        "receipt": "Receipt detected:",
        "food_packaging": "Food packaging detected:",
        "business_card": "Business card detected:",
    }
    parts.append(type_hints.get(text_type, f"Text detected ({text_type}):"))

    if confidence < 0.5:
        parts.append("(Note: text quality is poor, some characters may be inaccurate)")

    # Structured formatting for specialized document types
    medicine = result.get("medicine_info")
    if medicine and isinstance(medicine, dict):
        drug = medicine.get("drug_name", "")
        dosage = medicine.get("dosage", "")
        freq = medicine.get("frequency", "")
        warnings = medicine.get("warnings", [])
        expiry = medicine.get("expiry_date")
        if drug:
            med_line = f"Medication: {drug}"
            if dosage:
                med_line += f", {dosage}"
            parts.append(med_line)
        if freq:
            parts.append(f"  Take: {freq}")
        if warnings:
            parts.append(f"  Warnings: {'; '.join(warnings)}")
        if expiry:
            parts.append(f"  Expires: {expiry}")

    receipt = result.get("receipt_info")
    if receipt and isinstance(receipt, dict):
        store = receipt.get("store_name", "")
        r_items = receipt.get("items", [])
        total = receipt.get("total", "")
        payment = receipt.get("payment_method")
        change = receipt.get("change")
        if store:
            parts.append(f"Store: {store}")
        for ri in r_items:
            parts.append(f"  - {ri}")
        if total:
            parts.append(f"  Total: {total}")
        if payment:
            parts.append(f"  Paid by: {payment}")
        if change:
            parts.append(f"  Change: {change}")

    nutrition = result.get("nutrition_info")
    if nutrition and isinstance(nutrition, dict):
        product = nutrition.get("product_name", "")
        allergens = nutrition.get("allergens", [])
        calories = nutrition.get("calories", "")
        serving = nutrition.get("serving_size", "")
        ingredients = nutrition.get("ingredients")
        if product:
            parts.append(f"Product: {product}")
        if allergens:
            parts.append(f"  ALLERGENS: {', '.join(allergens)}")
        if calories:
            cal_line = f"  Nutrition: {calories} cal"
            if serving:
                cal_line += f" per {serving}"
            parts.append(cal_line)
        if ingredients:
            parts.append(f"  Ingredients: {ingredients}")

    contact = result.get("contact_info")
    if contact and isinstance(contact, dict):
        name = contact.get("name", "")
        title = contact.get("title")
        company = contact.get("company")
        phone = contact.get("phone")
        email = contact.get("email")
        address = contact.get("address")
        if name:
            name_line = name
            if title:
                name_line += f", {title}"
            if company:
                name_line += f" at {company}"
            parts.append(f"  {name_line}")
        if phone:
            parts.append(f"  Phone: {phone}")
        if email:
            parts.append(f"  Email: {email}")
        if address:
            parts.append(f"  Address: {address}")

    # Fallback to generic items/text for non-specialized types
    if not any(result.get(k) for k in ("medicine_info", "receipt_info", "nutrition_info", "contact_info")):
        items = result.get("items", [])
        if items:
            for item in items:
                parts.append(f"  - {item}")
        else:
            text = result.get("text", "")
            if text:
                parts.append(text)

    return "\n".join(parts)

