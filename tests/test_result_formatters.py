"""Tests for result formatters (vision, OCR, face)."""

import pytest

from formatters.result_formatters import (
    _format_face_results,
    _format_ocr_result,
    _format_vision_result,
)

# ---------------------------------------------------------------------------
# Tests: Vision result formatter — new fields
# ---------------------------------------------------------------------------


class TestVisionResultFormatter:
    def test_light_level_included_at_lod2(self):
        result = {
            "safety_warnings": [],
            "scene_description": "A room",
            "light_level": "warm_ambient",
        }
        formatted = _format_vision_result(result, lod=2)
        assert "Light: warm_ambient" in formatted

    def test_light_level_excluded_at_lod1(self):
        result = {
            "safety_warnings": [],
            "scene_description": "",
            "light_level": "bright_daylight",
        }
        formatted = _format_vision_result(result, lod=1)
        assert "Light:" not in formatted

    def test_emotions_included_at_lod2(self):
        result = {
            "safety_warnings": [],
            "scene_description": "",
            "emotions": [
                {"person_position": "2 o'clock, 2m", "expression": "smiling"},
                {"person_position": "10 o'clock, 3m", "expression": "focused"},
            ],
        }
        formatted = _format_vision_result(result, lod=2)
        assert "Expressions:" in formatted
        assert "smiling" in formatted
        assert "focused" in formatted

    def test_emotions_excluded_at_lod1(self):
        result = {
            "safety_warnings": [],
            "scene_description": "",
            "emotions": [
                {"person_position": "2 o'clock", "expression": "smiling"},
            ],
        }
        formatted = _format_vision_result(result, lod=1)
        assert "Expressions:" not in formatted

    def test_currency_always_included(self):
        result = {
            "safety_warnings": [],
            "scene_description": "",
            "currency_detected": ["US $20 bill", "US $5 bill"],
        }
        formatted = _format_vision_result(result, lod=1)
        assert "Currency:" in formatted
        assert "$20" in formatted
        assert "$5" in formatted

    def test_motion_direction_in_spatial_objects(self):
        result = {
            "safety_warnings": [],
            "scene_description": "",
            "spatial_objects": [
                {
                    "label": "vehicle",
                    "clock_position": 3,
                    "distance_estimate": "2m",
                    "salience": "safety",
                    "motion_direction": "approaching",
                },
            ],
        }
        formatted = _format_vision_result(result, lod=1)
        assert "(approaching)" in formatted

    def test_no_new_fields_returns_normal(self):
        """Results without new fields should format normally."""
        result = {
            "safety_warnings": ["stairs at 12"],
            "scene_description": "hallway",
            "people_count": 2,
            "spatial_objects": [
                {"label": "stairs", "clock_position": 12, "distance_estimate": "2m", "salience": "safety"},
            ],
        }
        formatted = _format_vision_result(result, lod=2)
        assert "[VISION ANALYSIS]" in formatted
        assert "stairs" in formatted


# ---------------------------------------------------------------------------
# Tests: OCR result formatter — specialized document types
# ---------------------------------------------------------------------------


class TestOCRResultFormatter:
    def test_medicine_label_formatting(self):
        result = {
            "text": "Ibuprofen 200mg",
            "text_type": "medicine_label",
            "items": [],
            "confidence": 0.95,
            "medicine_info": {
                "drug_name": "Ibuprofen",
                "dosage": "200mg",
                "frequency": "1-2 tablets every 6 hours",
                "warnings": ["Do not exceed 6 tablets in 24 hours", "Take with food"],
                "expiry_date": "2027-03-15",
            },
        }
        formatted = _format_ocr_result(result)
        assert "Medication label detected:" in formatted
        assert "Medication: Ibuprofen, 200mg" in formatted
        assert "Take: 1-2 tablets every 6 hours" in formatted
        assert "Do not exceed" in formatted
        assert "Expires: 2027-03-15" in formatted

    def test_receipt_formatting(self):
        result = {
            "text": "Walmart Receipt",
            "text_type": "receipt",
            "items": [],
            "confidence": 0.90,
            "receipt_info": {
                "store_name": "Walmart",
                "items": ["Milk - $3.99", "Bread - $2.49", "Eggs - $4.29"],
                "total": "$10.77",
                "payment_method": "Visa ending 4321",
                "change": None,
            },
        }
        formatted = _format_ocr_result(result)
        assert "Receipt detected:" in formatted
        assert "Store: Walmart" in formatted
        assert "Milk - $3.99" in formatted
        assert "Total: $10.77" in formatted
        assert "Paid by: Visa ending 4321" in formatted

    def test_food_packaging_formatting(self):
        result = {
            "text": "Granola Bar",
            "text_type": "food_packaging",
            "items": [],
            "confidence": 0.88,
            "nutrition_info": {
                "product_name": "Nature Valley Granola Bar",
                "allergens": ["Peanuts", "Tree Nuts", "Wheat"],
                "calories": "190",
                "serving_size": "1 bar (42g)",
                "ingredients": "Whole grain oats, sugar, peanut butter...",
            },
        }
        formatted = _format_ocr_result(result)
        assert "Food packaging detected:" in formatted
        assert "Product: Nature Valley Granola Bar" in formatted
        assert "ALLERGENS: Peanuts, Tree Nuts, Wheat" in formatted
        assert "190 cal" in formatted
        assert "1 bar (42g)" in formatted

    def test_business_card_formatting(self):
        result = {
            "text": "Jane Smith, VP Engineering",
            "text_type": "business_card",
            "items": [],
            "confidence": 0.92,
            "contact_info": {
                "name": "Jane Smith",
                "title": "VP Engineering",
                "company": "TechCorp",
                "phone": "+1-555-123-4567",
                "email": "jane@techcorp.com",
                "address": "123 Innovation Way, SF",
            },
        }
        formatted = _format_ocr_result(result)
        assert "Business card detected:" in formatted
        assert "Jane Smith, VP Engineering at TechCorp" in formatted
        assert "Phone: +1-555-123-4567" in formatted
        assert "Email: jane@techcorp.com" in formatted

    def test_regular_menu_still_works(self):
        result = {
            "text": "Coffee Menu",
            "text_type": "menu",
            "items": ["Latte - $4.50", "Cappuccino - $4.00"],
            "confidence": 0.95,
        }
        formatted = _format_ocr_result(result)
        assert "Menu text detected" in formatted
        assert "Latte - $4.50" in formatted

    def test_low_confidence_warning(self):
        result = {
            "text": "blurry text",
            "text_type": "unknown",
            "items": [],
            "confidence": 0.3,
        }
        formatted = _format_ocr_result(result)
        assert "poor" in formatted.lower()

    def test_medicine_without_expiry(self):
        result = {
            "text": "Aspirin",
            "text_type": "medicine_label",
            "items": [],
            "confidence": 0.9,
            "medicine_info": {
                "drug_name": "Aspirin",
                "dosage": "325mg",
                "frequency": "every 4 hours",
                "warnings": [],
                "expiry_date": None,
            },
        }
        formatted = _format_ocr_result(result)
        assert "Aspirin" in formatted
        assert "Expires:" not in formatted


# ---------------------------------------------------------------------------
# Tests: Face result formatter (unchanged, regression check)
# ---------------------------------------------------------------------------


class TestFaceResultFormatter:
    def test_high_confidence_face(self):
        faces = [{"person_name": "Alice", "relationship": "friend", "similarity": 0.92}]
        formatted = _format_face_results(faces)
        assert "[FACE ID]" in formatted
        assert "Alice" in formatted
        assert "high confidence" in formatted

    def test_low_confidence_face(self):
        faces = [{"person_name": "Bob", "similarity": 0.55}]
        formatted = _format_face_results(faces)
        assert "low confidence" in formatted
