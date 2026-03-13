"""Health router extracted from ``server.py``."""

from __future__ import annotations

import os

from fastapi import APIRouter

from app_globals import LIVE_MODEL, _face_available, _ocr_available, _vision_available

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    """Health check endpoint for Cloud Run readiness probes."""
    return {
        "status": "ok",
        "model": LIVE_MODEL,
        "phase": 6,
        "capabilities": {
            "vision": _vision_available,
            "ocr": _ocr_available,
            "face": _face_available,
            "plus_codes": True,
            "elevation": True,
            "street_view": True,
            "address_validation": True,
            "accessibility": True,
            "maps_grounding": bool(os.getenv("GOOGLE_MAPS_API_KEY")),
            "weather": True,
            "haptics": True,
            "depth": True,
        },
    }

