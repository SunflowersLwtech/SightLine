"""Profile API router extracted from ``server.py``."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app_globals import session_manager
from firestore_client import get_firestore_client

logger = logging.getLogger("sightline.server")

router = APIRouter(prefix="/api")

ALLOWED_FIELDS = {
    "vision_status",
    "blindness_onset",
    "onset_age",
    "has_guide_dog",
    "has_white_cane",
    "tts_speed",
    "verbosity_preference",
    "language",
    "description_priority",
    "color_description",
    "om_level",
    "travel_frequency",
    "preferred_name",
}


@router.get("/profile/{user_id}")
async def api_get_profile(user_id: str) -> JSONResponse:
    """Get the UserProfile from Firestore."""
    try:
        db = get_firestore_client()
        doc = db.collection("user_profiles").document(user_id).get()
        if not doc.exists:
            return JSONResponse({"error": "Profile not found"}, status_code=404)
        data = doc.to_dict()
        # Convert timestamps to ISO strings
        for key in ("created_at", "updated_at"):
            if key in data and hasattr(data[key], "isoformat"):
                data[key] = data[key].isoformat()
        return JSONResponse(data)
    except Exception as e:
        logger.exception("Get profile failed for %s", user_id)
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/profile/{user_id}")
async def api_save_profile(user_id: str, request: Request) -> JSONResponse:
    """Create or update a UserProfile in Firestore.

    Body JSON — any of:
        vision_status: str (totally_blind / low_vision)
        blindness_onset: str (congenital / acquired)
        onset_age: int | null
        has_guide_dog: bool
        has_white_cane: bool
        tts_speed: float
        verbosity_preference: str (concise / detailed)
        language: str
        description_priority: str (spatial / object)
        color_description: bool
        om_level: str (beginner / intermediate / advanced)
        travel_frequency: str (daily / weekly / rarely)
        preferred_name: str
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    filtered = {k: v for k, v in body.items() if k in ALLOWED_FIELDS}
    if not filtered:
        return JSONResponse({"error": "No valid fields provided"}, status_code=400)

    try:
        from google.cloud import firestore as _fs

        db = get_firestore_client()
        doc_ref = db.collection("user_profiles").document(user_id)
        filtered["updated_at"] = _fs.SERVER_TIMESTAMP
        # Merge so we don't overwrite fields not included in this request
        doc_ref.set(filtered, merge=True)
        session_manager.invalidate_user_profile(user_id)
        logger.info(
            "REST profile save for user %s: %s (cache invalidated)",
            user_id,
            list(filtered.keys()),
        )
        return JSONResponse({"status": "saved", "user_id": user_id, "fields": list(filtered.keys())})
    except Exception as e:
        logger.exception("Save profile failed for %s", user_id)
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/users")
async def api_list_users() -> JSONResponse:
    """List all user IDs from Firestore."""
    try:
        db = get_firestore_client()
        docs = db.collection("user_profiles").stream()
        user_ids = sorted(doc.id for doc in docs)
        return JSONResponse({"users": user_ids, "count": len(user_ids)})
    except Exception as e:
        logger.exception("List users failed")
        return JSONResponse({"error": str(e)}, status_code=500)

