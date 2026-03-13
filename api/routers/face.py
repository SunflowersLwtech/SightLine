"""Face API router extracted from ``server.py``."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from api.utils import _coerce_bool
from app_globals import _face_available

logger = logging.getLogger("sightline.server")

router = APIRouter(prefix="/api/face")


@router.post("/register")
async def api_register_face(request: Request) -> JSONResponse:
    """Register a face via REST (for iOS FaceRegistrationView).

    Body JSON:
        user_id: str
        person_name: str
        relationship: str
        image_base64: str  (JPEG base64-encoded)
        photo_index: int (optional, default 0)
        consent_confirmed: bool (optional, default false)
        store_reference_photo: bool (optional, default false)

    Returns the face_id and metadata on success.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    user_id = body.get("user_id")
    person_name = body.get("person_name")
    relationship = body.get("relationship", "")
    image_base64 = body.get("image_base64")
    photo_index = body.get("photo_index", 0)
    consent_confirmed = _coerce_bool(body.get("consent_confirmed"), default=False)
    store_reference_photo = _coerce_bool(body.get("store_reference_photo"), default=False)

    if not all([user_id, person_name, image_base64]):
        return JSONResponse(
            {"error": "Missing required fields: user_id, person_name, image_base64"},
            status_code=400,
        )

    if store_reference_photo and not consent_confirmed:
        return JSONResponse(
            {"error": "consent_confirmed must be true when store_reference_photo is enabled"},
            status_code=400,
        )

    if not _face_available:
        return JSONResponse(
            {"error": "Face recognition is not available on this server"},
            status_code=503,
        )

    try:
        from tools.face_tools import register_face

        result = await asyncio.to_thread(
            register_face,
            user_id=user_id,
            person_name=person_name,
            relationship=relationship,
            image_base64=image_base64,
            photo_index=photo_index,
            consent_confirmed=consent_confirmed,
            store_reference_photo=store_reference_photo,
        )
        logger.info("REST face register: %s for user %s", result.get("face_id"), user_id)
        return JSONResponse(result, status_code=201)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    except Exception as e:
        logger.exception("Face registration failed")
        return JSONResponse({"error": f"Registration failed: {str(e)}"}, status_code=500)


@router.get("/list/{user_id}")
async def api_list_faces(user_id: str) -> JSONResponse:
    """List all registered faces for a user (without embeddings)."""
    if not _face_available:
        return JSONResponse({"error": "Face recognition not available"}, status_code=503)

    try:
        from tools.face_tools import list_faces

        faces = await asyncio.to_thread(list_faces, user_id)
        return JSONResponse({"faces": faces, "count": len(faces)})
    except Exception as e:
        logger.exception("List faces failed for user %s", user_id)
        return JSONResponse({"error": str(e)}, status_code=500)


@router.delete("/{user_id}/{face_id}")
async def api_delete_face(user_id: str, face_id: str) -> JSONResponse:
    """Delete a single face entry from the library."""
    if not _face_available:
        return JSONResponse({"error": "Face recognition not available"}, status_code=503)

    try:
        from tools.face_tools import delete_face

        deleted = await asyncio.to_thread(delete_face, user_id, face_id)
        if deleted:
            return JSONResponse({"status": "deleted", "face_id": face_id})
        return JSONResponse({"error": "Face not found"}, status_code=404)
    except Exception as e:
        logger.exception("Delete face failed")
        return JSONResponse({"error": str(e)}, status_code=500)


@router.delete("/{user_id}")
async def api_clear_face_library(user_id: str, request: Request) -> JSONResponse:
    """Clear all faces or delete a specific person from the user's library.

    Query params:
        person_name: Optional. If provided, only delete that person's entries.
    """
    if not _face_available:
        return JSONResponse({"error": "Face recognition not available"}, status_code=503)

    person_name = request.query_params.get("person_name")

    try:
        from tools.face_tools import delete_all_faces

        count = await asyncio.to_thread(delete_all_faces, user_id, person_name)
        if person_name:
            return JSONResponse({
                "status": "deleted",
                "person_name": person_name,
                "deleted_count": count,
            })
        return JSONResponse({"status": "cleared", "deleted_count": count})
    except Exception as e:
        logger.exception("Clear face library failed")
        return JSONResponse({"error": str(e)}, status_code=500)

