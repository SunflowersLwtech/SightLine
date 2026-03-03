"""SightLine face recognition agent.

Provides face detection, embedding generation, and identity matching
using InsightFace (buffalo_l) with ArcFace 512-D embeddings.

Privacy: Raw images are NEVER stored — only L2-normalized embeddings.
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Optional

import cv2
import numpy as np
from insightface.app import FaceAnalysis
import os
from tools.tool_behavior import ToolBehavior, behavior_to_text

logger = logging.getLogger(__name__)

# Cosine similarity threshold for positive match
MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.4"))

# Singleton FaceAnalysis instance (lazy-initialized)
_face_app: Optional[FaceAnalysis] = None


def _get_face_app() -> FaceAnalysis:
    """Return a lazily-initialized FaceAnalysis singleton."""
    global _face_app
    if _face_app is None:
        _face_app = FaceAnalysis(
            name="buffalo_l",
            root=os.path.expanduser("~/.insightface"),
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("InsightFace buffalo_l model loaded")
    return _face_app


def _decode_image(image_base64: str) -> np.ndarray:
    """Decode a base64-encoded image to a BGR numpy array (OpenCV format)."""
    raw = base64.b64decode(image_base64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image from base64 data")
    return img


def detect_faces(image_bytes: bytes) -> list[dict]:
    """Detect faces in a raw image (JPEG/PNG bytes).

    Args:
        image_bytes: Raw image file bytes.

    Returns:
        List of dicts with keys: bbox, score, embedding, face_image.
        - bbox: [x1, y1, x2, y2] pixel coordinates
        - score: detection confidence (0-1)
        - embedding: 512-D L2-normalized numpy array
        - face_image: cropped+aligned face as numpy array (for display only, not stored)
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes")

    app = _get_face_app()
    faces = app.get(img)

    results = []
    for face in faces:
        results.append({
            "bbox": face.bbox.tolist(),
            "score": float(face.det_score),
            "embedding": face.normed_embedding,
            "face_image": face.normed_embedding is not None,  # flag only
        })
    return results


def generate_embedding(face_image: np.ndarray) -> np.ndarray:
    """Generate a 512-D ArcFace embedding from a BGR face image.

    The image is passed through InsightFace detection+recognition.
    Returns the normed_embedding of the first detected face.

    Args:
        face_image: BGR numpy array containing a face.

    Returns:
        512-D L2-normalized embedding as numpy array.

    Raises:
        ValueError: If no face is detected in the image.
    """
    app = _get_face_app()
    faces = app.get(face_image)
    if not faces:
        raise ValueError("No face detected in the provided image")
    return faces[0].normed_embedding


def match_face(
    embedding: np.ndarray,
    library: list[dict],
) -> Optional[dict]:
    """Match an embedding against a face library using cosine similarity.

    Args:
        embedding: 512-D L2-normalized query embedding.
        library: List of dicts with at least 'embedding', 'person_name',
                 'face_id' keys. Each 'embedding' is a 512-D numpy array.

    Returns:
        Best match dict with added 'similarity' key, or None if no match
        exceeds MATCH_THRESHOLD.
    """
    if not library:
        return None

    # Stack all library embeddings for vectorized cosine similarity
    lib_embeddings = np.stack([entry["embedding"] for entry in library])
    # Both are L2-normalized, so dot product == cosine similarity
    similarities = lib_embeddings @ embedding

    best_idx = int(np.argmax(similarities))
    best_sim = float(similarities[best_idx])

    if best_sim > MATCH_THRESHOLD:
        result = {**library[best_idx], "similarity": best_sim}
        # Remove the raw embedding from the returned result
        result.pop("embedding", None)
        return result
    return None


def identify_persons_in_frame(
    image_base64: str,
    user_id: str,
    face_library: Optional[list[dict]] = None,
    behavior: ToolBehavior | str = ToolBehavior.SILENT,
) -> list[dict]:
    """End-to-end pipeline: detect faces -> embed -> match against library.

    Args:
        image_base64: Base64-encoded JPEG/PNG image.
        user_id: User ID (used if face_library is not pre-loaded).
        face_library: Pre-loaded face library (list of dicts with
                      'embedding', 'person_name', etc.). If None, an empty
                      list is used (caller should pre-load via
                      face_tools.load_face_library).
        behavior: Delivery policy for downstream integration. Defaults to
                  SILENT to avoid hard interruption during dialogue.

    Returns:
        List of dicts, one per detected face, with keys:
        - bbox: [x1, y1, x2, y2]
        - score: detection confidence
        - person_name: matched name or "unknown"
        - relationship: matched relationship or None
        - similarity: cosine similarity score (0 if unknown)
    """
    t0 = time.monotonic()

    img = _decode_image(image_base64)
    app = _get_face_app()
    faces = app.get(img)

    if face_library is None:
        face_library = []

    results = []
    for face in faces:
        emb = face.normed_embedding
        match = match_face(emb, face_library) if face_library else None

        results.append({
            "bbox": face.bbox.tolist(),
            "score": float(face.det_score),
            "person_name": match["person_name"] if match else "unknown",
            "relationship": match.get("relationship") if match else None,
            "similarity": match["similarity"] if match else 0.0,
        })

    elapsed_ms = (time.monotonic() - t0) * 1000
    logger.info(
        "identify_persons_in_frame: %d faces, %.0f ms, behavior=%s",
        len(results),
        elapsed_ms,
        behavior_to_text(behavior),
    )
    return results
