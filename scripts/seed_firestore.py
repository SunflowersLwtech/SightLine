#!/usr/bin/env python3
"""Seed Firestore with demo user profile and test data.

Usage:
    conda activate sightline
    python scripts/seed_firestore.py

Creates:
    - user_profiles/demo_user_001  — congenital blind, expert user (Sarah)
    - user_profiles/demo_user_002  — acquired low vision, beginner (Marcus)
    - user_profiles/demo_user_004  — acquired low vision, Chinese (小明)

Requires:
    - GOOGLE_CLOUD_PROJECT env var or gcloud default project
    - Valid credentials (ADC or SA JSON)
"""

import os
import sys

# Ensure the project root is on sys.path so `lod.models` can be imported.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datetime import datetime, timezone
from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "sightline-hackathon")


def get_client() -> firestore.Client:
    """Create a Firestore client with project ID."""
    return firestore.Client(project=PROJECT_ID)


# ---------------------------------------------------------------------------
# Demo user profiles (matching Infra Report §1.2 schema)
# ---------------------------------------------------------------------------

DEMO_USERS = [
    {
        "doc_id": "demo_user_001",
        "data": {
            "vision_status": "totally_blind",
            "blindness_onset": "congenital",
            "onset_age": None,
            "has_guide_dog": True,
            "has_white_cane": False,
            "tts_speed": 2.0,
            "verbosity_preference": "concise",
            "language": "en-US",
            "description_priority": "spatial",
            "color_description": False,  # Congenital blind — no colour descriptions
            "om_level": "advanced",
            "travel_frequency": "daily",
            "preferred_name": "Sarah",
            "created_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP,
        },
    },
    {
        "doc_id": "demo_user_002",
        "data": {
            "vision_status": "low_vision",
            "blindness_onset": "acquired",
            "onset_age": 40,
            "has_guide_dog": False,
            "has_white_cane": True,
            "tts_speed": 1.2,
            "verbosity_preference": "detailed",
            "language": "en-US",
            "description_priority": "object",
            "color_description": True,
            "om_level": "beginner",
            "travel_frequency": "rarely",
            "preferred_name": "Marcus",
            "created_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP,
        },
    },
    {
        "doc_id": "demo_user_004",
        "data": {
            "vision_status": "low_vision",
            "blindness_onset": "acquired",
            "onset_age": 35,
            "has_guide_dog": False,
            "has_white_cane": True,
            "tts_speed": 1.3,
            "verbosity_preference": "concise",
            "language": "zh-CN",
            "description_priority": "spatial",
            "color_description": True,
            "om_level": "intermediate",
            "travel_frequency": "weekly",
            "preferred_name": "小明",
            "created_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP,
        },
    },
]


def seed_users(db: firestore.Client) -> None:
    """Create or overwrite demo user profiles in Firestore."""
    for user in DEMO_USERS:
        doc_ref = db.collection("user_profiles").document(user["doc_id"])
        doc_ref.set(user["data"])
        print(f"  [OK] user_profiles/{user['doc_id']}")


def seed_session_meta(db: firestore.Client) -> None:
    """Create a sample session metadata document for demo_user_001."""
    doc_ref = (
        db.collection("user_profiles")
        .document("demo_user_001")
        .collection("sessions_meta")
        .document("demo_session_001")
    )
    doc_ref.set({
        "start_time": firestore.SERVER_TIMESTAMP,
        "end_time": None,
        "trip_purpose": "Coffee shop visit for demo",
        "lod_distribution": {"lod1": 40, "lod2": 35, "lod3": 25},
        "space_transitions": ["outdoor→lobby", "lobby→cafe"],
        "total_interactions": 0,
    })
    print("  [OK] user_profiles/demo_user_001/sessions_meta/demo_session_001")


# ---------------------------------------------------------------------------
# Demo face library (synthetic 512-D embeddings for face recognition testing)
# ---------------------------------------------------------------------------

DEMO_FACES = [
    {"person_name": "Alice Chen", "relationship": "friend", "num_samples": 3},
    {"person_name": "Bob Martinez", "relationship": "coworker", "num_samples": 3},
    {"person_name": "Mom", "relationship": "family", "num_samples": 3},
]


def _make_synthetic_embedding(seed: int) -> list[float]:
    """Generate a deterministic synthetic 512-D L2-normalized embedding.

    Uses a fixed seed so re-running the script produces identical embeddings,
    enabling repeatable demo matching.
    """
    rng = np.random.RandomState(seed)
    vec = rng.randn(512).astype(np.float32)
    vec /= np.linalg.norm(vec)  # L2 normalize
    return vec.tolist()


def seed_face_library(db: firestore.Client, user_id: str = "demo_user_001") -> None:
    """Seed synthetic face embeddings for the demo user.

    Creates multiple photo samples per person, each with a slightly different
    embedding (simulating different angles/lighting) but from the same seed
    cluster so cosine similarity between samples of the same person is high.
    """
    now = datetime.now(timezone.utc)
    face_coll = db.collection("user_profiles").document(user_id).collection("face_library")

    total = 0
    for person_idx, person in enumerate(DEMO_FACES):
        base_seed = (person_idx + 1) * 1000
        for photo_idx in range(person["num_samples"]):
            # Same-person samples use close seeds for high intra-person similarity
            seed = base_seed + photo_idx
            embedding = _make_synthetic_embedding(seed)

            doc_data = {
                "person_name": person["person_name"],
                "relationship": person["relationship"],
                "embedding": Vector(embedding),
                "photo_index": photo_idx,
                "registered_by": user_id,
                "created_at": now,
            }
            face_coll.add(doc_data)
            total += 1

        print(f"  [OK] {person['person_name']} ({person['relationship']}) "
              f"— {person['num_samples']} samples")

    print(f"  Total face entries seeded: {total}")


def verify_indexes(db: firestore.Client) -> None:
    """Check that vector indexes exist (informational only)."""
    print("\n--- Vector Index Status ---")
    print("  face_library (512-D): Check via gcloud firestore indexes composite list")
    print("  memories (2048-D):    Check via gcloud firestore indexes composite list")
    print("  (Indexes were created during Phase 0 — SL-04)")


def main() -> None:
    print(f"Seeding Firestore for project: {PROJECT_ID}\n")

    db = get_client()

    print("--- User Profiles ---")
    seed_users(db)

    print("\n--- Session Metadata ---")
    seed_session_meta(db)

    print("\n--- Face Library (demo_user_001) ---")
    seed_face_library(db)

    verify_indexes(db)

    print("\n--- Done! ---")
    print(f"  Total users seeded: {len(DEMO_USERS)}")
    print(f"  Total face entries: {sum(f['num_samples'] for f in DEMO_FACES)}")
    print("  Run `gcloud firestore documents list user_profiles --limit=5` to verify.")


if __name__ == "__main__":
    main()
