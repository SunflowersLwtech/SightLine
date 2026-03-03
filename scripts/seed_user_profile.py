#!/usr/bin/env python3
"""Seed Firestore with a demo UserProfile.

Usage:
    conda activate sightline
    python scripts/seed_user_profile.py <user_id> [--preset <preset_name>]

Examples:
    python scripts/seed_user_profile.py demo_user_001 --preset congenital_blind
    python scripts/seed_user_profile.py demo_user_002 --preset low_vision_acquired
    python scripts/seed_user_profile.py my_user --preset detailed_low_vision

Presets:
    congenital_blind    — Born blind, no color descriptions, concise, guide dog
    low_vision_acquired — Low vision, acquired, detailed, color enabled
    detailed_low_vision — Low vision, acquired, beginner O&M, detailed verbosity

Requires:
    - SA JSON at SightLine/sightline-backend-sa.json
"""

import argparse
import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.cloud import firestore
from google.oauth2 import service_account

SA_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sightline-backend-sa.json",
)
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "sightline-hackathon")

# ---------------------------------------------------------------------------
# Preset profiles
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict] = {
    "congenital_blind": {
        "vision_status": "totally_blind",
        "blindness_onset": "congenital",
        "onset_age": None,
        "has_guide_dog": True,
        "has_white_cane": False,
        "tts_speed": 2.0,
        "verbosity_preference": "concise",
        "language": "en-US",
        "description_priority": "spatial",
        "color_description": False,
        "om_level": "intermediate",
        "travel_frequency": "daily",
        "preferred_name": "Demo User",
    },
    "low_vision_acquired": {
        "vision_status": "low_vision",
        "blindness_onset": "acquired",
        "onset_age": 30,
        "has_guide_dog": False,
        "has_white_cane": True,
        "tts_speed": 1.3,
        "verbosity_preference": "detailed",
        "language": "en-US",
        "description_priority": "object",
        "color_description": True,
        "om_level": "intermediate",
        "travel_frequency": "weekly",
        "preferred_name": "Demo User",
    },
    "detailed_low_vision": {
        "vision_status": "low_vision",
        "blindness_onset": "acquired",
        "onset_age": 45,
        "has_guide_dog": False,
        "has_white_cane": False,
        "tts_speed": 1.2,
        "verbosity_preference": "detailed",
        "language": "en-US",
        "description_priority": "object",
        "color_description": True,
        "om_level": "beginner",
        "travel_frequency": "rarely",
        "preferred_name": "Demo User",
    },
}

# Default profile (same as congenital_blind preset)
DEFAULT_PROFILE: dict = PRESETS["congenital_blind"].copy()


def get_client() -> firestore.Client:
    """Create a Firestore client using the SA JSON."""
    if os.path.exists(SA_JSON):
        creds = service_account.Credentials.from_service_account_file(SA_JSON)
        return firestore.Client(project=PROJECT_ID, credentials=creds)
    return firestore.Client(project=PROJECT_ID)


def seed_user_profile(user_id: str, profile_data: dict) -> None:
    """Write a UserProfile document to Firestore."""
    db = get_client()
    doc_ref = db.collection("user_profiles").document(user_id)

    data = {**profile_data}
    data["created_at"] = firestore.SERVER_TIMESTAMP
    data["updated_at"] = firestore.SERVER_TIMESTAMP

    doc_ref.set(data)
    print(f"[OK] user_profiles/{user_id}")
    print(f"     vision_status:        {data.get('vision_status')}")
    print(f"     blindness_onset:      {data.get('blindness_onset')}")
    print(f"     verbosity_preference: {data.get('verbosity_preference')}")
    print(f"     color_description:    {data.get('color_description')}")
    print(f"     om_level:             {data.get('om_level')}")
    print(f"     language:             {data.get('language')}")
    print(f"     preferred_name:       {data.get('preferred_name')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed a UserProfile into Firestore.",
    )
    parser.add_argument(
        "user_id",
        help="Firestore document ID for the user (e.g. demo_user_001)",
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default=None,
        help="Use a preset profile instead of the default",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available presets and exit",
    )
    args = parser.parse_args()

    if args.list_presets:
        print("Available presets:")
        for name, data in PRESETS.items():
            print(f"  {name}:")
            for k, v in data.items():
                print(f"    {k}: {v}")
            print()
        return

    profile = PRESETS[args.preset].copy() if args.preset else DEFAULT_PROFILE.copy()
    print(f"Seeding UserProfile for project: {PROJECT_ID}")
    print(f"Preset: {args.preset or 'default (congenital_blind)'}\n")
    seed_user_profile(args.user_id, profile)
    print("\nDone!")


if __name__ == "__main__":
    main()
