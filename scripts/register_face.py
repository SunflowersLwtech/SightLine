#!/usr/bin/env python3
"""Register real face photos into the SightLine Firestore face library.

Usage:
    conda activate sightline
    cd SightLine/

    # Register 3 photos of one person:
    python scripts/register_face.py \
        --user demo_user_001 \
        --name "David" \
        --relationship "boss" \
        --consent \
        --store-photo \
        --photos /path/to/david1.jpg /path/to/david2.jpg /path/to/david3.jpg

    # List all registered faces:
    python scripts/register_face.py --user demo_user_001 --list

    # Clear all faces for a user:
    python scripts/register_face.py --user demo_user_001 --clear

    # Clear all faces for one specific person:
    python scripts/register_face.py --user demo_user_001 --delete-person "David"
"""

import argparse
import base64
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.face_tools import register_face, list_faces, clear_face_library, delete_all_faces


def read_image_as_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def main():
    parser = argparse.ArgumentParser(description="SightLine face registration tool")
    parser.add_argument("--user", required=True, help="User ID (e.g. demo_user_001)")
    parser.add_argument("--name", help="Person's name")
    parser.add_argument("--relationship", default="friend", help="Relationship (e.g. friend, spouse, coworker)")
    parser.add_argument("--photos", nargs="+", help="Paths to photo files (3-5 recommended)")
    parser.add_argument("--consent", action="store_true", help="Confirm consent for storing familiar-face data")
    parser.add_argument("--store-photo", action="store_true", help="Store compressed reference photo in Firestore")
    parser.add_argument("--list", action="store_true", help="List all registered faces")
    parser.add_argument("--clear", action="store_true", help="Clear ALL faces for this user")
    parser.add_argument("--delete-person", help="Delete all entries for a specific person name")
    args = parser.parse_args()

    if args.list:
        faces = list_faces(args.user)
        if not faces:
            print(f"No faces registered for user '{args.user}'")
        else:
            print(f"Registered faces for user '{args.user}':")
            from itertools import groupby
            faces.sort(key=lambda x: x["person_name"])
            for name, group in groupby(faces, key=lambda x: x["person_name"]):
                entries = list(group)
                print(f"  {name} ({entries[0]['relationship']}) — {len(entries)} photo(s)")
        return

    if args.clear:
        count = clear_face_library(args.user)
        print(f"Cleared {count} face(s) for user '{args.user}'")
        return

    if args.delete_person:
        count = delete_all_faces(args.user, args.delete_person)
        print(f"Deleted {count} face(s) for '{args.delete_person}'")
        return

    if not args.name or not args.photos:
        parser.error("--name and --photos are required for registration")

    if args.store_photo and not args.consent:
        parser.error("--store-photo requires --consent")

    if len(args.photos) < 3:
        print(f"Warning: only {len(args.photos)} photo(s) provided, 3-5 recommended for best accuracy")

    print(f"Registering '{args.name}' ({args.relationship}) for user '{args.user}'...")
    success = 0
    for i, photo_path in enumerate(args.photos):
        if not os.path.exists(photo_path):
            print(f"  [SKIP] File not found: {photo_path}")
            continue
        try:
            image_b64 = read_image_as_base64(photo_path)
            result = register_face(
                user_id=args.user,
                person_name=args.name,
                relationship=args.relationship,
                image_base64=image_b64,
                photo_index=i,
                consent_confirmed=args.consent,
                store_reference_photo=args.store_photo,
            )
            print(f"  [OK] Photo {i+1}: face_id={result['face_id']}")
            success += 1
        except ValueError as e:
            print(f"  [FAIL] Photo {i+1} ({photo_path}): {e}")
        except Exception as e:
            print(f"  [ERROR] Photo {i+1} ({photo_path}): {e}")

    print(f"Done: {success}/{len(args.photos)} photos registered successfully.")
    if success >= 3:
        print(f"✓ '{args.name}' is ready for real-time face recognition.")
    else:
        print(f"⚠ Need at least 3 successful registrations. Please add more photos.")


if __name__ == "__main__":
    main()
