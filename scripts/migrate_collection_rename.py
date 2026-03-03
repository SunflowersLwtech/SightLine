#!/usr/bin/env python3
"""Migrate Firestore collection: users → user_profiles.

Copies all documents (including subcollections: memories, face_library,
sessions_meta) from `users/{id}` to `user_profiles/{id}`, preserving
all fields including Vector embeddings.

Usage:
    conda activate sightline
    python scripts/migrate_collection_rename.py              # dry-run + backup + copy
    python scripts/migrate_collection_rename.py --delete-old # also remove old collection

Requires:
    - SA JSON at SightLine/sightline-backend-sa.json
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.cloud import firestore
from google.oauth2 import service_account

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SA_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sightline-backend-sa.json",
)
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "sightline-hackathon")

OLD_COLLECTION = "users"
NEW_COLLECTION = "user_profiles"
SUBCOLLECTIONS = ["memories", "face_library", "sessions_meta"]


def get_client() -> firestore.Client:
    """Create a Firestore client using SA credentials."""
    if os.path.exists(SA_JSON):
        creds = service_account.Credentials.from_service_account_file(SA_JSON)
        return firestore.Client(project=PROJECT_ID, credentials=creds)
    return firestore.Client(project=PROJECT_ID)


def _serialize_for_json(data: dict) -> dict:
    """Convert Firestore-specific types to JSON-serializable equivalents."""
    out = {}
    for k, v in data.items():
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        elif hasattr(v, "to_map_value"):
            # Firestore Vector → list of floats
            out[k] = list(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [
                x.isoformat() if hasattr(x, "isoformat") else x for x in v
            ]
        elif isinstance(v, dict):
            out[k] = _serialize_for_json(v)
        else:
            out[k] = v
    return out


def backup_collection(db: firestore.Client, backup_dir: Path) -> dict:
    """Backup all documents from the old collection to JSON files.

    Returns a summary dict with counts per document and subcollection.
    """
    backup_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    docs = list(db.collection(OLD_COLLECTION).stream())
    logger.info("Found %d documents in '%s'", len(docs), OLD_COLLECTION)

    for doc in docs:
        doc_id = doc.id
        data = doc.to_dict()
        entry: dict = {"profile": _serialize_for_json(data), "subcollections": {}}

        for sub_name in SUBCOLLECTIONS:
            sub_docs = list(
                db.collection(OLD_COLLECTION)
                .document(doc_id)
                .collection(sub_name)
                .stream()
            )
            entry["subcollections"][sub_name] = {
                sd.id: _serialize_for_json(sd.to_dict()) for sd in sub_docs
            }

        summary[doc_id] = {
            "profile_fields": len(data),
            **{
                f"sub_{s}": len(entry["subcollections"][s])
                for s in SUBCOLLECTIONS
            },
        }

        backup_path = backup_dir / f"{doc_id}.json"
        with open(backup_path, "w") as f:
            json.dump(entry, f, indent=2, default=str)
        logger.info("  Backed up %s → %s", doc_id, backup_path)

    return summary


def copy_collection(db: firestore.Client) -> dict:
    """Copy all documents and subcollections from old to new collection.

    Returns counts: {doc_id: {profile: bool, sub_name: int, ...}}
    """
    results: dict = {}
    docs = list(db.collection(OLD_COLLECTION).stream())

    for doc in docs:
        doc_id = doc.id
        data = doc.to_dict()
        entry: dict = {"profile_copied": False}

        # Copy top-level document
        db.collection(NEW_COLLECTION).document(doc_id).set(data)
        entry["profile_copied"] = True
        logger.info("  Copied %s/%s → %s/%s", OLD_COLLECTION, doc_id, NEW_COLLECTION, doc_id)

        # Copy subcollections
        for sub_name in SUBCOLLECTIONS:
            sub_docs = list(
                db.collection(OLD_COLLECTION)
                .document(doc_id)
                .collection(sub_name)
                .stream()
            )
            count = 0
            for sd in sub_docs:
                (
                    db.collection(NEW_COLLECTION)
                    .document(doc_id)
                    .collection(sub_name)
                    .document(sd.id)
                    .set(sd.to_dict())
                )
                count += 1
            entry[f"sub_{sub_name}"] = count
            if count > 0:
                logger.info(
                    "    Copied %d docs: %s/%s/%s → %s/%s/%s",
                    count, OLD_COLLECTION, doc_id, sub_name,
                    NEW_COLLECTION, doc_id, sub_name,
                )

        results[doc_id] = entry

    return results


def verify_migration(db: firestore.Client) -> bool:
    """Verify that new collection has same document counts as old."""
    old_docs = {doc.id for doc in db.collection(OLD_COLLECTION).stream()}
    new_docs = {doc.id for doc in db.collection(NEW_COLLECTION).stream()}

    if old_docs != new_docs:
        logger.error(
            "Document ID mismatch! old=%s new=%s",
            old_docs - new_docs, new_docs - old_docs,
        )
        return False

    for doc_id in old_docs:
        for sub_name in SUBCOLLECTIONS:
            old_sub = {
                sd.id
                for sd in db.collection(OLD_COLLECTION)
                .document(doc_id)
                .collection(sub_name)
                .stream()
            }
            new_sub = {
                sd.id
                for sd in db.collection(NEW_COLLECTION)
                .document(doc_id)
                .collection(sub_name)
                .stream()
            }
            if old_sub != new_sub:
                logger.error(
                    "Subcollection mismatch for %s/%s: old=%d new=%d",
                    doc_id, sub_name, len(old_sub), len(new_sub),
                )
                return False

    # Spot-check embedding dimensions
    for doc_id in old_docs:
        _verify_embeddings(db, doc_id)

    logger.info("Verification PASSED: all documents and subcollections match")
    return True


def _verify_embeddings(db: firestore.Client, doc_id: str) -> None:
    """Spot-check that embedding dimensions are preserved."""
    # Check face embeddings (512-D)
    face_docs = list(
        db.collection(NEW_COLLECTION)
        .document(doc_id)
        .collection("face_library")
        .limit(1)
        .stream()
    )
    for fd in face_docs:
        emb = fd.to_dict().get("embedding")
        if emb is not None:
            dim = len(list(emb)) if hasattr(emb, "__iter__") else 0
            if dim == 512:
                logger.info("  Face embedding check OK: %s (512-D)", doc_id)
            else:
                logger.warning("  Face embedding unexpected dim=%d for %s", dim, doc_id)

    # Check memory embeddings (2048-D)
    mem_docs = list(
        db.collection(NEW_COLLECTION)
        .document(doc_id)
        .collection("memories")
        .limit(1)
        .stream()
    )
    for md in mem_docs:
        emb = md.to_dict().get("embedding")
        if emb is not None:
            dim = len(list(emb)) if hasattr(emb, "__iter__") else 0
            if dim == 2048:
                logger.info("  Memory embedding check OK: %s (2048-D)", doc_id)
            else:
                logger.warning("  Memory embedding unexpected dim=%d for %s", dim, doc_id)


def delete_old_collection(db: firestore.Client) -> int:
    """Delete all documents (including subcollection docs) from the old collection."""
    total = 0
    docs = list(db.collection(OLD_COLLECTION).stream())

    for doc in docs:
        # Delete subcollection documents first
        for sub_name in SUBCOLLECTIONS:
            sub_docs = list(
                db.collection(OLD_COLLECTION)
                .document(doc.id)
                .collection(sub_name)
                .stream()
            )
            for sd in sub_docs:
                sd.reference.delete()
                total += 1

        # Delete the parent document
        doc.reference.delete()
        total += 1
        logger.info("  Deleted %s/%s (and subcollections)", OLD_COLLECTION, doc.id)

    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Migrate Firestore collection: {OLD_COLLECTION} → {NEW_COLLECTION}",
    )
    parser.add_argument(
        "--delete-old",
        action="store_true",
        help=f"Delete the old '{OLD_COLLECTION}' collection after successful migration",
    )
    args = parser.parse_args()

    db = get_client()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = Path(__file__).parent.parent / "backups" / f"migration_{ts}"

    # Step 1: Backup
    print(f"\n=== Step 1: Backup '{OLD_COLLECTION}' → {backup_dir} ===")
    summary = backup_collection(db, backup_dir)
    print(f"Backup summary: {json.dumps(summary, indent=2)}")

    # Step 2: Copy
    print(f"\n=== Step 2: Copy '{OLD_COLLECTION}' → '{NEW_COLLECTION}' ===")
    results = copy_collection(db)
    print(f"Copy results: {json.dumps(results, indent=2)}")

    # Step 3: Verify
    print(f"\n=== Step 3: Verify migration ===")
    ok = verify_migration(db)
    if not ok:
        print("VERIFICATION FAILED — aborting. Old collection untouched.")
        sys.exit(1)

    # Step 4: Optionally delete old collection
    if args.delete_old:
        print(f"\n=== Step 4: Delete old '{OLD_COLLECTION}' collection ===")
        deleted = delete_old_collection(db)
        print(f"Deleted {deleted} documents from '{OLD_COLLECTION}'")
    else:
        print(f"\n=== Step 4: Skipped (use --delete-old to remove '{OLD_COLLECTION}') ===")

    print("\nMigration complete!")


if __name__ == "__main__":
    main()
