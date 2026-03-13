"""Shared Firestore client factory for lightweight runtime consumers."""

from __future__ import annotations

from functools import lru_cache

from google.cloud import firestore

from config import get_google_cloud_project


@lru_cache(maxsize=1)
def get_firestore_client() -> firestore.Client:
    """Return a cached Firestore client."""
    return firestore.Client(project=get_google_cloud_project())


def reset_firestore_client_for_testing() -> None:
    """Clear the shared Firestore client cache for tests."""
    get_firestore_client.cache_clear()


__all__ = ["get_firestore_client", "reset_firestore_client_for_testing"]
