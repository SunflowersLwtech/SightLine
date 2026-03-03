"""SightLine Narrative Snapshot manager (SL-37).

Saves a checkpoint when LOD downgrades during an active task and
restores it when LOD upgrades again — so the user can resume
reading a menu, document, etc. from where they left off.

TTL: 10 minutes.  After that the snapshot expires and the task
restarts from scratch.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from lod.models import NarrativeSnapshot, SessionContext

logger = logging.getLogger("sightline.narrative")

SNAPSHOT_TTL = timedelta(minutes=10)


def save_snapshot(
    session: SessionContext,
    task_type: str,
    progress: str,
    remaining: list[str],
) -> None:
    """Save a narrative snapshot on LOD downgrade.

    Called when the LOD drops while ``session.active_task`` is set.
    """
    session.narrative_snapshot = NarrativeSnapshot(
        task_type=task_type,
        progress=progress,
        remaining=remaining,
        timestamp=datetime.now(timezone.utc),
    )
    logger.info(
        "Narrative snapshot saved: task=%s progress=%s",
        task_type,
        progress,
    )


def try_restore_snapshot(session: SessionContext) -> str | None:
    """Attempt to restore a narrative snapshot on LOD upgrade.

    Returns a prompt-injection string if the snapshot is valid,
    or ``None`` if expired / absent.
    """
    snap = session.narrative_snapshot
    if snap is None:
        return None

    age = datetime.now(timezone.utc) - snap.timestamp
    if age > SNAPSHOT_TTL:
        logger.info(
            "Narrative snapshot expired (age=%s > TTL=%s), discarding",
            age,
            SNAPSHOT_TTL,
        )
        session.narrative_snapshot = None
        return None

    # Build resume prompt
    remaining_str = ", ".join(snap.remaining) if snap.remaining else "unknown"
    prompt = (
        f"[RESUME] The user was previously doing '{snap.task_type}'. "
        f"Progress so far: {snap.progress}. "
        f"Remaining items: {remaining_str}. "
        f"Please continue from where they left off — do NOT restart from the beginning. "
        f"IMPORTANT: Still interrupt immediately for any safety hazards."
    )

    # Consume the snapshot
    session.narrative_snapshot = None
    logger.info("Narrative snapshot restored: %s", snap.task_type)
    return prompt


def on_lod_change(
    session: SessionContext,
    old_lod: int,
    new_lod: int,
) -> str | None:
    """Handle LOD transitions for narrative continuity.

    - Downgrade (e.g. LOD 3 → LOD 1): save snapshot if task active.
    - Upgrade (e.g. LOD 1 → LOD 3): try to restore snapshot.

    Returns an optional prompt-injection string for the restore case.
    """
    if new_lod < old_lod:
        # LOD downgrade — save if there's an active task
        if session.active_task:
            save_snapshot(
                session,
                task_type=session.active_task,
                progress="in progress",
                remaining=[],
            )
    elif new_lod > old_lod:
        # LOD upgrade — try to restore
        return try_restore_snapshot(session)

    return None
