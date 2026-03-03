"""Tests for SightLine narrative snapshot manager."""

from datetime import datetime, timedelta, timezone

from lod.models import NarrativeSnapshot, SessionContext
from lod.narrative_snapshot import (
    SNAPSHOT_TTL,
    on_lod_change,
    save_snapshot,
    try_restore_snapshot,
)


def test_save_snapshot():
    session = SessionContext()
    save_snapshot(session, task_type="menu_reading", progress="Read items 1-3", remaining=["item 4", "item 5"])
    assert session.narrative_snapshot is not None
    assert session.narrative_snapshot.task_type == "menu_reading"
    assert session.narrative_snapshot.progress == "Read items 1-3"


def test_restore_within_ttl():
    session = SessionContext()
    save_snapshot(session, task_type="document_reading", progress="Page 2", remaining=["page 3"])
    result = try_restore_snapshot(session)
    assert result is not None
    assert "[RESUME]" in result
    assert "document_reading" in result


def test_restore_expired():
    session = SessionContext()
    save_snapshot(session, task_type="test", progress="p1", remaining=[])
    # Manually expire the snapshot
    session.narrative_snapshot.timestamp = datetime.now(timezone.utc) - timedelta(minutes=11)
    result = try_restore_snapshot(session)
    assert result is None


def test_restore_consumes_snapshot():
    session = SessionContext()
    save_snapshot(session, task_type="test", progress="p1", remaining=[])
    result1 = try_restore_snapshot(session)
    assert result1 is not None
    result2 = try_restore_snapshot(session)
    assert result2 is None


def test_on_lod_downgrade_saves():
    session = SessionContext(active_task="reading menu", current_lod=3)
    result = on_lod_change(session, old_lod=3, new_lod=1)
    assert result is None  # downgrade returns None
    assert session.narrative_snapshot is not None
    assert session.narrative_snapshot.task_type == "reading menu"


def test_on_lod_upgrade_restores():
    session = SessionContext()
    save_snapshot(session, task_type="menu_reading", progress="item 3", remaining=["item 4"])
    result = on_lod_change(session, old_lod=1, new_lod=3)
    assert result is not None
    assert "[RESUME]" in result


def test_on_lod_downgrade_no_task():
    session = SessionContext(active_task=None, current_lod=3)
    on_lod_change(session, old_lod=3, new_lod=1)
    assert session.narrative_snapshot is None
