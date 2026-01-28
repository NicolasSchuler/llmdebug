"""Tests for output and file management."""

import json

from llmdebug.config import SnapshotConfig
from llmdebug.output import get_latest_snapshot, write_bundle


def test_write_bundle_creates_files(tmp_path):
    """Test that write_bundle creates JSON and traceback files."""
    cfg = SnapshotConfig(out_dir=str(tmp_path))
    payload = {
        "name": "test_snapshot",
        "exception": {"type": "ValueError", "message": "test"},
        "traceback": "Traceback (most recent call last):\n  ...",
        "frames": [],
    }

    json_path = write_bundle(payload, cfg)

    assert json_path.exists()
    assert json_path.suffix == ".json"

    # Check traceback file was created
    tb_path = json_path.with_suffix("").with_suffix(".traceback.txt")
    assert tb_path.exists()

    # Check latest.json
    latest = tmp_path / "latest.json"
    assert latest.exists() or latest.is_symlink()


def test_get_latest_snapshot(tmp_path):
    """Test reading the latest snapshot."""
    cfg = SnapshotConfig(out_dir=str(tmp_path))
    payload = {
        "name": "test",
        "exception": {"type": "Error", "message": "msg"},
        "frames": [],
    }

    write_bundle(payload, cfg)

    result = get_latest_snapshot(str(tmp_path))
    assert result is not None
    assert result["name"] == "test"


def test_get_latest_snapshot_missing(tmp_path):
    """Test that missing snapshot returns None."""
    result = get_latest_snapshot(str(tmp_path))
    assert result is None


def test_atomic_write(tmp_path):
    """Test that writes are atomic (no partial files)."""
    cfg = SnapshotConfig(out_dir=str(tmp_path))
    payload = {
        "name": "atomic_test",
        "frames": [],
    }

    json_path = write_bundle(payload, cfg)

    # No .tmp files should remain
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert len(tmp_files) == 0

    # JSON should be valid
    content = json.loads(json_path.read_text())
    assert content["name"] == "atomic_test"


def test_snapshot_cleanup(tmp_path):
    """Test that old snapshots are cleaned up when max_snapshots is exceeded."""
    import time

    cfg = SnapshotConfig(out_dir=str(tmp_path), max_snapshots=3)

    # Create 5 snapshots
    for i in range(5):
        payload = {
            "name": f"snapshot_{i}",
            "frames": [],
            "traceback": f"traceback {i}",
        }
        write_bundle(payload, cfg)
        time.sleep(0.01)  # Ensure different timestamps

    # Should only have 3 snapshots (+ latest.json)
    json_files = [p for p in tmp_path.glob("*.json") if p.name != "latest.json"]
    assert len(json_files) == 3

    # The newest snapshots should be kept (snapshot_2, snapshot_3, snapshot_4)
    names = sorted(json.loads(f.read_text())["name"] for f in json_files)
    assert names == ["snapshot_2", "snapshot_3", "snapshot_4"]


def test_snapshot_cleanup_unlimited(tmp_path):
    """Test that max_snapshots=0 means unlimited."""
    cfg = SnapshotConfig(out_dir=str(tmp_path), max_snapshots=0)

    # Create 10 snapshots
    for i in range(10):
        payload = {"name": f"snapshot_{i}", "frames": []}
        write_bundle(payload, cfg)

    # All 10 should exist
    json_files = [p for p in tmp_path.glob("*.json") if p.name != "latest.json"]
    assert len(json_files) == 10


def test_snapshot_cleanup_removes_traceback_files(tmp_path):
    """Test that traceback files are also cleaned up."""
    import time

    cfg = SnapshotConfig(out_dir=str(tmp_path), max_snapshots=2)

    for i in range(4):
        payload = {
            "name": f"snap_{i}",
            "frames": [],
            "traceback": f"tb {i}",
        }
        write_bundle(payload, cfg)
        time.sleep(0.01)

    # Should have 2 JSON files and 2 traceback files
    json_files = [p for p in tmp_path.glob("*.json") if p.name != "latest.json"]
    tb_files = list(tmp_path.glob("*.traceback.txt"))

    assert len(json_files) == 2
    assert len(tb_files) == 2
