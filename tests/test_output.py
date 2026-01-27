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
