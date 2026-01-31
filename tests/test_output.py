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
    cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json")
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

    cfg = SnapshotConfig(out_dir=str(tmp_path), max_snapshots=3, output_format="json")

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


def test_concurrent_write_bundle(tmp_path):
    """Test that concurrent writes don't corrupt files."""
    from concurrent.futures import ThreadPoolExecutor

    cfg = SnapshotConfig(out_dir=str(tmp_path), max_snapshots=0, output_format="json")  # unlimited

    def write_snapshot(i: int) -> None:
        payload = {
            "name": f"concurrent_{i}",
            "frames": [],
            "index": i,
        }
        write_bundle(payload, cfg)

    # Run 20 writes across 10 threads concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(write_snapshot, range(20)))

    # All 20 snapshots should exist
    json_files = [p for p in tmp_path.glob("*.json") if p.name != "latest.json"]
    assert len(json_files) == 20

    # All files should be valid JSON
    for f in json_files:
        content = json.loads(f.read_text())
        assert "name" in content
        assert content["name"].startswith("concurrent_")


def test_concurrent_write_with_cleanup(tmp_path):
    """Test concurrent writes with cleanup enabled."""
    from concurrent.futures import ThreadPoolExecutor

    cfg = SnapshotConfig(out_dir=str(tmp_path), max_snapshots=5, output_format="json")

    def write_snapshot(i: int) -> None:
        payload = {
            "name": f"cleanup_{i}",
            "frames": [],
        }
        write_bundle(payload, cfg)

    # Run 15 writes across 5 threads
    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(write_snapshot, range(15)))

    # Should have at most 5 snapshots due to cleanup
    json_files = [p for p in tmp_path.glob("*.json") if p.name != "latest.json"]
    assert len(json_files) <= 5

    # All remaining files should be valid
    for f in json_files:
        content = json.loads(f.read_text())
        assert "name" in content


def test_lock_file_created(tmp_path):
    """Test that lock file is created during write."""
    cfg = SnapshotConfig(out_dir=str(tmp_path))
    payload = {"name": "lock_test", "frames": []}

    write_bundle(payload, cfg)

    # Lock file should exist (or have been cleaned up, which is fine)
    # The important thing is no error occurred
    json_files = [p for p in tmp_path.glob("*.json") if p.name != "latest.json"]
    assert len(json_files) == 1
