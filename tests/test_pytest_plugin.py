"""Tests for pytest plugin."""

import subprocess
import sys

from llmdebug import get_latest_snapshot


def test_plugin_captures_on_failure(tmp_path):
    """Test that the pytest plugin captures snapshots on test failures."""
    # Create a test file
    test_file = tmp_path / "test_example.py"
    test_file.write_text(
        """
def test_failing():
    x = [1, 2, 3]
    assert False, "intentional failure"
"""
    )

    # Run pytest in subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    # Test should fail
    assert result.returncode != 0

    # Check that snapshot was created
    llmdebug_dir = tmp_path / ".llmdebug"
    assert llmdebug_dir.exists(), f"No .llmdebug dir. stderr: {result.stderr}"

    snapshots = list(llmdebug_dir.glob("*.json"))
    assert len(snapshots) >= 1  # At least the snapshot (and maybe latest.json)

    # Check that repro command is present (use get_latest_snapshot to auto-expand keys)
    snapshot = get_latest_snapshot(str(llmdebug_dir))
    assert snapshot is not None, "Failed to read snapshot"
    repro = snapshot.get("pytest", {}).get("repro")
    assert isinstance(repro, list)
    assert "-m" in repro
    assert "pytest" in repro
    assert "test_example.py::test_failing" in repro


def test_no_snapshot_marker(tmp_path):
    """Test that no_snapshot marker prevents capture."""
    test_file = tmp_path / "test_no_capture.py"
    test_file.write_text(
        """
import pytest

@pytest.mark.no_snapshot
def test_failing_no_capture():
    assert False, "should not capture"
"""
    )

    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    # Test should fail
    assert result.returncode != 0

    # Check that no snapshot was created for this specific test
    llmdebug_dir = tmp_path / ".llmdebug"
    if llmdebug_dir.exists():
        snapshots = list(llmdebug_dir.glob("*test_failing_no_capture*.json"))
        assert len(snapshots) == 0, "Snapshot should not be created for no_snapshot marker"


def test_passing_test_no_snapshot(tmp_path):
    """Test that passing tests don't create snapshots."""
    test_file = tmp_path / "test_passing.py"
    test_file.write_text(
        """
def test_passing():
    assert True
"""
    )

    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    # Test should pass
    assert result.returncode == 0

    # Check that no snapshot was created
    llmdebug_dir = tmp_path / ".llmdebug"
    assert not llmdebug_dir.exists() or len(list(llmdebug_dir.glob("*.json"))) == 0
