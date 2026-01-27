"""Tests for exception capture."""

import pytest

from llmdebug import debug_snapshot, snapshot_section


def test_decorator_captures_exception(tmp_path):
    """Test that decorator captures exception and still raises."""

    @debug_snapshot(out_dir=str(tmp_path))
    def failing_func():
        x = [1, 2, 3]  # noqa: F841 - intentional for testing locals capture
        raise ValueError("test error")

    with pytest.raises(ValueError, match="test error"):
        failing_func()

    # Check snapshot was created
    snapshots = list(tmp_path.glob("*.json"))
    assert len(snapshots) == 2  # snapshot + latest.json


def test_context_manager_captures_exception(tmp_path):
    """Test that context manager captures exception."""
    with pytest.raises(RuntimeError):
        with snapshot_section("test_section", out_dir=str(tmp_path)):
            data = {"key": "value"}  # noqa: F841 - intentional for testing locals capture
            raise RuntimeError("context error")

    snapshots = list(tmp_path.glob("*test_section*.json"))
    assert len(snapshots) == 1


def test_no_exception_no_snapshot(tmp_path):
    """Test that no snapshot is created when no exception."""

    @debug_snapshot(out_dir=str(tmp_path))
    def passing_func():
        return 42

    result = passing_func()
    assert result == 42

    snapshots = list(tmp_path.glob("*.json"))
    assert len(snapshots) == 0
