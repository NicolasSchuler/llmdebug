"""Tests for exception capture."""

import pytest

from llmdebug import debug_snapshot, get_latest_snapshot, snapshot_section


def test_decorator_captures_exception(tmp_path):
    """Test that decorator captures exception and still raises."""

    @debug_snapshot(out_dir=str(tmp_path))
    def failing_func():
        x = [1, 2, 3]  # noqa: F841 - intentional for testing locals capture
        raise ValueError("test error")

    with pytest.raises(ValueError, match="test error"):
        failing_func()

    # Check snapshot was created
    snapshots = [path for path in tmp_path.glob("*.json") if path.name != "latest.json"]
    assert len(snapshots) == 1


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


def test_snapshot_contains_correct_structure(tmp_path):
    """Test that snapshot JSON contains all expected fields."""

    @debug_snapshot(out_dir=str(tmp_path))
    def failing_with_locals():
        my_list = [1, 2, 3]  # noqa: F841
        my_dict = {"key": "value"}  # noqa: F841
        my_number = 42  # noqa: F841
        raise ValueError("test error message")

    with pytest.raises(ValueError):
        failing_with_locals()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    # Check top-level structure
    assert snapshot["name"] == "failing_with_locals"
    assert "timestamp_utc" in snapshot
    assert snapshot["exception"]["type"] == "ValueError"
    assert snapshot["exception"]["qualified_type"] == "builtins.ValueError"
    assert snapshot["exception"]["message"] == "test error message"
    assert "traceback" in snapshot
    assert "frames" in snapshot
    assert snapshot["crash_frame_index"] == 0
    assert "env" in snapshot

    # Check frames structure (crash site should be first)
    assert len(snapshot["frames"]) > 0
    crash_frame = snapshot["frames"][0]
    assert "file" in crash_frame
    assert "line" in crash_frame
    assert "function" in crash_frame
    assert crash_frame["function"] == "failing_with_locals"
    assert "locals" in crash_frame

    # Check that locals were captured
    locals_dict = crash_frame["locals"]
    assert "my_list" in locals_dict
    assert locals_dict["my_list"] == [1, 2, 3]
    assert "my_dict" in locals_dict
    assert "my_number" in locals_dict
    assert locals_dict["my_number"] == 42


def test_snapshot_captures_array_shapes(tmp_path):
    """Test that numpy arrays are summarized with shape info."""
    pytest.importorskip("numpy")
    import numpy as np

    @debug_snapshot(out_dir=str(tmp_path))
    def failing_with_array():
        arr = np.zeros((10, 20, 30))  # noqa: F841
        raise RuntimeError("array error")

    with pytest.raises(RuntimeError):
        failing_with_array()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    crash_frame = snapshot["frames"][0]
    arr_info = crash_frame["locals"]["arr"]

    # Should have array summary, not raw data
    assert "__array__" in arr_info
    assert arr_info["shape"] == [10, 20, 30]
    assert arr_info["dtype"] == "float64"


def test_locals_mode_none_skips_locals(tmp_path):
    """Test that locals_mode='none' doesn't capture locals."""

    @debug_snapshot(out_dir=str(tmp_path), locals_mode="none")
    def failing_no_locals():
        secret = "password123"  # noqa: F841
        raise ValueError("error")

    with pytest.raises(ValueError):
        failing_no_locals()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    crash_frame = snapshot["frames"][0]
    assert "locals" not in crash_frame


def test_locals_mode_meta_captures_metadata(tmp_path):
    """Test that locals_mode='meta' captures locals_meta only."""

    @debug_snapshot(out_dir=str(tmp_path), locals_mode="meta")
    def failing_meta():
        secret = "password123"  # noqa: F841
        items = [1, 2, 3]  # noqa: F841
        raise ValueError("error")

    with pytest.raises(ValueError):
        failing_meta()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    crash_frame = snapshot["frames"][0]
    assert "locals" not in crash_frame
    assert "locals_meta" in crash_frame


def test_source_mode_crash_only(tmp_path):
    """Test that source is captured only for crash frame."""

    @debug_snapshot(out_dir=str(tmp_path), frames=2, source_mode="crash_only")
    def outer():
        def inner():
            raise RuntimeError("boom")

        inner()

    with pytest.raises(RuntimeError):
        outer()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    assert len(snapshot["frames"]) == 2
    crash_frame = snapshot["frames"][0]
    parent_frame = snapshot["frames"][1]
    assert "source" in crash_frame
    assert "source" not in parent_frame


def test_exception_args_and_notes(tmp_path):
    """Test that exception args and notes are captured."""

    @debug_snapshot(out_dir=str(tmp_path))
    def failing_with_args():
        exc = ValueError("a", "b")
        if hasattr(exc, "add_note"):
            exc.add_note("note")
        raise exc

    with pytest.raises(ValueError):
        failing_with_args()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    exc = snapshot["exception"]
    assert exc["args"] == ["a", "b"]
    if hasattr(BaseException(), "add_note"):
        assert "notes" in exc


def test_locals_mode_invalid_raises():
    """Test that invalid locals_mode values raise a ValueError."""
    with pytest.raises(ValueError, match="locals_mode must be 'safe', 'meta', or 'none'"):
        debug_snapshot(locals_mode="invalid")  # type: ignore[arg-type]
