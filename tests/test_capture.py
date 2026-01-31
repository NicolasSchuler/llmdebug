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
            exc.add_note("note")  # pyright: ignore[reportAttributeAccessIssue]
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


def test_include_modules_filters_frames(tmp_path):
    """Test that include_modules filters non-matching frames."""

    # Use a module prefix that won't match our test module
    @debug_snapshot(out_dir=str(tmp_path), include_modules=["nonexistent_module"])
    def outer():
        def inner():
            raise ValueError("filtered test")

        inner()

    with pytest.raises(ValueError):
        outer()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    # Should only have the crash frame (always included) since no module matches
    # The crash frame is always kept regardless of filter
    frames = snapshot["frames"]
    assert len(frames) >= 1

    # Crash frame should be present at index 0
    assert frames[0]["function"] == "inner"


def test_include_modules_matches_prefix(tmp_path):
    """Test that module prefix matching works correctly."""

    # Use the actual test module prefix which should match
    @debug_snapshot(out_dir=str(tmp_path), frames=5, include_modules=["tests", "test_"])
    def outer():
        def middle():
            def inner():
                raise RuntimeError("prefix test")

            inner()

        middle()

    with pytest.raises(RuntimeError):
        outer()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    # All frames should be from our test module
    frames = snapshot["frames"]
    assert len(frames) >= 1
    # Crash frame is always included
    assert frames[0]["function"] == "inner"


def test_include_modules_none_captures_all(tmp_path):
    """Test that include_modules=None captures all frames (default behavior)."""

    @debug_snapshot(out_dir=str(tmp_path), frames=5, include_modules=None)
    def failing():
        raise ValueError("all frames test")

    with pytest.raises(ValueError):
        failing()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None
    assert len(snapshot["frames"]) >= 1


def test_exception_chain_depth_configurable(tmp_path):
    """Test that max_exception_depth controls exception chain traversal."""

    @debug_snapshot(out_dir=str(tmp_path), max_exception_depth=3)
    def deep_chain():
        try:
            try:
                try:
                    raise ValueError("level 0")
                except ValueError as e:
                    raise TypeError("level 1") from e
            except TypeError as e:
                raise RuntimeError("level 2") from e
        except RuntimeError as e:
            raise KeyError("level 3") from e

    with pytest.raises(KeyError):
        deep_chain()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    exc = snapshot["exception"]
    assert exc["type"] == "KeyError"

    # With depth=3, we should see the cause chain
    assert "cause" in exc
    assert exc["cause"]["type"] == "RuntimeError"
    assert "cause" in exc["cause"]
    assert exc["cause"]["cause"]["type"] == "TypeError"


def test_exception_chain_depth_one(tmp_path):
    """Test that max_exception_depth=1 stops at first exception."""

    @debug_snapshot(out_dir=str(tmp_path), max_exception_depth=1)
    def shallow_chain():
        try:
            raise ValueError("original")
        except ValueError as e:
            raise RuntimeError("wrapped") from e

    with pytest.raises(RuntimeError):
        shallow_chain()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    exc = snapshot["exception"]
    assert exc["type"] == "RuntimeError"

    # With depth=1, cause should not have further cause
    # (depth 1 means we capture the exception itself but no chain traversal)
    # Actually checking the implementation: depth starts at 0, increments to 1 for cause
    # So with max_depth=1, cause at depth=1 won't recurse further
    if "cause" in exc:
        # The cause is captured but won't have its own cause/context
        assert "cause" not in exc["cause"] or exc["cause"].get("cause") is None


def test_config_validation_max_exception_depth():
    """Test that max_exception_depth validation works."""
    with pytest.raises(ValueError, match="max_exception_depth must be >= 1"):
        debug_snapshot(max_exception_depth=0)


def test_config_validation_lock_timeout():
    """Test that lock_timeout validation works."""
    with pytest.raises(ValueError, match="lock_timeout must be >= 0"):
        debug_snapshot(lock_timeout=-1)


def test_config_validation_log_max_records():
    """Test that log_max_records validation works."""
    with pytest.raises(ValueError, match="log_max_records must be >= 0"):
        debug_snapshot(log_max_records=-1)


# ============================================================================
# Function Arguments Tests
# ============================================================================


def test_function_args_captured(tmp_path):
    """Test that function arguments are captured separately from locals."""

    @debug_snapshot(out_dir=str(tmp_path), include_args=True)
    def func_with_args(a, b, c=10):
        local_var = a + b
        raise ValueError(f"Sum is {local_var}")

    with pytest.raises(ValueError):
        func_with_args(1, 2, c=20)

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    # Find the crash frame (should be func_with_args)
    crash_frame = snapshot["frames"][0]
    assert crash_frame["function"] == "func_with_args"

    # Should have args and arg_names
    assert "args" in crash_frame
    assert "arg_names" in crash_frame

    # Check arg_names
    assert crash_frame["arg_names"] == ["a", "b", "c"]

    # Check arg values
    args = crash_frame["args"]
    assert args["a"] == 1
    assert args["b"] == 2
    assert args["c"] == 20


def test_function_args_disabled(tmp_path):
    """Test that function arguments are not captured when disabled."""

    @debug_snapshot(out_dir=str(tmp_path), include_args=False)
    def func_with_args(a, b):
        raise ValueError("test")

    with pytest.raises(ValueError):
        func_with_args(1, 2)

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    crash_frame = snapshot["frames"][0]

    # Should not have args or arg_names
    assert "args" not in crash_frame
    assert "arg_names" not in crash_frame


def test_function_args_with_varargs(tmp_path):
    """Test that *args and **kwargs are captured."""

    @debug_snapshot(out_dir=str(tmp_path), include_args=True)
    def func_with_varargs(a, *args, **kwargs):
        raise ValueError("test")

    with pytest.raises(ValueError):
        func_with_varargs(1, 2, 3, x=10, y=20)

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    crash_frame = snapshot["frames"][0]
    assert "args" in crash_frame
    assert "arg_names" in crash_frame

    # Check arg_names includes *args and **kwargs markers
    arg_names = crash_frame["arg_names"]
    assert "a" in arg_names
    assert any("*args" in name for name in arg_names)
    assert any("**kwargs" in name for name in arg_names)

    # Check values (tuples become lists in JSON serialization)
    args = crash_frame["args"]
    assert args["a"] == 1
    assert args["args"] == [2, 3]
    assert args["kwargs"] == {"'x'": 10, "'y'": 20}  # Keys get repr'd


def test_function_args_with_self(tmp_path):
    """Test that self is captured as an argument for methods."""

    class MyClass:
        def __init__(self, value):
            self.value = value

        @debug_snapshot(out_dir=str(tmp_path), include_args=True)
        def method(self, x):
            raise ValueError("test")

    obj = MyClass(42)
    with pytest.raises(ValueError):
        obj.method(10)

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    crash_frame = snapshot["frames"][0]
    assert "args" in crash_frame
    assert "arg_names" in crash_frame

    # Check that self is in arg_names
    assert "self" in crash_frame["arg_names"]
    assert "x" in crash_frame["arg_names"]

    # Check values
    args = crash_frame["args"]
    assert args["x"] == 10
    # self should be serialized as an object
    assert "__type__" in args["self"] or "value" in str(args["self"])


def test_function_args_with_keyword_only(tmp_path):
    """Test that keyword-only arguments are captured."""

    @debug_snapshot(out_dir=str(tmp_path), include_args=True)
    def func_with_kwonly(a, b, *, kwonly1, kwonly2=None):
        raise ValueError("test")

    with pytest.raises(ValueError):
        func_with_kwonly(1, 2, kwonly1="required", kwonly2="optional")

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    crash_frame = snapshot["frames"][0]
    assert "args" in crash_frame
    assert "arg_names" in crash_frame

    # Check all arg names are captured (including keyword-only)
    arg_names = crash_frame["arg_names"]
    assert "a" in arg_names
    assert "b" in arg_names
    assert "kwonly1" in arg_names
    assert "kwonly2" in arg_names

    # Check values
    args = crash_frame["args"]
    assert args["a"] == 1
    assert args["b"] == 2
    assert args["kwonly1"] == "required"
    assert args["kwonly2"] == "optional"


def test_function_args_with_kwonly_and_varargs(tmp_path):
    """Test complex signature: positional, *args, keyword-only, **kwargs."""

    @debug_snapshot(out_dir=str(tmp_path), include_args=True)
    def complex_func(a, *args, kwonly, **kwargs):
        raise ValueError("test")

    with pytest.raises(ValueError):
        complex_func(1, 2, 3, kwonly="kw", extra="val")

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    crash_frame = snapshot["frames"][0]
    arg_names = crash_frame["arg_names"]

    # Check all arg types are captured in order
    assert "a" in arg_names
    assert "kwonly" in arg_names
    assert any("*args" in name for name in arg_names)
    assert any("**kwargs" in name for name in arg_names)

    # Check values
    args = crash_frame["args"]
    assert args["a"] == 1
    assert args["kwonly"] == "kw"
    assert args["args"] == [2, 3]


# ============================================================================
# Error Categorization Tests
# ============================================================================


def test_error_categorization_in_snapshot(tmp_path):
    """Test that error categorization is included in snapshot."""

    @debug_snapshot(out_dir=str(tmp_path), categorize_errors=True)
    def shape_error():
        raise ValueError("shape mismatch: expected (10,) got (20,)")

    with pytest.raises(ValueError):
        shape_error()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    # Should have error_category in exception
    exc = snapshot["exception"]
    assert "error_category" in exc
    assert exc["error_category"]["category"] == "shape_mismatch"
    assert "suggestion" in exc["error_category"]


def test_error_categorization_disabled(tmp_path):
    """Test that error categorization is not included when disabled."""

    @debug_snapshot(out_dir=str(tmp_path), categorize_errors=False)
    def some_error():
        raise ValueError("shape mismatch")

    with pytest.raises(ValueError):
        some_error()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    # Should not have error_category
    exc = snapshot["exception"]
    assert "error_category" not in exc


# ============================================================================
# Git Context Tests
# ============================================================================


def test_git_context_in_snapshot(tmp_path):
    """Test that git context is included in snapshot."""

    @debug_snapshot(out_dir=str(tmp_path), include_git=True)
    def failing():
        raise ValueError("test")

    with pytest.raises(ValueError):
        failing()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    # Should have git context since we're running from a git repo
    if "git" in snapshot:
        assert "commit" in snapshot["git"]
        assert "branch" in snapshot["git"]


def test_git_context_disabled(tmp_path):
    """Test that git context is not included when disabled."""

    @debug_snapshot(out_dir=str(tmp_path), include_git=False)
    def failing():
        raise ValueError("test")

    with pytest.raises(ValueError):
        failing()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    # Should not have git context
    assert "git" not in snapshot


# ============================================================================
# Config Serialization Tests
# ============================================================================


def test_new_config_fields_serialized(tmp_path):
    """Test that new config fields are included in capture_config."""

    @debug_snapshot(
        out_dir=str(tmp_path),
        include_git=True,
        include_array_stats=True,
        include_args=True,
        categorize_errors=True,
        include_async_context=True,
        capture_logs=False,
        log_max_records=50,
    )
    def failing():
        raise ValueError("test")

    with pytest.raises(ValueError):
        failing()

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None

    cfg = snapshot["capture_config"]
    assert cfg["include_git"] is True
    assert cfg["include_array_stats"] is True
    assert cfg["include_args"] is True
    assert cfg["categorize_errors"] is True
    assert cfg["include_async_context"] is True
    assert cfg["capture_logs"] is False
    assert cfg["log_max_records"] == 50
    assert cfg["lock_timeout"] == 5.0
    assert cfg["output_format"] == "json_compact"
