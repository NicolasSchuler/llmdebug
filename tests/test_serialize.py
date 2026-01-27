"""Tests for serialization."""

import pytest

from llmdebug.config import SnapshotConfig
from llmdebug.serialize import summarize_array, to_jsonlike


@pytest.fixture
def cfg():
    return SnapshotConfig()


def test_primitives(cfg):
    """Test primitive type serialization."""
    assert to_jsonlike(None, cfg) is None
    assert to_jsonlike(True, cfg) is True
    assert to_jsonlike(42, cfg) == 42
    assert to_jsonlike(3.14, cfg) == 3.14
    assert to_jsonlike("hello", cfg) == "hello"


def test_string_truncation(cfg):
    """Test long strings are truncated."""
    long_str = "x" * 1000
    result = to_jsonlike(long_str, cfg)
    assert isinstance(result, str)
    assert len(result) <= cfg.max_str
    assert "TRUNC" in result


def test_list_serialization(cfg):
    """Test list serialization with truncation."""
    small_list = [1, 2, 3]
    assert to_jsonlike(small_list, cfg) == [1, 2, 3]

    large_list = list(range(100))
    result = to_jsonlike(large_list, cfg)
    assert isinstance(result, list)
    assert len(result) == cfg.max_items + 1  # items + TRUNC marker
    assert result[-1] == "...[TRUNC]"


def test_dict_serialization(cfg):
    """Test dict serialization."""
    d = {"a": 1, "b": "hello"}
    result = to_jsonlike(d, cfg)
    assert isinstance(result, dict)
    assert result["'a'"] == 1
    assert result["'b'"] == "hello"


@pytest.mark.skipif(
    not pytest.importorskip("numpy", reason="numpy not installed"),
    reason="numpy not installed",
)
def test_numpy_array_summary():
    """Test numpy array summarization."""
    import numpy as np

    cfg = SnapshotConfig()
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    result = summarize_array(arr, cfg)
    assert result["__array__"] == "numpy.ndarray"
    assert result["shape"] == [2, 3]
    assert result["dtype"] == "float32"
    assert "head" in result
