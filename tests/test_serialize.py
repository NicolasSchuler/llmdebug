"""Tests for serialization."""

import pytest

from llmdebug.config import SnapshotConfig
from llmdebug.serialize import summarize_array, to_jsonlike


def _has_module(name: str) -> bool:
    """Check if a module is importable."""
    try:
        __import__(name)
        return True
    except ImportError:
        return False


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


@pytest.mark.skipif(not _has_module("numpy"), reason="numpy not installed")
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


@pytest.mark.skipif(not _has_module("numpy"), reason="numpy not installed")
def test_numpy_nan_detection():
    """Test NaN/Inf detection in numpy arrays."""
    import numpy as np

    cfg = SnapshotConfig()

    # Array with NaN and Inf
    arr = np.array([1.0, float("nan"), float("inf"), float("-inf"), 2.0])
    result = summarize_array(arr, cfg)

    assert "anomalies" in result
    assert result["anomalies"]["nan"] == 1
    assert result["anomalies"]["inf"] == 2  # +inf and -inf

    # Clean array should have no anomalies key
    clean_arr = np.array([1.0, 2.0, 3.0])
    clean_result = summarize_array(clean_arr, cfg)
    assert "anomalies" not in clean_result


@pytest.mark.skipif(not _has_module("torch"), reason="torch not installed")
def test_pytorch_tensor_summary():
    """Test PyTorch tensor with requires_grad and anomalies."""
    import torch  # type: ignore[import-not-found]

    cfg = SnapshotConfig()

    # Tensor with requires_grad and anomalies
    t = torch.tensor([1.0, float("nan"), float("inf")], requires_grad=True)
    result = summarize_array(t, cfg)

    assert result["__array__"] == "torch.Tensor"
    assert result["requires_grad"] is True
    assert "anomalies" in result
    assert result["anomalies"]["nan"] == 1
    assert result["anomalies"]["inf"] == 1

    # CPU device should not appear (it's the default)
    assert "device" not in result or result.get("device") is None

    # Tensor without requires_grad
    t2 = torch.tensor([1.0, 2.0, 3.0], requires_grad=False)
    result2 = summarize_array(t2, cfg)
    assert "requires_grad" not in result2
    assert "anomalies" not in result2


@pytest.mark.skipif(not _has_module("torch"), reason="torch not installed")
def test_pytorch_cuda_device():
    """Test that CUDA device is captured when available."""
    import torch  # type: ignore[import-not-found]

    cfg = SnapshotConfig()

    if torch.cuda.is_available():
        t = torch.tensor([1.0, 2.0], device="cuda")
        result = summarize_array(t, cfg)
        assert "device" in result
        assert "cuda" in result["device"]
    else:
        # Just test that CPU device is omitted
        t = torch.tensor([1.0, 2.0], device="cpu")
        result = summarize_array(t, cfg)
        assert "device" not in result


@pytest.mark.skipif(not _has_module("jax"), reason="jax not installed")
def test_jax_array_summary():
    """Test JAX array with anomalies."""
    import jax.numpy as jnp  # type: ignore[import-not-found]

    cfg = SnapshotConfig()

    # JAX array with anomalies
    arr = jnp.array([1.0, float("nan"), float("inf")])
    result = summarize_array(arr, cfg)

    assert "jax" in result["__array__"].lower()
    assert "anomalies" in result
    assert result["anomalies"]["nan"] == 1
    assert result["anomalies"]["inf"] == 1

    # JAX doesn't have requires_grad (uses functional transforms)
    assert "requires_grad" not in result
