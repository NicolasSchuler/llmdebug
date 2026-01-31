"""Tests for serialization."""

import re

import pytest

from llmdebug.config import SnapshotConfig
from llmdebug.serialize import (
    compile_redactors,
    compute_array_stats,
    redact_text,
    serialize_locals,
    summarize_array,
    to_jsonlike,
)


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


# ============================================================================
# Redaction Tests
# ============================================================================


def test_redact_text_basic():
    """Test direct redact_text function with simple pattern."""
    redactors = compile_redactors(("password",))
    result = redact_text("my password is secret", redactors)
    assert result == "my [REDACTED] is secret"


def test_redact_text_compiled_pattern():
    """Test redact_text with pre-compiled regex Pattern."""
    pattern = re.compile(r"api[_-]?key", re.IGNORECASE)
    redactors = compile_redactors((pattern,))

    result = redact_text("my API_KEY is abc123", redactors)
    assert "[REDACTED]" in result
    assert "API_KEY" not in result


def test_redact_text_multiple_patterns():
    """Test redact_text with multiple patterns."""
    redactors = compile_redactors(("password", "secret", "token"))
    result = redact_text("password=abc, secret=xyz, token=123", redactors)
    assert result == "[REDACTED]=abc, [REDACTED]=xyz, [REDACTED]=123"


def test_serialize_locals_redacts_password():
    """Test that sensitive patterns are redacted in serialized locals.

    Note: redaction replaces the pattern wherever it appears in the JSON,
    which means keys like "password" become "[REDACTED]", not their values.
    """
    cfg = SnapshotConfig(redact=("password", "secret"))
    redactors = compile_redactors(cfg.redact)

    local_vars = {
        "password": "hunter2",
        "username": "alice",
        "secret_key": "abc123",
    }

    result = serialize_locals(local_vars, cfg, redactors)
    result_str = str(result)

    # The key "password" should be redacted to "[REDACTED]"
    assert "password" not in result_str
    assert "[REDACTED]" in result_str
    # The key "secret_key" should become "[REDACTED]_key"
    assert "secret_key" not in result_str
    # Non-sensitive values should remain
    assert result["username"] == "alice"


def test_serialize_locals_redacts_nested_dict():
    """Test that redaction works in nested structures.

    The redaction replaces the pattern "password" in the JSON key,
    not the value. To redact values, use patterns matching the values themselves.
    """
    cfg = SnapshotConfig(redact=("password",))
    redactors = compile_redactors(cfg.redact)

    local_vars = {
        "config": {
            "database": {
                "password": "secret123",
                "host": "localhost",
            }
        }
    }

    result = serialize_locals(local_vars, cfg, redactors)

    # The nested key "password" should be redacted
    result_str = str(result)
    assert "password" not in result_str
    assert "[REDACTED]" in result_str
    assert "localhost" in result_str


def test_serialize_locals_redacts_in_arrays():
    """Test that redaction works in array/list elements."""
    cfg = SnapshotConfig(redact=("password",))
    redactors = compile_redactors(cfg.redact)

    local_vars = {
        "credentials": ["user1:password123", "user2:safe"],
    }

    result = serialize_locals(local_vars, cfg, redactors)

    result_str = str(result)
    # "password123" should be partially redacted (the "password" part)
    assert "[REDACTED]" in result_str


def test_serialize_locals_no_redaction_when_empty():
    """Test that empty redact list doesn't modify values."""
    cfg = SnapshotConfig(redact=())
    redactors = compile_redactors(cfg.redact)

    local_vars = {
        "password": "hunter2",
        "secret": "abc123",
    }

    result = serialize_locals(local_vars, cfg, redactors)

    # Without redaction patterns, values should be preserved
    assert result["password"] == "hunter2"
    assert result["secret"] == "abc123"


def test_redact_text_regex_pattern():
    """Test redact_text with regex metacharacters."""
    # Pattern to match API keys like "sk-abc123..."
    pattern = re.compile(r"sk-[a-zA-Z0-9]+")
    redactors = compile_redactors((pattern,))

    result = redact_text("API key: sk-abc123XYZ", redactors)
    assert result == "API key: [REDACTED]"
    assert "sk-" not in result


# ============================================================================
# Array Statistics Tests
# ============================================================================


@pytest.mark.skipif(not _has_module("numpy"), reason="numpy not installed")
def test_compute_array_stats_numpy():
    """Test statistical computation for numpy arrays."""
    import numpy as np

    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = compute_array_stats(arr)

    assert stats is not None
    assert stats["min"] == 1.0
    assert stats["max"] == 5.0
    assert stats["mean"] == 3.0
    # std is population std
    assert abs(stats["std"] - 1.4142135) < 0.001


@pytest.mark.skipif(not _has_module("numpy"), reason="numpy not installed")
def test_compute_array_stats_empty_array():
    """Test that empty arrays return None for stats."""
    import numpy as np

    arr = np.array([])
    stats = compute_array_stats(arr)

    assert stats is None


@pytest.mark.skipif(not _has_module("numpy"), reason="numpy not installed")
def test_compute_array_stats_non_numeric():
    """Test that non-numeric arrays return None for stats."""
    import numpy as np

    arr = np.array(["a", "b", "c"])
    stats = compute_array_stats(arr)

    assert stats is None


@pytest.mark.skipif(not _has_module("numpy"), reason="numpy not installed")
def test_compute_array_stats_integer():
    """Test stats for integer arrays."""
    import numpy as np

    arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    stats = compute_array_stats(arr)

    assert stats is not None
    assert stats["min"] == 1.0
    assert stats["max"] == 5.0


@pytest.mark.skipif(not _has_module("numpy"), reason="numpy not installed")
def test_summarize_array_with_stats():
    """Test array summary includes stats when configured."""
    import numpy as np

    cfg = SnapshotConfig(include_array_stats=True)
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    result = summarize_array(arr, cfg)

    assert "stats" in result
    assert result["stats"]["min"] == 1.0
    assert result["stats"]["max"] == 5.0


@pytest.mark.skipif(not _has_module("numpy"), reason="numpy not installed")
def test_summarize_array_without_stats():
    """Test array summary excludes stats by default."""
    import numpy as np

    cfg = SnapshotConfig(include_array_stats=False)
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    result = summarize_array(arr, cfg)

    assert "stats" not in result


@pytest.mark.skipif(not _has_module("torch"), reason="torch not installed")
def test_compute_array_stats_pytorch():
    """Test statistical computation for PyTorch tensors."""
    import torch  # type: ignore[import-not-found]

    arr = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = compute_array_stats(arr)

    assert stats is not None
    assert stats["min"] == 1.0
    assert stats["max"] == 5.0
    assert stats["mean"] == 3.0


@pytest.mark.skipif(not _has_module("jax"), reason="jax not installed")
def test_compute_array_stats_jax():
    """Test statistical computation for JAX arrays."""
    import jax.numpy as jnp  # type: ignore[import-not-found]

    arr = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = compute_array_stats(arr)

    assert stats is not None
    assert stats["min"] == 1.0
    assert stats["max"] == 5.0
