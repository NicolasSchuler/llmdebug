"""Safe serialization with smart array handling."""

from __future__ import annotations

import inspect
import itertools
import json
import re
from re import Pattern
from typing import Any

from .config import SnapshotConfig

JsonLike = None | bool | int | float | str | dict[str, Any] | list[Any]


def compile_redactors(patterns: tuple) -> list[Pattern[str]]:
    """Compile regex patterns for redaction."""
    return [re.compile(p) if isinstance(p, str) else p for p in patterns]


def redact_text(text: str, redactors: list[Pattern[str]]) -> str:
    """Apply redaction patterns to text."""
    for rx in redactors:
        text = rx.sub("[REDACTED]", text)
    return text


def truncate_str(s: str, max_len: int) -> str:
    """Truncate string with marker."""
    if len(s) <= max_len:
        return s
    return s[: max_len - 12] + "...[TRUNC]"


def safe_repr(x: Any, cfg: SnapshotConfig) -> str:
    """Get repr with fallback for unreprable objects."""
    try:
        r = repr(x)
    except Exception:
        r = f"<unreprable {type(x).__name__}>"
    return truncate_str(r, cfg.max_str)


def is_array_like(x: Any) -> bool:
    """Check if object is an array type (numpy, jax, torch, etc)."""
    module = type(x).__module__
    return any(
        module.startswith(prefix)
        for prefix in ("numpy", "jax", "torch", "tensorflow", "cupy")
    ) and hasattr(x, "shape")


def is_dataframe(x: Any) -> bool:
    """Check if object is a pandas DataFrame."""
    return type(x).__module__.startswith("pandas") and type(x).__name__ == "DataFrame"


def is_series(x: Any) -> bool:
    """Check if object is a pandas Series."""
    return type(x).__module__.startswith("pandas") and type(x).__name__ == "Series"


def get_tensor_device(arr: Any) -> str | None:
    """Get device string for PyTorch/JAX tensors.

    Returns None for CPU (default) or if device cannot be determined.
    Only returns non-None for non-CPU devices to highlight noteworthy info.
    """
    try:
        # PyTorch: arr.device -> device(type='cuda', index=0)
        if hasattr(arr, "device"):
            device = arr.device
            # PyTorch device object
            if hasattr(device, "type"):
                if device.type != "cpu":
                    return str(device)
                return None
            # JAX SingleDeviceSharding or similar
            device_str = str(device)
            if "cpu" not in device_str.lower():
                return device_str
            return None
        # JAX: arr.devices() -> {gpu(id=0)}
        if hasattr(arr, "devices"):
            devices = arr.devices()
            if devices:
                # Check if any non-CPU device
                devices_str = str(devices)
                if "cpu" not in devices_str.lower():
                    return devices_str
    except Exception:
        pass
    return None


def get_requires_grad(arr: Any) -> bool | None:
    """Check if tensor requires gradients (PyTorch only).

    Returns True only if requires_grad is True, None otherwise.
    """
    try:
        if hasattr(arr, "requires_grad") and arr.requires_grad:
            return True
    except Exception:
        pass
    return None


def count_anomalies(arr: Any) -> dict[str, int] | None:
    """Count NaN and Inf values in tensor.

    Returns dict with counts only if anomalies exist, None otherwise.
    Works with NumPy, PyTorch, and JAX arrays.
    """
    try:
        module = type(arr).__module__

        nan_count = 0
        inf_count = 0

        if module.startswith("torch"):
            # PyTorch
            import torch  # type: ignore[import-not-found]

            nan_count = int(torch.isnan(arr).sum().item())
            inf_count = int(torch.isinf(arr).sum().item())
        elif module.startswith("jax"):
            # JAX
            import jax.numpy as jnp  # type: ignore[import-not-found]

            nan_count = int(jnp.isnan(arr).sum())
            inf_count = int(jnp.isinf(arr).sum())
        elif module.startswith("numpy"):
            # NumPy
            import numpy as np

            nan_count = int(np.isnan(arr).sum())
            inf_count = int(np.isinf(arr).sum())
        else:
            return None

        if nan_count > 0 or inf_count > 0:
            result: dict[str, int] = {}
            if nan_count > 0:
                result["nan"] = nan_count
            if inf_count > 0:
                result["inf"] = inf_count
            return result
    except Exception:
        pass
    return None


def compute_array_stats(arr: Any) -> dict[str, float] | None:
    """Compute min/max/mean/std statistics for numeric arrays.

    Returns:
        Dict with statistics rounded to 6 decimal places, or None if not applicable.
        Returns None for empty arrays, non-numeric dtypes, or on error.
    """
    try:
        module = type(arr).__module__

        # Check if numeric dtype
        dtype_str = str(getattr(arr, "dtype", ""))
        # Complex types are intentionally excluded: "min/max/mean/std" are ambiguous
        # without choosing a projection (real/imag/magnitude), and converting to float
        # would silently drop information.
        if not any(t in dtype_str for t in ("float", "int", "uint")):
            return None

        # Check for empty array
        if hasattr(arr, "size") and arr.size == 0:
            return None

        if module.startswith("torch"):
            # PyTorch - move to CPU for stats
            import torch  # type: ignore[import-not-found]

            arr_cpu = arr.detach().cpu().float()
            return {
                "min": round(float(torch.min(arr_cpu).item()), 6),
                "max": round(float(torch.max(arr_cpu).item()), 6),
                "mean": round(float(torch.mean(arr_cpu).item()), 6),
                # Match NumPy/JAX default semantics (population std, not unbiased).
                "std": round(float(torch.std(arr_cpu, unbiased=False).item()), 6),
            }
        elif module.startswith("jax"):
            # JAX
            import jax.numpy as jnp  # type: ignore[import-not-found]

            arr_float = arr.astype(jnp.float32)
            return {
                "min": round(float(jnp.min(arr_float)), 6),
                "max": round(float(jnp.max(arr_float)), 6),
                "mean": round(float(jnp.mean(arr_float)), 6),
                "std": round(float(jnp.std(arr_float)), 6),
            }
        elif module.startswith(("numpy", "cupy")):
            # NumPy/CuPy
            import numpy as np

            # Handle CuPy by converting to numpy
            if module.startswith("cupy"):
                arr = arr.get()
            arr_float = arr.astype(np.float64)
            return {
                "min": round(float(np.min(arr_float)), 6),
                "max": round(float(np.max(arr_float)), 6),
                "mean": round(float(np.mean(arr_float)), 6),
                "std": round(float(np.std(arr_float)), 6),
            }
    except Exception:
        pass
    return None


def summarize_array(arr: Any, cfg: SnapshotConfig) -> dict:
    """Summarize an array-like object.

    Returns basic info (type, shape, dtype, head) plus ML-specific metadata:
    - device: only if non-CPU (highlights CUDA/TPU tensors)
    - requires_grad: only if True (highlights trainable tensors)
    - anomalies: NaN/Inf counts only if present (highlights corruption)
    """
    type_name = f"{type(arr).__module__}.{type(arr).__name__}"
    summary: dict[str, Any] = {"__array__": type_name}

    if hasattr(arr, "shape"):
        try:
            summary["shape"] = list(arr.shape)
        except Exception:
            pass

    if hasattr(arr, "dtype"):
        try:
            summary["dtype"] = str(arr.dtype)
        except Exception:
            pass

    # Sample first N elements
    try:
        flat = arr.flatten() if hasattr(arr, "flatten") else arr.ravel()
        sample = [float(x) for x in flat[:5]]
        summary["head"] = sample
        if len(flat) > 5:
            summary["head_truncated"] = True
    except Exception:
        pass

    # ML-specific metadata (only include if noteworthy)
    device = get_tensor_device(arr)
    if device is not None:
        summary["device"] = device

    requires_grad = get_requires_grad(arr)
    if requires_grad:
        summary["requires_grad"] = True

    anomalies = count_anomalies(arr)
    if anomalies:
        summary["anomalies"] = anomalies

    # Optional statistical summaries
    if cfg.include_array_stats:
        stats = compute_array_stats(arr)
        if stats:
            summary["stats"] = stats

    return summary


def summarize_dataframe(df: Any, cfg: SnapshotConfig) -> dict:
    """Summarize a pandas DataFrame."""
    summary: dict[str, Any] = {"__dataframe__": True}

    try:
        summary["shape"] = list(df.shape)
    except Exception:
        pass

    try:
        summary["columns"] = list(df.columns)[: cfg.max_items]
        if len(df.columns) > cfg.max_items:
            summary["columns_truncated"] = True
    except Exception:
        pass

    try:
        head_rows = min(3, len(df))
        summary["head"] = df.head(head_rows).values.tolist()
    except Exception:
        pass

    return summary


def summarize_series(series: Any, cfg: SnapshotConfig) -> dict:
    """Summarize a pandas Series."""
    summary: dict[str, Any] = {"__series__": True}

    try:
        summary["shape"] = list(series.shape)
    except Exception:
        pass

    if hasattr(series, "name") and series.name is not None:
        summary["name"] = str(series.name)

    if hasattr(series, "dtype"):
        summary["dtype"] = str(series.dtype)

    try:
        summary["head"] = series.head(5).tolist()
        if len(series) > 5:
            summary["head_truncated"] = True
    except Exception:
        pass

    return summary


def to_jsonlike(x: Any, cfg: SnapshotConfig, depth: int = 0) -> JsonLike:
    """Convert arbitrary Python object to JSON-serializable form."""
    # Primitives
    if x is None or isinstance(x, (bool, int, float)):
        return x
    if isinstance(x, str):
        return truncate_str(x, cfg.max_str)

    # Max depth reached
    if depth >= 2:
        return safe_repr(x, cfg)

    # Special types
    if is_array_like(x):
        return summarize_array(x, cfg)
    if is_dataframe(x):
        return summarize_dataframe(x, cfg)
    if is_series(x):
        return summarize_series(x, cfg)

    # Collections
    if isinstance(x, (list, tuple, set, frozenset)):
        truncated = False

        if isinstance(x, (list, tuple)):
            items = x[: cfg.max_items + 1]
        else:
            items = list(itertools.islice(x, cfg.max_items + 1))

        if len(items) > cfg.max_items:
            truncated = True
            items = items[: cfg.max_items]

        result = [to_jsonlike(i, cfg, depth + 1) for i in items]
        if truncated:
            result.append("...[TRUNC]")
        return result

    if isinstance(x, dict):
        out: dict[str, Any] = {}
        for i, (k, v) in enumerate(x.items()):
            if i >= cfg.max_items:
                out["...[TRUNC]"] = f"+{len(x) - cfg.max_items} more keys"
                break
            out[safe_repr(k, cfg)] = to_jsonlike(v, cfg, depth + 1)
        return out

    # Objects with __dict__
    if hasattr(x, "__dict__") and depth < 1:
        d = getattr(x, "__dict__", {})
        if isinstance(d, dict):
            out = {"__type__": type(x).__name__}
            for i, (k, v) in enumerate(d.items()):
                if i >= min(cfg.max_items, 20):
                    out["...[TRUNC]"] = "more fields omitted"
                    break
                out[str(k)] = to_jsonlike(v, cfg, depth + 1)
            return out

    return safe_repr(x, cfg)


def serialize_locals(
    local_vars: dict, cfg: SnapshotConfig, redactors: list[Pattern[str]]
) -> dict:
    """Serialize local variables with redaction."""
    result, _, _ = serialize_locals_with_stats(local_vars, cfg, redactors)
    return result


def serialize_locals_with_stats(
    local_vars: dict, cfg: SnapshotConfig, redactors: list[Pattern[str]]
) -> tuple[dict[str, Any], bool, int]:
    """Serialize local variables with redaction and truncation stats."""
    result: dict[str, Any] = {}
    total_count: int | None = None
    try:
        total_count = len(local_vars)
    except Exception:
        total_count = None

    for i, (k, v) in enumerate(local_vars.items()):
        if i >= cfg.max_items:
            break
        # Skip modules, functions, classes
        if inspect.ismodule(v) or inspect.isfunction(v) or inspect.isclass(v):
            result[k] = f"<{type(v).__name__}>"
        else:
            result[k] = to_jsonlike(v, cfg)

    # Apply redaction
    raw = json.dumps(result, ensure_ascii=False, default=str)
    raw = redact_text(raw, redactors)
    cleaned = json.loads(raw)

    truncated = False
    truncated_keys = 0
    if total_count is not None and total_count > cfg.max_items:
        truncated = True
        truncated_keys = total_count - cfg.max_items

    return cleaned, truncated, truncated_keys


def locals_metadata(local_vars: dict, cfg: SnapshotConfig) -> dict[str, dict[str, Any]]:
    """Lightweight metadata for local variables (type + size hints)."""
    meta: dict[str, dict[str, Any]] = {}

    for i, (k, v) in enumerate(local_vars.items()):
        if i >= cfg.max_items:
            break

        entry: dict[str, Any] = {
            "type": f"{type(v).__module__}.{type(v).__name__}",
        }

        # Length hints for common containers/strings
        try:
            entry["len"] = int(len(v))  # type: ignore[arg-type]
        except Exception:
            pass

        # Array/dataframe shape hints (redundant but quick to scan)
        if is_array_like(v) and hasattr(v, "shape"):
            try:
                entry["shape"] = list(v.shape)
            except Exception:
                pass
        if is_dataframe(v) and hasattr(v, "shape"):
            try:
                entry["shape"] = list(v.shape)
            except Exception:
                pass
        if is_series(v) and hasattr(v, "shape"):
            try:
                entry["shape"] = list(v.shape)
            except Exception:
                pass

        meta[str(k)] = entry

    return meta
