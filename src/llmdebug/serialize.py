"""Safe serialization with smart array handling."""

from __future__ import annotations

import inspect
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


def summarize_array(arr: Any, cfg: SnapshotConfig) -> dict:
    """Summarize an array-like object."""
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
        items = list(x)[: cfg.max_items]
        result = [to_jsonlike(i, cfg, depth + 1) for i in items]
        if len(list(x)) > cfg.max_items:
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
    result: dict[str, Any] = {}

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
    return json.loads(raw)
