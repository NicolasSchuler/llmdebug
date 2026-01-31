"""Key abbreviation mappings for compact JSON output format."""

from __future__ import annotations

from typing import Any

# Full key -> Short key mapping
# Organized by category for maintainability
# IMPORTANT: Short keys must be unique and unlikely to appear as user variable names
# We use prefixed keys (_X format) to avoid collisions with user data
KEY_MAP: dict[str, str] = {
    # Top-level keys
    "schema_version": "_v",
    "timestamp_utc": "_ts",
    "exception": "_exc",
    "traceback": "_tb",
    "frames": "_fr",
    "crash_frame_index": "_cfi",
    "capture_config": "_cfg",
    "llmdebug_version": "_ldv",
    "env": "_env",
    "name": "_n",
    "pytest": "_pt",
    "git": "_git",
    "async": "_async",
    "recent_logs": "_logs",
    # Frame keys
    "file": "_f",
    "file_rel": "_frel",
    "line": "_l",
    "function": "_fn",
    "module": "_mod",
    "code": "_c",
    "source": "_src",
    "locals": "_loc",
    "locals_meta": "_lm",
    "locals_truncated": "_lt",
    "locals_truncated_keys": "_ltk",
    "args": "_args",
    "args_meta": "_am",
    "arg_names": "_an",
    "is_coroutine": "_coro",
    # Exception keys
    "type": "_t",
    "qualified_type": "_qt",
    "message": "_msg",
    "suppress_context": "_sc",
    "error_category": "_ecat",
    "category": "_cat",
    "suggestion": "_sug",
    "cause": "_cau",
    "context": "_ctx",
    # Source keys
    "start": "_s",
    "end": "_e",
    "snippet": "_snip",
    "lineno": "_ln",
    # Array summary keys
    "__array__": "_arr",
    "__dataframe__": "_df",
    "__series__": "_ser",
    "__type__": "_typ",
    "shape": "_sh",
    "dtype": "_dt",
    "head": "_hd",
    "head_truncated": "_ht",
    "device": "_dev",
    "requires_grad": "_rg",
    "anomalies": "_anom",
    "stats": "_st",
    "mean": "_mu",
    "std": "_sd",
    "min": "_mn",
    "max": "_mx",
    # Env keys
    "python": "_py",
    "executable": "_exe",
    "platform": "_plat",
    "cwd": "_wd",
    "argv": "_av",
    # Git keys
    "commit": "_cm",
    "commit_full": "_cmf",
    "branch": "_br",
    "dirty": "_d",
    # Log keys
    "level": "_lv",
    "logger": "_lgr",
    "filename": "_fname",
    "created": "_cr",
    # Pytest keys
    "nodeid": "_nid",
    "outcome": "_out",
    "when": "_w",
    "longrepr": "_lr",
    "capstdout": "_cout",
    "capstderr": "_cerr",
    "params": "_p",
    "repro": "_rp",
}

# Reverse mapping for decoding
REVERSE_KEY_MAP: dict[str, str] = {v: k for k, v in KEY_MAP.items()}

# Keys that contain user data (locals, args) - don't transform their contents
USER_DATA_KEYS = {
    # Full keys
    "locals",
    "locals_meta",
    "args",
    "args_meta",
    "params",
    # Compact keys
    "_loc",
    "_lm",
    "_args",
    "_am",
    "_p",
}


def compact_keys(
    data: dict[str, Any] | list[Any] | Any,
    _in_user_data: bool = False,
) -> dict[str, Any] | list[Any] | Any:
    """Recursively replace keys with short versions.

    Args:
        data: Data structure to compact (dict, list, or primitive)
        _in_user_data: Internal flag to track if we're inside user data

    Returns:
        Same structure with keys replaced by short versions
    """
    if isinstance(data, dict):
        result = {}
        for k, v in data.items():
            # Determine new key
            new_key = k if _in_user_data else KEY_MAP.get(k, k)
            # Check if this key holds user data
            is_user_data = k in USER_DATA_KEYS or new_key in USER_DATA_KEYS
            # Recurse with appropriate flag
            result[new_key] = compact_keys(v, _in_user_data=is_user_data or _in_user_data)
        return result
    elif isinstance(data, list):
        return [compact_keys(item, _in_user_data=_in_user_data) for item in data]
    else:
        return data


def expand_keys(
    data: dict[str, Any] | list[Any] | Any,
    _in_user_data: bool = False,
) -> dict[str, Any] | list[Any] | Any:
    """Recursively restore original keys from short versions.

    Args:
        data: Data structure with short keys
        _in_user_data: Internal flag to track if we're inside user data

    Returns:
        Same structure with keys restored to full versions
    """
    if isinstance(data, dict):
        result = {}
        for k, v in data.items():
            # Determine new key
            new_key = k if _in_user_data else REVERSE_KEY_MAP.get(k, k)
            # Check if this key holds user data (check both compact and full key names)
            is_user_data = k in USER_DATA_KEYS or new_key in USER_DATA_KEYS
            # Recurse with appropriate flag
            result[new_key] = expand_keys(v, _in_user_data=is_user_data or _in_user_data)
        return result
    elif isinstance(data, list):
        return [expand_keys(item, _in_user_data=_in_user_data) for item in data]
    else:
        return data


def is_compact_snapshot(data: dict[str, Any]) -> bool:
    """Detect if a snapshot uses compact key format.

    Checks for presence of short keys that would only exist in compact format.
    """
    # Check for compact top-level keys (using underscore-prefixed format)
    compact_indicators = {"_v", "_fr", "_exc", "_ts", "_cfi"}
    return bool(compact_indicators & set(data.keys()))
