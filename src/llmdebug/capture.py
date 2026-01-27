"""Core exception capture logic."""

from __future__ import annotations

import datetime as dt
import os
import platform
import sys
import traceback
from datetime import timezone
from pathlib import Path
from typing import Any

from .config import SnapshotConfig
from .output import write_bundle
from .serialize import compile_redactors, locals_metadata, serialize_locals, truncate_str


SCHEMA_VERSION = "1.0"


try:  # Python 3.11+
    BaseExceptionGroup  # type: ignore[name-defined]  # noqa: B018
except NameError:  # pragma: no cover - Python 3.10 fallback
    BaseExceptionGroup = None  # type: ignore[assignment]


def _serialize_config(cfg: SnapshotConfig) -> dict[str, Any]:
    redact = []
    for item in cfg.redact:
        if isinstance(item, str):
            redact.append(item)
        else:
            redact.append(item.pattern)
    return {
        "out_dir": cfg.out_dir,
        "frames": cfg.frames,
        "source_context": cfg.source_context,
        "locals_mode": cfg.locals_mode,
        "max_str": cfg.max_str,
        "max_items": cfg.max_items,
        "redact": redact,
        "include_env": cfg.include_env,
        "debug": cfg.debug,
    }


def _summarize_exception(
    exc: BaseException,
    cfg: SnapshotConfig,
    *,
    depth: int = 0,
    max_depth: int = 2,
    seen: set[int] | None = None,
) -> dict[str, Any]:
    if seen is None:
        seen = set()
    if id(exc) in seen:
        return {"type": type(exc).__name__, "message": "...[CYCLE]"}
    seen.add(id(exc))

    summary: dict[str, Any] = {
        "type": type(exc).__name__,
        "message": truncate_str(str(exc), cfg.max_str),
    }

    if depth >= max_depth:
        return summary

    if BaseExceptionGroup is not None and isinstance(exc, BaseExceptionGroup):
        exceptions = list(exc.exceptions)
        limit = min(len(exceptions), cfg.max_items)
        summary["is_exception_group"] = True
        summary["exceptions"] = [
            _summarize_exception(e, cfg, depth=depth + 1, max_depth=max_depth, seen=seen)
            for e in exceptions[:limit]
        ]
        if len(exceptions) > limit:
            summary["exceptions_truncated"] = True

    cause = getattr(exc, "__cause__", None)
    if cause is not None:
        summary["cause"] = _summarize_exception(
            cause, cfg, depth=depth + 1, max_depth=max_depth, seen=seen
        )

    context = getattr(exc, "__context__", None)
    if context is not None:
        summary["context"] = _summarize_exception(
            context, cfg, depth=depth + 1, max_depth=max_depth, seen=seen
        )

    summary["suppress_context"] = bool(getattr(exc, "__suppress_context__", False))
    return summary


def get_env_info(cfg: SnapshotConfig) -> dict[str, Any]:
    """Collect environment information."""
    argv = []
    try:
        argv = [truncate_str(str(a), cfg.max_str) for a in sys.argv]
    except Exception:
        argv = []
    return {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "argv": argv,
    }


def get_source_snippet(filename: str, lineno: int, ctx: int) -> dict[str, Any]:
    """Extract source code around a line."""
    try:
        if not filename or not os.path.exists(filename):
            return {}
        with open(filename, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        start = max(lineno - 1 - ctx, 0)
        end = min(lineno - 1 + ctx + 1, len(lines))
        snippet = [
            {"lineno": i + 1, "code": lines[i].rstrip("\n")} for i in range(start, end)
        ]
        return {"start": start + 1, "end": end, "snippet": snippet}
    except Exception:
        return {}


def collect_frames(tb, cfg: SnapshotConfig) -> list[dict[str, Any]]:
    """Collect stack frames from traceback."""
    redactors = compile_redactors(cfg.redact)
    frames_out = []

    # Extract frame info
    extracted = traceback.extract_tb(tb)[-cfg.frames :]

    # Walk tb objects to get locals
    tb_list = []
    cur = tb
    while cur is not None:
        tb_list.append(cur)
        cur = cur.tb_next
    tb_list = tb_list[-cfg.frames :]

    for tb_item, ex_item in zip(tb_list, extracted, strict=True):
        frame = tb_item.tb_frame
        lineno = ex_item.lineno or 0
        file_rel = None
        try:
            cwd = os.getcwd()
            file_rel = os.path.relpath(ex_item.filename, cwd)
            if file_rel.startswith(".."):
                file_rel = None
        except Exception:
            file_rel = None

        frame_dict: dict[str, Any] = {
            "file": ex_item.filename,
            "file_rel": file_rel,
            "line": lineno,
            "function": ex_item.name,
            "module": frame.f_globals.get("__name__"),
            "code": ex_item.line,
            "source": get_source_snippet(ex_item.filename, lineno, cfg.source_context),
        }

        if cfg.locals_mode == "safe":
            frame_dict["locals"] = serialize_locals(frame.f_locals, cfg, redactors)
            frame_dict["locals_meta"] = locals_metadata(frame.f_locals, cfg)

        frames_out.append(frame_dict)

    # Reverse so crash site is first (index 0)
    return list(reversed(frames_out))


def capture_exception(
    name: str,
    exc: BaseException,
    tb,
    cfg: SnapshotConfig,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Capture exception details and write snapshot."""
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "timestamp_utc": dt.datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "exception": _summarize_exception(exc, cfg),
        "traceback": "".join(traceback.format_exception(type(exc), exc, tb)),
        "frames": collect_frames(tb, cfg),
        "capture_config": _serialize_config(cfg),
    }

    if cfg.include_env:
        payload["env"] = get_env_info(cfg)

    if extra:
        payload.update(extra)

    try:
        from . import __version__ as llmdebug_version
    except Exception:
        llmdebug_version = "unknown"
    payload["llmdebug_version"] = llmdebug_version

    return write_bundle(payload, cfg)
