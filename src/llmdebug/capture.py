"""Core exception capture logic."""

from __future__ import annotations

import datetime as dt
import inspect as _inspect_module
import os
import platform
import sys
import traceback
from datetime import timezone
from pathlib import Path
from typing import Any

from .config import SnapshotConfig
from .output import write_bundle
from .serialize import (
    compile_redactors,
    locals_metadata,
    serialize_locals_with_stats,
    truncate_str,
)

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
        "source_mode": cfg.source_mode,
        "locals_mode": cfg.locals_mode,
        "max_str": cfg.max_str,
        "max_items": cfg.max_items,
        "redact": redact,
        "include_env": cfg.include_env,
        "debug": cfg.debug,
        "include_modules": list(cfg.include_modules) if cfg.include_modules else None,
        "max_exception_depth": cfg.max_exception_depth,
        "lock_timeout": cfg.lock_timeout,
        "include_git": cfg.include_git,
        "include_array_stats": cfg.include_array_stats,
        "include_args": cfg.include_args,
        "categorize_errors": cfg.categorize_errors,
        "include_async_context": cfg.include_async_context,
        "capture_logs": cfg.capture_logs,
        "log_max_records": cfg.log_max_records,
        "output_format": cfg.output_format,
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
        exc_type = type(exc)
        return {
            "type": exc_type.__name__,
            "qualified_type": f"{exc_type.__module__}.{exc_type.__name__}",
            "message": "...[CYCLE]",
        }
    seen.add(id(exc))

    exc_type = type(exc)
    summary: dict[str, Any] = {
        "type": exc_type.__name__,
        "qualified_type": f"{exc_type.__module__}.{exc_type.__name__}",
        "message": truncate_str(str(exc), cfg.max_str),
    }
    try:
        if exc.args:
            summary["args"] = [
                truncate_str(str(a), cfg.max_str) for a in list(exc.args)[: cfg.max_items]
            ]
    except Exception:
        pass
    try:
        notes = getattr(exc, "__notes__", None)
        if notes:
            summary["notes"] = [
                truncate_str(str(n), cfg.max_str) for n in list(notes)[: cfg.max_items]
            ]
    except Exception:
        pass

    if depth >= max_depth:
        return summary

    if BaseExceptionGroup is not None and isinstance(exc, BaseExceptionGroup):
        exceptions = list(exc.exceptions)  # pyright: ignore[reportAttributeAccessIssue]
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

    # Error categorization (only at top level)
    if depth == 0 and cfg.categorize_errors:
        try:
            from .error_categories import categorize_exception

            category = categorize_exception(exc)
            if category:
                summary["error_category"] = category
        except Exception:
            pass

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


def extract_function_args(frame) -> tuple[list[str], dict[str, Any]] | None:
    """Extract function argument names and values from a frame.

    Returns:
        Tuple of (arg_names, arg_values) where arg_names is a list of argument
        names in order, and arg_values is a dict mapping arg names to values.
        Returns None on error.

    Note: varnames layout is [positional][keyword-only][*args][**kwargs][locals]
    """
    try:
        code = frame.f_code
        arg_count = code.co_argcount
        kwonly_count = code.co_kwonlyargcount
        varnames = code.co_varnames
        flags = code.co_flags

        arg_names: list[str] = []
        arg_values: dict[str, Any] = {}

        # Regular positional/keyword args
        for name in varnames[:arg_count]:
            arg_names.append(name)
            if name in frame.f_locals:
                arg_values[name] = frame.f_locals[name]

        # Track position after positional args
        pos = arg_count

        # Keyword-only args (come after positional args in varnames)
        for name in varnames[pos : pos + kwonly_count]:
            arg_names.append(name)
            if name in frame.f_locals:
                arg_values[name] = frame.f_locals[name]
        pos += kwonly_count

        # Check for *args (CO_VARARGS = 0x04)
        if flags & _inspect_module.CO_VARARGS:
            args_name = varnames[pos]
            arg_names.append(f"*{args_name}")
            if args_name in frame.f_locals:
                arg_values[args_name] = frame.f_locals[args_name]
            pos += 1

        # Check for **kwargs (CO_VARKEYWORDS = 0x08)
        if flags & _inspect_module.CO_VARKEYWORDS:
            kwargs_name = varnames[pos]
            arg_names.append(f"**{kwargs_name}")
            if kwargs_name in frame.f_locals:
                arg_values[kwargs_name] = frame.f_locals[kwargs_name]

        return arg_names, arg_values
    except Exception:
        return None


def collect_frames(tb, cfg: SnapshotConfig) -> list[dict[str, Any]]:
    """Collect stack frames from traceback."""
    redactors = compile_redactors(cfg.redact) if cfg.locals_mode == "safe" else []
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

    # Pair tb_items with extracted items
    paired = list(zip(tb_list, extracted, strict=True))

    # Apply module filtering if configured
    if cfg.include_modules is not None:
        filtered = []
        for i, (tb_item, ex_item) in enumerate(paired):
            frame = tb_item.tb_frame
            module = frame.f_globals.get("__name__", "")
            is_crash_frame = i == len(paired) - 1
            matches_filter = any(module.startswith(p) for p in cfg.include_modules)
            if is_crash_frame or matches_filter:
                filtered.append((tb_item, ex_item))
        paired = filtered

    total_frames = len(paired)
    for i, (tb_item, ex_item) in enumerate(paired):
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
        }

        # Mark coroutine frames for async context
        if cfg.include_async_context:
            try:
                from .async_context import is_coroutine_frame

                if is_coroutine_frame(frame):
                    frame_dict["is_coroutine"] = True
            except Exception:
                pass

        if cfg.source_mode == "all":
            frame_dict["source"] = get_source_snippet(
                ex_item.filename, lineno, cfg.source_context
            )
        elif cfg.source_mode == "crash_only" and i == total_frames - 1:
            frame_dict["source"] = get_source_snippet(
                ex_item.filename, lineno, cfg.source_context
            )

        if cfg.locals_mode == "safe":
            locals_dict, truncated, truncated_keys = serialize_locals_with_stats(
                frame.f_locals, cfg, redactors
            )
            frame_dict["locals"] = locals_dict
            frame_dict["locals_meta"] = locals_metadata(frame.f_locals, cfg)
            if truncated:
                frame_dict["locals_truncated"] = True
                frame_dict["locals_truncated_keys"] = truncated_keys

            # Separate function arguments
            if cfg.include_args:
                arg_info = extract_function_args(frame)
                if arg_info:
                    arg_names, arg_values = arg_info
                    serialized_args, _, _ = serialize_locals_with_stats(arg_values, cfg, redactors)
                    frame_dict["args"] = serialized_args
                    frame_dict["arg_names"] = arg_names
        elif cfg.locals_mode == "meta":
            frame_dict["locals_meta"] = locals_metadata(frame.f_locals, cfg)
            try:
                total_locals = len(frame.f_locals)
            except Exception:
                total_locals = None
            if total_locals is not None and total_locals > cfg.max_items:
                frame_dict["locals_truncated"] = True
                frame_dict["locals_truncated_keys"] = total_locals - cfg.max_items

            # Separate function arguments (metadata only)
            if cfg.include_args:
                arg_info = extract_function_args(frame)
                if arg_info:
                    arg_names, arg_values = arg_info
                    frame_dict["args_meta"] = locals_metadata(arg_values, cfg)
                    frame_dict["arg_names"] = arg_names

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
    frames = collect_frames(tb, cfg)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "timestamp_utc": dt.datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "exception": _summarize_exception(exc, cfg, max_depth=cfg.max_exception_depth),
        "traceback": "".join(traceback.format_exception(type(exc), exc, tb)),
        "frames": frames,
        "capture_config": _serialize_config(cfg),
    }
    if frames:
        payload["crash_frame_index"] = 0

    if cfg.include_env:
        payload["env"] = get_env_info(cfg)

    # Git context
    if cfg.include_git:
        try:
            from .git_context import get_git_context

            git_ctx = get_git_context()
            if git_ctx:
                payload["git"] = git_ctx
        except Exception:
            pass

    # Async context
    if cfg.include_async_context:
        try:
            from .async_context import get_async_context

            async_ctx = get_async_context()
            if async_ctx:
                payload["async"] = async_ctx
        except Exception:
            pass

    # Recent logs
    if cfg.capture_logs:
        try:
            from .log_capture import get_recent_logs

            logs = get_recent_logs(cfg.log_max_records)
            if logs:
                payload["recent_logs"] = logs
        except Exception:
            pass

    if extra:
        payload.update(extra)

    try:
        from . import __version__ as llmdebug_version
    except Exception:
        llmdebug_version = "unknown"
    payload["llmdebug_version"] = llmdebug_version

    return write_bundle(payload, cfg)
