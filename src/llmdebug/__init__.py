"""Structured debug snapshots for LLM-assisted debugging."""

from __future__ import annotations

import contextlib
import functools
import sys
from collections.abc import Callable, Iterable
from re import Pattern
from typing import Any

from .capture import capture_exception
from .config import SnapshotConfig
from .log_capture import install_log_handler as enable_log_capture
from .output import get_latest_snapshot

__version__ = "2.0.0"
__all__ = [
    "debug_snapshot",
    "snapshot_section",
    "SnapshotConfig",
    "get_latest_snapshot",
    "enable_log_capture",
]


def debug_snapshot(
    *,
    name: str | None = None,
    out_dir: str = ".llmdebug",
    frames: int = 5,
    source_context: int = 3,
    source_mode: str = "all",
    locals_mode: str = "safe",
    max_str: int = 500,
    max_items: int = 50,
    redact: Iterable[str | Pattern[str]] = (),
    include_env: bool = True,
    debug: bool = False,
    max_snapshots: int = 50,
    include_modules: Iterable[str] | None = None,
    max_exception_depth: int = 5,
    lock_timeout: float = 5.0,
    include_git: bool = True,
    include_array_stats: bool = False,
    include_args: bool = True,
    categorize_errors: bool = True,
    include_async_context: bool = True,
    capture_logs: bool = False,
    log_max_records: int = 20,
    output_format: str = "json_compact",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that captures debug snapshots on exception.

    Usage:
        @debug_snapshot()
        def main():
            ...

    Args:
        name: Snapshot name (defaults to function name)
        out_dir: Output directory for snapshots
        frames: Number of stack frames to capture
        source_context: Lines of source before/after crash
        source_mode: "all" | "crash_only" | "none"
        locals_mode: "safe" | "meta" | "none"
        max_str: Max string length before truncation
        max_items: Max collection items to capture
        redact: Regex patterns to redact from output
        include_env: Include Python/platform info
        debug: Warn on capture failure instead of silent
        max_snapshots: Max snapshots to keep (0 = unlimited)
        include_modules: Only capture frames from these module prefixes (None = all)
        max_exception_depth: Max depth for exception chain traversal
        lock_timeout: Seconds to wait for file lock
        include_git: Include git commit/branch/dirty status
        include_array_stats: Include min/max/mean/std for arrays
        include_args: Separate function arguments from locals
        categorize_errors: Auto-classify errors with suggestions
        include_async_context: Include asyncio task info
        capture_logs: Capture recent log records (requires enable_log_capture())
        log_max_records: Max log records to capture
        output_format: Output format: "json", "json_compact", or "toon"
    """
    cfg = SnapshotConfig(
        out_dir=out_dir,
        frames=frames,
        source_context=source_context,
        source_mode=source_mode,
        locals_mode=locals_mode,
        max_str=max_str,
        max_items=max_items,
        redact=tuple(redact),
        include_env=include_env,
        debug=debug,
        max_snapshots=max_snapshots,
        include_modules=tuple(include_modules) if include_modules else None,
        max_exception_depth=max_exception_depth,
        lock_timeout=lock_timeout,
        include_git=include_git,
        include_array_stats=include_array_stats,
        include_args=include_args,
        categorize_errors=categorize_errors,
        include_async_context=include_async_context,
        capture_logs=capture_logs,
        log_max_records=log_max_records,
        output_format=output_format,
    )

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        snap_name = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                _, _, tb = sys.exc_info()
                try:
                    capture_exception(snap_name, e, tb, cfg)
                except Exception:
                    if cfg.debug:
                        sys.stderr.write(f"llmdebug: capture failed for {snap_name}\n")
                raise

        return wrapper

    return decorator


@contextlib.contextmanager
def snapshot_section(
    section_name: str,
    *,
    out_dir: str = ".llmdebug",
    frames: int = 5,
    source_context: int = 3,
    source_mode: str = "all",
    locals_mode: str = "safe",
    max_str: int = 500,
    max_items: int = 50,
    redact: Iterable[str | Pattern[str]] = (),
    include_env: bool = True,
    debug: bool = False,
    max_snapshots: int = 50,
    include_modules: Iterable[str] | None = None,
    max_exception_depth: int = 5,
    lock_timeout: float = 5.0,
    include_git: bool = True,
    include_array_stats: bool = False,
    include_args: bool = True,
    categorize_errors: bool = True,
    include_async_context: bool = True,
    capture_logs: bool = False,
    log_max_records: int = 20,
    output_format: str = "json_compact",
):
    """Context manager that captures debug snapshots on exception.

    Usage:
        with snapshot_section("data_loading"):
            ...
    """
    cfg = SnapshotConfig(
        out_dir=out_dir,
        frames=frames,
        source_context=source_context,
        source_mode=source_mode,
        locals_mode=locals_mode,
        max_str=max_str,
        max_items=max_items,
        redact=tuple(redact),
        include_env=include_env,
        debug=debug,
        max_snapshots=max_snapshots,
        include_modules=tuple(include_modules) if include_modules else None,
        max_exception_depth=max_exception_depth,
        lock_timeout=lock_timeout,
        include_git=include_git,
        include_array_stats=include_array_stats,
        include_args=include_args,
        categorize_errors=categorize_errors,
        include_async_context=include_async_context,
        capture_logs=capture_logs,
        log_max_records=log_max_records,
        output_format=output_format,
    )
    try:
        yield
    except Exception as e:
        _, _, tb = sys.exc_info()
        try:
            capture_exception(section_name, e, tb, cfg)
        except Exception:
            if cfg.debug:
                sys.stderr.write(f"llmdebug: capture failed for {section_name}\n")
        raise
