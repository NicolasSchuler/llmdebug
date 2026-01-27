"""Structured debug snapshots for LLM-assisted debugging."""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Callable, Iterable
from re import Pattern
from typing import Any

from .capture import capture_exception
from .config import SnapshotConfig
from .output import get_latest_snapshot

__version__ = "0.1.2"
__all__ = ["debug_snapshot", "snapshot_section", "SnapshotConfig", "get_latest_snapshot"]


def debug_snapshot(
    *,
    name: str | None = None,
    out_dir: str = ".llmdebug",
    frames: int = 5,
    source_context: int = 3,
    locals_mode: str = "safe",
    max_str: int = 500,
    max_items: int = 50,
    redact: Iterable[str | Pattern[str]] = (),
    include_env: bool = True,
    debug: bool = False,
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
        locals_mode: "safe" to capture locals, "none" to skip
        max_str: Max string length before truncation
        max_items: Max collection items to capture
        redact: Regex patterns to redact from output
        include_env: Include Python/platform info
        debug: Warn on capture failure instead of silent
    """
    cfg = SnapshotConfig(
        out_dir=out_dir,
        frames=frames,
        source_context=source_context,
        locals_mode=locals_mode,
        max_str=max_str,
        max_items=max_items,
        redact=tuple(redact),
        include_env=include_env,
        debug=debug,
    )

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        snap_name = name or fn.__name__

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

        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper

    return decorator


@contextlib.contextmanager
def snapshot_section(
    section_name: str,
    *,
    out_dir: str = ".llmdebug",
    frames: int = 5,
    source_context: int = 3,
    locals_mode: str = "safe",
    max_str: int = 500,
    max_items: int = 50,
    redact: Iterable[str | Pattern[str]] = (),
    include_env: bool = True,
    debug: bool = False,
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
        locals_mode=locals_mode,
        max_str=max_str,
        max_items=max_items,
        redact=tuple(redact),
        include_env=include_env,
        debug=debug,
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
