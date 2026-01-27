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
from .serialize import compile_redactors, serialize_locals


def get_env_info() -> dict[str, Any]:
    """Collect environment information."""
    return {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
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
        frame_dict: dict[str, Any] = {
            "file": ex_item.filename,
            "line": lineno,
            "function": ex_item.name,
            "code": ex_item.line,
            "source": get_source_snippet(ex_item.filename, lineno, cfg.source_context),
        }

        if cfg.locals_mode == "safe":
            frame_dict["locals"] = serialize_locals(frame.f_locals, cfg, redactors)

        frames_out.append(frame_dict)

    # Reverse so crash site is first (index 0)
    return list(reversed(frames_out))


def capture_exception(name: str, exc: BaseException, tb, cfg: SnapshotConfig) -> Path:
    """Capture exception details and write snapshot."""
    payload: dict[str, Any] = {
        "name": name,
        "timestamp_utc": dt.datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "exception": {
            "type": type(exc).__name__,
            "message": str(exc),
        },
        "traceback": "".join(traceback.format_exception(type(exc), exc, tb)),
        "frames": collect_frames(tb, cfg),
    }

    if cfg.include_env:
        payload["env"] = get_env_info()

    return write_bundle(payload, cfg)
