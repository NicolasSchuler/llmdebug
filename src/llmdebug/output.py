"""File output and latest.json management."""

from __future__ import annotations

import datetime as dt
import json
import sys
from datetime import timezone
from pathlib import Path
from typing import Any

from .config import SnapshotConfig


def _timestamp() -> str:
    """Generate timestamp string."""
    return dt.datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _update_latest_symlink(out_dir: Path, target_name: str) -> None:
    """Update latest.json symlink/copy."""
    latest = out_dir / "latest.json"
    target = out_dir / target_name

    if sys.platform == "win32":
        # Windows: copy instead of symlink
        try:
            if latest.exists():
                latest.unlink()
            latest.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass
    else:
        # POSIX: atomic symlink update
        try:
            tmp = out_dir / ".latest.json.tmp"
            tmp.unlink(missing_ok=True)
            tmp.symlink_to(target_name)
            tmp.rename(latest)
        except Exception:
            pass


def _cleanup_old_snapshots(out_dir: Path, max_snapshots: int) -> None:
    """Remove old snapshots beyond the limit."""
    if max_snapshots <= 0:
        return  # Unlimited

    # Find all snapshot JSON files (exclude latest.json)
    snapshots = [
        p for p in out_dir.glob("*.json")
        if p.name != "latest.json" and not p.name.endswith(".tmp")
    ]

    if len(snapshots) <= max_snapshots:
        return

    # Sort by modification time (oldest first)
    snapshots.sort(key=lambda p: p.stat().st_mtime)

    # Delete oldest snapshots beyond the limit
    to_delete = snapshots[: len(snapshots) - max_snapshots]
    for snap in to_delete:
        try:
            snap.unlink()
            # Also delete associated traceback file
            tb_file = snap.with_suffix("").with_suffix(".traceback.txt")
            if tb_file.exists():
                tb_file.unlink()
        except Exception:
            pass  # Best effort cleanup


def write_bundle(payload: dict[str, Any], cfg: SnapshotConfig) -> Path:
    """Write snapshot bundle to disk."""
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    name = payload.get("name", "snapshot").replace(" ", "_").replace("/", "_")
    base = f"{_timestamp()}_{name}"
    json_path = out_dir / f"{base}.json"

    # Atomic write
    tmp = json_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.rename(json_path)

    # Update latest.json
    _update_latest_symlink(out_dir, json_path.name)

    # Write human-readable traceback
    if "traceback" in payload:
        tb_path = out_dir / f"{base}.traceback.txt"
        tb_path.write_text(payload["traceback"], encoding="utf-8")

    # Cleanup old snapshots
    _cleanup_old_snapshots(out_dir, cfg.max_snapshots)

    return json_path


def get_latest_snapshot(out_dir: str = ".llmdebug") -> dict[str, Any] | None:
    """Read the latest snapshot if it exists."""
    latest = Path(out_dir) / "latest.json"
    if not latest.exists():
        return None
    try:
        return json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        return None
