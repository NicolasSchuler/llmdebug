"""File output and latest.json management."""

from __future__ import annotations

import datetime as dt
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timezone
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

from .config import SnapshotConfig


@contextmanager
def _directory_lock(out_dir: Path, timeout: float) -> Iterator[bool]:
    """Acquire lock for output directory operations.

    Yields True if lock acquired, False if timeout.
    """
    lock = FileLock(out_dir / ".llmdebug.lock", timeout=timeout)
    try:
        lock.acquire()
        yield True
    except Timeout:
        yield False
    finally:
        if lock.is_locked:
            lock.release()


def _timestamp() -> str:
    """Generate timestamp string."""
    return dt.datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _update_latest_symlink(out_dir: Path, target_name: str, ext: str = ".json") -> None:
    """Update latest symlink/copy.

    Args:
        out_dir: Output directory
        target_name: Name of the target file
        ext: File extension for latest file (default: .json)
    """
    latest = out_dir / f"latest{ext}"
    target = out_dir / target_name

    # Clean up stale latest file from other format to avoid confusion
    opposite_ext = ".toon" if ext == ".json" else ".json"
    stale_latest = out_dir / f"latest{opposite_ext}"
    try:
        stale_latest.unlink(missing_ok=True)
    except Exception:
        pass  # Best effort cleanup

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
            tmp = out_dir / f".latest{ext}.tmp"
            tmp.unlink(missing_ok=True)
            tmp.symlink_to(target_name)
            tmp.rename(latest)
        except Exception:
            pass


def _cleanup_old_snapshots(out_dir: Path, max_snapshots: int) -> None:
    """Remove old snapshots beyond the limit."""
    if max_snapshots <= 0:
        return  # Unlimited

    # Find all snapshot files (JSON and TOON), exclude latest.*
    snapshots = [
        p for p in out_dir.iterdir()
        if p.is_file()
        and p.suffix in {".json", ".toon"}
        and not p.name.startswith("latest")
        and not p.name.endswith(".tmp")
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
    """Write snapshot bundle to disk in configured format."""
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    name = payload.get("name", "snapshot").replace(" ", "_").replace("/", "_")
    base = f"{_timestamp()}_{name}"

    # Determine content and extension based on format
    if cfg.output_format == "json":
        content = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        ext = ".json"
    elif cfg.output_format == "json_compact":
        from .compact_keys import compact_keys
        compacted = compact_keys(payload)
        content = json.dumps(compacted, separators=(",", ":"), ensure_ascii=False, default=str)
        ext = ".json"  # Still valid JSON, just minified with short keys
    elif cfg.output_format == "toon":
        from .toon_encoder import encode_snapshot_toon
        content = encode_snapshot_toon(payload)
        ext = ".toon"
    else:
        raise ValueError(f"Unknown output_format: {cfg.output_format}")

    snapshot_path = out_dir / f"{base}{ext}"

    # Atomic write
    tmp = snapshot_path.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.rename(snapshot_path)

    # Write human-readable traceback
    if "traceback" in payload:
        tb_path = out_dir / f"{base}.traceback.txt"
        tb_path.write_text(payload["traceback"], encoding="utf-8")

    # Lock for symlink update and cleanup (these can race with concurrent writers)
    with _directory_lock(out_dir, cfg.lock_timeout) as acquired:
        if acquired:
            _update_latest_symlink(out_dir, snapshot_path.name, ext)
            _cleanup_old_snapshots(out_dir, cfg.max_snapshots)
        # If lock not acquired, skip symlink/cleanup - snapshot file is still written

    return snapshot_path


def get_latest_snapshot(out_dir: str = ".llmdebug") -> dict[str, Any] | None:
    """Read the latest snapshot if it exists, auto-detecting format.

    If both latest.json and latest.toon exist (e.g. stale leftovers from
    a previous run), prefers the newest by mtime. Automatically expands
    compact JSON keys to full keys for compatibility.

    Args:
        out_dir: Output directory containing snapshots

    Returns:
        Snapshot data with full keys, or None if not found
    """
    out_path = Path(out_dir)

    latest_json = out_path / "latest.json"
    latest_toon = out_path / "latest.toon"
    candidates: list[tuple[str, Path, float]] = []
    for kind, path in (("json", latest_json), ("toon", latest_toon)):
        if not path.exists():
            continue
        try:
            candidates.append((kind, path, path.stat().st_mtime))
        except Exception:
            candidates.append((kind, path, 0.0))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[2], reverse=True)

    def _read_json(path: Path) -> dict[str, Any] | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        try:
            from .compact_keys import expand_keys, is_compact_snapshot
            if is_compact_snapshot(data):
                expanded = expand_keys(data)
                return expanded if isinstance(expanded, dict) else None
        except Exception:
            pass
        return data

    def _read_toon(path: Path) -> dict[str, Any] | None:
        try:
            from .toon_encoder import decode_snapshot_toon
            data = decode_snapshot_toon(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    for kind, path, _ in candidates:
        if kind == "json":
            snapshot = _read_json(path)
        else:
            snapshot = _read_toon(path)
        if snapshot is not None:
            return snapshot

    return None
