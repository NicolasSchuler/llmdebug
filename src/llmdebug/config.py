"""Configuration dataclass for snapshot capture."""

from __future__ import annotations

from dataclasses import dataclass, field
from re import Pattern


@dataclass(frozen=True)
class SnapshotConfig:
    """Configuration for debug snapshot capture."""

    out_dir: str = ".llmdebug"
    frames: int = 5
    source_context: int = 3
    source_mode: str = "all"  # "all" | "crash_only" | "none"
    locals_mode: str = "safe"  # "safe" | "meta" | "none"
    max_str: int = 500
    max_items: int = 50
    redact: tuple[str | Pattern[str], ...] = field(default_factory=tuple)
    include_env: bool = True
    debug: bool = False
    max_snapshots: int = 50  # 0 = unlimited
    include_modules: tuple[str, ...] | None = None  # Filter frames by module prefix
    max_exception_depth: int = 5  # Exception chain recursion limit
    lock_timeout: float = 5.0  # Seconds to wait for file lock
    # New feature flags
    include_git: bool = True  # Capture git commit/branch/dirty status
    include_array_stats: bool = False  # Compute min/max/mean/std for arrays
    include_args: bool = True  # Separate function arguments from locals
    categorize_errors: bool = True  # Auto-classify errors with suggestions
    include_async_context: bool = True  # Capture asyncio task info
    capture_logs: bool = False  # Capture recent log records
    log_max_records: int = 20  # Max log records to capture
    output_format: str = "json_compact"  # "json" | "json_compact" | "toon"

    def __post_init__(self) -> None:
        if self.output_format not in {"json", "json_compact", "toon"}:
            raise ValueError("output_format must be 'json', 'json_compact', or 'toon'")
        if self.locals_mode not in {"safe", "meta", "none"}:
            raise ValueError("locals_mode must be 'safe', 'meta', or 'none'")
        if self.source_mode not in {"all", "crash_only", "none"}:
            raise ValueError("source_mode must be 'all', 'crash_only', or 'none'")
        if self.max_snapshots < 0:
            raise ValueError("max_snapshots must be >= 0 (0 = unlimited)")
        if self.max_exception_depth < 1:
            raise ValueError("max_exception_depth must be >= 1")
        if self.lock_timeout < 0:
            raise ValueError("lock_timeout must be >= 0")
        if self.log_max_records < 0:
            raise ValueError("log_max_records must be >= 0")
