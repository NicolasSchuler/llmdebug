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

    def __post_init__(self) -> None:
        if self.locals_mode not in {"safe", "meta", "none"}:
            raise ValueError("locals_mode must be 'safe', 'meta', or 'none'")
        if self.source_mode not in {"all", "crash_only", "none"}:
            raise ValueError("source_mode must be 'all', 'crash_only', or 'none'")
