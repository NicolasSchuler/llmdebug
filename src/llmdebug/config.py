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
    locals_mode: str = "safe"  # "safe" | "none"
    max_str: int = 500
    max_items: int = 50
    redact: tuple[str | Pattern[str], ...] = field(default_factory=tuple)
    include_env: bool = True
    debug: bool = False
