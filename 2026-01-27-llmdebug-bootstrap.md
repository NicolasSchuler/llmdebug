# llmdebug Bootstrap Guide

This document contains everything needed to create the `llmdebug` repository from scratch.

## Repository Structure

```
llmdebug/
├── src/
│   └── llmdebug/
│       ├── __init__.py
│       ├── capture.py
│       ├── serialize.py
│       ├── output.py
│       ├── config.py
│       ├── pytest_plugin.py
│       └── py.typed
├── tests/
│   ├── __init__.py
│   ├── test_capture.py
│   ├── test_serialize.py
│   ├── test_output.py
│   └── test_pytest_plugin.py
├── pyproject.toml
├── README.md
├── LICENSE
├── CLAUDE.md
└── .gitignore
```

## Complete pyproject.toml

```toml
[project]
name = "llmdebug"
version = "0.1.0"
description = "Structured debug snapshots for LLM-assisted debugging"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.10"
authors = [{ name = "Nicolas", email = "your@email.com" }]
keywords = ["debugging", "llm", "crash-reporting", "pytest"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Software Development :: Debuggers",
    "Framework :: Pytest",
]
dependencies = []

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "numpy>=1.20",
    "ruff>=0.1",
    "pyright>=1.1",
]

[project.urls]
Homepage = "https://github.com/youruser/llmdebug"
Repository = "https://github.com/youruser/llmdebug"

[project.entry-points.pytest11]
llmdebug = "llmdebug.pytest_plugin"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/llmdebug"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.pyright]
include = ["src", "tests"]
pythonVersion = "3.10"
typeCheckingMode = "standard"
```

## src/llmdebug/__init__.py

```python
"""Structured debug snapshots for LLM-assisted debugging."""

from __future__ import annotations

from .capture import capture_exception
from .config import SnapshotConfig
from .output import get_latest_snapshot

import contextlib
import sys
from typing import Any, Callable, Iterable, Pattern, Union

__version__ = "0.1.0"
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
    redact: Iterable[Union[str, Pattern[str]]] = (),
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
    redact: Iterable[Union[str, Pattern[str]]] = (),
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
```

## src/llmdebug/config.py

```python
"""Configuration dataclass for snapshot capture."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Pattern, Tuple, Union


@dataclass(frozen=True)
class SnapshotConfig:
    """Configuration for debug snapshot capture."""

    out_dir: str = ".llmdebug"
    frames: int = 5
    source_context: int = 3
    locals_mode: str = "safe"  # "safe" | "none"
    max_str: int = 500
    max_items: int = 50
    redact: Tuple[Union[str, Pattern[str]], ...] = field(default_factory=tuple)
    include_env: bool = True
    debug: bool = False
```

## src/llmdebug/serialize.py

```python
"""Safe serialization with smart array handling."""

from __future__ import annotations

import inspect
import json
import re
from typing import Any, Dict, List, Pattern, Union

from .config import SnapshotConfig

JsonLike = Union[None, bool, int, float, str, Dict[str, Any], List[Any]]


def compile_redactors(patterns: tuple) -> list[Pattern[str]]:
    """Compile regex patterns for redaction."""
    return [re.compile(p) if isinstance(p, str) else p for p in patterns]


def redact_text(text: str, redactors: list[Pattern[str]]) -> str:
    """Apply redaction patterns to text."""
    for rx in redactors:
        text = rx.sub("[REDACTED]", text)
    return text


def truncate_str(s: str, max_len: int) -> str:
    """Truncate string with marker."""
    if len(s) <= max_len:
        return s
    return s[: max_len - 12] + "...[TRUNC]"


def safe_repr(x: Any, cfg: SnapshotConfig) -> str:
    """Get repr with fallback for unreprable objects."""
    try:
        r = repr(x)
    except Exception:
        r = f"<unreprable {type(x).__name__}>"
    return truncate_str(r, cfg.max_str)


def is_array_like(x: Any) -> bool:
    """Check if object is an array type (numpy, jax, torch, etc)."""
    module = type(x).__module__
    return any(
        module.startswith(prefix)
        for prefix in ("numpy", "jax", "torch", "tensorflow", "cupy")
    ) and hasattr(x, "shape")


def is_dataframe(x: Any) -> bool:
    """Check if object is a pandas DataFrame."""
    return type(x).__module__.startswith("pandas") and type(x).__name__ == "DataFrame"


def is_series(x: Any) -> bool:
    """Check if object is a pandas Series."""
    return type(x).__module__.startswith("pandas") and type(x).__name__ == "Series"


def summarize_array(arr: Any, cfg: SnapshotConfig) -> dict:
    """Summarize an array-like object."""
    type_name = f"{type(arr).__module__}.{type(arr).__name__}"
    summary: Dict[str, Any] = {"__array__": type_name}

    if hasattr(arr, "shape"):
        try:
            summary["shape"] = list(arr.shape)
        except Exception:
            pass

    if hasattr(arr, "dtype"):
        try:
            summary["dtype"] = str(arr.dtype)
        except Exception:
            pass

    # Sample first N elements
    try:
        flat = arr.flatten() if hasattr(arr, "flatten") else arr.ravel()
        sample = [float(x) for x in flat[:5]]
        summary["head"] = sample
        if len(flat) > 5:
            summary["head_truncated"] = True
    except Exception:
        pass

    return summary


def summarize_dataframe(df: Any, cfg: SnapshotConfig) -> dict:
    """Summarize a pandas DataFrame."""
    summary: Dict[str, Any] = {"__dataframe__": True}

    try:
        summary["shape"] = list(df.shape)
    except Exception:
        pass

    try:
        summary["columns"] = list(df.columns)[: cfg.max_items]
        if len(df.columns) > cfg.max_items:
            summary["columns_truncated"] = True
    except Exception:
        pass

    try:
        head_rows = min(3, len(df))
        summary["head"] = df.head(head_rows).values.tolist()
    except Exception:
        pass

    return summary


def summarize_series(series: Any, cfg: SnapshotConfig) -> dict:
    """Summarize a pandas Series."""
    summary: Dict[str, Any] = {"__series__": True}

    try:
        summary["shape"] = list(series.shape)
    except Exception:
        pass

    if hasattr(series, "name") and series.name is not None:
        summary["name"] = str(series.name)

    if hasattr(series, "dtype"):
        summary["dtype"] = str(series.dtype)

    try:
        summary["head"] = series.head(5).tolist()
        if len(series) > 5:
            summary["head_truncated"] = True
    except Exception:
        pass

    return summary


def to_jsonlike(x: Any, cfg: SnapshotConfig, depth: int = 0) -> JsonLike:
    """Convert arbitrary Python object to JSON-serializable form."""
    # Primitives
    if x is None or isinstance(x, (bool, int, float)):
        return x
    if isinstance(x, str):
        return truncate_str(x, cfg.max_str)

    # Max depth reached
    if depth >= 2:
        return safe_repr(x, cfg)

    # Special types
    if is_array_like(x):
        return summarize_array(x, cfg)
    if is_dataframe(x):
        return summarize_dataframe(x, cfg)
    if is_series(x):
        return summarize_series(x, cfg)

    # Collections
    if isinstance(x, (list, tuple, set, frozenset)):
        items = list(x)[: cfg.max_items]
        result = [to_jsonlike(i, cfg, depth + 1) for i in items]
        if len(list(x)) > cfg.max_items:
            result.append("...[TRUNC]")
        return result

    if isinstance(x, dict):
        out: Dict[str, Any] = {}
        for i, (k, v) in enumerate(x.items()):
            if i >= cfg.max_items:
                out["...[TRUNC]"] = f"+{len(x) - cfg.max_items} more keys"
                break
            out[safe_repr(k, cfg)] = to_jsonlike(v, cfg, depth + 1)
        return out

    # Objects with __dict__
    if hasattr(x, "__dict__") and depth < 1:
        d = getattr(x, "__dict__", {})
        if isinstance(d, dict):
            out = {"__type__": type(x).__name__}
            for i, (k, v) in enumerate(d.items()):
                if i >= min(cfg.max_items, 20):
                    out["...[TRUNC]"] = "more fields omitted"
                    break
                out[str(k)] = to_jsonlike(v, cfg, depth + 1)
            return out

    return safe_repr(x, cfg)


def serialize_locals(
    local_vars: dict, cfg: SnapshotConfig, redactors: list[Pattern[str]]
) -> dict:
    """Serialize local variables with redaction."""
    result: Dict[str, Any] = {}

    for i, (k, v) in enumerate(local_vars.items()):
        if i >= cfg.max_items:
            break
        # Skip modules, functions, classes
        if inspect.ismodule(v) or inspect.isfunction(v) or inspect.isclass(v):
            result[k] = f"<{type(v).__name__}>"
        else:
            result[k] = to_jsonlike(v, cfg)

    # Apply redaction
    raw = json.dumps(result, ensure_ascii=False, default=str)
    raw = redact_text(raw, redactors)
    return json.loads(raw)
```

## src/llmdebug/capture.py

```python
"""Core exception capture logic."""

from __future__ import annotations

import datetime as dt
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

from .config import SnapshotConfig
from .output import write_bundle
from .serialize import compile_redactors, serialize_locals


def get_env_info() -> Dict[str, Any]:
    """Collect environment information."""
    return {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
    }


def get_source_snippet(filename: str, lineno: int, ctx: int) -> Dict[str, Any]:
    """Extract source code around a line."""
    try:
        if not filename or not os.path.exists(filename):
            return {}
        with open(filename, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        start = max(lineno - 1 - ctx, 0)
        end = min(lineno - 1 + ctx + 1, len(lines))
        snippet = [
            {"lineno": i + 1, "code": lines[i].rstrip("\n")} for i in range(start, end)
        ]
        return {"start": start + 1, "end": end, "snippet": snippet}
    except Exception:
        return {}


def collect_frames(tb, cfg: SnapshotConfig) -> list[Dict[str, Any]]:
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

    for tb_item, ex_item in zip(tb_list, extracted):
        frame = tb_item.tb_frame
        frame_dict: Dict[str, Any] = {
            "file": ex_item.filename,
            "line": ex_item.lineno,
            "function": ex_item.name,
            "code": ex_item.line,
            "source": get_source_snippet(ex_item.filename, ex_item.lineno, cfg.source_context),
        }

        if cfg.locals_mode == "safe":
            frame_dict["locals"] = serialize_locals(frame.f_locals, cfg, redactors)

        frames_out.append(frame_dict)

    # Reverse so crash site is first (index 0)
    return list(reversed(frames_out))


def capture_exception(name: str, exc: BaseException, tb, cfg: SnapshotConfig) -> Path:
    """Capture exception details and write snapshot."""
    payload: Dict[str, Any] = {
        "name": name,
        "timestamp_utc": dt.datetime.utcnow().isoformat() + "Z",
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
```

## src/llmdebug/output.py

```python
"""File output and latest.json management."""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from .config import SnapshotConfig


def _timestamp() -> str:
    """Generate timestamp string."""
    return dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


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


def write_bundle(payload: Dict[str, Any], cfg: SnapshotConfig) -> Path:
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

    return json_path


def get_latest_snapshot(out_dir: str = ".llmdebug") -> Dict[str, Any] | None:
    """Read the latest snapshot if it exists."""
    latest = Path(out_dir) / "latest.json"
    if not latest.exists():
        return None
    try:
        return json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        return None
```

## src/llmdebug/pytest_plugin.py

```python
"""Pytest plugin for automatic snapshot capture on test failures."""

from __future__ import annotations

import pytest

from .capture import capture_exception
from .config import SnapshotConfig


def pytest_configure(config):
    """Register the no_snapshot marker."""
    config.addinivalue_line(
        "markers",
        "no_snapshot: disable llmdebug snapshot capture for this test",
    )


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Capture snapshot on test failure."""
    outcome = yield
    report = outcome.get_result()

    # Only capture on test call failures (not setup/teardown)
    if report.when != "call" or not report.failed:
        return

    # Respect opt-out marker
    if item.get_closest_marker("no_snapshot"):
        return

    # Get exception info
    if call.excinfo is None:
        return

    exc = call.excinfo.value
    tb = call.excinfo.tb

    # Use test node ID as snapshot name
    name = item.nodeid.replace("/", "_").replace("::", "_")

    cfg = SnapshotConfig()
    try:
        capture_exception(name, exc, tb, cfg)
    except Exception:
        pass  # Silent fallback - never break test runs
```

## src/llmdebug/py.typed

```
# PEP 561 marker file
```

## tests/test_capture.py

```python
"""Tests for exception capture."""

import pytest

from llmdebug import debug_snapshot, snapshot_section


def test_decorator_captures_exception(tmp_path):
    """Test that decorator captures exception and still raises."""

    @debug_snapshot(out_dir=str(tmp_path))
    def failing_func():
        x = [1, 2, 3]
        raise ValueError("test error")

    with pytest.raises(ValueError, match="test error"):
        failing_func()

    # Check snapshot was created
    snapshots = list(tmp_path.glob("*.json"))
    assert len(snapshots) == 2  # snapshot + latest.json


def test_context_manager_captures_exception(tmp_path):
    """Test that context manager captures exception."""
    with pytest.raises(RuntimeError):
        with snapshot_section("test_section", out_dir=str(tmp_path)):
            data = {"key": "value"}
            raise RuntimeError("context error")

    snapshots = list(tmp_path.glob("*test_section*.json"))
    assert len(snapshots) == 1


def test_no_exception_no_snapshot(tmp_path):
    """Test that no snapshot is created when no exception."""

    @debug_snapshot(out_dir=str(tmp_path))
    def passing_func():
        return 42

    result = passing_func()
    assert result == 42

    snapshots = list(tmp_path.glob("*.json"))
    assert len(snapshots) == 0
```

## tests/test_serialize.py

```python
"""Tests for serialization."""

import pytest

from llmdebug.config import SnapshotConfig
from llmdebug.serialize import to_jsonlike, summarize_array


@pytest.fixture
def cfg():
    return SnapshotConfig()


def test_primitives(cfg):
    """Test primitive type serialization."""
    assert to_jsonlike(None, cfg) is None
    assert to_jsonlike(True, cfg) is True
    assert to_jsonlike(42, cfg) == 42
    assert to_jsonlike(3.14, cfg) == 3.14
    assert to_jsonlike("hello", cfg) == "hello"


def test_string_truncation(cfg):
    """Test long strings are truncated."""
    long_str = "x" * 1000
    result = to_jsonlike(long_str, cfg)
    assert len(result) <= cfg.max_str
    assert "TRUNC" in result


def test_list_serialization(cfg):
    """Test list serialization with truncation."""
    small_list = [1, 2, 3]
    assert to_jsonlike(small_list, cfg) == [1, 2, 3]

    large_list = list(range(100))
    result = to_jsonlike(large_list, cfg)
    assert len(result) == cfg.max_items + 1  # items + TRUNC marker
    assert result[-1] == "...[TRUNC]"


def test_dict_serialization(cfg):
    """Test dict serialization."""
    d = {"a": 1, "b": "hello"}
    result = to_jsonlike(d, cfg)
    assert result["'a'"] == 1
    assert result["'b'"] == "hello"


@pytest.mark.skipif(
    not pytest.importorskip("numpy", reason="numpy not installed"),
    reason="numpy not installed",
)
def test_numpy_array_summary():
    """Test numpy array summarization."""
    import numpy as np

    cfg = SnapshotConfig()
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    result = summarize_array(arr, cfg)
    assert result["__array__"] == "numpy.ndarray"
    assert result["shape"] == [2, 3]
    assert result["dtype"] == "float32"
    assert "head" in result
```

## .gitignore

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/
.eggs/

# Virtual environments
.venv/
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Testing
.pytest_cache/
.coverage
htmlcov/

# llmdebug output (for development)
.llmdebug/

# OS
.DS_Store
Thumbs.db
```

## README.md

```markdown
# llmdebug

Structured debug snapshots for LLM-assisted debugging.

When your code fails, `llmdebug` captures the exception, stack frames, local variables, and environment info in a JSON format that LLMs can easily parse and reason about.

## Installation

```bash
pip install llmdebug
```

## Quick Start

### Decorator

```python
from llmdebug import debug_snapshot

@debug_snapshot()
def main():
    data = load_data()
    process(data)

if __name__ == "__main__":
    main()
```

### Context Manager

```python
from llmdebug import snapshot_section

with snapshot_section("data_processing"):
    result = transform(data)
```

### Pytest (automatic)

Just install the package - test failures automatically generate snapshots.

```bash
pytest  # Failures create .llmdebug/*.json
```

## Output

On failure, find your snapshot at `.llmdebug/latest.json`:

```json
{
  "name": "main",
  "exception": {"type": "ValueError", "message": "..."},
  "frames": [
    {
      "file": "app.py",
      "line": 42,
      "function": "process",
      "locals": {
        "data": {"__array__": "numpy.ndarray", "shape": [0, 512]}
      }
    }
  ]
}
```

## For Claude Code Users

Add this to your `CLAUDE.md`:

```markdown
## Debug Snapshots

On any failure:
1. Read `.llmdebug/latest.json`
2. Check exception type, crash frame, and locals
3. Look for: empty arrays, None values, type mismatches
4. Form hypothesis, fix, and re-run
```

## Configuration

```python
@debug_snapshot(
    out_dir=".llmdebug",      # Output directory
    frames=5,                  # Stack frames to capture
    locals_mode="safe",        # "safe" or "none"
    redact=[r"api_key=.*"],    # Patterns to redact
)
```

## License

MIT
```

## CLAUDE.md

```markdown
# CLAUDE.md

## Debug Snapshots (llmdebug)

This project uses `llmdebug` for structured crash diagnostics.

### On any failure:
1. Read `.llmdebug/latest.json`
2. Analyze the snapshot:
   - **Exception type/message** - what went wrong
   - **Closest frame** - where it happened (index 0 is crash site)
   - **Locals** - variable values at crash time
   - **Array shapes** - look for empty arrays, shape mismatches, wrong dtypes
3. Form 2-3 hypotheses based on evidence
4. Fix the most likely cause, re-run to verify

### Key signals to look for:
- `shape: [0, ...]` - empty array, upstream data issue
- `None` where object expected - initialization/ordering bug
- Type mismatch in locals - wrong function called or bad return value

### Don't:
- Guess without reading the snapshot first
- Make multiple speculative changes - one hypothesis at a time
```

## Next Steps

1. Create GitHub repository
2. Copy all files from this bootstrap guide
3. Run `uv sync` or `pip install -e ".[dev]"`
4. Run `pytest` to verify tests pass
5. Publish to PyPI: `uv build && uv publish`
