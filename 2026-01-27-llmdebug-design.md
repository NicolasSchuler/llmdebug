# llmdebug Design Document

**Date:** 2026-01-27
**Status:** Approved
**Author:** Nicolas (with Claude)

## Overview

A lightweight Python package that captures structured "debug snapshots" on exceptions, designed specifically for LLM-assisted debugging workflows. When code fails, it writes a JSON file containing the traceback, stack frames, local variables, and environment info in a format optimized for LLM consumption.

## Goals

- **On exception**, capture a debug bundle: traceback, stack frames, selected locals (safely serialized), source context, and environment info
- **Low friction**: decorator and context manager API
- **LLM-friendly**: one JSON file per crash with `latest.json` pointer
- **Zero dependencies**: pure Python, works anywhere

## Non-Goals (for v1)

- Perfect serialization of arbitrary objects
- Full tracing/profiling
- Uploading anywhere
- Manual checkpoint marks (deferred to v1.1)

## Package Structure

```
llmdebug/
├── __init__.py          # Public API: debug_snapshot, snapshot_section
├── capture.py           # Core capture logic, frame collection
├── serialize.py         # Safe serialization, array summaries
├── output.py            # File writing, latest.json management
├── pytest_plugin.py     # pytest hook (auto-registered via entry point)
└── py.typed             # PEP 561 marker
```

## Public API

### Decorator

```python
from llmdebug import debug_snapshot

@debug_snapshot()
def main():
    ...
```

### Context Manager

```python
from llmdebug import snapshot_section

with snapshot_section("data_loading"):
    ...
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `out_dir` | `".llmdebug"` | Output directory for snapshots |
| `frames` | `5` | Number of stack frames to capture |
| `source_context` | `3` | Lines of source before/after crash line |
| `locals_mode` | `"safe"` | `"safe"` or `"none"` |
| `max_str` | `500` | Max string length before truncation |
| `max_items` | `50` | Max items in collections |
| `redact` | `()` | Regex patterns to redact from output |
| `include_env` | `True` | Include Python/platform info |
| `debug` | `False` | Warn on capture failure (vs silent) |

## Array Serialization

Smart detection and summarization for numpy, jax, torch, and pandas without hard dependencies:

```python
def _summarize_array(arr) -> dict:
    type_name = type(arr).__module__ + "." + type(arr).__name__
    summary = {"__array__": type_name}

    if hasattr(arr, "shape"):
        summary["shape"] = list(arr.shape)
    if hasattr(arr, "dtype"):
        summary["dtype"] = str(arr.dtype)

    # Sample first N elements
    try:
        flat = arr.flatten() if hasattr(arr, "flatten") else arr.ravel()
        summary["head"] = [float(x) for x in flat[:5]]
        if len(flat) > 5:
            summary["head_truncated"] = True
    except:
        pass

    return summary
```

Detection via `type(x).__module__` - works without importing the libraries.

**Pandas special handling:**
- DataFrames: `{"__dataframe__": True, "shape": [100, 5], "columns": [...], "head": [[...]]}`
- Series: `{"__series__": True, "name": "...", "shape": [100], "head": [...]}`

## Output Management

### File Structure

```
.llmdebug/
├── 20260127T143052Z_main.json
├── 20260127T143052Z_main.traceback.txt
├── 20260127T144512Z_data_loading.json
├── 20260127T144512Z_data_loading.traceback.txt
└── latest.json  → symlink to most recent .json
```

### Atomic Writes

1. Write to `.tmp` file
2. Rename to final path (atomic on POSIX)
3. Update `latest.json` symlink atomically
4. Fall back to copy on Windows (symlinks require privileges)

## Pytest Plugin

Auto-registered via entry point - no configuration needed.

```python
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    if report.when != "call" or not report.failed:
        return

    if item.get_closest_marker("no_snapshot"):
        return

    if call.excinfo is None:
        return

    name = item.nodeid.replace("/", "_").replace("::", "_")
    cfg = SnapshotConfig()

    try:
        capture_exception(name, call.excinfo.value, call.excinfo.tb, cfg)
    except Exception:
        pass  # Silent fallback
```

**Opt-out marker:**
```python
@pytest.mark.no_snapshot
def test_expected_failure():
    ...
```

## Snapshot JSON Schema

```json
{
  "name": "test_training_step",
  "timestamp_utc": "2026-01-27T14:30:52Z",
  "exception": {
    "type": "ValueError",
    "message": "operands could not be broadcast together..."
  },
  "traceback": "Traceback (most recent call last):\n  ...",
  "frames": [
    {
      "file": "/path/to/training.py",
      "line": 142,
      "function": "train_step",
      "code": "output = model(x) + residual",
      "source": {
        "start": 139,
        "end": 145,
        "snippet": [
          {"lineno": 139, "code": "def train_step(state, batch):"},
          {"lineno": 142, "code": "    output = model(x) + residual"}
        ]
      },
      "locals": {
        "x": {"__array__": "jax.Array", "shape": [32, 64], "dtype": "float32", "head": [0.1, -0.2]},
        "residual": {"__array__": "jax.Array", "shape": [32, 128], "dtype": "float32"},
        "state": "<TrainState>"
      }
    }
  ],
  "env": {
    "python": "3.13.1",
    "platform": "Darwin-25.2.0-arm64",
    "cwd": "/Users/nicolas/project"
  }
}
```

**Key design choices:**
- `frames` ordered innermost-first (crash site at index 0)
- Array summaries use `__array__` prefix for LLM pattern recognition
- Complex objects become `"<ClassName>"`
- Source snippet provides context around crash line

## Failure Resilience

Silent fallback on capture failure:

```python
try:
    capture_exception(name, exc, tb, cfg)
except Exception:
    if cfg.debug:
        sys.stderr.write(f"llmdebug: capture failed\n")
    # Never interfere with original exception
```

The tool should never make debugging harder.

## Claude Code Integration

### CLAUDE.md Snippet

```markdown
## Debug Snapshots (llmdebug)

This project uses `llmdebug` for structured crash diagnostics.

### On any failure:
1. Read `.llmdebug/latest.json`
2. Analyze the snapshot:
   - **Exception type/message** - what went wrong
   - **Closest frame** - where it happened (last frame is crash site)
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

## Packaging

```toml
[project]
name = "llmdebug"
version = "0.1.0"
description = "Structured debug snapshots for LLM-assisted debugging"
requires-python = ">=3.10"
dependencies = []

[project.optional-dependencies]
dev = ["pytest", "numpy", "ruff", "pyright"]

[project.entry-points.pytest11]
llmdebug = "llmdebug.pytest_plugin"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

## Future Enhancements (v1.1+)

- **Manual marks:** `mark("checkpoint", locals={"x": x})` for non-exception debugging
- **Pluggable serializers:** Registry for custom type handlers
- **Claude Code skill:** `/debug` command that auto-reads and analyzes snapshots
- **pytest.ini configuration:** Custom defaults per-project
