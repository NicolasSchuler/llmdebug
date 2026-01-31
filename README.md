<p align="center">
  <img src="logo/bird.png" alt="llmdebug logo" width="200">
</p>

# llmdebug

Structured debug snapshots for LLM-assisted debugging.

When your code fails, `llmdebug` captures the exception, stack frames, local variables, and environment info in a JSON format optimized for LLM consumption. This enables **evidence-based debugging** instead of the "guess → patch → rerun" loop.

## Why?

Without observability, LLMs debug by guessing:
```
fail → guess patch → rerun → repeat (LLM roulette)
```

With `llmdebug`, failures produce rich snapshots automatically:
```
fail → read snapshot → ranked hypotheses → minimal patch → verify
```

The key insight: **baseline instrumentation should always be on**, so the first failure already has the evidence needed to diagnose it.

## Installation

```bash
pip install llmdebug
```

## Quick Start

### Pytest (automatic - recommended)

Just install the package. Test failures automatically generate snapshots.

```bash
pytest  # Failures create .llmdebug/latest.json
```

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

For targeted instrumentation when you need more detail:

```python
from llmdebug import snapshot_section

with snapshot_section("data_processing"):
    result = transform(data)
```

## Output

On failure, find your snapshot at `.llmdebug/latest.json`:

```json
{
  "name": "test_training_step",
  "timestamp_utc": "2026-01-27T14:30:52Z",
  "exception": {
    "type": "ValueError",
    "message": "operands could not be broadcast together..."
  },
  "frames": [
    {
      "file": "training.py",
      "line": 42,
      "function": "train_step",
      "code": "output = model(x) + residual",
      "locals": {
        "x": {"__array__": "jax.Array", "shape": [32, 64], "dtype": "float32"},
        "residual": {"__array__": "jax.Array", "shape": [32, 128], "dtype": "float32"}
      }
    }
  ],
  "env": {"python": "3.12.0", "platform": "Darwin-24.0.0-arm64"}
}
```

**Key features:**
- Crash frame is at index 0 (most relevant first)
- Arrays summarized with `shape` and `dtype` (not raw data)
- Source snippet around the failing line
- Environment info for reproducibility

### Snapshot metadata (new)

The snapshot includes extra fields to reduce LLM guesswork:
- `schema_version` and `llmdebug_version` for compatibility
- `crash_frame_index` to mark the exact crash frame in `frames`
- `capture_config` (frames, locals_mode, truncation limits, redaction patterns)
- `exception` may include `qualified_type`, `args`, `notes`, `cause`, `context`, and `exceptions` (ExceptionGroup)
- Each frame may include `module`, `file_rel`, and `locals_meta` (type/size hints)
- Frames may include `locals_truncated` and `locals_truncated_keys` when locals are omitted
- Pytest runs add `pytest` context (`longrepr`, `capstdout`, `capstderr`, params, `repro`)
- `env.argv` records the command-line invocation

## For Claude Code / LLM Users

Add this to your project's `CLAUDE.md`:

```markdown
## Debug Snapshots (llmdebug)

This project uses `llmdebug` for structured crash diagnostics.

### On any failure:
1. **Read `.llmdebug/latest.json` first** - never patch before reading
2. Analyze the snapshot:
   - **Exception type/message** - what went wrong
   - **Crash frame (index 0)** - where it happened
   - **Locals** - variable values at crash time
   - **Array shapes** - look for empty arrays, shape mismatches
3. **Produce 2-4 ranked hypotheses** based on evidence
4. Apply minimal fix for the most likely hypothesis
5. Re-run to verify

### Key signals:
- `shape: [0, ...]` - empty array, upstream data issue
- `None` where object expected - initialization bug
- Shape mismatch in binary op - broadcasting error
- `i=10` with `len(arr)=10` - off-by-one

### When the snapshot isn't enough:
If locals show the symptom but not the cause:
1. Add `with snapshot_section("stage_x")` around suspect code
2. Re-run to get a better snapshot
3. Repeat hypothesis→patch loop

### Don't:
- Guess without reading the snapshot first
- Make multiple speculative changes at once
- Refactor until tests pass
```

## Configuration

```python
@debug_snapshot(
    out_dir=".llmdebug",       # Output directory
    frames=5,                   # Stack frames to capture
    source_context=3,           # Lines of source before/after crash
    source_mode="all",          # "all" | "crash_only" | "none"
    locals_mode="safe",         # "safe" | "meta" | "none"
    max_str=500,                # Truncate long strings
    max_items=50,               # Truncate large collections
    redact=[r"api_key=.*"],     # Regex patterns to redact
    include_env=True,           # Include Python/platform info
    max_snapshots=50,           # Auto-cleanup old snapshots (0 = unlimited)
    output_format="json_compact", # "json" | "json_compact" | "toon"
)
```

### Output Formats

llmdebug supports multiple output formats to optimize for different use cases:

| Format | Size | Best For |
|--------|------|----------|
| `json` | baseline | Human readability, external tools |
| `json_compact` (default) | ~40% smaller | LLM context efficiency |
| `toon` | ~50% smaller | Maximum token savings |

**Compact JSON** uses abbreviated keys (e.g., `_exc` instead of `exception`) to reduce token usage. The `get_latest_snapshot()` function auto-expands keys, so your code works identically regardless of format.

**TOON format** requires an optional dependency:
```bash
pip install llmdebug[toon]
```

Set format via environment variable for pytest:
```bash
LLMDEBUG_OUTPUT_FORMAT=json pytest  # Use pretty JSON
```

### Pytest Opt-out

Skip snapshot capture for specific tests:

```python
import pytest

@pytest.mark.no_snapshot
def test_expected_failure():
    ...
```

## API

```python
from llmdebug import debug_snapshot, snapshot_section, get_latest_snapshot

# Read the most recent snapshot programmatically
snapshot = get_latest_snapshot()  # Returns dict or None
```

## License

MIT
