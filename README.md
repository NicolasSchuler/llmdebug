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
