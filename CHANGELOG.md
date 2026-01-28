# CHANGELOG

## v0.1.4 (2026-01-27)

### Features

- ML tensor introspection: NaN/Inf detection, device tracking, requires_grad
- Pytest-specific context capture: test nodeid, outcome, stdout/stderr
- Project logo

### Bug Fixes

- Deprecated datetime.utcnow() usage
- Handle None line numbers in frame extraction

## v0.1.0 (2026-01-27)

### Features

- Initial release
- Exception capture with structured JSON snapshots
- Pytest plugin for automatic capture on test failures
- Smart array summarization for NumPy arrays
- Atomic file writes for snapshot output
