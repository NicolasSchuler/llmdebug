"""Pytest plugin for automatic snapshot capture on test failures."""

from __future__ import annotations

import os
import sys

import pytest

from .capture import capture_exception
from .config import SnapshotConfig
from .serialize import safe_repr


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

    debug = os.getenv("LLMDEBUG_DEBUG", "").lower() in {"1", "true", "yes", "on"}

    # Parse include_modules from comma-separated env var
    include_modules_env = os.getenv("LLMDEBUG_INCLUDE_MODULES", "")
    include_modules = (
        tuple(m.strip() for m in include_modules_env.split(",") if m.strip())
        if include_modules_env
        else None
    )

    # Parse max_exception_depth from env var
    try:
        max_exception_depth = int(os.getenv("LLMDEBUG_MAX_EXCEPTION_DEPTH", "5"))
    except ValueError:
        max_exception_depth = 5

    # Parse new feature flags from env vars
    include_git = os.getenv("LLMDEBUG_INCLUDE_GIT", "true").lower() in {"1", "true", "yes", "on"}
    include_array_stats = os.getenv("LLMDEBUG_ARRAY_STATS", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    include_args = os.getenv("LLMDEBUG_INCLUDE_ARGS", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    categorize_errors = os.getenv("LLMDEBUG_CATEGORIZE_ERRORS", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    include_async_context = os.getenv("LLMDEBUG_ASYNC_CONTEXT", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    capture_logs = os.getenv("LLMDEBUG_CAPTURE_LOGS", "").lower() in {"1", "true", "yes", "on"}
    try:
        log_max_records = int(os.getenv("LLMDEBUG_LOG_MAX_RECORDS", "20"))
    except ValueError:
        log_max_records = 20

    # Output format: json, json_compact (default), or toon
    output_format = os.getenv("LLMDEBUG_OUTPUT_FORMAT", "json_compact")
    if output_format not in {"json", "json_compact", "toon"}:
        output_format = "json_compact"

    cfg = SnapshotConfig(
        debug=debug,
        include_modules=include_modules,
        max_exception_depth=max_exception_depth,
        include_git=include_git,
        include_array_stats=include_array_stats,
        include_args=include_args,
        categorize_errors=categorize_errors,
        include_async_context=include_async_context,
        capture_logs=capture_logs,
        log_max_records=log_max_records,
        output_format=output_format,
    )

    # Pytest-specific context to reduce guesswork
    params = None
    callspec = getattr(item, "callspec", None)
    if callspec is not None:
        try:
            params = {k: safe_repr(v, cfg) for k, v in callspec.params.items()}
        except Exception:
            params = None

    pytest_context = {
        "nodeid": report.nodeid,
        "outcome": report.outcome,
        "when": report.when,
        "longrepr": getattr(report, "longreprtext", None),
        "capstdout": getattr(report, "capstdout", None),
        "capstderr": getattr(report, "capstderr", None),
        "params": params,
        "repro": [sys.executable, "-m", "pytest", report.nodeid, "-q"],
    }
    try:
        capture_exception(name, exc, tb, cfg, extra={"pytest": pytest_context})
    except Exception:
        if cfg.debug:
            sys.stderr.write(f"llmdebug: capture failed for {name}\n")
