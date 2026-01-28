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
    cfg = SnapshotConfig(debug=debug)

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
