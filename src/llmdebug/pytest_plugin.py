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
