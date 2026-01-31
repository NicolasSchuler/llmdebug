"""Git repository context capture for debug snapshots."""

from __future__ import annotations

import subprocess
from typing import Any


def _run_git_command(args: list[str], timeout: float = 2.0) -> str | None:
    """Run a git command with timeout, return stdout or None on failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def get_git_context() -> dict[str, Any] | None:
    """Capture git repository context.

    Returns:
        Dict with commit, branch, dirty status, or None if not in a git repo.

    Example output:
        {
            "commit": "abc123",
            "commit_full": "abc123def456...",
            "branch": "main",
            "dirty": True
        }
    """
    # Check if we're in a git repo
    if _run_git_command(["rev-parse", "--git-dir"]) is None:
        return None

    context: dict[str, Any] = {}

    # Get commit hash (short and full)
    commit_short = _run_git_command(["rev-parse", "--short", "HEAD"])
    if commit_short:
        context["commit"] = commit_short

    commit_full = _run_git_command(["rev-parse", "HEAD"])
    if commit_full:
        context["commit_full"] = commit_full

    # Get branch name
    branch = _run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])
    if branch:
        context["branch"] = branch
        # Check for detached HEAD
        if branch == "HEAD":
            context["detached"] = True

    # Check if working directory is dirty
    status = _run_git_command(["status", "--porcelain"])
    if status is not None:
        context["dirty"] = len(status) > 0

    return context if context else None
