"""Tests for git context capture."""

import os
import subprocess

from llmdebug.git_context import get_git_context


def test_git_context_in_repo():
    """Test git context capture when in a git repository."""
    # This test runs from the llmdebug repo itself
    context = get_git_context()

    # Should have context since we're in a git repo
    assert context is not None
    assert "commit" in context
    assert "commit_full" in context
    assert "branch" in context
    assert "dirty" in context

    # Commit should be a short hash (7+ chars)
    assert len(context["commit"]) >= 7
    # Full commit should be 40 chars
    assert len(context["commit_full"]) == 40


def test_git_context_not_in_repo(tmp_path, monkeypatch):
    """Test git context capture when not in a git repository."""
    # Change to a non-git directory
    monkeypatch.chdir(tmp_path)

    context = get_git_context()

    # Should return None when not in a git repo
    assert context is None


def test_git_context_dirty_status(tmp_path):
    """Test that dirty status is correctly detected."""
    # Initialize a new git repo
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )

    # Create and commit a file
    test_file = tmp_path / "test.txt"
    test_file.write_text("content")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )

    # Save current dir and change to temp repo
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        # Should be clean
        context = get_git_context()
        assert context is not None
        assert context["dirty"] is False

        # Make it dirty
        test_file.write_text("modified")
        context = get_git_context()
        assert context is not None
        assert context["dirty"] is True
    finally:
        os.chdir(old_cwd)


def test_git_context_detached_head(tmp_path):
    """Test git context when HEAD is detached."""
    # Initialize a new git repo
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )

    # Create and commit a file
    test_file = tmp_path / "test.txt"
    test_file.write_text("content")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )

    # Detach HEAD
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    commit_hash = result.stdout.strip()
    subprocess.run(
        ["git", "checkout", commit_hash],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )

    # Save current dir and change to temp repo
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        context = get_git_context()
        assert context is not None
        assert context.get("detached") is True
    finally:
        os.chdir(old_cwd)
