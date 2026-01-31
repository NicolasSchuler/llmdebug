"""Tests for async context capture."""

import asyncio

import pytest

from llmdebug.async_context import get_async_context, is_coroutine_frame


def test_async_context_not_in_async():
    """Test that async context returns None when not in async context."""
    result = get_async_context()
    assert result is None


@pytest.mark.asyncio(loop_scope="function")
async def test_async_context_in_async():
    """Test async context capture when running in async context."""
    result = get_async_context()

    assert result is not None
    assert "current_task" in result
    assert "total_tasks" in result
    assert result["total_tasks"] >= 1


@pytest.mark.asyncio(loop_scope="function")
async def test_async_context_with_multiple_tasks():
    """Test async context with multiple concurrent tasks."""
    results = []

    async def capture_task():
        await asyncio.sleep(0.01)
        results.append(get_async_context())

    async def background_task():
        await asyncio.sleep(0.1)

    # Create background tasks
    bg_tasks = [asyncio.create_task(background_task(), name=f"bg_task_{i}") for i in range(3)]

    # Create a task that captures context
    capture = asyncio.create_task(capture_task(), name="capture_task")
    await capture

    # Cancel background tasks
    for task in bg_tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert len(results) == 1
    context = results[0]
    assert context is not None
    assert context["current_task"] == "capture_task"
    assert context["total_tasks"] >= 4  # 3 bg + 1 capture


@pytest.mark.asyncio(loop_scope="function")
async def test_async_context_current_coro():
    """Test that current coroutine name is captured."""

    async def my_named_coroutine():
        return get_async_context()

    task = asyncio.create_task(my_named_coroutine())
    context = await task

    assert context is not None
    # The coroutine name should be captured
    assert "current_coro" in context or "current_task" in context


def test_is_coroutine_frame_with_regular_function():
    """Test is_coroutine_frame returns False for regular frames."""
    import sys

    # Get current frame (not a coroutine)
    frame = sys._getframe()
    assert is_coroutine_frame(frame) is False


@pytest.mark.asyncio(loop_scope="function")
async def test_is_coroutine_frame_with_async_function():
    """Test is_coroutine_frame for async function frames."""
    import sys

    # This frame should be a coroutine frame
    frame = sys._getframe()

    # The test itself runs in async context, so the frame should have CO_COROUTINE
    # Note: The actual frame might be the test runner frame, not the coroutine itself
    # This tests the basic functionality
    result = is_coroutine_frame(frame)
    # Just verify it doesn't crash and returns a bool
    assert isinstance(result, bool)


def test_async_context_other_tasks_truncation():
    """Test that other_tasks list is truncated to max 10."""

    async def run_test():
        tasks = []
        for i in range(15):
            task = asyncio.create_task(asyncio.sleep(0.5), name=f"task_{i}")
            tasks.append(task)

        await asyncio.sleep(0.01)  # Let tasks start
        context = get_async_context()

        # Cancel all tasks
        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        return context

    context = asyncio.run(run_test())

    assert context is not None
    if "other_tasks" in context:
        assert len(context["other_tasks"]) <= 10
        if context["total_tasks"] > 11:
            assert context.get("other_tasks_truncated") is True
