"""A hung isolated test fails instead of holding the run."""

from __future__ import annotations

import time

import pytest


@pytest.mark.xfail(strict=True, reason="the timeout fails it")
@pytest.mark.isolated(timeout=2)
def test_hang_is_killed_and_fails() -> None:
    time.sleep(600)


@pytest.mark.isolated(timeout=60)
def test_timeout_leaves_a_quick_test_alone() -> None:
    assert True
