"""Hypothesis keeps its files off NFS, in every kind of test process."""

from __future__ import annotations

import os
from pathlib import Path

from hypothesis.configuration import storage_directory
import pytest

from tests.conftest import HYPOTHESIS_STORAGE_NAMED


def _check_storage() -> None:
    path = storage_directory("x", intent_to_write=False).path
    assert path.parent == Path(os.environ["HYPOTHESIS_STORAGE_DIRECTORY"])
    assert Path(__file__).resolve().parent.parent not in path.resolve().parents


def test_storage_is_the_configured_directory() -> None:
    _check_storage()
    if not HYPOTHESIS_STORAGE_NAMED:
        assert os.environ["HYPOTHESIS_STORAGE_DIRECTORY"].startswith("/dev/shm/")


@pytest.mark.isolated
def test_storage_is_the_configured_directory_in_an_isolated_child() -> None:
    _check_storage()
