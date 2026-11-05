"""Shared pytest fixtures. real_data skips when data files are absent."""
from __future__ import annotations

import pytest

from src.paths import INTERIM

_NO_TESTS_COLLECTED = 5


@pytest.fixture
def real_data() -> None:
    if not any(INTERIM.iterdir()):
        pytest.skip("real data files absent — run make fetch first")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Treat "no tests collected" as success, not failure.

    Lives here rather than in the Makefile: the old `|| [ $$? -eq 5 ]` guard was
    POSIX sh, and make invokes cmd.exe on Windows, where `[` is not a command.
    Real test failures still fail.
    """
    if exitstatus == _NO_TESTS_COLLECTED:
        session.exitstatus = 0
