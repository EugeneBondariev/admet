"""Shared pytest fixtures. real_data skips when data files are absent."""
from __future__ import annotations

import pytest

from src.paths import RAW


@pytest.fixture
def real_data() -> None:
    if not any(RAW.iterdir()):
        pytest.skip("real data files absent — run make fetch first")
