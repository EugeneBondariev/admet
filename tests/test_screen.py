"""Unit tests for virtual screening thresholds.

_passes() is a pure function mapping (endpoint, predicted_value) → bool.
Tests cover the pass/fail boundary for every endpoint and verify that
the boundary is strict (not ≤) where the threshold docstring says so.
"""
from __future__ import annotations

import pytest

from src.constants import Endpoint
from src.screening.screen import _passes


@pytest.mark.parametrize("endpoint,pred,expected", [
    # SOLUBILITY: logS > -5.0
    (Endpoint.SOLUBILITY, -4.0,  True),
    (Endpoint.SOLUBILITY, -6.0,  False),
    (Endpoint.SOLUBILITY, -5.0,  False),   # boundary: not strictly >

    # BBB: p(penetrant) < 0.50  (non-CNS drugs want to stay out of brain)
    (Endpoint.BBB,        0.4,   True),
    (Endpoint.BBB,        0.6,   False),
    (Endpoint.BBB,        0.5,   False),   # boundary: not strictly <

    # LIPOPHILICITY: -1.0 ≤ logD ≤ 5.0
    (Endpoint.LIPOPHILICITY,  2.0, True),
    (Endpoint.LIPOPHILICITY, -2.0, False),  # below lower bound
    (Endpoint.LIPOPHILICITY,  6.0, False),  # above upper bound
    (Endpoint.LIPOPHILICITY, -1.0, True),   # exactly at lower bound (≤ includes it)
    (Endpoint.LIPOPHILICITY,  5.0, True),   # exactly at upper bound (≤ includes it)

    # PPBR: < 90%
    (Endpoint.PPBR, 89.0, True),
    (Endpoint.PPBR, 91.0, False),
    (Endpoint.PPBR, 90.0, False),          # boundary: not strictly <

    # VDSS: < 20 L/kg
    (Endpoint.VDSS, 15.0,  True),
    (Endpoint.VDSS, 25.0,  False),
    (Endpoint.VDSS, 20.0,  False),         # boundary: not strictly <

    # CYP3A4: p(inhibitor) < 0.50
    (Endpoint.CYP3A4, 0.4, True),
    (Endpoint.CYP3A4, 0.6, False),
    (Endpoint.CYP3A4, 0.5, False),

    # HLM: p(stable) > 0.50  (Y=1 = stable = good)
    (Endpoint.HLM, 0.6, True),
    (Endpoint.HLM, 0.4, False),
    (Endpoint.HLM, 0.5, False),            # boundary: not strictly >

    # CLEARANCE: < 150 mL/min/kg
    (Endpoint.CLEARANCE, 100.0, True),
    (Endpoint.CLEARANCE, 200.0, False),
    (Endpoint.CLEARANCE, 150.0, False),    # boundary: not strictly <

    # HERG: p(blocker) < 0.30  (strict: cardiac safety)
    (Endpoint.HERG, 0.2,  True),
    (Endpoint.HERG, 0.4,  False),
    (Endpoint.HERG, 0.3,  False),          # boundary: not strictly <

    # LD50: log(LD50) > -2.0  (higher = less toxic)
    (Endpoint.LD50, -1.0, True),
    (Endpoint.LD50, -3.0, False),
    (Endpoint.LD50, -2.0, False),          # boundary: not strictly >

    # DILI: p(liver injury) < 0.30  (strict: leading withdrawal reason)
    (Endpoint.DILI, 0.2,  True),
    (Endpoint.DILI, 0.4,  False),
    (Endpoint.DILI, 0.3,  False),          # boundary: not strictly <

    # CARCINOGENS: p(carcinogen) < 0.20  (strict: regulatory hard stop)
    (Endpoint.CARCINOGENS, 0.1,  True),
    (Endpoint.CARCINOGENS, 0.3,  False),
    (Endpoint.CARCINOGENS, 0.2,  False),   # boundary: not strictly <
])
def test_passes_thresholds(endpoint: Endpoint, pred: float, expected: bool) -> None:
    assert _passes(endpoint, pred) is expected
