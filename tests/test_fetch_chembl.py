"""Unit tests for ChEMBL bioactivity curation logic.

_curate() is a pure transformation (DataFrame → DataFrame) that can be exercised
without any network access.  _fetch_raw() and _merge_with_tdc() touch disk/network
so they are not tested here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import Endpoint
from src.data.fetch_chembl_bioactivity import _curate, _Query


def _ic50_query() -> _Query:
    return _Query(
        target_chembl_id="CHEMBL240",
        standard_type="IC50",
        assay_type="B",
        confidence_score_gte=8,
        unit_multipliers={"nM": 1.0, "uM": 1_000.0},
        threshold=10_000.0,
        positive_if_lt=True,
    )


def test_curate_binarizes_at_threshold() -> None:
    df = pd.DataFrame({
        "smiles": ["CCO", "c1ccccc1"],
        "value":  [1_000.0, 20_000.0],  # 1000 nM < 10000 → Y=1; 20000 nM > 10000 → Y=0
        "units":  ["nM", "nM"],
    })

    result = _curate(df=df, query=_ic50_query(), endpoint=Endpoint.HERG)

    assert len(result) == 2
    assert set(result["Y"].tolist()) == {0.0, 1.0}


def test_curate_normalizes_units() -> None:
    df = pd.DataFrame({
        "smiles": ["CCO"],
        "value":  [5.0],    # 5 uM × 1000 = 5000 nM < 10000 → Y=1
        "units":  ["uM"],
    })

    result = _curate(df=df, query=_ic50_query(), endpoint=Endpoint.HERG)

    assert len(result) == 1
    assert result["Y"].iloc[0] == 1.0


def test_curate_drops_unrecognized_units() -> None:
    df = pd.DataFrame({
        "smiles": ["CCO"],
        "value":  [100.0],
        "units":  ["mg/L"],  # not in unit_multipliers
    })

    result = _curate(df=df, query=_ic50_query(), endpoint=Endpoint.HERG)

    assert result.empty


def test_curate_deduplicates_by_taking_median() -> None:
    # Same canonical SMILES measured twice — median(1000, 3000) = 2000 nM → Y=1
    df = pd.DataFrame({
        "smiles": ["CCO", "CCO"],
        "value":  [1_000.0, 3_000.0],
        "units":  ["nM", "nM"],
    })

    result = _curate(df=df, query=_ic50_query(), endpoint=Endpoint.HERG)

    assert len(result) == 1
    assert result["Y"].iloc[0] == 1.0  # median 2000 nM < 10000 → positive


def test_curate_regression_preserves_values() -> None:
    logd_query = _Query(
        target_chembl_id=None,
        standard_type="LogD",
        assay_type="P",
        confidence_score_gte=None,
    )
    df = pd.DataFrame({
        "smiles": ["CCO", "c1ccccc1"],
        "value":  [0.5, 2.1],
        "units":  ["", ""],
    })

    result = _curate(df=df, query=logd_query, endpoint=Endpoint.LIPOPHILICITY)

    assert len(result) == 2
    assert np.isclose(sorted(result["Y"].tolist()), [0.5, 2.1]).all()
