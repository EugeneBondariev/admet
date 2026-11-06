"""Unit tests for 3D graph featurization and graph-build caching.

Skipped automatically on the system Python where torch is not installed.
Run with the venv Python: .venv/Scripts/python.exe -m pytest tests/test_mol_graph.py
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("torch_geometric")

import torch  # noqa: E402

from src.features.mol_graph import load_or_build_dataset, smiles_to_graph  # noqa: E402

_SMILES = ["CC", "CCC", "CCCC", "CCO", "c1ccccc1"]
_DF = pd.DataFrame({"Drug": _SMILES, "Y": [float(i) for i in range(len(_SMILES))]})


# ---------------------------------------------------------------------------
# smiles_to_graph
# ---------------------------------------------------------------------------

def test_smiles_to_graph_returns_data_for_valid_smiles() -> None:
    g = smiles_to_graph(smiles="CCO", y=1.5)

    assert g is not None
    assert g.z.shape[0] == g.pos.shape[0]
    assert g.pos.shape[1] == 3
    assert g.y.item() == pytest.approx(1.5)


def test_smiles_to_graph_atomic_numbers_are_positive() -> None:
    g = smiles_to_graph(smiles="c1ccccc1", y=0.0)

    assert g is not None
    assert (g.z > 0).all()


def test_smiles_to_graph_returns_none_on_invalid_smiles() -> None:
    assert smiles_to_graph(smiles="notamolecule", y=0.0) is None


def test_smiles_to_graph_returns_none_on_empty_string() -> None:
    assert smiles_to_graph(smiles="", y=0.0) is None


# ---------------------------------------------------------------------------
# load_or_build_dataset — cache creation and reuse
# ---------------------------------------------------------------------------

def test_load_or_build_creates_cache_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.features.mol_graph.PROC_GRAPHS", tmp_path)

    load_or_build_dataset(endpoint="test", df=_DF)

    assert (tmp_path / "test_graphs.pt").exists()


def test_load_or_build_returns_one_graph_per_valid_smiles(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.features.mol_graph.PROC_GRAPHS", tmp_path)
    df = pd.DataFrame({"Drug": ["CCO", "notvalid", "c1ccccc1"], "Y": [1.0, 2.0, 3.0]})

    graphs, smiles = load_or_build_dataset(endpoint="test", df=df)

    assert len(graphs) == 2
    assert smiles == ["CCO", "c1ccccc1"]


def test_load_or_build_loads_from_cache_on_second_call(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.features.mol_graph.PROC_GRAPHS", tmp_path)
    monkeypatch.setattr("src.features.mol_graph.INTERIM", tmp_path)
    _DF.to_parquet(tmp_path / "test.parquet")

    graphs1, smiles1 = load_or_build_dataset(endpoint="test", df=_DF)
    graphs2, smiles2 = load_or_build_dataset(endpoint="test", df=_DF)

    assert smiles1 == smiles2
    assert len(graphs1) == len(graphs2)


def test_load_or_build_y_values_match_input(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.features.mol_graph.PROC_GRAPHS", tmp_path)
    df = pd.DataFrame({"Drug": ["CCO", "c1ccccc1"], "Y": [0.5, 2.5]})

    graphs, _ = load_or_build_dataset(endpoint="test", df=df)

    assert graphs[0].y.item() == pytest.approx(0.5)
    assert graphs[1].y.item() == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# load_or_build_dataset — checkpoint resume
# ---------------------------------------------------------------------------

def test_load_or_build_resumes_from_checkpoint(tmp_path, monkeypatch) -> None:
    """Pre-create a checkpoint with 2 molecules done; verify the remaining 3 are added."""
    monkeypatch.setattr("src.features.mol_graph.PROC_GRAPHS", tmp_path)

    checkpoint_smiles = _SMILES[:2]
    checkpoint_graphs = [
        g for s, y in zip(checkpoint_smiles, range(2))
        if (g := smiles_to_graph(s, float(y))) is not None
    ]
    torch.save(
        {"graphs": checkpoint_graphs, "smiles": checkpoint_smiles, "processed_count": 2},
        tmp_path / "test_graphs_ckpt.pt",
    )

    graphs, smiles_out = load_or_build_dataset(endpoint="test", df=_DF)

    # All 5 molecules should be present (they're all valid SMILES).
    assert len(graphs) == 5
    # The first two entries come from the checkpoint — ordering preserved.
    assert smiles_out[:2] == checkpoint_smiles
    # Checkpoint file is cleaned up once the full cache is written.
    assert not (tmp_path / "test_graphs_ckpt.pt").exists()
    assert (tmp_path / "test_graphs.pt").exists()


def test_load_or_build_checkpoint_cleaned_up_after_full_build(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.features.mol_graph.PROC_GRAPHS", tmp_path)

    # Plant a checkpoint so the code takes the resume path.
    g = smiles_to_graph("CC", 0.0)
    torch.save({"graphs": [g], "smiles": ["CC"], "processed_count": 1}, tmp_path / "test_graphs_ckpt.pt")

    load_or_build_dataset(endpoint="test", df=_DF)

    assert not (tmp_path / "test_graphs_ckpt.pt").exists()
