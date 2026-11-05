"""Unit tests for 3D graph featurization.

Skipped automatically on the system Python where torch is not installed.
Run with the venv Python: .venv/Scripts/python.exe -m pytest tests/test_mol_graph.py
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("torch_geometric")

from src.features.mol_graph import build_dataset, smiles_to_graph  # noqa: E402


def test_smiles_to_graph_returns_data_for_valid_smiles() -> None:
    g = smiles_to_graph(smiles="CCO", y=1.5)

    assert g is not None
    assert g.z.shape[0] == g.pos.shape[0]  # one atomic number per atom
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


def test_build_dataset_excludes_failed_molecules() -> None:
    df = pd.DataFrame({"Drug": ["CCO", "notamolecule", "c1ccccc1"], "Y": [1.0, 2.0, 3.0]})

    graphs, smiles = build_dataset(df)

    assert len(graphs) == 2
    assert smiles == ["CCO", "c1ccccc1"]


def test_build_dataset_y_values_match_input() -> None:
    df = pd.DataFrame({"Drug": ["CCO", "c1ccccc1"], "Y": [0.5, 2.5]})

    graphs, _ = build_dataset(df)

    assert graphs[0].y.item() == pytest.approx(0.5)
    assert graphs[1].y.item() == pytest.approx(2.5)


def test_build_dataset_smiles_aligned_to_graphs() -> None:
    df = pd.DataFrame({"Drug": ["CCO", "c1ccccc1", "CC(=O)O"], "Y": [1.0, 2.0, 3.0]})

    graphs, smiles = build_dataset(df)

    assert len(graphs) == len(smiles)
    # Verify y alignment: each graph's y matches the SMILES' original row Y
    y_map = dict(zip(df["Drug"], df["Y"]))
    for g, smi in zip(graphs, smiles):
        assert g.y.item() == pytest.approx(y_map[smi])
