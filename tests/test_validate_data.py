"""Unit tests for data validation helpers."""
from __future__ import annotations

from src.data.validate_data import is_valid_smiles


def test_is_valid_smiles_accepts_common_molecules() -> None:
    assert is_valid_smiles("CCO")                              # ethanol
    assert is_valid_smiles("c1ccccc1")                        # benzene
    assert is_valid_smiles("CC(=O)Oc1ccccc1C(=O)O")          # aspirin
    assert is_valid_smiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O")     # ibuprofen


def test_is_valid_smiles_rejects_garbage() -> None:
    assert not is_valid_smiles("not_a_molecule")
    assert not is_valid_smiles("%%%")
    assert not is_valid_smiles("ZZZZZZ")
    assert not is_valid_smiles("C(C)(C)(C)(C)(C)")  # carbon with 6 bonds — invalid valence
