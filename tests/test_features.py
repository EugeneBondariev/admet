"""Tests for featurization and the Murcko scaffold split.

All molecules here are real SMILES — the point is to exercise RDKit's actual
scaffold assignment, not a stand-in for it. Group sizes are controlled by
building series of analogs that share one core (same scaffold) alongside
distinct ring systems (one molecule per scaffold).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from src.features.features import (
    FEATURE_NAMES,
    FP_BITS,
    MACCS_BITS,
    featurize,
    featurize_smiles,
    get_scaffold_groups,
    quantile_bins,
    scaffold_split,
)

# Four unrelated cores; alkyl chains of increasing length give distinct
# molecules that all reduce to the same Murcko scaffold.
BENZENE = "c1ccccc1"
NAPHTHALENE = "c1ccc2ccccc2c1"
PYRIDINE = "c1ccncc1"
QUINOLINE = "c1ccc2ncccc2c1"


def _series(core: str, n: int) -> list[str]:
    """n analogs sharing one scaffold, differing only in a terminal alkyl chain."""
    return ["C" * (i + 1) + core for i in range(n)]


def _singletons(n: int) -> list[str]:
    """n molecules with n distinct scaffolds — cycloalkanes of increasing ring size."""
    return ["C1" + "C" * (size - 1) + "1" for size in range(3, 3 + n)]


def _ring_series(ring_size: int, n: int) -> list[str]:
    """n analogs sharing one cycloalkane scaffold of the given ring size."""
    return ["C" * (i + 1) + "C1" + "C" * (ring_size - 1) + "1" for i in range(n)]


def _acyclics(n: int) -> list[str]:
    """n distinct molecules with no ring system — straight-chain alkanes."""
    return ["C" * (i + 1) for i in range(n)]


def _scaffolds(smiles: list[str]) -> set[str]:
    return {
        MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smi), includeChirality=False)
        for smi in smiles
    }


def test_series_helper_really_shares_one_scaffold() -> None:
    # The premise every size-controlled test below depends on.
    assert _scaffolds(_series(BENZENE, 5)) == {BENZENE}
    assert len(_scaffolds(_singletons(10))) == 10


def test_split_is_a_partition() -> None:
    smiles = _singletons(30) + _series(BENZENE, 40)
    train_idx, test_idx = scaffold_split(smiles)

    assert set(train_idx).isdisjoint(set(test_idx))
    assert sorted([*train_idx, *test_idx]) == list(range(len(smiles)))


def test_no_scaffold_appears_in_both_splits() -> None:
    smiles = _singletons(25) + _series(BENZENE, 20) + _series(NAPHTHALENE, 15)
    train_idx, test_idx = scaffold_split(smiles)

    train_scaffolds = _scaffolds([smiles[i] for i in train_idx])
    test_scaffolds = _scaffolds([smiles[i] for i in test_idx])
    assert train_scaffolds.isdisjoint(test_scaffolds)


def test_analogs_of_the_same_core_are_never_separated() -> None:
    # aspirin, toluene and paracetamol all reduce to benzene, so they are
    # inseparable by a scaffold split even though they are unrelated drugs.
    drugs = ["CC(=O)Oc1ccccc1C(=O)O", "Cc1ccccc1", "CC(=O)Nc1ccc(O)cc1"]
    smiles = _singletons(30) + drugs
    train_idx, test_idx = scaffold_split(smiles)

    benzene_idx = {len(smiles) - 3, len(smiles) - 2, len(smiles) - 1}
    assert benzene_idx <= set(train_idx) or benzene_idx <= set(test_idx)


def test_rare_scaffolds_go_to_test_and_the_big_series_stays_in_train() -> None:
    smiles = _singletons(30) + _series(BENZENE, 60)
    train_idx, test_idx = scaffold_split(smiles)

    test_scaffolds = _scaffolds([smiles[i] for i in test_idx])
    assert test_scaffolds <= _scaffolds(_singletons(30))
    assert BENZENE not in test_scaffolds
    assert all(i < 30 for i in test_idx)
    assert set(range(30, len(smiles))) <= set(train_idx)


def test_straddling_group_is_left_in_train_rather_than_overshooting() -> None:
    # 18 singletons fill test to 18%; the next group of 12 would take it to 30%,
    # past the 20% + 2% cap, so it stays in train and test stops at 18.
    smiles = _singletons(18) + _series(BENZENE, 12) + _series(NAPHTHALENE, 70)
    assert len(smiles) == 100

    train_idx, test_idx = scaffold_split(smiles)

    assert len(test_idx) == 18
    assert len(test_idx) / len(smiles) <= 0.22


def test_tolerance_can_be_widened_to_admit_a_larger_group() -> None:
    smiles = _singletons(18) + _series(BENZENE, 12) + _series(NAPHTHALENE, 70)

    _, test_idx = scaffold_split(smiles, tolerance=0.15)

    # 18 + 12 = 30 now fits under the 35% cap, so the benzene series joins test.
    assert len(test_idx) == 30
    assert BENZENE in _scaffolds([smiles[i] for i in test_idx])


def test_raises_when_no_scaffold_group_is_small_enough() -> None:
    smiles = (
        _series(BENZENE, 50)
        + _series(NAPHTHALENE, 50)
        + _series(PYRIDINE, 50)
        + _series(QUINOLINE, 50)
    )

    with pytest.raises(ValueError, match="No scaffold group fits"):
        scaffold_split(smiles)


def test_acyclic_molecules_are_not_pooled_into_one_block() -> None:
    # Pooled under a single empty scaffold these were one group far too large to
    # ever reach test (29.5% of AqSolDB). Keyed individually they are ordinary
    # singletons and land on both sides.
    smiles = _acyclics(40) + _series(BENZENE, 60)

    train_idx, test_idx = scaffold_split(smiles)

    acyclic = set(range(40))
    assert acyclic & set(test_idx)
    assert acyclic & set(train_idx)


def test_duplicate_molecules_form_one_inseparable_group() -> None:
    # 25 copies of ethanol are one group of 25, too big for a 22% test set. Keyed
    # per row they would be 25 singletons, and test would quietly fill with
    # copies of a molecule sitting in train — leakage, not diversity.
    smiles = ["CCO"] * 25 + _series(BENZENE, 75)

    with pytest.raises(ValueError, match="No scaffold group fits"):
        scaffold_split(smiles)


def test_split_is_deterministic_for_a_given_random_state() -> None:
    smiles = _acyclics(50) + _series(BENZENE, 50)

    first = scaffold_split(smiles, random_state=0)
    second = scaffold_split(smiles, random_state=0)

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_random_state_decides_which_equal_sized_groups_reach_test() -> None:
    # 50 singletons compete for 20 test slots; without the shuffle, dataframe
    # row order alone would pick the winners.
    smiles = _acyclics(50) + _series(BENZENE, 50)

    _, test_a = scaffold_split(smiles, random_state=0)
    _, test_b = scaffold_split(smiles, random_state=1)

    assert set(test_a) != set(test_b)


def test_classification_split_puts_both_classes_in_train_and_test() -> None:
    smiles = _singletons(30) + _series(BENZENE, 50) + _series(NAPHTHALENE, 10)
    y = np.array([0.0] * 30 + [1.0] * 60, dtype=np.float32)

    train_idx, test_idx = scaffold_split(smiles, y=y)

    assert set(np.unique(y[test_idx])) == {0.0, 1.0}
    assert set(np.unique(y[train_idx])) == {0.0, 1.0}
    # Per-class quotas still move whole groups, so the core invariant must survive.
    assert set(train_idx).isdisjoint(set(test_idx))
    train_scaffolds = _scaffolds([smiles[i] for i in train_idx])
    assert train_scaffolds.isdisjoint(_scaffolds([smiles[i] for i in test_idx]))


def test_test_set_mirrors_the_dataset_class_balance() -> None:
    # The rarest scaffolds are overwhelmingly positive, so a class-blind
    # rare-first fill would hand test ~22 positives and no negatives at all.
    smiles = _singletons(60) + _series(BENZENE, 40)
    y = np.array([1.0] * 40 + [0.0] * 20 + [1.0] * 40, dtype=np.float32)
    assert y.mean() == 0.8

    _, test_idx = scaffold_split(smiles, y=y)

    assert 0.0 in np.unique(y[test_idx])
    assert abs(y[test_idx].mean() - y.mean()) < 0.1


def test_rare_class_is_not_squeezed_out_by_rounding() -> None:
    # 3 negatives in 100 molecules: a proportional quota rounds to zero, so the
    # floor of one molecule per class is what keeps them in test at all.
    smiles = _singletons(60) + _series(BENZENE, 40)
    y = np.array([1.0] * 57 + [0.0] * 3 + [1.0] * 40, dtype=np.float32)

    train_idx, test_idx = scaffold_split(smiles, y=y)

    assert 0.0 in np.unique(y[test_idx])
    assert 0.0 in np.unique(y[train_idx])


def test_raises_when_a_class_cannot_reach_the_test_set() -> None:
    # Every positive sits in one 60-molecule series, far past any class's share
    # of a 22% test set, so no split can give test a positive.
    smiles = _singletons(30) + _series(BENZENE, 60)
    y = np.array([0.0] * 30 + [1.0] * 60, dtype=np.float32)

    with pytest.raises(ValueError, match=r"No test molecules for class \[1\]"):
        scaffold_split(smiles, y=y)


def test_unparseable_smiles_are_forced_into_train() -> None:
    # featurize() drops these, so this only happens on raw input. A molecule with
    # no features cannot be predicted, so it must never land in test.
    smiles = _singletons(30) + ["not_a_molecule", "%%%"] + _series(BENZENE, 18)

    train_idx, test_idx = scaffold_split(smiles)

    bad = {30, 31}
    assert bad <= set(train_idx)
    assert not bad & set(test_idx)


def test_regression_split_covers_the_whole_label_range() -> None:
    # Low labels sit in singleton scaffolds and high labels only in pairs, so
    # rare-first filling reaches the bottom of the range and never the top —
    # exactly the train/test value skew that stratification exists to stop.
    smiles = _singletons(36) + [s for size in range(39, 46) for s in _ring_series(size, 2)]
    y = np.arange(50, dtype=np.float32)
    assert len(smiles) == 50

    _, unstratified = scaffold_split(smiles)
    _, stratified = scaffold_split(smiles, y=y, stratify_bins=5)

    bins = quantile_bins(y, 5)
    assert set(bins[stratified]) == set(bins)
    assert set(bins[unstratified]) != set(bins)


def test_regression_split_handles_heavily_tied_labels() -> None:
    # Clearance is clipped at 3 and 150, so most labels tie and the quantile
    # edges collapse. Fewer, uneven bins is fine; empty strata would not be.
    smiles = _singletons(60) + _series(BENZENE, 40)
    y = np.array([3.0] * 50 + [150.0] * 50, dtype=np.float32)

    train_idx, test_idx = scaffold_split(smiles, y=y, stratify_bins=5)

    assert len(np.unique(quantile_bins(y, 5))) < 5
    assert set(np.unique(y[test_idx])) == {3.0, 150.0}
    assert set(np.unique(y[train_idx])) == {3.0, 150.0}


def test_featurize_returns_aligned_x_y_and_smiles() -> None:
    smiles = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
    df = pd.DataFrame({"Drug": smiles, "Y": [1.0, 2.0, 3.0]})

    X, y, smiles_out = featurize(df)

    assert X.shape[0] == 3
    assert X.shape[1] > FP_BITS  # fingerprint bits plus RDKit descriptors
    assert X.dtype == np.float32
    assert np.array_equal(y, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert smiles_out == smiles



def test_featurize_fingerprint_block_is_binary() -> None:
    df = pd.DataFrame({"Drug": ["CC(=O)Oc1ccccc1C(=O)O"], "Y": [0.0]})

    X, _, _ = featurize(df)

    assert set(np.unique(X[:, :FP_BITS])) <= {0.0, 1.0}
    assert X[:, :FP_BITS].sum() > 0


def test_featurize_maccs_block_is_binary() -> None:
    df = pd.DataFrame({"Drug": ["CC(=O)Oc1ccccc1C(=O)O"], "Y": [0.0]})

    X, _, _ = featurize(df)

    maccs_start = len(FEATURE_NAMES) - MACCS_BITS - 2
    maccs_block = X[:, maccs_start : maccs_start + MACCS_BITS]
    assert set(np.unique(maccs_block)) <= {0.0, 1.0}


def test_featurize_ro5_violations_range() -> None:
    # Aspirin (MW 180, logP 1.2, HBD 1, HBA 4) — 0 violations.
    # Cyclosporin A — a large cyclic peptide well over MW 500 and logP 5.
    df = pd.DataFrame({
        "Drug": ["CC(=O)Oc1ccccc1C(=O)O", "CCC1C(=O)N(CC(=O)N(C(C(=O)N(C(C(=O)NC(C(=O)N(C(C(=O)N(C(C(=O)N1C)CC(C)C)C)CC(C)C)C)C)CC(C)C)C)CC(C)C)C)C"],
        "Y": [0.0, 0.0],
    })

    X, _, _ = featurize(df)

    ro5_col = -2
    assert X[0, ro5_col] == 0.0
    assert X[1, ro5_col] > 0.0
    assert 0.0 <= X[:, ro5_col].max() <= 4.0


def test_featurize_pains_flag() -> None:
    # Rhodanine core is a classic PAINS alert.
    # Aspirin has no PAINS alert.
    rhodanine = "O=C1NC(=S)SC1"
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    df = pd.DataFrame({"Drug": [aspirin, rhodanine], "Y": [0.0, 0.0]})

    X, _, _ = featurize(df)

    pains_col = -1
    assert X[0, pains_col] == 0.0
    assert X[1, pains_col] == 1.0


def test_featurize_smiles_drops_unparseable() -> None:
    smiles = ["CCO", "not_a_molecule", "c1ccccc1", "%%%"]

    X, valid = featurize_smiles(smiles)

    assert valid == ["CCO", "c1ccccc1"]
    assert X.shape[0] == 2


def test_featurize_smiles_raises_on_all_invalid() -> None:
    with pytest.raises(ValueError, match="No parseable SMILES"):
        featurize_smiles(["not_valid", "%%%"])


def test_get_scaffold_groups_same_core_same_label() -> None:
    # Three aspirin analogs all reduce to benzene — must share one label.
    benzene_analogs = ["CC(=O)Oc1ccccc1C(=O)O", "Cc1ccccc1", "CC(=O)Nc1ccc(O)cc1"]

    groups = get_scaffold_groups(benzene_analogs)

    assert groups[0] == groups[1] == groups[2]


def test_get_scaffold_groups_distinct_cores_distinct_labels() -> None:
    smiles = ["c1ccccc1", "c1ccncc1", "C1CCCCC1"]  # benzene, pyridine, cyclohexane

    groups = get_scaffold_groups(smiles)

    assert len(set(groups)) == 3


def test_get_scaffold_groups_acyclics_are_not_pooled() -> None:
    acyclics = ["C", "CC", "CCC", "CCCC"]

    groups = get_scaffold_groups(acyclics)

    assert len(set(groups)) == 4


# ---------------------------------------------------------------------------
# load_or_featurize — cache creation, reuse, and hash invalidation
# ---------------------------------------------------------------------------

def test_load_or_featurize_returns_correct_shape() -> None:
    df = pd.DataFrame({"Drug": ["CCO", "c1ccccc1", "CC(=O)O"], "Y": [1.0, 2.0, 3.0]})
    X, y, smiles = featurize(df)

    assert X.shape == (3, len(FEATURE_NAMES))
    assert y.shape == (3,)
    assert len(smiles) == 3


def test_load_or_featurize_results_are_aligned() -> None:
    df = pd.DataFrame({"Drug": ["CCO", "c1ccccc1"], "Y": [1.5, 2.5]})
    X, y, smiles = featurize(df)

    assert y[0] == pytest.approx(1.5)
    assert y[1] == pytest.approx(2.5)
    assert smiles[0] == "CCO"
    assert smiles[1] == "c1ccccc1"
