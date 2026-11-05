"""Molecular featurization: ECFP4 Morgan fingerprints + RDKit physicochemical descriptors.

Features: 2048-bit Morgan fingerprint (radius=2) concatenated with ~210 RDKit descriptors.
Rows with unparseable SMILES are dropped and logged.
"""
from __future__ import annotations

import logging
import warnings
from collections import Counter
from math import ceil

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.warning")

logger = logging.getLogger(__name__)

FP_RADIUS = 2
FP_BITS = 2048

_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
_DESC_FUNCS = Descriptors.descList  # list of (name, callable) pairs
FEATURE_NAMES: list[str] = (
    [f"fp_{i}" for i in range(FP_BITS)] + [name for name, _ in _DESC_FUNCS]
)


def _mol_to_features(mol: Chem.Mol) -> np.ndarray:
    fp = _morgan_gen.GetFingerprintAsNumPy(mol).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        desc = np.array([fn(mol) for _, fn in _DESC_FUNCS], dtype=np.float32)
    return np.concatenate([fp, desc])


def featurize(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return X, y, smiles for valid rows. Drops rows with unparseable SMILES."""
    rows, labels, smiles_out = [], [], []
    n_dropped = 0
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row["Drug"]))
        if mol is None:
            n_dropped += 1
            continue
        rows.append(_mol_to_features(mol))
        labels.append(float(row["Y"]))
        smiles_out.append(str(row["Drug"]))

    if n_dropped:
        logger.warning("Dropped %d rows with unparseable SMILES", n_dropped)
    logger.info("Featurized %d molecules (%d features)", len(rows), len(FEATURE_NAMES))

    return np.vstack(rows), np.array(labels, dtype=np.float32), smiles_out


def scaffold_split(
    smiles: list[str],
    test_size: float = 0.2,
    y: np.ndarray | None = None,
    tolerance: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Split by Murcko scaffold — no scaffold appears in both train and test.

    Assigns rarest scaffolds (smallest groups) to test first so the test set
    covers diverse chemical space rather than a single dominant series.
    Scaffolds move as whole groups, so the test set can only land on exact
    test_size by luck; a group is skipped (left in train) if adding it would
    push test past test_size + tolerance, keeping the realized split close
    enough to compare metrics across endpoints and against TDC.
    If y is provided, each class gets its own share of the test set in proportion
    to the full dataset, so a classification test set mirrors the dataset's class
    balance instead of whatever the rarest scaffolds happen to carry. Without this,
    an imbalanced endpoint like hERG fills test almost entirely with one class and
    its ROC-AUC / AUPRC become uninterpretable. Raises ValueError if a class cannot
    reach the test set because every scaffold group carrying it is too large.
    """
    scaffold_to_indices: dict[str, list[int]] = {}
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        scaffold = (
            MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            if mol is not None
            else ""
        )
        scaffold_to_indices.setdefault(scaffold, []).append(i)

    groups = sorted(scaffold_to_indices.values(), key=len)
    n_test_target = int(len(smiles) * test_size)
    n_test_cap = int(len(smiles) * (test_size + tolerance))

    # Per-class size quotas. With y=None this collapses to one pseudo-class and
    # the quotas reduce to the plain size target/cap, so regression is unchanged.
    labels = np.zeros(len(smiles), dtype=int) if y is None else y.astype(int)
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    shares = {int(c): int(n) / len(smiles) for c, n in zip(unique_labels, label_counts)}

    # Floor of 1 so a class too rare to earn a proportional slot still reaches test.
    class_target = {cls: max(1, round(n_test_target * share)) for cls, share in shares.items()}
    # Each cap leaves room for every other class's target, so the majority class
    # cannot consume the whole test budget before a rare class gets in.
    class_cap: dict[int, int] = {}
    for cls, share in shares.items():
        reserved = sum(t for c, t in class_target.items() if c != cls)
        class_cap[cls] = max(
            class_target[cls], min(ceil(n_test_cap * share), n_test_cap - reserved)
        )

    test_idx: list[int] = []
    train_idx: list[int] = []
    skipped: list[int] = []
    test_counts: Counter[int] = Counter()

    for group in groups:
        group_counts = Counter(int(labels[i]) for i in group)
        if all(test_counts[c] >= target for c, target in class_target.items()):
            train_idx.extend(group)
        elif len(test_idx) + len(group) <= n_test_cap and all(
            test_counts[c] + n <= class_cap[c] for c, n in group_counts.items()
        ):
            test_idx.extend(group)
            test_counts.update(group_counts)
        else:
            skipped.append(len(group))
            train_idx.extend(group)

    if not test_idx:
        raise ValueError(
            f"No scaffold group fits in a {100 * (test_size + tolerance):.0f}% test set "
            f"(smallest group has {min(len(g) for g in groups)} of {len(smiles)} molecules) — "
            f"raise tolerance or test_size"
        )

    if skipped:
        logger.info(
            "Left %d scaffold group(s) in train (largest %d molecules) — would have pushed "
            "test, or one class's share of it, past %.0f%%",
            len(skipped),
            max(skipped),
            100 * (test_size + tolerance),
        )

    if y is not None:
        missing = sorted(cls for cls in class_target if test_counts[cls] == 0)
        if missing:
            raise ValueError(
                f"Class(es) {missing} absent from the test set — every scaffold group "
                f"carrying them is larger than that class's share of a "
                f"{100 * (test_size + tolerance):.0f}% test set; raise tolerance or test_size"
            )
        logger.info(
            "Test class balance: %s (dataset: %s)",
            dict(sorted(test_counts.items())),
            dict(sorted(Counter(int(c) for c in labels).items())),
        )

    index_to_scaffold = {i: s for s, indices in scaffold_to_indices.items() for i in indices}
    realized = len(test_idx) / len(smiles)
    logger.info(
        "Scaffold split: %d train / %d test (%.1f%% test) — %d train scaffolds / %d test scaffolds",
        len(train_idx),
        len(test_idx),
        100 * realized,
        len({index_to_scaffold[i] for i in train_idx}),
        len({index_to_scaffold[i] for i in test_idx}),
    )
    if abs(realized - test_size) > tolerance:
        logger.warning(
            "Realized test fraction %.1f%% is outside %.0f%% ± %.0f%% — scaffold group sizes "
            "prevent a closer split; metrics are not directly comparable to other endpoints",
            100 * realized,
            100 * test_size,
            100 * tolerance,
        )
    return np.array(train_idx), np.array(test_idx)
