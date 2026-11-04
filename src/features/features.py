"""Molecular featurization: ECFP4 Morgan fingerprints + RDKit physicochemical descriptors.

Features: 2048-bit Morgan fingerprint (radius=2) concatenated with ~210 RDKit descriptors.
Rows with unparseable SMILES are dropped and logged.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from src.paths import PROCESSED, RAW

RDLogger.DisableLog("rdApp.warning")

logger = logging.getLogger(__name__)

FP_RADIUS = 2
FP_BITS = 2048

_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
_DESC_FUNCS = Descriptors.descList  # list of (name, callable) pairs
FEATURE_NAMES: list[str] = (
    [f"fp_{i}" for i in range(FP_BITS)] + [name for name, _ in _DESC_FUNCS]
)


def load_or_featurize(
    endpoint: str, df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load cached features from data/processed/ or featurize and cache."""
    cache = PROCESSED / f"{endpoint}_features.npz"
    if cache.exists():
        if (RAW / f"{endpoint}.parquet").stat().st_mtime > cache.stat().st_mtime:
            logger.warning(
                "Raw data for %s is newer than feature cache — delete %s to rebuild",
                endpoint,
                cache,
            )
        logger.info("Loading cached features for %s", endpoint)
        data = np.load(cache)
        return data["X"], data["y"], list(data["smiles"])
    X, y, smiles = featurize(df)
    np.savez_compressed(str(cache), X=X, y=y, smiles=np.array(smiles))
    logger.info("Cached features → %s", cache.name)
    return X, y, smiles


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
    If y is provided, ensures all classes appear in the test set (important for
    small classification datasets like hERG).
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
    test_idx: list[int] = []
    train_idx: list[int] = []
    skipped: list[int] = []

    for group in groups:
        if len(test_idx) >= n_test_target:
            train_idx.extend(group)
        elif len(test_idx) + len(group) <= n_test_cap:
            test_idx.extend(group)
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
            "test past %.0f%%",
            len(skipped),
            max(skipped),
            100 * (test_size + tolerance),
        )

    if y is not None:
        train_set = set(train_idx)
        test_set = set(test_idx)
        missing_classes = set(np.unique(y).astype(int)) - set(y[list(test_set)].astype(int))
        if missing_classes:
            logger.warning("Test set missing classes %s — moving scaffolds to fix", missing_classes)
            for cls in missing_classes:
                for indices in scaffold_to_indices.values():
                    if any(int(y[i]) == cls for i in indices if i in train_set):
                        for i in indices:
                            train_set.discard(i)
                            test_set.add(i)
                        break
        train_idx = list(train_set)
        test_idx = list(test_set)

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
