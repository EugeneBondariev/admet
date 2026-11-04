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
) -> tuple[np.ndarray, np.ndarray]:
    """Split by Murcko scaffold — no scaffold appears in both train and test.

    Assigns rarest scaffolds (smallest groups) to test first so the test set
    covers diverse chemical space rather than a single dominant series.
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
    test_idx, train_idx = [], []

    for group in groups:
        if len(test_idx) < n_test_target:
            test_idx.extend(group)
        else:
            train_idx.extend(group)

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

    logger.info(
        "Scaffold split: %d train / %d test (%.0f%% test)",
        len(train_idx),
        len(test_idx),
        100 * len(test_idx) / len(smiles),
    )
    return np.array(train_idx), np.array(test_idx)
