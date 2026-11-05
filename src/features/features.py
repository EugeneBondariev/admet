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

from src.paths import INTERIM, PROCESSED

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")

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
        if (INTERIM / f"{endpoint}.parquet").stat().st_mtime > cache.stat().st_mtime:
            logger.warning(
                "Interim data for %s is newer than feature cache — delete %s to rebuild",
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
    for smi, label in zip(df["Drug"], df["Y"]):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            n_dropped += 1
            continue
        rows.append(_mol_to_features(mol))
        labels.append(float(label))
        smiles_out.append(str(smi))

    if n_dropped:
        logger.warning("Dropped %d rows with unparseable SMILES", n_dropped)
    logger.info("Featurized %d molecules (%d features)", len(rows), len(FEATURE_NAMES))

    return np.vstack(rows), np.array(labels, dtype=np.float32), smiles_out


def _quantile_bins(y: np.ndarray, n_bins: int) -> np.ndarray:
    """Cut continuous labels into n_bins roughly equal-count quantile bins.

    Duplicate edges are collapsed, so a label with many tied values — Clearance
    is clipped at 3 and 150 — yields fewer, uneven bins rather than empty ones.
    """
    edges = np.unique(np.quantile(y, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]))
    return np.searchsorted(edges, y, side="right").astype(int)


def scaffold_split(
    smiles: list[str],
    test_size: float = 0.2,
    tolerance: float = 0.02,
    y: np.ndarray | None = None,
    stratify_bins: int | None = None,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split by Murcko scaffold — no scaffold appears in both train and test.

    Assigns rarest scaffolds (smallest groups) to test first so the test set
    covers diverse chemical space rather than a single dominant series.
    Equal-sized groups are ordered by a random_state-seeded shuffle, so which
    of them reaches test does not depend on row order in the dataframe.
    Scaffolds move as whole groups, so the test set can only land on exact
    test_size by luck; a group is skipped (left in train) if adding it would
    push test past test_size + tolerance, keeping the realized split close
    enough to compare metrics across endpoints and against TDC.
    A molecule with no ring system has no Murcko scaffold. Those are keyed
    individually rather than pooled under one empty scaffold: acyclic molecules
    share no chemical core, and pooling them makes a large inseparable block
    that never reaches test (29.5% of AqSolDB, and the soluble end of it).
    If y is provided, each label stratum gets its own share of the test set in
    proportion to the full dataset, so test mirrors the real label distribution
    instead of whatever the rarest scaffolds happen to carry. With
    stratify_bins=None y is read as discrete classes (classification); with an
    int, y is continuous and is cut into that many quantile bins (regression).
    Both need it: hERG's test set otherwise fills almost entirely with one class
    and its ROC-AUC / AUPRC stop meaning anything, while scaffold and label
    correlate for regression too, so test can drift to one end of the range.
    Raises ValueError if a stratum cannot reach test because every scaffold group
    carrying it is too large.
    """
    scaffold_to_indices: dict[str, list[int]] = {}
    unparseable: list[int] = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            # featurize() already drops these, so this only fires on raw input.
            # Pin them to train: a molecule with no features cannot be predicted,
            # so letting it reach test would corrupt the metric.
            unparseable.append(i)
            continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        # Keyed by canonical SMILES so true duplicates still share a group —
        # the same molecule either side of the split is leakage, not diversity.
        key = scaffold or f"acyclic:{Chem.MolToSmiles(mol)}"
        scaffold_to_indices.setdefault(key, []).append(i)

    if unparseable:
        logger.warning(
            "%d unparseable SMILES forced into train (first index %d) — featurize() "
            "normally drops these",
            len(unparseable),
            unparseable[0],
        )

    # Shuffle, then stable-sort by size: random order within each size band,
    # deterministic given random_state. Without the shuffle, ties are broken by
    # dataframe row order, which decides the split for datasets made mostly of
    # singleton scaffolds.
    group_list = list(scaffold_to_indices.values())
    rng = np.random.default_rng(random_state)
    groups = sorted((group_list[i] for i in rng.permutation(len(group_list))), key=len)
    n_test_target = int(len(smiles) * test_size)
    n_test_cap = int(len(smiles) * (test_size + tolerance))

    # Per-stratum size quotas. With y=None this collapses to one pseudo-stratum
    # and the quotas reduce to the plain size target and cap.
    if y is None:
        labels = np.zeros(len(smiles), dtype=int)
    elif stratify_bins is None:
        labels = y.astype(int)
    else:
        labels = _quantile_bins(y, stratify_bins)
    # Shares come from the splittable rows only — a stratum whose molecules are
    # all unparseable can never reach test, and should not be given a quota.
    splittable = [i for g in scaffold_to_indices.values() for i in g]
    unique_labels, label_counts = np.unique(labels[splittable], return_counts=True)
    shares = {int(c): int(n) / len(splittable) for c, n in zip(unique_labels, label_counts)}

    # Floor of 1 so a stratum too rare to earn a proportional slot still reaches test.
    class_target = {cls: max(1, round(n_test_target * share)) for cls, share in shares.items()}
    # Each cap leaves room for every other stratum's target, so a common stratum
    # cannot consume the whole test budget before a rare one gets in.
    class_cap: dict[int, int] = {}
    for cls, share in shares.items():
        reserved = sum(t for c, t in class_target.items() if c != cls)
        class_cap[cls] = max(
            class_target[cls], min(ceil(n_test_cap * share), n_test_cap - reserved)
        )

    test_idx: list[int] = []
    test_counts: Counter[int] = Counter()
    counts_per_group = [Counter(int(labels[i]) for i in g) for g in groups]
    taken = [False] * len(groups)
    rejected: set[int] = set()

    # Pass 1 admits a group only while it keeps every stratum at or under target,
    # which reserves each stratum's slots before the budget is spent. Pass 2 then
    # tops test up toward the overall target, now allowing growth up to each cap.
    # One pass alone fails whichever stratum only occurs in chunky groups.
    for limits in (class_target, class_cap):
        for gi, group in enumerate(groups):
            if all(test_counts[c] >= t for c, t in class_target.items()):
                break
            if taken[gi]:
                continue
            group_counts = counts_per_group[gi]
            if len(test_idx) + len(group) <= n_test_cap and all(
                test_counts[c] + n <= limits[c] for c, n in group_counts.items()
            ):
                test_idx.extend(group)
                test_counts.update(group_counts)
                taken[gi] = True
                rejected.discard(gi)
            else:
                rejected.add(gi)

    train_idx = unparseable + [i for gi, g in enumerate(groups) if not taken[gi] for i in g]
    skipped = [len(groups[gi]) for gi in sorted(rejected)]

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
        kind = "label bin" if stratify_bins is not None else "class"
        missing = sorted(cls for cls in class_target if test_counts[cls] == 0)
        if missing:
            raise ValueError(
                f"No test molecules for {kind} {missing} — every scaffold group carrying "
                f"them is larger than that {kind}'s share of a "
                f"{100 * (test_size + tolerance):.0f}% test set; raise tolerance or test_size"
            )
        logger.info(
            "Test %s balance: %s (dataset: %s)",
            kind,
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
        # Unparseable rows have no scaffold, so they are not counted here.
        len({index_to_scaffold[i] for i in train_idx if i in index_to_scaffold}),
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
