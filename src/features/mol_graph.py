"""3D molecular graph featurization for PyTorch Geometric.

Generates 3D conformers from SMILES using RDKit ETKDGv3, builds PyG Data
objects with atomic numbers and positions for use with SchNet.
Caches graph datasets to data/processed/{endpoint}_graphs.pt.
Checkpoints every 500 molecules so interrupted builds resume from where they stopped.
"""
from __future__ import annotations

import logging
import warnings

import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from tqdm import tqdm

from src.paths import INTERIM, PROC_GRAPHS

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")

logger = logging.getLogger(__name__)

_ETKDG = AllChem.ETKDGv3()
_ETKDG.randomSeed = 42

CHECKPOINT_EVERY = 500


def smiles_to_graph(smiles: str, y: float) -> Data | None:
    """Generate a 3D conformer and return a PyG Data object, or None on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, _ETKDG) == -1:
        fallback = AllChem.ETKDGv3()
        fallback.randomSeed = 42
        fallback.enforceChirality = False
        if AllChem.EmbedMolecule(mol, fallback) == -1:
            return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        AllChem.MMFFOptimizeMolecule(mol)

    mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer()

    z = torch.tensor(
        [atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long
    )
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float)
    return Data(z=z, pos=pos, y=torch.tensor([y], dtype=torch.float))


def load_or_build_dataset(
    endpoint: str, df: pd.DataFrame
) -> tuple[list[Data], list[str]]:
    """Load cached graph dataset or build and cache it, resuming from checkpoint if interrupted."""
    cache = PROC_GRAPHS / f"{endpoint}_graphs.pt"
    ckpt = PROC_GRAPHS / f"{endpoint}_graphs_ckpt.pt"

    if cache.exists():
        if (INTERIM / f"{endpoint}.parquet").stat().st_mtime > cache.stat().st_mtime:
            logger.warning(
                "Interim data for %s is newer than graph cache — delete %s to rebuild",
                endpoint, cache,
            )
        logger.info("Loading cached graphs for %s (%.1f MB) ...", endpoint, cache.stat().st_size / 1e6)
        saved = torch.load(cache, weights_only=False)
        return saved["graphs"], saved["smiles"]

    graphs: list[Data] = []
    smiles_out: list[str] = []
    start_idx = 0
    n_failed = 0

    if ckpt.exists():
        saved_ckpt = torch.load(ckpt, weights_only=False)
        graphs = saved_ckpt["graphs"]
        smiles_out = saved_ckpt["smiles"]
        start_idx = saved_ckpt["processed_count"]
        logger.info("Resuming from checkpoint: %d/%d rows already done", start_idx, len(df))

    rows = zip(df["Drug"].iloc[start_idx:], df["Y"].iloc[start_idx:])
    with tqdm(rows, desc="building graphs", unit="mol", initial=start_idx, total=len(df)) as pbar:
        for local_i, (smi, label) in enumerate(pbar):
            g = smiles_to_graph(smiles=smi, y=float(label))
            if g is None:
                n_failed += 1
            else:
                graphs.append(g)
                smiles_out.append(smi)

            processed = start_idx + local_i + 1
            if processed % CHECKPOINT_EVERY == 0:
                torch.save({"graphs": graphs, "smiles": smiles_out, "processed_count": processed}, ckpt)

    if n_failed:
        logger.warning("Failed 3D conformer for %d molecules", n_failed)
    logger.info("Built %d/%d molecular graphs", len(graphs), len(df))

    torch.save({"graphs": graphs, "smiles": smiles_out}, cache)
    logger.info("Cached %d graphs → %s", len(graphs), cache.name)
    if ckpt.exists():
        ckpt.unlink()

    return graphs, smiles_out
