"""3D molecular graph featurization for PyTorch Geometric.

Generates 3D conformers from SMILES using RDKit ETKDGv3, builds PyG Data
objects with atomic numbers and positions for use with SchNet.
Caches graph datasets to data/processed/{endpoint}_graphs.pt.
"""
from __future__ import annotations

import logging
import warnings

import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from torch_geometric.data import Data

from src.paths import PROCESSED, RAW

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")

logger = logging.getLogger(__name__)

_ETKDG = AllChem.ETKDGv3()
_ETKDG.randomSeed = 42


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


def build_dataset(df: pd.DataFrame) -> tuple[list[Data], list[str]]:
    """Build PyG graphs from Drug/Y DataFrame. Returns (graphs, smiles) for valid rows."""
    graphs: list[Data] = []
    smiles_out: list[str] = []
    n_failed = 0
    for _, row in df.iterrows():
        smi = str(row["Drug"])
        g = smiles_to_graph(smiles=smi, y=float(row["Y"]))
        if g is None:
            n_failed += 1
        else:
            graphs.append(g)
            smiles_out.append(smi)
    if n_failed:
        logger.warning("Failed 3D conformer for %d molecules", n_failed)
    logger.info("Built %d molecular graphs", len(graphs))
    return graphs, smiles_out


def load_or_build_dataset(
    endpoint: str, df: pd.DataFrame
) -> tuple[list[Data], list[str]]:
    """Load cached graph dataset or build and cache it."""
    cache = PROCESSED / f"{endpoint}_graphs.pt"
    if cache.exists():
        if (RAW / f"{endpoint}.parquet").stat().st_mtime > cache.stat().st_mtime:
            logger.warning(
                "Raw data for %s is newer than graph cache — delete %s to rebuild",
                endpoint,
                cache,
            )
        logger.info("Loading cached graphs for %s", endpoint)
        saved = torch.load(str(cache), weights_only=False)
        return saved["graphs"], saved["smiles"]
    graphs, smiles = build_dataset(df)
    torch.save({"graphs": graphs, "smiles": smiles}, str(cache))
    logger.info("Cached %d graphs → %s", len(graphs), cache.name)
    return graphs, smiles
