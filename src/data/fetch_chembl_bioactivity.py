"""Fetch ADMET bioactivity data from ChEMBL to supplement TDC training sets.

For 4 of the 12 endpoints (hERG, CYP3A4, Lipophilicity, Solubility), queries the
ChEMBL activity API, normalizes units, deduplicates by canonical SMILES (median),
and saves:
  data/external/chembl_activity_{endpoint}.parquet  — raw API response (download cache)
  data/interim/{endpoint}_chembl.parquet            — curated ChEMBL-only
  data/interim/{endpoint}_merged.parquet            — TDC + ChEMBL, deduplicated
                                                      (TDC value kept for overlaps)

HLM and Clearance are skipped: their ChEMBL entries span too many assay formats
(microsomal vs hepatocyte, t½ vs % remaining, different substrate concentrations)
to curate reliably without per-assay domain knowledge.

Run: python -m src.data.fetch_chembl_bioactivity [--endpoint ENDPOINT] [--force]
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field

import pandas as pd
import requests
from rdkit import Chem

from src.constants import Endpoint
from src.data.validate_data import is_valid_smiles
from src.logging_config import setup_logging
from src.paths import EXTERNAL, INTERIM

logger = logging.getLogger(__name__)

CHEMBL_ACTIVITY_API = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
_BATCH = 1000
_REQUEST_DELAY = 0.1  # seconds between paginated requests


@dataclass
class _Query:
    target_chembl_id: str | None  # None for physicochemical assays (no specific target)
    standard_type: str            # e.g. "IC50", "LogD", "LogS"
    assay_type: str               # B=binding, F=functional, P=physicochemical
    confidence_score_gte: int | None = None  # None = no filter (physicochemical assays)
    # unit -> multiplier to normalize to base unit (nM for IC50).
    # Empty dict = accept all unit strings at face value (dimensionless: LogD, LogS).
    unit_multipliers: dict[str, float] = field(default_factory=dict)
    threshold: float | None = None      # None = regression; float = binarize at this value
    positive_if_lt: bool | None = None  # True: Y=1 if value < threshold (e.g. IC50 -> blocker)
    value_bounds: tuple[float, float] | None = None  # drop rows outside (lo, hi) before deduplication


_QUERIES: dict[Endpoint, _Query] = {
    Endpoint.HERG: _Query(
        target_chembl_id="CHEMBL240",  # KCNH2 — the hERG potassium channel
        standard_type="IC50",
        assay_type="B",
        confidence_score_gte=8,
        unit_multipliers={"pM": 0.001, "nM": 1.0, "uM": 1_000.0, "mM": 1_000_000.0},
        threshold=10_000.0,   # 10 µM in nM — matches herg_karim TDC binarization
        positive_if_lt=True,  # IC50 < 10 µM -> blocker (Y=1)
    ),
    Endpoint.CYP3A4: _Query(
        target_chembl_id="CHEMBL340",  # CYP3A4 liver enzyme
        standard_type="IC50",
        assay_type="B",
        confidence_score_gte=8,
        unit_multipliers={"pM": 0.001, "nM": 1.0, "uM": 1_000.0, "mM": 1_000_000.0},
        threshold=10_000.0,   # 10 µM in nM — matches CYP3A4_Veith TDC binarization
        positive_if_lt=True,  # IC50 < 10 µM -> inhibitor (Y=1)
    ),
    Endpoint.LIPOPHILICITY: _Query(
        target_chembl_id=None,
        standard_type="LogD",
        assay_type="P",
        confidence_score_gte=None,
        value_bounds=(-5.0, 10.0),  # physically valid LogD range; filters ChEMBL data-entry errors
    ),
}

_SKIPPED: frozenset[Endpoint] = frozenset({
    Endpoint.SOLUBILITY,  # ChEMBL stores solubility as "Solubility" in µg/mL, not LogS
    Endpoint.HLM,         # mixed microsomal/hepatocyte formats, inconsistent units
    Endpoint.CLEARANCE,   # hepatocyte vs microsomal vs in vivo — too heterogeneous
    Endpoint.BBB,         # in vivo PK parameter, not in ChEMBL activity format
    Endpoint.PPBR,        # % binding assays vary too much across labs/conditions
    Endpoint.VDSS,        # in vivo PK parameter, not in ChEMBL activity format
    Endpoint.LD50,        # acute toxicity studies not systematically in ChEMBL
    Endpoint.DILI,        # clinical outcome label, not measurable via ChEMBL activity
    Endpoint.CARCINOGENS, # regulatory/clinical label, not in ChEMBL activity format
})


def _fetch_raw(query: _Query, endpoint: Endpoint) -> pd.DataFrame:
    """Download activities from ChEMBL with pagination; cache raw response to external/."""
    cache = EXTERNAL / f"chembl_activity_{endpoint}.parquet"
    if cache.exists():
        logger.info("Loading %s raw activities from cache (%s)", endpoint, cache.name)
        return pd.read_parquet(cache)

    logger.info("Fetching %s activities from ChEMBL ...", endpoint)
    rows: list[dict] = []
    offset = 0

    while True:
        params: dict = {
            "format": "json",
            "limit": _BATCH,
            "offset": offset,
            "standard_type": query.standard_type,
            "standard_relation": "=",
            "assay_type": query.assay_type,
        }
        if query.target_chembl_id:
            params["target_chembl_id"] = query.target_chembl_id
        if query.confidence_score_gte is not None:
            params["confidence_score__gte"] = query.confidence_score_gte

        resp = requests.get(CHEMBL_ACTIVITY_API, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        for act in data["activities"]:
            smiles = act.get("canonical_smiles")
            value = act.get("standard_value")
            if not smiles or value is None:
                continue
            if act.get("data_validity_comment"):
                continue  # flagged as potentially erroneous by ChEMBL curators
            rows.append({
                "smiles": str(smiles),
                "value": float(value),
                "units": act.get("standard_units") or "",
            })

        if len(data["activities"]) < _BATCH:
            break
        offset += _BATCH
        if offset % 10_000 == 0:
            logger.info("  ... %d activities fetched", len(rows))
        time.sleep(_REQUEST_DELAY)

    df = pd.DataFrame(rows)
    df.to_parquet(cache, index=False)
    logger.info("Fetched %d raw activities for %s -> %s", len(df), endpoint, cache.name)
    return df


def _curate(df: pd.DataFrame, query: _Query, endpoint: Endpoint) -> pd.DataFrame:
    """Normalize units, canonicalize SMILES, deduplicate (median), binarize if classification."""
    if df.empty:
        return pd.DataFrame(columns=["Drug", "Y"])

    # Unit normalization for IC50 endpoints; dimensionless endpoints skip this block
    if query.unit_multipliers:
        before = len(df)
        df = df[df["units"].isin(query.unit_multipliers)].copy()
        if dropped := before - len(df):
            logger.info("  Dropped %d rows with unrecognized units", dropped)
        if df.empty:
            return pd.DataFrame(columns=["Drug", "Y"])
        df["value"] = df.apply(
            lambda r: r["value"] * query.unit_multipliers[r["units"]], axis=1
        )

    if query.value_bounds is not None:
        lo, hi = query.value_bounds
        before = len(df)
        df = df[(df["value"] >= lo) & (df["value"] <= hi)].copy()
        if dropped := before - len(df):
            logger.info("  Dropped %d rows outside value bounds [%g, %g]", dropped, lo, hi)
        if df.empty:
            return pd.DataFrame(columns=["Drug", "Y"])

    # Canonicalize SMILES via RDKit so duplicates across different SMILES representations merge
    canonical: list[str | None] = []
    for smi in df["smiles"]:
        mol = Chem.MolFromSmiles(str(smi))
        canonical.append(Chem.MolToSmiles(mol) if mol is not None else None)
    df = df.copy()
    df["Drug"] = canonical
    df = df.dropna(subset=["Drug"])

    # Same molecule measured multiple times in different assays -> take median
    df = df.groupby("Drug")[["value"]].median().reset_index()

    # Binarize for classification endpoints
    if query.threshold is not None and query.positive_if_lt is not None:
        df["Y"] = (df["value"] < query.threshold if query.positive_if_lt
                   else df["value"] >= query.threshold).astype(float)
    else:
        df["Y"] = df["value"].astype(float)

    df = df[["Drug", "Y"]].copy()

    before = len(df)
    df = df[df["Drug"].apply(is_valid_smiles)].reset_index(drop=True)
    if dropped := before - len(df):
        logger.warning("  Dropped %d rows with invalid SMILES after curation", dropped)

    if query.threshold is not None:
        pos = int(df["Y"].sum())
        logger.info(
            "%s curated: %d molecules, %d positive (%.1f%%)",
            endpoint, len(df), pos, 100 * pos / len(df) if len(df) else 0,
        )
    else:
        logger.info(
            "%s curated: %d molecules, Y in [%.2f, %.2f]",
            endpoint, len(df), float(df["Y"].min()), float(df["Y"].max()),
        )
    return df


def _merge_with_tdc(chembl_df: pd.DataFrame, endpoint: Endpoint) -> pd.DataFrame:
    """Combine TDC and ChEMBL; TDC values take priority for overlapping molecules."""
    tdc_path = INTERIM / f"{endpoint}.parquet"
    if not tdc_path.exists():
        logger.warning("TDC data not found for %s — merged = ChEMBL only", endpoint)
        return chembl_df

    tdc_df = pd.read_parquet(tdc_path)[["Drug", "Y"]].copy()

    # Canonicalize TDC SMILES for fair overlap detection
    canonical: list[str | None] = []
    for smi in tdc_df["Drug"]:
        mol = Chem.MolFromSmiles(str(smi))
        canonical.append(Chem.MolToSmiles(mol) if mol is not None else None)
    tdc_df["Drug"] = canonical
    tdc_df = tdc_df.dropna(subset=["Drug"])

    tdc_smiles = set(tdc_df["Drug"])
    chembl_new = chembl_df[~chembl_df["Drug"].isin(tdc_smiles)]

    merged = pd.concat([tdc_df, chembl_new], ignore_index=True)
    logger.info(
        "%s merged: %d TDC + %d new from ChEMBL = %d total",
        endpoint, len(tdc_df), len(chembl_new), len(merged),
    )
    return merged


def fetch_endpoint(endpoint: Endpoint, force: bool = False) -> None:
    if endpoint in _SKIPPED:
        logger.info(
            "Skipping %s — HLM/Clearance assay formats too heterogeneous in ChEMBL", endpoint
        )
        return

    chembl_path = INTERIM / f"{endpoint}_chembl.parquet"
    merged_path = INTERIM / f"{endpoint}_merged.parquet"
    if chembl_path.exists() and merged_path.exists() and not force:
        logger.info("Skipping %s (files exist; use --force to re-fetch)", endpoint)
        return

    query = _QUERIES[endpoint]
    raw_df = _fetch_raw(query=query, endpoint=endpoint)
    if raw_df.empty:
        logger.warning("No activities found for %s", endpoint)
        return

    curated_df = _curate(df=raw_df, query=query, endpoint=endpoint)
    if curated_df.empty:
        logger.warning("No molecules survived curation for %s", endpoint)
        return

    curated_df.to_parquet(chembl_path, index=False)
    logger.info("ChEMBL-only -> %s  (%d molecules)", chembl_path.name, len(curated_df))

    merged_df = _merge_with_tdc(chembl_df=curated_df, endpoint=endpoint)
    merged_df.to_parquet(merged_path, index=False)
    logger.info("Merged     -> %s  (%d molecules)", merged_path.name, len(merged_df))


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Fetch ChEMBL bioactivity data to supplement TDC training sets"
    )
    parser.add_argument("--endpoint", default=None, help="Single endpoint (default: all)")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cache exists")
    args = parser.parse_args()

    endpoints = [Endpoint(args.endpoint)] if args.endpoint else list(Endpoint)
    for ep in endpoints:
        fetch_endpoint(endpoint=ep, force=args.force)
    logger.info("Done. Merged parquets saved to %s", INTERIM)


if __name__ == "__main__":
    main()
