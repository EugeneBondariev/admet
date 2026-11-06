"""Validate the ADMET parquet files converted from TDC downloads.

Checks: expected columns, no nulls in key fields, row count minimums,
RDKit-parseable SMILES, label ranges per task type.
Run: python -m src.data.validate_data
"""
from __future__ import annotations

import logging

import pandas as pd
from rdkit import Chem

from src.constants import ENDPOINT_TASK, Endpoint, Task
from src.logging_config import setup_logging
from src.paths import INTERIM

logger = logging.getLogger(__name__)


def is_valid_smiles(smiles: str) -> bool:
    return Chem.MolFromSmiles(smiles) is not None


MIN_ROWS: dict[Endpoint, int] = {
    Endpoint.SOLUBILITY: 5000,
    Endpoint.LIPOPHILICITY: 2000,
    Endpoint.CYP3A4: 5000,
    Endpoint.HLM: 3000,
    Endpoint.CLEARANCE: 500,
    Endpoint.HERG: 8000,
}

REGRESSION_BOUNDS: dict[Endpoint, tuple[float, float]] = {
    Endpoint.SOLUBILITY: (-15.0, 5.0),       # logS range
    Endpoint.LIPOPHILICITY: (-5.0, 10.0),     # logD range
    Endpoint.CLEARANCE: (0.0, 5000.0),        # mL/min/kg
}


def validate_endpoint(endpoint: Endpoint) -> bool:
    path = INTERIM / f"{endpoint}.parquet"
    if not path.exists():
        logger.error("[%s] file not found — run make fetch first", endpoint)
        return False

    df = pd.read_parquet(path)
    ok = True

    for col in ("Drug_ID", "Drug", "Y"):
        if col not in df.columns:
            logger.error("[%s] missing column: %s", endpoint, col)
            ok = False

    if not ok:
        return False

    null_smiles = df["Drug"].isna().sum()
    null_y = df["Y"].isna().sum()
    if null_smiles:
        logger.error("[%s] %d null SMILES", endpoint, null_smiles)
        ok = False
    if null_y:
        logger.error("[%s] %d null labels", endpoint, null_y)
        ok = False

    if len(df) < MIN_ROWS[endpoint]:
        logger.error("[%s] only %d rows (expected >=%d)", endpoint, len(df), MIN_ROWS[endpoint])
        ok = False

    bad_smiles = (~df["Drug"].apply(is_valid_smiles)).sum()
    if bad_smiles:
        logger.error("[%s] %d unparseable SMILES — fetch_data should have removed these", endpoint, bad_smiles)
        ok = False

    task = ENDPOINT_TASK[endpoint]
    if task == Task.CLASSIFICATION:
        unexpected = set(df["Y"].unique()) - {0, 1}
        if unexpected:
            logger.error("[%s] unexpected label values: %s", endpoint, unexpected)
            ok = False
    else:
        lo, hi = REGRESSION_BOUNDS[endpoint]
        out_of_range = ((df["Y"] < lo) | (df["Y"] > hi)).sum()
        if out_of_range:
            logger.warning("[%s] %d Y values outside expected range [%g, %g]", endpoint, out_of_range, lo, hi)

    if ok:
        logger.info("[%s] OK — %d rows", endpoint, len(df))
    return ok


def main() -> None:
    setup_logging()
    results = {ep: validate_endpoint(ep) for ep in Endpoint}
    failed = [ep for ep, ok in results.items() if not ok]
    if failed:
        logger.error("Validation failed for: %s", ", ".join(failed))
        raise SystemExit(1)
    logger.info("All endpoints passed validation.")


if __name__ == "__main__":
    main()
