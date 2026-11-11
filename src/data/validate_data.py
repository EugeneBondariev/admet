"""Validate the ADMET parquet files converted from TDC downloads.

Checks: expected columns, no nulls in key fields, row count minimums,
RDKit-parseable SMILES, label ranges per task type.
Run: python -m src.data.validate_data
"""
from __future__ import annotations

import logging

import pandas as pd
from rdkit import Chem, RDLogger

from src.constants import ENDPOINT_TASK, Endpoint, Task
from src.logging_config import setup_logging
from src.paths import INTERIM

RDLogger.DisableLog("rdApp.warning")

logger = logging.getLogger(__name__)


def is_valid_smiles(smiles: str) -> bool:
    return Chem.MolFromSmiles(smiles) is not None


MIN_ROWS: dict[Endpoint, int] = {
    Endpoint.SOLUBILITY:    5_000,
    Endpoint.BBB:           1_500,
    Endpoint.LIPOPHILICITY: 2_000,
    Endpoint.PPBR:          1_500,
    Endpoint.VDSS:            800,
    Endpoint.CYP3A4:        5_000,
    Endpoint.HLM:           3_000,
    Endpoint.CLEARANCE:       500,
    Endpoint.HERG:          8_000,
    Endpoint.LD50:          5_000,
    Endpoint.DILI:            350,
    Endpoint.CARCINOGENS:     200,
}

REGRESSION_BOUNDS: dict[Endpoint, tuple[float, float]] = {
    Endpoint.SOLUBILITY:    (-15.0, 5.0),     # logS (mol/L)
    Endpoint.LIPOPHILICITY: (-5.0,  10.0),    # logD (pH 7.4)
    Endpoint.PPBR:          (0.0,   100.0),   # plasma protein binding (%)
    Endpoint.VDSS:          (0.0,   1_000.0), # volume of distribution (L/kg); highly lipophilic drugs can exceed 500
    Endpoint.CLEARANCE:     (0.0,   5_000.0), # hepatic clearance (mL/min/kg)
    Endpoint.LD50:          (0.0,   7.0),     # log10(LD50 mg/kg); TDC docs say mol/kg but data is mg/kg
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
            logger.warning(
                "[%s] %d Y values (%.1f%%) outside expected range [%g, %g]",
                endpoint, out_of_range, 100 * out_of_range / len(df), lo, hi,
            )

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
