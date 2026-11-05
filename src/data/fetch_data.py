"""Download all ADMET datasets from TDC and convert them to parquet.

TDC writes its own .tab download into data/raw/ (immutable source of truth);
this module converts each to data/interim/{endpoint}.parquet.

Data source: Therapeutics Data Commons (https://tdcommons.ai)
Run: python -m src.data.fetch_data
"""
from __future__ import annotations

import logging

from tdc.single_pred import ADME, Tox

from src.constants import ENDPOINT_TDC_GROUP, Endpoint, TDCGroup
from src.data.validate_data import is_valid_smiles
from src.paths import INTERIM, RAW

logger = logging.getLogger(__name__)

_TDC_CLASS = {TDCGroup.ADME: ADME, TDCGroup.TOX: Tox}


def fetch_endpoint(endpoint: Endpoint, force: bool = False) -> None:
    out = INTERIM / f"{endpoint}.parquet"
    if out.exists() and not force:
        logger.info("Skipping %s (already exists)", endpoint)
        return
    logger.info("Fetching %s ...", endpoint)
    cls = _TDC_CLASS[ENDPOINT_TDC_GROUP[endpoint]]
    # path=RAW keeps TDC's .tab download in data/raw/ instead of ./data, which
    # it defaults to — that default is relative to the shell's cwd, not the project.
    df = cls(name=endpoint, path=str(RAW)).get_data()
    df["Drug"] = df["Drug"].astype(str)
    df["Y"] = df["Y"].astype(float)
    before = len(df)
    df = df[df["Drug"].apply(is_valid_smiles)].reset_index(drop=True)
    if dropped := before - len(df):
        logger.warning("  Dropped %d rows with unparseable SMILES", dropped)
    df.to_parquet(out, index=False)
    logger.info("  %d rows -> %s", len(df), out.name)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for endpoint in Endpoint:
        fetch_endpoint(endpoint)


if __name__ == "__main__":
    main()
