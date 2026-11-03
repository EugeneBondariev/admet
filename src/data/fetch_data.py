"""Download all ADMET datasets from TDC and save to data/raw/ as parquet.

Data source: Therapeutics Data Commons (https://tdcommons.ai)
Run: python -m src.data.fetch_data
"""
from __future__ import annotations

import logging

from tdc.single_pred import ADME, Tox

from src.constants import ENDPOINT_TDC_GROUP, Endpoint, TDCGroup
from src.paths import RAW

logger = logging.getLogger(__name__)

_TDC_CLASS = {TDCGroup.ADME: ADME, TDCGroup.TOX: Tox}


def fetch_endpoint(endpoint: Endpoint, force: bool = False) -> None:
    out = RAW / f"{endpoint}.parquet"
    if out.exists() and not force:
        logger.info("Skipping %s (already exists)", endpoint)
        return
    logger.info("Fetching %s ...", endpoint)
    cls = _TDC_CLASS[ENDPOINT_TDC_GROUP[endpoint]]
    df = cls(name=endpoint).get_data()
    df.to_parquet(out, index=False)
    logger.info("  %d rows -> %s", len(df), out.name)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for endpoint in Endpoint:
        fetch_endpoint(endpoint)


if __name__ == "__main__":
    main()
