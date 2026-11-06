"""Virtual screening against ChEMBL lead-like compounds.

Loads 6 trained LightGBM models (from make predict), featurizes SMILES from
ChEMBL (MW ≤ 500, logP ≤ 5), applies drug-likeness thresholds to each
endpoint, and reports how many candidates survive all filters.
Run: python -m src.screening.screen [--n-compounds N]

Requires models saved by make predict.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime

import joblib
import numpy as np
import requests
from lightgbm import LGBMClassifier, LGBMRegressor

from src.constants import ENDPOINT_TASK, LOG_TRANSFORM, Endpoint, Task
from src.features.features import FEATURE_HASH, featurize_smiles
from src.logging_config import setup_logging
from src.paths import EXTERNAL, MODELS_LGBM, PROC_FEATURES, REPORTS_SCREENING

logger = logging.getLogger(__name__)

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"

# Thresholds for each endpoint.
# Regression: threshold on predicted value (after expm1 for log-transformed endpoints).
# Classification: threshold on p(positive class = 1).
#   CYP3A4  Y=1 → inhibitor     → want p < 0.50
#   HLM     Y=1 → stable        → want p > 0.50  (verify: 0=unstable, 1=stable in TDC hlm)
#   hERG    Y=1 → blocker       → want p < 0.30  (strict cardiac safety)
THRESHOLD_LABEL: dict[Endpoint, str] = {
    Endpoint.SOLUBILITY:    "logS > -5.0 mol/L",
    Endpoint.LIPOPHILICITY: "-1.0 ≤ logD ≤ 5.0",
    Endpoint.CYP3A4:        "p(inhibitor) < 0.50",
    Endpoint.HLM:           "p(stable) > 0.50",
    Endpoint.CLEARANCE:     "clearance < 150 mL/min/kg",
    Endpoint.HERG:          "p(hERG blocker) < 0.30",
}


def _passes(endpoint: Endpoint, pred: float) -> bool:
    if endpoint == Endpoint.SOLUBILITY:
        return pred > -5.0
    if endpoint == Endpoint.LIPOPHILICITY:
        return -1.0 <= pred <= 5.0
    if endpoint == Endpoint.CLEARANCE:
        return pred < 150.0
    if endpoint == Endpoint.CYP3A4:
        return pred < 0.5
    if endpoint == Endpoint.HLM:
        return pred > 0.5
    if endpoint == Endpoint.HERG:
        return pred < 0.3
    raise ValueError(f"No threshold defined for {endpoint}")


def _load_model(endpoint: Endpoint) -> LGBMClassifier | LGBMRegressor:
    path = MODELS_LGBM / f"{endpoint}_lgbm.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path} — run 'make predict' first"
        )
    return joblib.load(path)  # type: ignore[return-value]


def _predict(
    model: LGBMClassifier | LGBMRegressor, X: np.ndarray, endpoint: Endpoint
) -> np.ndarray:
    task = ENDPOINT_TASK[endpoint]
    if task == Task.CLASSIFICATION:
        assert isinstance(model, LGBMClassifier)
        return np.asarray(model.predict_proba(X))[:, 1]
    assert isinstance(model, LGBMRegressor)
    preds = np.asarray(model.predict(X))
    if endpoint in LOG_TRANSFORM:
        preds = np.expm1(preds)
    return preds


def _fetch_chembl(n: int) -> list[str]:
    cache = EXTERNAL / f"chembl_lead_like_{n}.smi"
    if cache.exists():
        lines = cache.read_text().splitlines()
        if len(lines) < n:
            logger.warning(
                "Cached %s has %d compounds (requested %d) — "
                "ChEMBL may have fewer matching compounds, or a previous download was interrupted; "
                "delete it to re-download",
                cache.name, len(lines), n,
            )
        logger.info("Loading %d compounds from cache (%s)", len(lines), cache.name)
        return lines

    logger.info("Downloading %d lead-like compounds from ChEMBL (MW ≤ 500, logP ≤ 5) ...", n)
    smiles: list[str] = []
    offset = 0
    batch = 1000
    while len(smiles) < n:
        resp = requests.get(
            CHEMBL_API,
            params={  # type: ignore[arg-type]
                "format": "json",
                "limit": min(batch, n - len(smiles)),
                "offset": offset,
                "molecule_properties__full_mwt__lte": 500,
                "molecule_properties__alogp__lte": 5,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        for mol in data["molecules"]:
            structs = mol.get("molecule_structures")
            smi = structs.get("canonical_smiles") if structs else None
            if smi:
                smiles.append(smi)
        if len(data["molecules"]) < batch:
            break
        offset += batch
        if offset % 10_000 == 0:
            logger.info("  ... %d downloaded", len(smiles))
    cache.write_text("\n".join(smiles))
    logger.info("Downloaded %d SMILES → %s", len(smiles), cache.name)
    return smiles



def _load_or_featurize(smiles: list[str], n: int) -> tuple[np.ndarray, list[str]]:
    cache = PROC_FEATURES / f"chembl_lead_like_{n}_features.npz"
    smiles_file = EXTERNAL / f"chembl_lead_like_{n}.smi"
    if cache.exists():
        data = np.load(cache, allow_pickle=False)
        cached_hash = str(data["feature_hash"]) if "feature_hash" in data else None
        if cached_hash != FEATURE_HASH:
            logger.warning(
                "Feature set changed (%s → %s) — rebuilding ChEMBL feature cache",
                cached_hash, FEATURE_HASH,
            )
        elif smiles_file.exists() and smiles_file.stat().st_mtime > cache.stat().st_mtime:
            logger.warning("SMILES file is newer than feature cache — rebuilding")
        else:
            logger.info("Loading cached features from %s", cache.name)
            return data["X"], list(data["smiles"])
    X, valid_smiles = featurize_smiles(smiles)
    np.savez_compressed(cache, X=X, smiles=np.array(valid_smiles), feature_hash=np.array(FEATURE_HASH))
    logger.info("Cached features → %s", cache.name)
    return X, valid_smiles


def screen(smiles: list[str], n: int) -> dict:
    """Run all 6 models on SMILES, apply thresholds, return structured results."""
    X, valid_smiles = _load_or_featurize(smiles, n)
    n_valid = len(valid_smiles)

    models = {ep: _load_model(ep) for ep in Endpoint}
    preds: dict[Endpoint, np.ndarray] = {
        ep: _predict(models[ep], X, ep) for ep in Endpoint
    }

    pass_masks: dict[Endpoint, np.ndarray] = {
        ep: np.array([_passes(ep, float(p)) for p in preds[ep]]) for ep in Endpoint
    }
    all_pass = np.ones(n_valid, dtype=bool)
    for mask in pass_masks.values():
        all_pass &= mask

    hits = [smi for smi, ok in zip(valid_smiles, all_pass) if ok]

    return {
        "n_input": len(smiles),
        "n_valid": n_valid,
        "per_endpoint": {
            ep: {
                "label": THRESHOLD_LABEL[ep],
                "n_pass": int(pass_masks[ep].sum()),
                "pct_pass": float(pass_masks[ep].mean() * 100),
            }
            for ep in Endpoint
        },
        "n_hits": len(hits),
        "pct_hits": float(len(hits) / n_valid * 100) if n_valid else 0.0,
        "hits": hits,
    }


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Virtual screening against ChEMBL lead-like compounds")
    parser.add_argument(
        "--n-compounds", type=int, default=100_000,
        help="Number of compounds to download from ChEMBL (default 100000). Usage: N=value",
    )
    args = parser.parse_args()

    try:
        smiles = _fetch_chembl(args.n_compounds)
        source = f"ChEMBL lead-like (n={args.n_compounds})"
    except Exception as e:
        logger.error("ChEMBL download failed: %s", e)
        raise SystemExit(1)

    results = screen(smiles, n=args.n_compounds)

    logger.info("\n=== VIRTUAL SCREENING SUMMARY ===")
    logger.info("Source:  %s", source)
    logger.info("Input:   %d SMILES", results["n_input"])
    logger.info("Valid:   %d (RDKit-parseable)", results["n_valid"])
    logger.info("\nFilter results:")
    for _, ep_res in results["per_endpoint"].items():
        logger.info("  %-35s %6d pass  (%5.1f%%)", ep_res["label"], ep_res["n_pass"], ep_res["pct_pass"])
    logger.info("\nAll 6 filters passed: %d compounds (%.1f%%)", results["n_hits"], results["pct_hits"])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    hits_path = REPORTS_SCREENING / f"chembl_hits_{ts}.smi"
    hits_path.write_text("\n".join(results["hits"]))
    logger.info("Hits saved → %s", hits_path)

    report = {
        "timestamp": datetime.now().isoformat(),
        "source": source,
        "n_input": results["n_input"],
        "n_valid": results["n_valid"],
        "per_endpoint": {str(ep): v for ep, v in results["per_endpoint"].items()},
        "n_hits": results["n_hits"],
        "pct_hits": results["pct_hits"],
    }
    report_path = REPORTS_SCREENING / f"screen_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Report  → %s", report_path)


if __name__ == "__main__":
    main()
