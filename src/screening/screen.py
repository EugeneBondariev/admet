"""Virtual screening against ChEMBL drug-like compounds.

Loads 12 trained LightGBM models (from make predict), featurizes SMILES from
ChEMBL (all 4 Ro5 rules: MW ≤ 500, logP ≤ 5, HBD ≤ 5, HBA ≤ 10), applies drug-likeness thresholds to each
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
import pandas as pd
import requests
from lightgbm import LGBMClassifier, LGBMRegressor
from rdkit import Chem

from src.constants import ENDPOINT_TASK, LOG_TRANSFORM, Endpoint, Task
from src.features.features import FEATURE_HASH, featurize_smiles
from src.logging_config import setup_logging
from src.paths import EXTERNAL, INTERIM, MODELS_LGBM, PROC_FEATURES, REPORTS_SCREENING

logger = logging.getLogger(__name__)

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"

# Thresholds for each endpoint.
# Regression: threshold on predicted value (after expm1 for log-transformed endpoints).
# Classification: threshold on p(positive class = 1).
#   CYP3A4      Y=1 → inhibitor (blocks the enzyme)      → BAD  → want p < 0.50
#   HLM         Y=1 → stable (survives liver enzymes)     → GOOD → want p > 0.50
#   hERG        Y=1 → blocker (blocks cardiac channel)    → BAD  → want p < 0.30  (strict: causes arrhythmia)
#   BBB         Y=1 → crosses blood-brain barrier         → context-dependent; p < 0.50 for non-CNS screening
#   DILI        Y=1 → drug-induced liver injury           → BAD  → want p < 0.30  (strict: major withdrawal reason)
#   CARCINOGENS Y=1 → carcinogen                         → BAD  → want p < 0.20  (strict: safety-critical)
#
# Endpoint descriptions:
#   SOLUBILITY    — how well the drug dissolves in water (logS); too insoluble → never absorbed from gut
#   BBB           — blood-brain barrier penetration; undesirable for non-CNS drugs (CNS side effects)
#   LIPOPHILICITY — fat vs water affinity (logD); too polar → can't cross membranes; too greasy → won't dissolve
#   PPBR          — plasma protein binding rate (%); >90% bound → little free drug reaches target
#   VDSS          — volume of distribution (L/kg); very high → accumulates in tissues, slow elimination
#   CYP3A4        — liver enzyme metabolising ~50% of drugs; inhibiting it causes dangerous drug-drug interactions
#   HLM           — human liver microsome stability; unstable drugs are destroyed before reaching their target
#   CLEARANCE     — rate the liver removes drug from blood (mL/min/kg); too high → too short half-life
#   HERG          — cardiac potassium channel; blocking it causes fatal arrhythmia; major safety filter
#   LD50          — acute oral toxicity; Y = log(LD50 mol/kg); higher → less toxic (needs more to kill)
#   DILI          — drug-induced liver injury; leading cause of post-market drug withdrawal
#   CARCINOGENS   — cancer-causing potential; regulatory hard stop
THRESHOLD_LABEL: dict[Endpoint, str] = {
    Endpoint.SOLUBILITY:    "logS > -5.0 mol/L",
    Endpoint.BBB:           "p(BBB penetrant) < 0.50",
    Endpoint.LIPOPHILICITY: "-1.0 ≤ logD ≤ 5.0",
    Endpoint.PPBR:          "PPBR < 90%",
    Endpoint.VDSS:          "VDss < 20 L/kg",
    Endpoint.CYP3A4:        "p(inhibitor) < 0.50",
    Endpoint.HLM:           "p(stable) > 0.50",
    Endpoint.CLEARANCE:     "clearance < 150 mL/min/kg",
    Endpoint.HERG:          "p(hERG blocker) < 0.30",
    Endpoint.LD50:          "log(LD50) > -2.0 mol/kg",
    Endpoint.DILI:          "p(DILI) < 0.30",
    Endpoint.CARCINOGENS:   "p(carcinogen) < 0.20",
}


def _passes(endpoint: Endpoint, pred: float) -> bool:
    match endpoint:
        case Endpoint.SOLUBILITY:
            return pred > -5.0
        case Endpoint.BBB:
            return pred < 0.5
        case Endpoint.LIPOPHILICITY:
            return -1.0 <= pred <= 5.0
        case Endpoint.PPBR:
            return pred < 90.0
        case Endpoint.VDSS:
            return pred < 20.0
        case Endpoint.CYP3A4:
            return pred < 0.5
        case Endpoint.HLM:
            return pred > 0.5
        case Endpoint.CLEARANCE:
            return pred < 150.0
        case Endpoint.HERG:
            return pred < 0.3
        case Endpoint.LD50:
            return pred > -2.0
        case Endpoint.DILI:
            return pred < 0.3
        case Endpoint.CARCINOGENS:
            return pred < 0.2
        case _:
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
    cache = EXTERNAL / f"chembl_drug_like_{n}.smi"
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

    logger.info("Downloading %d drug-like compounds from ChEMBL (MW ≤ 500, logP ≤ 5) ...", n)
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
                "molecule_properties__full_mwt__lte": 500,  # MW (molecular weight) ≤ 500 Da
                "molecule_properties__alogp__lte": 5,       # alogP (atomic logP: lipophilicity from atom contributions) ≤ 5
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
    cache = PROC_FEATURES / f"chembl_drug_like_{n}_features.npz"
    smiles_file = EXTERNAL / f"chembl_drug_like_{n}.smi"
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


def _training_set_canonical_smiles() -> set[str]:
    """Return canonical SMILES of every molecule seen during training."""
    seen: set[str] = set()
    for ep in Endpoint:
        path = INTERIM / f"{ep}.parquet"
        if not path.exists():
            continue
        for smi in pd.read_parquet(path)["Drug"]:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                seen.add(Chem.MolToSmiles(mol))
    logger.info("Training set: %d unique canonical SMILES across all endpoints", len(seen))
    return seen


def screen(smiles: list[str], n: int) -> dict:
    """Run all 12 models on SMILES, apply thresholds, return structured results."""
    X, valid_smiles = _load_or_featurize(smiles, n)
    n_valid = len(valid_smiles)
    training_smiles = _training_set_canonical_smiles()

    models = {ep: _load_model(ep) for ep in Endpoint}
    # shape (n_molecules,) per endpoint
    # classification → p(positive class), range [0, 1]
    # regression     → physical units after expm1 for log-transformed endpoints (Clearance only)
    #                   Solubility: logS (~[-10, 2]), Lipophilicity: logD (~[-2, 7]), Clearance: mL/min/kg (~[0, 300])
    preds: dict[Endpoint, np.ndarray] = {
        ep: _predict(models[ep], X, ep) for ep in Endpoint
    }

    pass_masks: dict[Endpoint, np.ndarray] = {
        ep: np.array([_passes(ep, float(p)) for p in preds[ep]]) for ep in Endpoint
    }
    n_filters_passed: np.ndarray = np.stack(
        [pass_masks[ep].astype(int) for ep in Endpoint]
    ).sum(axis=0)
    all_pass: np.ndarray = n_filters_passed == len(list(Endpoint))

    rows = []
    for i, smi in enumerate(valid_smiles):
        mol = Chem.MolFromSmiles(smi)
        canonical = Chem.MolToSmiles(mol) if mol is not None else smi
        row: dict = {"smiles": smi}
        for ep in Endpoint:
            row[f"pred_{ep}"] = float(preds[ep][i])
            row[f"pass_{ep}"] = bool(pass_masks[ep][i])
        row["n_filters_passed"] = int(n_filters_passed[i])
        row["is_hit"] = bool(all_pass[i])
        row["is_near_miss"] = int(n_filters_passed[i]) == len(list(Endpoint)) - 1
        row["in_training_set"] = canonical in training_smiles
        rows.append(row)

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
        "molecules": rows,
    }


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Virtual screening against ChEMBL drug-like compounds")
    parser.add_argument(
        "--n-compounds", type=int, default=100_000,
        help="Number of compounds to download from ChEMBL (default 100000). Usage: N=value",
    )
    args = parser.parse_args()

    try:
        smiles = _fetch_chembl(args.n_compounds)
        source = f"ChEMBL drug-like (n={args.n_compounds})"
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
    logger.info("\nAll 12 filters passed: %d compounds (%.1f%%)", results["n_hits"], results["pct_hits"])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = pd.DataFrame(results["molecules"])
    n_in_training = int(df["in_training_set"].sum())
    csv_path = REPORTS_SCREENING / f"screen_{ts}.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Molecules → %s  (%d hits, %d near-misses)",
                csv_path.name,
                results["n_hits"],
                int(df["is_near_miss"].sum()))
    logger.info("In training set: %d / %d  (%.1f%%)",
                n_in_training, results["n_valid"],
                100 * n_in_training / results["n_valid"] if results["n_valid"] else 0.0)

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
    logger.info("Summary → %s", report_path)


if __name__ == "__main__":
    main()
