"""Endpoint definitions, TDC dataset names, and task types."""
from __future__ import annotations

from enum import StrEnum


class TDCGroup(StrEnum):
    ADME = "ADME"
    TOX = "Tox"


class Endpoint(StrEnum):
    """Declared in ADMET order: Absorption, Distribution (x3), Metabolism (x2),
    Excretion, Toxicity (x4). Values are the exact TDC dataset names."""

    # Absorption
    SOLUBILITY = "Solubility_AqSolDB"
    BBB = "BBB_Martins"
    # Distribution
    LIPOPHILICITY = "Lipophilicity_AstraZeneca"
    PPBR = "PPBR_AZ"
    VDSS = "VDss_Lombardo"
    # Metabolism
    CYP3A4 = "CYP3A4_Veith"
    HLM = "hlm"
    # Excretion
    CLEARANCE = "Clearance_Hepatocyte_AZ"
    # Toxicity
    HERG = "herg_karim"
    LD50 = "LD50_Zhu"
    DILI = "DILI"
    CARCINOGENS = "Carcinogens_Lagunin"


ENDPOINT_TDC_GROUP: dict[Endpoint, TDCGroup] = {
    Endpoint.SOLUBILITY: TDCGroup.ADME,
    Endpoint.BBB: TDCGroup.ADME,
    Endpoint.LIPOPHILICITY: TDCGroup.ADME,
    Endpoint.PPBR: TDCGroup.ADME,
    Endpoint.VDSS: TDCGroup.ADME,
    Endpoint.CYP3A4: TDCGroup.ADME,
    Endpoint.HLM: TDCGroup.ADME,
    Endpoint.CLEARANCE: TDCGroup.ADME,
    Endpoint.HERG: TDCGroup.TOX,
    Endpoint.LD50: TDCGroup.TOX,
    Endpoint.DILI: TDCGroup.TOX,
    Endpoint.CARCINOGENS: TDCGroup.TOX,
}


class Task(StrEnum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


ENDPOINT_TASK: dict[Endpoint, Task] = {
    Endpoint.SOLUBILITY: Task.REGRESSION,
    Endpoint.BBB: Task.CLASSIFICATION,
    Endpoint.LIPOPHILICITY: Task.REGRESSION,
    Endpoint.PPBR: Task.REGRESSION,
    Endpoint.VDSS: Task.REGRESSION,
    Endpoint.CYP3A4: Task.CLASSIFICATION,
    Endpoint.HLM: Task.CLASSIFICATION,
    Endpoint.CLEARANCE: Task.REGRESSION,
    Endpoint.HERG: Task.CLASSIFICATION,
    Endpoint.LD50: Task.REGRESSION,
    Endpoint.DILI: Task.CLASSIFICATION,
    Endpoint.CARCINOGENS: Task.CLASSIFICATION,
}

# Endpoints whose raw Y is log1p-transformed before training and expm1-reversed at prediction.
# Use when Y values span multiple orders of magnitude (e.g. 0.04–150 L/kg for VDss).
LOG_TRANSFORM: frozenset[Endpoint] = frozenset({Endpoint.CLEARANCE, Endpoint.VDSS})

ENDPOINT_LABEL: dict[Endpoint, str] = {
    # higher → more soluble (good)
    Endpoint.SOLUBILITY: "logS (mol/L)",
    # Y=1 → crosses blood-brain barrier (good for CNS drugs, undesirable otherwise)
    Endpoint.BBB: "BBB penetrant (binary)",
    # sweet spot -1 to 3; too low → can't cross membranes; too high → won't dissolve
    Endpoint.LIPOPHILICITY: "logD (pH 7.4)",
    # lower → more free (unbound) drug in plasma → more available to reach target (good)
    Endpoint.PPBR: "plasma protein binding rate (%)",
    # typical therapeutic range 0.6–20 L/kg; very high → accumulates in tissues
    Endpoint.VDSS: "volume of distribution (L/kg)",
    # Y=1 → inhibitor (bad)
    Endpoint.CYP3A4: "CYP3A4 inhibitor (binary)",
    # Y=1 → stable (good)
    Endpoint.HLM: "human liver microsomal stability (binary)",
    # lower → slower removal → longer half-life (good)
    Endpoint.CLEARANCE: "hepatic clearance (mL/min/kg)",
    # Y=1 → blocker (bad)
    Endpoint.HERG: "hERG blocker (binary)",
    # higher → less toxic (needs more mol/kg to kill); Y is log(LD50 mol/kg)
    Endpoint.LD50: "acute toxicity LD50 (log mol/kg)",
    # Y=1 → causes liver injury (bad)
    Endpoint.DILI: "drug-induced liver injury (binary)",
    # Y=1 → carcinogen (bad)
    Endpoint.CARCINOGENS: "carcinogenicity (binary)",
}
