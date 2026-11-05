"""Endpoint definitions, TDC dataset names, and task types."""
from __future__ import annotations

from enum import StrEnum


class TDCGroup(StrEnum):
    ADME = "ADME"
    TOX = "Tox"


class Endpoint(StrEnum):
    """Declared in ADMET order: Absorption, Distribution, Metabolism (x2),
    Excretion, Toxicity. Values are the exact TDC dataset names."""

    SOLUBILITY = "Solubility_AqSolDB"
    LIPOPHILICITY = "Lipophilicity_AstraZeneca"
    CYP3A4 = "CYP3A4_Veith"
    HLM = "hlm"
    CLEARANCE = "Clearance_Hepatocyte_AZ"
    HERG = "herg_karim"


ENDPOINT_TDC_GROUP: dict[Endpoint, TDCGroup] = {
    Endpoint.SOLUBILITY: TDCGroup.ADME,
    Endpoint.LIPOPHILICITY: TDCGroup.ADME,
    Endpoint.CYP3A4: TDCGroup.ADME,
    Endpoint.HLM: TDCGroup.ADME,
    Endpoint.CLEARANCE: TDCGroup.ADME,
    Endpoint.HERG: TDCGroup.TOX,
}


class Task(StrEnum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


ENDPOINT_TASK: dict[Endpoint, Task] = {
    Endpoint.SOLUBILITY: Task.REGRESSION,
    Endpoint.LIPOPHILICITY: Task.REGRESSION,
    Endpoint.CYP3A4: Task.CLASSIFICATION,
    Endpoint.HLM: Task.CLASSIFICATION,
    Endpoint.CLEARANCE: Task.REGRESSION,
    Endpoint.HERG: Task.CLASSIFICATION,
}

LOG_TRANSFORM: frozenset[Endpoint] = frozenset({Endpoint.CLEARANCE})

ENDPOINT_LABEL: dict[Endpoint, str] = {
    Endpoint.SOLUBILITY: "logS (mol/L)",
    Endpoint.LIPOPHILICITY: "logD (pH 7.4)",
    Endpoint.CYP3A4: "CYP3A4 inhibitor (binary)",
    Endpoint.HLM: "human liver microsomal stability (binary)",
    Endpoint.CLEARANCE: "hepatic clearance (mL/min/kg)",
    Endpoint.HERG: "hERG blocker (binary)",
}
