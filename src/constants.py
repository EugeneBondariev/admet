"""Endpoint definitions, TDC dataset names, and task types."""
from __future__ import annotations

from enum import StrEnum


class Endpoint(StrEnum):
    SOLUBILITY = "ESOL"
    LIPOPHILICITY = "Lipophilicity_AstraZeneca"
    CYP3A4 = "CYP3A4_Veith"
    CLEARANCE = "Clearance_Hepatocyte_AZ"
    HERG = "hERG"


class Task(StrEnum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


ENDPOINT_TASK: dict[Endpoint, Task] = {
    Endpoint.SOLUBILITY: Task.REGRESSION,
    Endpoint.LIPOPHILICITY: Task.REGRESSION,
    Endpoint.CYP3A4: Task.CLASSIFICATION,
    Endpoint.CLEARANCE: Task.REGRESSION,
    Endpoint.HERG: Task.CLASSIFICATION,
}

ENDPOINT_LABEL: dict[Endpoint, str] = {
    Endpoint.SOLUBILITY: "logS (mol/L)",
    Endpoint.LIPOPHILICITY: "logD (pH 7.4)",
    Endpoint.CYP3A4: "CYP3A4 inhibitor (binary)",
    Endpoint.CLEARANCE: "hepatic clearance (mL/min/kg)",
    Endpoint.HERG: "hERG blocker (binary)",
}
