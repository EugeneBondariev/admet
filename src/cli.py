"""Shared CLI helpers for ADMET entry points."""
from __future__ import annotations

import argparse

from src.constants import Endpoint


def resolve_endpoints(parser: argparse.ArgumentParser, name: str | None) -> list[Endpoint]:
    """Resolve --endpoint CLI argument to a list of Endpoint values.

    Accepts a short key prefix (e.g. 'carc', 'cle') or the full TDC dataset name.
    Calls parser.error() (exits with code 2) on ambiguous or unknown input.
    """
    if name is None:
        return list(Endpoint)

    prefix = name.lower()
    matches = [e for e in Endpoint if e.name.lower().startswith(prefix) or e.value == name]
    if len(matches) == 1:
        return matches
    if len(matches) > 1:
        parser.error(f"{name!r} is ambiguous: {', '.join(e.name.lower() for e in matches)}")
    valid = ", ".join(e.name.lower() for e in Endpoint)
    parser.error(f"Unknown endpoint {name!r}. Valid short keys: {valid}")
