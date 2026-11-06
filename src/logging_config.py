"""Colored terminal logging formatter.

Endpoint headers (--- name ---) are bold cyan.
Per-fold metric lines are green.
Warnings are yellow, errors bold red.
"""
from __future__ import annotations

import logging
import sys

import colorama

colorama.init(strip=False, convert=False)

_RESET = "\033[0m"
_ENDPOINT = "\033[1;36m"   # bold cyan
_FOLD = "\033[32m"         # green
_WARN = "\033[33m"         # yellow
_ERROR = "\033[1;31m"      # bold red


class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        raw = record.getMessage()
        if record.levelno >= logging.ERROR:
            return f"{_ERROR}{msg}{_RESET}"
        if record.levelno == logging.WARNING:
            return f"{_WARN}{msg}{_RESET}"
        if raw.lstrip().startswith(("---", "===")):
            return f"{_ENDPOINT}{msg}{_RESET}"
        if raw.startswith("fold "):
            return f"{_FOLD}{msg}{_RESET}"
        return msg


def setup_logging(level: int = logging.INFO) -> None:
    logging.root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColorFormatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    logging.root.setLevel(level)
    logging.root.addHandler(handler)
