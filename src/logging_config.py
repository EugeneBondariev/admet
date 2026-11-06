"""Colored terminal logging formatter.

Endpoint headers (--- name ---) are bold cyan.
Per-fold metric lines are green.
Warnings are yellow, errors bold red.
"""
from __future__ import annotations

import ctypes
import logging
import sys

_RESET = "\033[0m"
_ENDPOINT = "\033[1;36m"   # bold cyan
_FOLD = "\033[32m"         # green
_WARN = "\033[33m"         # yellow
_ERROR = "\033[1;31m"      # bold red


def _enable_windows_ansi() -> None:
    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        for handle_id in (-11, -12):  # STD_OUTPUT_HANDLE, STD_ERROR_HANDLE
            handle = kernel32.GetStdHandle(handle_id)
            mode = ctypes.c_ulong()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING)
    except Exception:
        pass


class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if record.levelno >= logging.ERROR:
            return f"{_ERROR}{msg}{_RESET}"
        if record.levelno == logging.WARNING:
            return f"{_WARN}{msg}{_RESET}"
        if msg.startswith("---"):
            return f"{_ENDPOINT}{msg}{_RESET}"
        if msg.startswith("fold "):
            return f"{_FOLD}{msg}{_RESET}"
        return msg


def setup_logging(level: int = logging.INFO) -> None:
    _enable_windows_ansi()
    logging.root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColorFormatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    logging.root.setLevel(level)
    logging.root.addHandler(handler)
