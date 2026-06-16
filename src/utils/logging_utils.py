"""Small logging helper so every module logs in a consistent format."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring the root handler once.

    Output goes to stdout with a compact ``HH:MM:SS | LEVEL | name | message``
    layout. UTF-8 reconfiguration keeps team names with accents readable on the
    Windows console.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.addHandler(handler)
        _CONFIGURED = True
    return logging.getLogger(name)
