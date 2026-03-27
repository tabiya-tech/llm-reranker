"""Central logging setup (``LOG_LEVEL`` from environment)."""

from __future__ import annotations

import logging
import os
import re


def sanitize_mongodb_uri(uri: str) -> str:
    """Mask credentials in a MongoDB URI for safe log lines."""
    if not uri or "@" not in uri:
        return uri
    return re.sub(r":([^/@?]+)@", r":****@", uri, count=1)


def configure_logging() -> None:
    """
    Configure root logging once. Call from ``main`` before other app code.

    ``LOG_LEVEL``: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO).
    """
    if logging.getLogger().handlers:
        return
    raw = (os.environ.get("LOG_LEVEL") or "INFO").strip().upper()
    if raw not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raw = "INFO"
    level = getattr(logging, raw)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
