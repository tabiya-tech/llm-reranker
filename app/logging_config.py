"""Central logging setup (``LOG_LEVEL`` from environment) and shared pipeline step labels."""

from __future__ import annotations

import logging
import os
import re

# ---------------------------------------------------------------------------
# Pipeline step labels (used in main.py logs; reusable in tests / Cloud Run)
# ---------------------------------------------------------------------------
STEP_OCCUPATION_TAXONOMY = "occupation taxonomy (classifier + CSV relations)"
STEP_LLM1_SKILL_RANKER = "LLM call 1 — skill ranker (Gemini)"
STEP_LLM2_JOB_ATTRIBUTES = "LLM call 2 — job attributes + ONET work activities (Gemini)"
STEP_LLM3_OCC_SKILLS = "LLM call 3 — occupation-skills listing relevance (Gemini)"
STEP_MONGO_UPSERT = "MongoDB — build ranked payload + queue upsert"


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
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    logging.getLogger("google_genai.models").setLevel(logging.WARNING)
