"""Central logging setup (``LOG_LEVEL``/``LOG_FORMAT`` from environment) and shared pipeline step labels."""

from __future__ import annotations

import logging
import os
import re

from app.llm.base import LlmConfig
from app.logger import JsonLogFormatter

# ---------------------------------------------------------------------------
# Pipeline step labels (used in main.py logs; reusable in tests / Cloud Run)
# ---------------------------------------------------------------------------
STEP_OCCUPATION_TAXONOMY = "occupation taxonomy (classifier + CSV relations)"
STEP_MONGO_UPSERT = "MongoDB — build ranked payload + queue upsert"


def step_llm1_skill_ranker(llm: LlmConfig) -> str:
    """Step label for LLM call 1, naming the provider/model that actually ran."""
    return f"LLM call 1 — skill ranker ({llm.describe()})"


def step_llm2_job_attributes(llm: LlmConfig) -> str:
    """Step label for LLM call 2, naming the provider/model that actually ran."""
    return f"LLM call 2 — job attributes + ONET work activities ({llm.describe()})"


def step_llm3_occ_skills(llm: LlmConfig) -> str:
    """Step label for LLM call 3, naming the provider/model that actually ran."""
    return f"LLM call 3 — occupation-skills listing relevance ({llm.describe()})"


def sanitize_mongodb_uri(uri: str) -> str:
    """Mask credentials in a MongoDB URI for safe log lines."""
    if not uri or "@" not in uri:
        return uri
    return re.sub(r":([^/@?]+)@", r":****@", uri, count=1)


_LEVEL_NAMES = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def _log_level_from_env() -> int:
    """
    ``LOG_LEVEL``: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO). ``WARN`` is
    accepted as the alias Python's own ``logging`` module uses.
    """
    raw = (os.environ.get("LOG_LEVEL") or "INFO").strip().upper()
    if raw == "WARN":
        raw = "WARNING"
    if raw not in _LEVEL_NAMES:
        logging.getLogger(__name__).warning(
            "LOG_LEVEL=%r is not one of DEBUG/INFO/WARNING/ERROR/CRITICAL; using INFO", raw
        )
        raw = "INFO"
    return int(getattr(logging, raw))


def _log_format_from_env() -> str:
    """``LOG_FORMAT``: ``text`` (default, human-readable) or ``json`` (one object per line)."""
    raw = (os.environ.get("LOG_FORMAT") or "text").strip().lower()
    if raw not in ("text", "json"):
        logging.getLogger(__name__).warning(
            "LOG_FORMAT=%r is not 'text' or 'json'; using 'text'", raw
        )
        raw = "text"
    return raw


def configure_logging() -> None:
    """
    Configure root logging once, from ``LOG_LEVEL`` and ``LOG_FORMAT``. Call from
    ``main`` before other app code — in particular, before the first ``get_settings()``
    / ``Settings.from_env()`` call: ``main.py`` exports ``RERANKER_LANGUAGE`` from
    ``--language`` only after this runs, and ``get_settings()`` is cached, so building
    ``Settings`` here first would freeze the language at whatever the environment said
    before ``--language`` was parsed. Both variables are therefore read directly from
    the environment rather than through ``Settings``.
    """
    if logging.getLogger().handlers:
        return
    level = _log_level_from_env()
    if _log_format_from_env() == "json":
        handler = logging.StreamHandler()
        handler.setFormatter(JsonLogFormatter())
        logging.basicConfig(level=level, handlers=[handler])
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    logging.getLogger("google_genai.models").setLevel(logging.WARNING)


def reconfigure_uvicorn_logging() -> None:
    """Route uvicorn's own log lines through the app's handler (uvicorn installs its own)."""
    root_handlers = logging.getLogger().handlers
    if not root_handlers:
        return
    for name in ("uvicorn", "uvicorn.access"):
        target = logging.getLogger(name)
        target.handlers = list(root_handlers)
        target.propagate = False
