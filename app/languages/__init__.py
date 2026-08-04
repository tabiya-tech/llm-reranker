"""Language registry — the one place to edit when adding a language.

One deployed image ranks jobs in any registered language. The language is an *input*:

* ``--language es`` on the batch CLI, or ``RERANKER_LANGUAGE`` / ``TARGET_LANGUAGE`` in the
  Cloud Run Job's env overrides — which is what the ``job-pipeline`` GCP Workflow sets
  per country.
* ``language`` on the ``POST /rank`` body for the single-job service.
* Unset, it is derived from ``TARGET_COUNTRY`` (``argentina`` → ``es``), so the existing
  per-country wiring needs no extra variable.

What the language selects:

* the resource pack under ``resources/<language>/`` — taxonomy label CSVs, the prompt
  files, and the job-attributes schema;
* the language the prompts instruct the model to answer in.

Each registered language has an ``app/languages/{lang}_config.py`` exporting
``LANGUAGE_CONFIG``; same shape as ``scraper/config/{country}_config.py``.

Adding a language:

1. Add its code to :data:`LANGUAGE_REGISTRY`.
2. Add ``app/languages/{code}_config.py`` exporting ``LANGUAGE_CONFIG``.
3. Build its resource pack:
   ``python scripts/build_language_resources.py --taxonomy-dir <export> --language {code}``
"""

from __future__ import annotations

import importlib
import logging
import os
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# ── Language registry ─────────────────────────────────────────────────────────
# The ONLY place you need to edit when adding a new language. Each entry must have a
# matching `app/languages/{code}_config.py` exporting LANGUAGE_CONFIG.
LANGUAGE_REGISTRY: tuple[str, ...] = ("en", "es")

DEFAULT_LANGUAGE = "en"

# Language a country's job ads are written in, used when no language is given
# explicitly. Countries absent here default to English.
LANGUAGE_BY_COUNTRY: dict[str, str] = {
    "zambia": "en",
    "kenya": "en",
    "south_africa": "en",
    "argentina": "es",
}


def _key(value: str) -> str:
    return str(value).strip().lower().replace("_", "-")


@lru_cache(maxsize=None)
def get_language_config(language: str) -> dict[str, Any]:
    """``LANGUAGE_CONFIG`` for a registered language code.

    :raises ValueError: if the language is not in :data:`LANGUAGE_REGISTRY`.
    """
    code = _key(language)
    if code not in LANGUAGE_REGISTRY:
        raise ValueError(
            f"Unsupported language {language!r}. Registered: {', '.join(LANGUAGE_REGISTRY)}"
        )
    mod = importlib.import_module(f"app.languages.{code}_config")
    # Config modules read env at import time; reloading on a cache miss means
    # `get_language_config.cache_clear()` re-reads the environment (what tests expect).
    mod = importlib.reload(mod)
    return dict(mod.LANGUAGE_CONFIG)


@lru_cache(maxsize=None)
def _locale_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for code in LANGUAGE_REGISTRY:
        index[code] = code
        for alias in get_language_config(code).get("locales", ()):
            index[_key(alias)] = code
    return index


def normalise_language(value: str | None, *, fallback: str = DEFAULT_LANGUAGE) -> str:
    """Resolve any locale spelling to a registered language code.

    Accepts a bare code (``es``), a taxonomy locale (``AR-es`` — the ``LOCALE`` column of
    ``model_info.csv``), a POSIX locale (``es_AR.UTF-8``), a country slug (``argentina``),
    or an English language name (``spanish``). Unknown values fall back with a warning
    rather than raising: a bad value should not fail a batch that would otherwise run.
    """
    if value is None:
        return fallback
    key = _key(value)
    if not key:
        return fallback
    index = _locale_index()
    if key in index:
        return index[key]
    for part in (key.split(".")[0], *reversed(key.split(".")[0].split("-"))):
        if part in index:
            return index[part]
    logger.warning(
        "Unknown language %r; using %r (registered: %s)",
        value,
        fallback,
        ", ".join(LANGUAGE_REGISTRY),
    )
    return fallback


def language_from_env(country: str | None = None) -> str:
    """Language for this run.

    Order: ``RERANKER_LANGUAGE`` → ``TARGET_LANGUAGE`` → the country's language
    (:data:`LANGUAGE_BY_COUNTRY`) → ``en``.
    """
    raw = (os.environ.get("RERANKER_LANGUAGE") or os.environ.get("TARGET_LANGUAGE") or "").strip()
    if raw:
        return normalise_language(raw)
    if country:
        derived = LANGUAGE_BY_COUNTRY.get(_key(country).replace("-", "_"))
        if derived:
            return derived
        # An unmapped country is not an error — it just means English until someone
        # adds it to LANGUAGE_BY_COUNTRY or sets RERANKER_LANGUAGE.
    return DEFAULT_LANGUAGE


def language_setting(language: str, key: str, default: Any = None) -> Any:
    """One value out of a language's ``LANGUAGE_CONFIG``."""
    return get_language_config(language).get(key, default)


__all__ = [
    "DEFAULT_LANGUAGE",
    "LANGUAGE_BY_COUNTRY",
    "LANGUAGE_REGISTRY",
    "get_language_config",
    "language_from_env",
    "language_setting",
    "normalise_language",
]
