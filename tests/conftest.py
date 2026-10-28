"""Shared pytest fixtures and bootstrap for the llm-reranker test suite.

Goals:
- Make ``import app.*`` and ``import utils.*`` work regardless of where ``pytest``
  is invoked from (CI runs ``python -m pytest tests/`` at ``llm-reranker/``).
- Give every test a clean, deterministic environment that does not depend on
  whatever happens to be in the developer's local ``.env`` file or shell. The repo
  ``.env`` is loaded (without overriding already-exported variables) at import time of
  ``app.config``; tests that rely on env defaults must therefore wipe the relevant
  variables and clear the ``Settings`` ``lru_cache``.
- Never reach the network: tests must not require MongoDB or Gemini.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterator

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# app.job_context.JobDescriptionContext has a TYPE_CHECKING-only forward ref to
# RankRequest (to dodge a circular import with app.job_ranking.routes.types) that only
# gets resolved as a side effect of build_job_description_context() running. Tests that
# construct JobDescriptionContext directly need it resolved up front, or they fail when
# collected/run before anything has called that function — regardless of test file order.
from app.job_context import JobDescriptionContext  # noqa: E402

JobDescriptionContext._ensure_rebuilt()


# Subset of env vars that ``app.config.Settings.from_env`` reads. We clear all of
# them between tests so callers get a predictable baseline (defaults from code).
_RERANKER_ENV_VARS = (
    "RERANKER_LANGUAGE",
    "TARGET_LANGUAGE",
    "RERANKER_PREFER_TRANSLATION",
    "RERANKER_LLM2_EXTRA_FIELDS",
    "MONGODB_URI",
    "MONGODB_DATABASE",
    "MONGODB_SOURCE_COLLECTION",
    "MONGODB_CLASSIFIED_COLLECTION",
    "MONGODB_TARGET_COLLECTION",
    "RERANKER_LOGS_COLLECTION",
    "RERANKER_STATE_COLLECTION",
    "MONGODB_TLS_CA_FILE",
    "MONGODB_TLS_INSECURE",
    "RESOURCES_DIR",
    "LLM_RERANKER_RESOURCES_DIR",
    "JOB_ATTRIBUTES_SCHEMA",
    "JOB_ATTRIBUTES_SCHEMA_FILE",
    "OCCUPATION_TO_SKILL_RELATIONS_CSV",
    "OCCUPATIONS_CSV",
    "OCCUPATIONS_CSV_LABEL_COLUMN",
    "OCCUPATIONS_CSV_INCLUDE_DESCRIPTIONS",
    "OCCUPATIONS_CSV_DESCRIPTION_COLUMN",
    "SKILLS_CSV",
    "SKILL_HIERARCHY_CSV",
    "SKILLS_CSV_INCLUDE_DESCRIPTIONS",
    "SKILLS_CSV_DESCRIPTION_COLUMN",
    "CLASSIFIED_MONGO_FIELD_JOB_ID",
    "CLASSIFIED_MONGO_FIELD_JOB_FINGERPRINT",
    "CLASSIFIED_KEYS_DESCRIPTION",
    "CLASSIFIED_KEYS_REQUIREMENTS",
    "CLASSIFIED_KEYS_SALARY",
    "CLASSIFIED_KEYS_LOCATION",
    "CLASSIFIED_KEYS_COUNTY",
    "CLASSIFIED_KEYS_EMPLOYMENT",
    "CLASSIFIED_KEYS_FINGERPRINT",
    "CLASSIFIED_FIELD_TITLE",
    "CLASSIFIED_FIELD_EMPLOYER",
    "CLASSIFIED_FIELD_CATEGORY",
    "CLASSIFIED_FIELD_APPLICATION_URL",
    "CLASSIFIED_FIELD_POSTED_DATE",
    "CLASSIFIED_FIELD_CLOSING_DATE",
    "CLASSIFIED_FIELD_SOURCE_PLATFORM",
    "GEMINI_API_KEY",
    "API_KEY",
    "GEMINI_MODEL",
    "VERTEX_MODEL",
    "GCP_PROJECT_ID",
    "GOOGLE_CLOUD_PROJECT",
    "GCP_LOCATION",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "VERTEX_SERVICE_ACCOUNT_JSON",
    "BULK_WRITE_BATCH_SIZE",
    "RERANKER_LLM3_MAX_MENU_SKILLS",
    "RERANKER_LLM3_CLASSIFIER_INTERSECT",
    "RERANKER_LLM3_MENU_SIGNAL_FILTER",
    "RERANKER_OCCUPATION_SKILLS_SIGNAL_INCLUDE",
    "OCCUPATION_RELATION_LOOKUP",
    "OCCUPATION_RELATION_ESCO_FALLBACK",
    "OCCUPATION_SKILL_RELATION_OCCUPATION_TYPES",
    "RERANKER_PARALLEL_WORKERS",
    "RERANKER_OVERLAP_LLM1_LLM2",
    "RERANKER_SKIP_ALREADY_RANKED",
    "RERANKER_ENABLE_JOB_EMBEDDING",
    "RERANKER_FAIL_ON_JOB_EMBEDDING_ERROR",
    "RERANKER_JOB_EMBEDDING_MODEL",
    "RERANKER_JOB_EMBEDDING_DIM",
    "RERANKER_JOB_EMBEDDING_TASK_TYPE",
    "RERANKER_JOB_EMBEDDING_L2_NORMALIZE",
    "RERANKER_JOB_EMBEDDING_BATCH_SIZE",
    "RERANKER_JOB_EMBEDDING_SLEEP_S",
    "RERANKER_JOB_EMBEDDING_SOURCE",
    "RERANKER_CONCAT_WHITENING_ENABLED",
    "RERANKER_CONCAT_WHITENING_PATH",
    "RERANKER_CONCAT_WHITENING_REQUIRED",
    "RERANKER_CONCAT_WHITENING_EXPECTED_SHA256",
    "CLASSIFIED_LINKED_ENTITY_TYPES",
    "CLASSIFIED_LINKED_ENTITY_MIN_SIMILARITY",
    "CLASSIFIER_LINKED_ENTITY_MIN_SIMILARITY",
    "CLASSIFIED_LINKED_ENTITY_MIN_SIMILARITY_STRICT",
    "CLASSIFIED_LINKED_ENTITY_LABEL_DEDUPE_TYPES",
    "TARGET_COUNTRY",
    "RERANKER_ENABLE_LLM3",
    "RERANKER_LLM3_MODE",
    "RERANKER_RUN_WA_ENRICHMENT",
    "RERANKER_SKIP_ALREADY_RANKED",
    "RERANKER_FAIL_ON_WA_ENRICHMENT_ERROR",
    "RERANKER_SYNC_IS_ACTIVE_FROM_CLOSING_DATE",
    "RERANKER_CUTOFF_DATE",
    "KENYA_RERANKER_CUTOFF_DATE",
    "ZAMBIA_RERANKER_CUTOFF_DATE",
    "GEMINI_TRANSIENT_MAX_ATTEMPTS",
    "RERANKER_LLM_PROVIDER",
    "RERANKER_LLM_CLIENT_CLASS",
    "RERANKER_LLM_MODEL",
    "RERANKER_LLM_API_KEY",
    "RERANKER_LLM_BASE_URL",
    "RERANKER_LLM_TEMPERATURE",
    "RERANKER_LLM_MAX_OUTPUT_TOKENS",
    "RERANKER_LLM_TIMEOUT_S",
    "RERANKER_LLM_MAX_ATTEMPTS",
    "RERANKER_LLM_OPTIONS",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "DASHSCOPE_API_KEY",
    "QWEN_API_KEY",
    "OLLAMA_API_KEY",
    "OLLAMA_BASE_URL",
    "OLLAMA_HOST",
    "RERANKER_ENABLE_TRACING",
    "RERANKER_LANGFUSE_PUBLIC_KEY",
    "RERANKER_LANGFUSE_SECRET_KEY",
    "RERANKER_LANGFUSE_HOST",
    "RERANKER_TRACING_ENVIRONMENT",
    "RERANKER_TRACING_RELEASE",
    "RERANKER_TRACING_CONFIG",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
    "LANGFUSE_BASE_URL",
    "K_REVISION",
    "LOG_LEVEL",
    "ENRICH_WA_DATABASE",
    "ENRICH_WA_TARGET_COLLECTION",
    "MONGODB_DATABASE_REFERENCE",
    "MONGODB_TARGET_COLLECTION_New",
    "ENRICH_WA_OCCUPATION_DB_PATH",
    "ENRICH_WA_BULK_BATCH_SIZE",
    "ENRICH_WA_LIMIT",
    "ENRICH_WA_DRY_RUN",
    "ENRICH_WA_ONLY_MISSING",
    "ENRICH_WA_DEFAULT_IMPORTANCE",
    "ENRICH_WA_DEFAULT_LEVEL",
    "RERANKER_LOGS_COLLECTION",
    "RERANKER_STATE_COLLECTION",
)


def _clear_settings_caches() -> None:
    """Drop ``lru_cache`` entries so ``Settings.from_env`` is re-evaluated."""
    from app import config as app_config

    app_config.get_settings.cache_clear()
    app_config.get_settings_for_language.cache_clear()
    try:
        from app import languages as app_languages

        # Language configs read env at import time; clearing lets a test change
        # RERANKER_LANGUAGE and per-language overrides and see them applied.
        app_languages.get_language_config.cache_clear()
        app_languages._locale_index.cache_clear()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _tracing_off(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep Langfuse off in every test, whatever the developer's ``.env`` says.

    ``api.lifespan`` initialises tracing from the environment; without this a local
    ``RERANKER_ENABLE_TRACING=1`` would make the test suite export traces. Tests that
    assert on spans use ``app.observability.testing.in_memory_tracing`` instead.
    """
    monkeypatch.setenv("RERANKER_ENABLE_TRACING", "0")
    yield
    from app.observability import tracing

    tracing._client = None
    tracing._config = tracing.TracingConfig()
    from app.job_ranking.get_job_ranking_service import reset_job_ranking_services

    # The API's ranking services are a process-wide singleton; a test must not inherit another's.
    reset_job_ranking_services()


@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    """Provide a clean reranker env: every documented env var is removed.

    Yields the same ``monkeypatch`` so the test can ``setenv`` the specific
    variables it cares about. ``Settings.from_env`` then sees only those.

    Caches are cleared on entry and exit so cached settings cannot leak
    across tests.
    """
    from app.config import LLM2_EXTRA_FIELD_ENV_PREFIX

    for name in _RERANKER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    # Each LLM2 extra field's text has its own variable, named after the field.
    for name in [n for n in os.environ if n.startswith(LLM2_EXTRA_FIELD_ENV_PREFIX)]:
        monkeypatch.delenv(name, raising=False)
    _clear_settings_caches()
    yield monkeypatch
    _clear_settings_caches()


@pytest.fixture
def make_settings(isolated_env: pytest.MonkeyPatch):
    """Build a ``Settings`` instance from a kwargs-style env override map.

    Example::

        s = make_settings(TARGET_COUNTRY="kenya", GEMINI_API_KEY="abc")
    """
    from app.config import Settings

    def _build(**env_overrides: str):
        for k, v in env_overrides.items():
            isolated_env.setenv(k, v)
        return Settings.from_env()

    return _build
