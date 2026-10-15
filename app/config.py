from __future__ import annotations

import logging
import os
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.languages import (
    DEFAULT_LANGUAGE,
    LANGUAGE_REGISTRY,
    get_language_config,
    language_from_env,
    language_setting,
    normalise_language,
)
from app.llm.base import (
    CUSTOM_PROVIDER,
    DEFAULT_TEMPERATURE,
    PROVIDER_API_KEY_ENV_VARS,
    PROVIDER_BASE_URL_ENV_VARS,
    LlmConfig,
    normalise_provider,
    resolve_provider_class_path,
)


def _repo_root() -> Path:
    """Repo root: parent of the ``app/`` package directory."""
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    """Directory containing ``resources/``, ``.env``, and ``scripts/`` (repo root)."""
    return _repo_root()


# override=False (the default): a shell-exported value always wins over .env.
load_dotenv(_repo_root() / ".env")

_logger = logging.getLogger(__name__)

# Total attempts per LLM call, including the first. The historical Gemini path clamped
# ``GEMINI_TRANSIENT_MAX_ATTEMPTS`` to 1..12; that clamp is preserved for every provider.
_LLM_MAX_ATTEMPTS_DEFAULT = 5
_LLM_MAX_ATTEMPTS_CEILING = 12

# Gemini model used when nothing is configured — the reranker's historical default.
_GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"

LLM3_MODE_FOLD = "fold"
LLM3_MODE_CALL = "call"
LLM3_MODES = (LLM3_MODE_FOLD, LLM3_MODE_CALL)

_TARGET_COUNTRY_SLUGS = frozenset({"zambia", "kenya", "south_africa", "argentina"})

# Canonical country name for prompts — keyed by ``TARGET_COUNTRY`` slug.
_COUNTRY_NAME_BY_SLUG: dict[str, str] = {
    "zambia": "Zambia",
    "kenya": "Kenya",
    "south_africa": "South Africa",
    "argentina": "Argentina",
}


def _country_name_from_slug(slug: str) -> str:
    """Human-readable country name derived from ``TARGET_COUNTRY`` only."""
    if slug in _COUNTRY_NAME_BY_SLUG:
        return _COUNTRY_NAME_BY_SLUG[slug]
    return slug.replace("_", " ").title()


def _normalize_classifier_entity_type_token(raw: str) -> str:
    """
    Map common env plural forms to classifier ``entity_type`` values (ESCO uses singular).

    Classifier JSON uses ``skill`` and ``occupation``; env often uses ``skills`` / ``occupations``.
    """
    t = raw.strip().lower()
    if t == "skills":
        return "skill"
    if t == "occupations":
        return "occupation"
    return t


def _comma_keys(value: str | None, default: str) -> tuple[str, ...]:
    """Split comma-separated keys; empty env / whitespace-only falls back to ``default``."""
    raw = (value if value is not None else default).strip()
    out = tuple(k.strip() for k in raw.split(",") if k.strip())
    if not out:
        out = tuple(k.strip() for k in default.strip().split(",") if k.strip())
    return out


def _entity_types_tuple(value: str | None, *, default: str = "skill,occupation") -> tuple[str, ...]:
    """Lowercase entity types from comma-separated env (Classifier ``entity_type`` values)."""
    t = _comma_keys(value, default)
    out: list[str] = []
    seen: set[str] = set()
    for x in t:
        if not str(x).strip():
            continue
        n = _normalize_classifier_entity_type_token(str(x))
        if n not in seen:
            seen.add(n)
            out.append(n)
    return tuple(out)


def _unique_key_order(*groups: tuple[str, ...]) -> tuple[str, ...]:
    """Concatenate key tuples, keeping first occurrence of each field name."""
    seen: set[str] = set()
    out: list[str] = []
    for g in groups:
        for k in g:
            if k not in seen:
                seen.add(k)
                out.append(k)
    return tuple(out)


def language_resource_dirs(resources_dir: Path, language: str) -> tuple[Path, ...]:
    """Directories searched for a resource basename, most specific first.

    ``resources/<language>/`` → ``resources/en/`` → ``resources/``. The English pack is a
    fallback so a language only has to ship the files that actually differ (the AR-es pack
    ships taxonomy labels and prompts, not the ONET reference data). ``resources/`` itself
    stays in the chain for the flat, pre-language layout and for language-neutral
    artifacts such as ``concat_whitening_gemini.npz``.
    """
    subdir = language_setting(language, "resources_subdir", language) or language
    dirs = [resources_dir / subdir]
    en_subdir = (
        language_setting(DEFAULT_LANGUAGE, "resources_subdir", DEFAULT_LANGUAGE) or DEFAULT_LANGUAGE
    )
    if en_subdir != subdir:
        dirs.append(resources_dir / en_subdir)
    dirs.append(resources_dir)
    return tuple(d.resolve() for d in dirs)


def _resolve_language_resource_path(
    repo_root_dir: Path,
    resources_dir: Path,
    language: str,
    raw: str | None,
    default_name: str,
    extra_fallbacks: tuple[Path, ...] = (),
) -> Path:
    """Absolute path for one resource file.

    With an explicit env value: an absolute path is used as given; a relative path is
    tried against each directory in :func:`language_resource_dirs`, then the repo root.
    Without one, ``default_name`` is looked up in the same directory order, then in
    ``extra_fallbacks``.

    When nothing exists, the path inside the *language's own* pack is returned, so a
    "not found" error names the file the pack is missing rather than a fallback.
    """
    candidates = language_resource_dirs(resources_dir, language)

    if raw is not None and str(raw).strip():
        p = Path(str(raw).strip())
        if p.is_absolute():
            return p.resolve()
        for directory in candidates:
            candidate = (directory / p).resolve()
            if candidate.is_file():
                return candidate
        under_root = (repo_root_dir / p).resolve()
        if under_root.is_file():
            return under_root
        return (candidates[0] / p).resolve()

    for directory in candidates:
        candidate = (directory / default_name).resolve()
        if candidate.is_file():
            return candidate
    for fallback in extra_fallbacks:
        if fallback.is_file():
            return fallback.resolve()
    return (candidates[0] / default_name).resolve()


def _resolve_job_attributes_schema_path(
    repo_root_dir: Path,
    resources_dir: Path,
    raw: str | None,
    language: str = DEFAULT_LANGUAGE,
) -> Path:
    """
    Path to the job-attributes JSON (same structure for any number of attributes).

    ``JOB_ATTRIBUTES_SCHEMA`` / ``JOB_ATTRIBUTES_SCHEMA_FILE``: absolute path, or relative
    to the language pack / ``resources/``, or relative to repo root. Default basename comes
    from the language config (``job_attributes_reduced.json``), so a language can ship its
    own translated attribute labels and currency levels.
    """
    default_name = (
        language_setting(language, "job_attributes_schema", "job_attributes_reduced.json")
        or "job_attributes_reduced.json"
    )
    return _resolve_language_resource_path(
        repo_root_dir, resources_dir, language, raw, default_name
    )


def _resolve_occupation_to_skill_relations_path(
    repo_root_dir: Path,
    resources_dir: Path,
    raw: str | None,
    language: str = DEFAULT_LANGUAGE,
) -> Path:
    """
    Path to occupation→skill relations CSV (see ``OCCUPATION_TO_SKILL_RELATIONS_CSV``).

    Default: ``resources/<language>/occupation_to_skill_relations.csv``.
    """
    return _resolve_language_resource_path(
        repo_root_dir, resources_dir, language, raw, "occupation_to_skill_relations.csv"
    )


def _resolve_occupations_csv_path(
    repo_root_dir: Path,
    resources_dir: Path,
    raw: str | None,
    language: str = DEFAULT_LANGUAGE,
) -> Path:
    """
    Tabiya ``occupations.csv``: ``ID`` + ``UUIDHISTORY`` / ``ORIGINURI``.

    Row ``ID`` must match ``OCCUPATIONID`` in ``occupation_to_skill_relations.csv`` — so
    this file and the relations file must come from the *same* language pack.

    ``OCCUPATIONS_CSV``; default ``resources/<language>/occupations.csv``.
    """
    return _resolve_language_resource_path(
        repo_root_dir, resources_dir, language, raw, "occupations.csv"
    )


def _resolve_skills_csv_path(
    repo_root_dir: Path,
    resources_dir: Path,
    raw: str | None,
    language: str = DEFAULT_LANGUAGE,
) -> Path:
    """
    Tabiya ``skills.csv``: ``ID`` + ``PREFERREDLABEL`` + ``UUIDHISTORY`` / ``ORIGINURI``.

    ``SKILLS_CSV``; default ``resources/<language>/skills.csv``.
    """
    return _resolve_language_resource_path(
        repo_root_dir, resources_dir, language, raw, "skills.csv"
    )


def _resolve_concat_whitening_path(
    repo_root_dir: Path,
    resources_dir: Path,
    raw: str | None,
    language: str = DEFAULT_LANGUAGE,
) -> Path:
    """
    Path to concat whitening artifact (``concat_whitening_gemini.npz``).

    Language-neutral (it transforms embeddings, not text), so it lives at ``resources/``.

    ``RERANKER_CONCAT_WHITENING_PATH``; default ``resources/concat_whitening_gemini.npz``.
    """
    return _resolve_language_resource_path(
        repo_root_dir, resources_dir, language, raw, "concat_whitening_gemini.npz"
    )


def _resolve_skill_hierarchy_csv_path(
    repo_root_dir: Path,
    resources_dir: Path,
    raw: str | None,
    language: str = DEFAULT_LANGUAGE,
) -> Path:
    """
    Path to ``skill_hierarchy.csv`` used for skill -> skill-group derivation.

    Resolution order:
      1) explicit env path (absolute or relative)
      2) ``resources/<language>/skill_hierarchy.csv``, then the English pack, then ``resources/``
      3) sibling repo fallback:
         ``../shp-matching-algorithm/backend/resources/skill_taxonomy/skill_hierarchy.csv``
    """
    sibling = (
        repo_root_dir.parent
        / "shp-matching-algorithm"
        / "backend"
        / "resources"
        / "skill_taxonomy"
        / "skill_hierarchy.csv"
    )
    return _resolve_language_resource_path(
        repo_root_dir,
        resources_dir,
        language,
        raw,
        "skill_hierarchy.csv",
        extra_fallbacks=(sibling,),
    )


def _resources_dir(repo_root_dir: Path, raw: str | None) -> Path:
    """
    Versioned prompts and schema live under ``resources/`` at the repo root.

    Override with ``RESOURCES_DIR`` (absolute path or path relative to the repo root),
    e.g. for a private checkout or overlay.
    """
    if raw and str(raw).strip():
        p = Path(str(raw).strip())
        return p.resolve() if p.is_absolute() else (repo_root_dir / p).resolve()
    return (repo_root_dir / "resources").resolve()


def _llm_api_key_from_env(provider: str, generic: str) -> str:
    """API key for ``provider``: the generic variable first, then that provider's own.

    The per-provider fallback is what lets an existing Gemini deployment keep working
    untouched — it already sets ``GEMINI_API_KEY`` and never hears of
    ``RERANKER_LLM_API_KEY``.
    """
    if generic:
        return generic
    for name in PROVIDER_API_KEY_ENV_VARS.get(provider, ()):
        v = (os.environ.get(name) or "").strip()
        if v:
            return v
    return ""


def _llm_base_url_from_env(provider: str, generic: str) -> str:
    """API endpoint for ``provider``: the generic variable first, then that provider's own.

    Only Ollama has one (``OLLAMA_BASE_URL`` / ``OLLAMA_HOST``), which is what lets a
    host that already runs Ollama point the reranker at it without a reranker-specific
    variable. ``OLLAMA_HOST`` is conventionally scheme-less (``127.0.0.1:11434``), so a
    scheme is added here; ``RERANKER_LLM_BASE_URL`` stays strict and is rejected by the
    client when it has none.
    """
    if generic:
        return generic.rstrip("/")
    for name in PROVIDER_BASE_URL_ENV_VARS.get(provider, ()):
        v = (os.environ.get(name) or "").strip()
        if v:
            if not v.startswith(("http://", "https://")):
                v = f"http://{v}"
            return v.rstrip("/")
    return ""


class Settings(BaseSettings):
    """
    Typed, validated application runtime configuration, read from environment variables
    (and ``.env``): Mongo connection, classified job field / key names, the backing LLM,
    language/resource resolution, and the reranker's feature toggles.

    Every field binds to exactly one env var (its uppercased name, e.g. ``mongodb_uri``
    <- ``MONGODB_URI``) and is never itself derived from another field. Derived values
    (``llm``, every ``*_path``, ``language``, ``gemini_api_key``, the folded-in
    ``enrich_wa_*`` values, …) are ``@cached_property`` attributes instead, computed once
    on first access. This split matters: pydantic-settings auto-binds any field without
    an explicit ``validation_alias`` to its own uppercased name regardless of what a
    sibling field already claims, so a field that is both "env-sourced" and "derived from
    other fields" would silently double-bind, and a stray env var matching its name could
    crash construction outright.
    """

    model_config = SettingsConfigDict(
        extra="ignore", env_file=None, arbitrary_types_allowed=True, frozen=True
    )

    @model_validator(mode="before")
    @classmethod
    def _blank_env_values_are_unset(cls, data: object) -> object:
        """A blank env var (``FOO=``) falls back to the field's default instead of
        failing validation on an empty string."""
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if not (isinstance(v, str) and v.strip() == "")}
        return data

    # -----------------------------------------------------------------------
    # Mongo
    # -----------------------------------------------------------------------
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "ZambiaJobs"
    # Legacy alias MONGODB_CLASSIFIED_COLLECTION removed 2026-10-14 (unused, undocumented).
    mongodb_source_collection: str = Field(
        default="classified_jobs",
        validation_alias="MONGODB_SOURCE_COLLECTION",
    )
    mongodb_target_collection: str = "ranked_jobs"
    mongodb_tls_ca_file: str | None = None
    mongodb_tls_insecure: bool = False
    reranker_logs_collection: str = "reranker_logs"
    reranker_state_collection: str = "reranker_state"

    # Legacy alias LLM_RERANKER_RESOURCES_DIR removed 2026-10-14; use RESOURCES_DIR.
    resources_dir_raw: str | None = Field(default=None, validation_alias="RESOURCES_DIR")

    # -----------------------------------------------------------------------
    # Classified job field / key names
    # -----------------------------------------------------------------------
    job_attributes_schema_raw: str | None = Field(
        default=None, validation_alias="JOB_ATTRIBUTES_SCHEMA"
    )
    job_attributes_schema_file_raw: str | None = Field(
        default=None, validation_alias="JOB_ATTRIBUTES_SCHEMA_FILE"
    )
    occupation_to_skill_relations_csv_raw: str | None = Field(
        default=None, validation_alias="OCCUPATION_TO_SKILL_RELATIONS_CSV"
    )
    occupations_csv_raw: str | None = Field(default=None, validation_alias="OCCUPATIONS_CSV")
    occupations_csv_label_column: str = "PREFERREDLABEL"
    occupations_csv_include_descriptions: bool = False
    occupations_csv_description_column: str = ""

    skills_csv_raw: str | None = Field(default=None, validation_alias="SKILLS_CSV")
    skill_hierarchy_csv_raw: str | None = Field(
        default=None, validation_alias="SKILL_HIERARCHY_CSV"
    )
    skills_csv_include_descriptions: bool = False
    skills_csv_description_column: str = ""

    mongo_field_job_id: str = Field(
        default="job_id", validation_alias="CLASSIFIED_MONGO_FIELD_JOB_ID"
    )
    mongo_field_job_fingerprint: str = Field(
        default="job_fingerprint", validation_alias="CLASSIFIED_MONGO_FIELD_JOB_FINGERPRINT"
    )

    keys_description_raw: str | None = Field(
        default=None, validation_alias="CLASSIFIED_KEYS_DESCRIPTION"
    )
    keys_requirements_raw: str | None = Field(
        default=None, validation_alias="CLASSIFIED_KEYS_REQUIREMENTS"
    )
    # Plain strings, not tuple[str, ...]: pydantic-settings JSON-decodes complex field
    # types before any validator runs, so a plain comma list would fail construction.
    # Split in the matching property instead.
    keys_salary_raw: str | None = Field(default=None, validation_alias="CLASSIFIED_KEYS_SALARY")
    keys_location_raw: str | None = Field(default=None, validation_alias="CLASSIFIED_KEYS_LOCATION")
    keys_county_raw: str | None = Field(default=None, validation_alias="CLASSIFIED_KEYS_COUNTY")
    keys_employment_raw: str | None = Field(
        default=None, validation_alias="CLASSIFIED_KEYS_EMPLOYMENT"
    )
    keys_fingerprint_raw: str | None = Field(
        default=None, validation_alias="CLASSIFIED_KEYS_FINGERPRINT"
    )

    field_title: str = Field(default="title", validation_alias="CLASSIFIED_FIELD_TITLE")
    field_employer: str = Field(default="employer", validation_alias="CLASSIFIED_FIELD_EMPLOYER")
    field_category: str = Field(default="category", validation_alias="CLASSIFIED_FIELD_CATEGORY")
    field_application_url: str = Field(
        default="application_url", validation_alias="CLASSIFIED_FIELD_APPLICATION_URL"
    )
    field_posted_date: str = Field(
        default="posted_date", validation_alias="CLASSIFIED_FIELD_POSTED_DATE"
    )
    field_closing_date: str = Field(
        default="closing_date", validation_alias="CLASSIFIED_FIELD_CLOSING_DATE"
    )
    field_source_platform: str = Field(
        default="source_platform", validation_alias="CLASSIFIED_FIELD_SOURCE_PLATFORM"
    )

    # -----------------------------------------------------------------------
    # The reranker's backing LLM. Read settings.llm (built from these raw fields
    # by the `llm` property below), not these fields directly.
    # -----------------------------------------------------------------------
    reranker_llm_client_class: str = ""
    reranker_llm_provider: str = ""
    reranker_llm_model: str = ""
    reranker_llm_api_key: str = ""
    reranker_llm_base_url: str = ""
    reranker_llm_temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    reranker_llm_max_output_tokens: int = Field(default=0, ge=0, le=1_000_000)
    reranker_llm_timeout_s: float = Field(default=0.0, ge=0.0, le=3600.0)
    # Clamped to 1..12 by the `llm` property, not rejected here — degrades, doesn't crash.
    reranker_llm_max_attempts: int | None = None
    # Historical alias consulted when ``RERANKER_LLM_MAX_ATTEMPTS`` is unset.
    gemini_transient_max_attempts: int | None = None
    reranker_llm_options: str = ""

    # gemini_api_key (the property) is read directly by the job-embedding step, which
    # always uses Gemini regardless of RERANKER_LLM_PROVIDER.
    gemini_api_key_raw: str = Field(default="", validation_alias="GEMINI_API_KEY")
    # Legacy alias for GEMINI_API_KEY. Remove on or after 2027-01-01.
    api_key_legacy: str = Field(default="", validation_alias="API_KEY")
    gemini_model_raw: str = Field(default="", validation_alias="GEMINI_MODEL")
    # Legacy alias for GEMINI_MODEL. Remove on or after 2027-01-01.
    vertex_model_legacy: str = Field(default="", validation_alias="VERTEX_MODEL")

    bulk_write_batch_size: int = Field(default=100, ge=1, le=10_000)

    # -----------------------------------------------------------------------
    # LLM3: the occupation-skills menu.
    # -----------------------------------------------------------------------
    reranker_enable_llm3: bool = True
    reranker_llm3_mode_raw: Literal["fold", "call"] = Field(
        default=LLM3_MODE_FOLD, validation_alias="RERANKER_LLM3_MODE"
    )
    reranker_llm3_max_menu_skills: int = Field(default=0, ge=0, le=50_000)
    reranker_llm3_classifier_intersect: bool = False
    reranker_llm3_menu_signal_filter_raw: Literal["off", "high", "medium", "low"] = Field(
        default="off", validation_alias="RERANKER_LLM3_MENU_SIGNAL_FILTER"
    )

    @field_validator(
        "reranker_llm3_mode_raw", "reranker_llm3_menu_signal_filter_raw", mode="before"
    )
    @classmethod
    def _lowercase_and_strip(cls, v: object) -> object:
        return v.strip().lower() if isinstance(v, str) else v

    # Applies even when reranker_enable_llm3 is off.
    reranker_occupation_skills_signal_include_raw: str | None = Field(
        default=None, validation_alias="RERANKER_OCCUPATION_SKILLS_SIGNAL_INCLUDE"
    )

    occupation_relation_lookup_raw: str = Field(
        default="tabiya_bridge", validation_alias="OCCUPATION_RELATION_LOOKUP"
    )
    occupation_relation_esco_fallback: bool = True

    occupation_skill_relation_occupation_types_raw: str | None = Field(
        default=None, validation_alias="OCCUPATION_SKILL_RELATION_OCCUPATION_TYPES"
    )

    # main.py --limit required; 1 = sequential.
    reranker_parallel_workers: int = Field(default=1, ge=1, le=32)
    reranker_overlap_llm1_llm2: bool = False
    # ORed with main.py --skip-already-ranked.
    reranker_skip_already_ranked: bool = False

    reranker_enable_job_embedding: bool = True
    reranker_fail_on_job_embedding_error: bool = False
    reranker_job_embedding_model: str = "gemini-embedding-001"
    reranker_job_embedding_dim: int = Field(default=3072, ge=1, le=10_000)
    reranker_job_embedding_task_type: str = "SEMANTIC_SIMILARITY"
    reranker_job_embedding_l2_normalize: bool = True
    reranker_job_embedding_batch_size: int = Field(default=100, ge=1, le=10_000)
    reranker_job_embedding_sleep_s: float = Field(default=0.12, ge=0.0, le=3600.0)
    reranker_job_embedding_source: str | None = "llm_classified_skills"

    reranker_concat_whitening_enabled: bool = False
    reranker_concat_whitening_path_raw: str | None = Field(
        default=None, validation_alias="RERANKER_CONCAT_WHITENING_PATH"
    )
    reranker_concat_whitening_required: bool = True

    classified_linked_entity_types_raw: str | None = Field(
        default=None, validation_alias="CLASSIFIED_LINKED_ENTITY_TYPES"
    )
    classified_linked_entity_min_similarity_raw: float | None = Field(
        default=None, validation_alias="CLASSIFIED_LINKED_ENTITY_MIN_SIMILARITY"
    )
    # Legacy alias for CLASSIFIED_LINKED_ENTITY_MIN_SIMILARITY. Remove on or after 2027-01-01.
    classifier_linked_entity_min_similarity_legacy: float | None = Field(
        default=None, validation_alias="CLASSIFIER_LINKED_ENTITY_MIN_SIMILARITY"
    )
    classified_linked_entity_min_similarity_strict: bool = False
    classified_linked_entity_label_dedupe_types_raw: str | None = Field(
        default=None, validation_alias="CLASSIFIED_LINKED_ENTITY_LABEL_DEDUPE_TYPES"
    )

    target_country_raw: str = Field(default="zambia", validation_alias="TARGET_COUNTRY")

    reranker_run_wa_enrichment: bool = True
    reranker_fail_on_wa_enrichment_error: bool = False
    reranker_sync_is_active_from_closing_date: bool = True

    # Folded in from utils/work_activities_enrichment.py's own Config.
    enrich_wa_database_raw: str = Field(default="", validation_alias="ENRICH_WA_DATABASE")
    enrich_wa_target_collection_raw: str = Field(
        default="", validation_alias="ENRICH_WA_TARGET_COLLECTION"
    )
    enrich_wa_occupation_db_path_raw: str = Field(
        default="", validation_alias="ENRICH_WA_OCCUPATION_DB_PATH"
    )
    enrich_wa_bulk_batch_size: int = Field(default=500, ge=1, le=100_000)
    enrich_wa_limit: int = Field(default=0, ge=0)
    enrich_wa_dry_run: bool = False
    enrich_wa_only_missing: bool = True
    enrich_wa_default_importance: float = 3.5
    enrich_wa_default_level: float = 3.5

    # Logging: DEBUG, INFO, WARNING, ERROR, CRITICAL.
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    @field_validator("log_level", mode="before")
    @classmethod
    def _uppercase_log_level(cls, v: object) -> object:
        if not isinstance(v, str):
            return v
        upper = v.strip().upper()
        # WARN is Python logging's own alias for WARNING.
        return "WARNING" if upper == "WARN" else upper

    # -----------------------------------------------------------------------
    # Derived / computed values. See the class docstring for why these are
    # properties rather than fields.
    # -----------------------------------------------------------------------

    @cached_property
    def _repo_root_path(self) -> Path:
        return _repo_root()

    @cached_property
    def target_country(self) -> str:
        """Validated ``TARGET_COUNTRY`` slug; an unrecognised value falls back to
        ``zambia`` rather than failing a batch that would otherwise run."""
        tc_raw = self.target_country_raw.strip().lower()
        return tc_raw if tc_raw in _TARGET_COUNTRY_SLUGS else "zambia"

    @cached_property
    def country_name(self) -> str:
        return _country_name_from_slug(self.target_country)

    @cached_property
    def language(self) -> str:
        """Selects the resources/<language>/ pack and the language prompts answer in.
        Order: an explicit language= override, then RERANKER_LANGUAGE/TARGET_LANGUAGE,
        then the country's language, then en."""
        if self._language_override is not None:
            return normalise_language(self._language_override)
        return language_from_env(self.target_country)

    @cached_property
    def _language_config(self) -> dict[str, object]:
        return get_language_config(self.language)

    @cached_property
    def language_name(self) -> str:
        """Human-readable name inserted into prompts as ``{language}``."""
        cfg = self._language_config
        return str(cfg.get("prompt_language") or cfg.get("name") or self.language)

    @cached_property
    def language_resources_dir(self) -> Path:
        """Resolved ``resources/<language>/`` for this run (files may fall back to English)."""
        return language_resource_dirs(self.resources_dir, self.language)[0]

    @cached_property
    def resources_dir(self) -> Path:
        return _resources_dir(self._repo_root_path, self.resources_dir_raw)

    @cached_property
    def job_attributes_schema_path(self) -> Path:
        raw = self.job_attributes_schema_raw or self.job_attributes_schema_file_raw
        return _resolve_job_attributes_schema_path(
            self._repo_root_path, self.resources_dir, raw, self.language
        )

    @cached_property
    def occupation_to_skill_relations_path(self) -> Path:
        return _resolve_occupation_to_skill_relations_path(
            self._repo_root_path,
            self.resources_dir,
            self.occupation_to_skill_relations_csv_raw,
            self.language,
        )

    @cached_property
    def occupations_csv_path(self) -> Path:
        return _resolve_occupations_csv_path(
            self._repo_root_path, self.resources_dir, self.occupations_csv_raw, self.language
        )

    @cached_property
    def skills_csv_path(self) -> Path:
        return _resolve_skills_csv_path(
            self._repo_root_path, self.resources_dir, self.skills_csv_raw, self.language
        )

    @cached_property
    def skill_hierarchy_csv_path(self) -> Path:
        return _resolve_skill_hierarchy_csv_path(
            self._repo_root_path, self.resources_dir, self.skill_hierarchy_csv_raw, self.language
        )

    @cached_property
    def reranker_concat_whitening_path(self) -> Path:
        return _resolve_concat_whitening_path(
            self._repo_root_path,
            self.resources_dir,
            self.reranker_concat_whitening_path_raw,
            self.language,
        )

    @cached_property
    def keys_salary(self) -> tuple[str, ...]:
        return _comma_keys(self.keys_salary_raw, "salary_text,salary,compensation,pay")

    @cached_property
    def keys_location(self) -> tuple[str, ...]:
        return _comma_keys(self.keys_location_raw, "city,location,town")

    @cached_property
    def keys_county(self) -> tuple[str, ...]:
        return _comma_keys(self.keys_county_raw, "province,county,district,region")

    @cached_property
    def keys_employment(self) -> tuple[str, ...]:
        return _comma_keys(self.keys_employment_raw, "employment_type,employmentType,contract_type")

    @cached_property
    def keys_fingerprint(self) -> tuple[str, ...]:
        return _comma_keys(self.keys_fingerprint_raw, "job_fingerprint,jobFingerprint")

    @cached_property
    def keys_description(self) -> tuple[str, ...]:
        return _unique_key_order(
            ("description", "job_description"),
            _comma_keys(
                self.keys_description_raw, "body,text,content,summary,details,snippet,html"
            ),
        )

    @cached_property
    def keys_requirements(self) -> tuple[str, ...]:
        return _unique_key_order(
            ("requirements",),
            _comma_keys(self.keys_requirements_raw, "job_requirements,qualifications"),
        )

    @cached_property
    def llm(self) -> LlmConfig:
        """Resolved, provider-agnostic model selection. Order: RERANKER_LLM_CLIENT_CLASS
        (dotted path) > RERANKER_LLM_PROVIDER (shipped alias) > Gemini. Both may be set:
        the class path picks the implementation, the provider alias still supplies the
        API-key fallback."""
        class_path_raw = self.reranker_llm_client_class.strip()
        provider_raw = self.reranker_llm_provider.strip()
        if provider_raw:
            provider = normalise_provider(provider_raw)
            # Validates the alias even when a class path overrides it, so a typo is never silent.
            alias_path = resolve_provider_class_path(provider, raw=provider_raw)
            client_class = class_path_raw or alias_path
        elif class_path_raw:
            provider = CUSTOM_PROVIDER
            client_class = class_path_raw
        else:
            provider = "gemini"
            client_class = resolve_provider_class_path(provider)

        explicit_model = self.reranker_llm_model.strip()
        if explicit_model:
            llm_model = explicit_model
        elif provider == "gemini":
            llm_model = self.gemini_model
        else:
            llm_model = ""

        if self.reranker_llm_max_attempts is not None:
            max_attempts = self.reranker_llm_max_attempts
        elif self.gemini_transient_max_attempts is not None:
            max_attempts = self.gemini_transient_max_attempts
        else:
            max_attempts = _LLM_MAX_ATTEMPTS_DEFAULT
        max_attempts = max(1, min(max_attempts, _LLM_MAX_ATTEMPTS_CEILING))

        explicit_api_key = self.reranker_llm_api_key.strip()
        if explicit_api_key:
            api_key = explicit_api_key
        elif provider == "gemini":
            api_key = self.gemini_api_key
        else:
            api_key = _llm_api_key_from_env(provider, "")

        return LlmConfig(
            provider=provider,
            client_class=client_class,
            model=llm_model,
            api_key=api_key,
            base_url=_llm_base_url_from_env(provider, self.reranker_llm_base_url.strip()),
            temperature=self.reranker_llm_temperature,
            max_output_tokens=self.reranker_llm_max_output_tokens,
            timeout_s=self.reranker_llm_timeout_s,
            max_attempts=max_attempts,
            options_json=self.reranker_llm_options.strip(),
        )

    @cached_property
    def gemini_api_key(self) -> str:
        return self.gemini_api_key_raw.strip() or self.api_key_legacy.strip()

    @cached_property
    def gemini_model(self) -> str:
        return (
            self.gemini_model_raw.strip()
            or self.vertex_model_legacy.strip()
            or _GEMINI_DEFAULT_MODEL
        )

    @cached_property
    def reranker_llm3_mode(self) -> str:
        return self.reranker_llm3_mode_raw

    @cached_property
    def reranker_llm3_menu_signal_filter(self) -> str:
        return self.reranker_llm3_menu_signal_filter_raw

    @cached_property
    def reranker_occupation_skills_signal_include(self) -> frozenset[str] | None:
        raw = (self.reranker_occupation_skills_signal_include_raw or "").strip().lower()
        if not raw or raw in ("all", "*", "any"):
            return None
        allowed = frozenset({"high", "medium", "low", "unlabelled"})
        parts = [x.strip() for x in raw.split(",") if x.strip()]
        chosen = frozenset(p for p in parts if p in allowed)
        unknown = [p for p in parts if p not in allowed]
        if unknown:
            _logger.warning(
                "RERANKER_OCCUPATION_SKILLS_SIGNAL_INCLUDE: ignoring unknown tokens %s "
                "(allowed: high, medium, low, unlabelled)",
                unknown,
            )
        return chosen if chosen else None

    @cached_property
    def classified_linked_entity_types(self) -> tuple[str, ...]:
        return _entity_types_tuple(self.classified_linked_entity_types_raw)

    @cached_property
    def classified_linked_entity_min_similarity(self) -> float | None:
        min_sim = self.classified_linked_entity_min_similarity_raw
        if min_sim is None:
            min_sim = self.classifier_linked_entity_min_similarity_legacy
        if min_sim is not None and not (0.0 <= min_sim <= 1.0):
            _logger.warning(
                "CLASSIFIED_LINKED_ENTITY_MIN_SIMILARITY: %s not in [0,1], ignoring", min_sim
            )
            return None
        return min_sim

    @cached_property
    def classified_linked_entity_label_dedupe_types(self) -> frozenset[str] | None:
        dedupe_raw = self.classified_linked_entity_label_dedupe_types_raw
        if not dedupe_raw or not dedupe_raw.strip():
            return None
        allowed = frozenset({"skill", "occupation"})
        seen: set[str] = set()
        chosen: list[str] = []
        for part in dedupe_raw.split(","):
            p = part.strip()
            if not p:
                continue
            n = _normalize_classifier_entity_type_token(p)
            if n not in allowed:
                _logger.warning(
                    "CLASSIFIED_LINKED_ENTITY_LABEL_DEDUPE_TYPES: unknown %r "
                    "(allowed: skill, occupation)",
                    p,
                )
                continue
            if n not in seen:
                seen.add(n)
                chosen.append(n)
        return frozenset(chosen) if chosen else None

    @cached_property
    def occupation_relation_lookup(self) -> Literal["tabiya_bridge", "esco_uuid_bridge"]:
        """``tabiya_bridge``: match via Tabiya ids (preferred id, URI, UUID history).
        ``esco_uuid_bridge``: match via ESCO uuid first, then fall back to the above."""
        raw_lookup = self.occupation_relation_lookup_raw.strip().lower()
        if raw_lookup in ("esco", "esco_uuid", "esco_uuid_bridge", "esco_bridge"):
            return "esco_uuid_bridge"
        return "tabiya_bridge"

    @cached_property
    def occupation_skill_relation_occupation_types(self) -> frozenset[str] | None:
        raw = (self.occupation_skill_relation_occupation_types_raw or "").strip()
        if not raw or raw.lower() in ("*", "all", "any"):
            return None
        parts = frozenset(x.strip().lower() for x in raw.split(",") if x.strip())
        return parts or None

    @cached_property
    def enrich_wa_database(self) -> str:
        """Defaults to mongodb_database when ENRICH_WA_DATABASE is unset."""
        return self.enrich_wa_database_raw.strip() or self.mongodb_database

    @cached_property
    def enrich_wa_target_collection(self) -> str:
        """Defaults to mongodb_target_collection when unset."""
        return self.enrich_wa_target_collection_raw.strip() or self.mongodb_target_collection

    @cached_property
    def enrich_wa_occupation_db_path(self) -> Path:
        occ_db_raw = self.enrich_wa_occupation_db_path_raw.strip()
        if occ_db_raw:
            p = Path(occ_db_raw)
            return (p if p.is_absolute() else (self._repo_root_path / p)).resolve()
        name = "combined_occupation_database_with_wa.json"
        candidates = language_resource_dirs(self.resources_dir, self.language)
        return next(
            (d / name for d in candidates if (d / name).is_file()), candidates[0] / name
        ).resolve()

    # Private attribute, not a field — set in __init__, never bound from the environment.
    _language_override: str | None

    def __init__(self, *, language_override: str | None = None, **data: Any) -> None:
        super().__init__(**data)
        object.__setattr__(self, "_language_override", language_override)
        # Eagerly validate: an unknown RERANKER_LLM_PROVIDER must fail construction, not
        # wait for the first settings.llm access from inside request handling.
        self.llm  # noqa: B018 — property access for its validation side effect

    def __setattr__(self, name: str, value: object) -> None:
        """frozen=True only guards declared fields; cached_property writes straight to
        __dict__ and would otherwise stay reassignable, corrupting the shared
        get_settings() singleton for every other caller."""
        raise AttributeError(
            f"Settings is immutable: cannot set {name!r} after construction "
            "(cached properties and fields alike)."
        )

    @staticmethod
    def from_env(language: str | None = None) -> Settings:
        """Build Settings from the environment. ``language``, when given, overrides
        RERANKER_LANGUAGE/TARGET_LANGUAGE/country for this instance only."""
        return Settings(language_override=language)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()


@lru_cache(maxsize=len(LANGUAGE_REGISTRY) + 1)
def get_settings_for_language(language: str | None) -> Settings:
    """Settings for one language.

    The batch CLI runs a whole collection in one language (``--language`` exports
    ``RERANKER_LANGUAGE`` before anything reads settings), so it uses
    :func:`get_settings`. The HTTP service can be asked for a different language per
    request, which is what this is for. ``None`` means "whatever the environment says".
    """
    if language is None:
        return get_settings()
    lang = normalise_language(language)
    base = get_settings()
    if lang == base.language:
        return base
    return Settings.from_env(language=lang)


def log_ranker_run_configuration(
    log: logging.Logger,
    settings: Settings,
    *,
    bulk_write_batch_size: int,
    cli_limit: int,
    log_prompt_debug: bool,
    invocation: str = "batch",
) -> None:
    """
    Log effective ranking-related env / settings (taxonomy filtering, LLM3 toggles, batch parallelism).
    """
    occ_types = (
        "all"
        if settings.occupation_skill_relation_occupation_types is None
        else ",".join(sorted(settings.occupation_skill_relation_occupation_types))
    )
    sig_inc = (
        "all"
        if settings.reranker_occupation_skills_signal_include is None
        else ",".join(sorted(settings.reranker_occupation_skills_signal_include))
    )
    workers = settings.reranker_parallel_workers
    if invocation != "batch":
        parallel_desc = "n/a"
    elif workers > 1 and log_prompt_debug:
        parallel_desc = f"no(prompt-debug, workers={workers})"
    elif workers > 1:
        parallel_desc = f"yes(workers={workers})"
    else:
        parallel_desc = f"no(workers={workers})"

    key_tail = settings.llm.api_key_tail()
    loc = f"{settings.target_country} ({settings.country_name})"

    log.info(
        "Ranker run configuration | invocation=%s | TARGET_COUNTRY=%s | "
        "RERANKER_LANGUAGE=%s | resources=%s | "
        "RERANKER_LLM_PROVIDER=%s | RERANKER_LLM_MODEL=%s | RERANKER_LLM_CLIENT_CLASS=%s | "
        "llm_api_key=...%s | "
        "MONGODB_SOURCE_COLLECTION=%s | MONGODB_TARGET_COLLECTION=%s | "
        "RERANKER_ENABLE_LLM3=%s | RERANKER_LLM3_MODE=%s | RERANKER_OCCUPATION_SKILLS_SIGNAL_INCLUDE=%s | "
        "RERANKER_LLM3_MAX_MENU_SKILLS=%s | RERANKER_LLM3_MENU_SIGNAL_FILTER=%s | "
        "RERANKER_LLM3_CLASSIFIER_INTERSECT=%s | OCCUPATION_SKILL_RELATION_OCCUPATION_TYPES=%s | "
        "OCCUPATION_RELATION_LOOKUP=%s | OCCUPATION_RELATION_ESCO_FALLBACK=%s | "
        "CLASSIFIED_LINKED_ENTITY_MIN_SIMILARITY=%s | CLASSIFIED_LINKED_ENTITY_MIN_SIMILARITY_STRICT=%s | "
        "CLASSIFIED_LINKED_ENTITY_LABEL_DEDUPE_TYPES=%s | "
        "RERANKER_PARALLEL_WORKERS=%s | parallel_batch=%s | RERANKER_OVERLAP_LLM1_LLM2=%s | BULK_WRITE_BATCH_SIZE=%s | "
        "RERANKER_RUN_WA_ENRICHMENT=%s | RERANKER_FAIL_ON_WA_ENRICHMENT_ERROR=%s | "
        "RERANKER_SYNC_IS_ACTIVE_FROM_CLOSING_DATE=%s | "
        "RERANKER_CONCAT_WHITENING_ENABLED=%s | RERANKER_CONCAT_WHITENING_PATH=%s | "
        "RERANKER_CONCAT_WHITENING_REQUIRED=%s | "
        "cli_limit=%s | log_prompt_debug=%s",
        invocation,
        loc,
        f"{settings.language} ({settings.language_name})",
        settings.language_resources_dir.name,
        settings.llm.provider,
        settings.llm.model,
        settings.llm.client_class,
        key_tail,
        settings.mongodb_source_collection,
        settings.mongodb_target_collection,
        int(settings.reranker_enable_llm3),
        settings.reranker_llm3_mode,
        sig_inc,
        settings.reranker_llm3_max_menu_skills,
        settings.reranker_llm3_menu_signal_filter,
        int(settings.reranker_llm3_classifier_intersect),
        occ_types,
        settings.occupation_relation_lookup,
        int(settings.occupation_relation_esco_fallback),
        (
            "off"
            if settings.classified_linked_entity_min_similarity is None
            else str(settings.classified_linked_entity_min_similarity)
        ),
        int(settings.classified_linked_entity_min_similarity_strict),
        (
            "off"
            if not settings.classified_linked_entity_label_dedupe_types
            else ",".join(sorted(settings.classified_linked_entity_label_dedupe_types))
        ),
        workers,
        parallel_desc,
        int(settings.reranker_overlap_llm1_llm2),
        bulk_write_batch_size,
        int(settings.reranker_run_wa_enrichment),
        int(settings.reranker_fail_on_wa_enrichment_error),
        int(settings.reranker_sync_is_active_from_closing_date),
        int(settings.reranker_concat_whitening_enabled),
        settings.reranker_concat_whitening_path,
        int(settings.reranker_concat_whitening_required),
        cli_limit if invocation == "batch" else 0,
        int(log_prompt_debug),
    )
