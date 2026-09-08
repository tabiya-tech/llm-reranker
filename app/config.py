from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

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


# Repo `.env` should win over variables already exported in the shell (override=True),
# so edits like MONGODB_TARGET_COLLECTION apply when you run main.py from this project.
load_dotenv(_repo_root() / ".env", override=True)

_logger = logging.getLogger(__name__)


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


def _comma_keys(value: str | None, default: str) -> tuple[str, ...]:
    """Split comma-separated keys; empty env / whitespace-only falls back to ``default``."""
    raw = (value if value is not None else default).strip()
    out = tuple(k.strip() for k in raw.split(",") if k.strip())
    if not out:
        out = tuple(k.strip() for k in default.strip().split(",") if k.strip())
    return out


def _int_env(name: str, default: int, *, min_v: int = 1, max_v: int = 10_000) -> int:
    """Parse a positive int from the environment with bounds."""
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        n = int(str(raw).strip())
    except ValueError:
        return default
    return max(min_v, min(n, max_v))


def _int_env_min0(name: str, default: int = 0, *, max_v: int = 50_000) -> int:
    """Non-negative int; negative values fall back to ``default``. Used for optional caps (0 = off)."""
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        n = int(str(raw).strip())
    except ValueError:
        return default
    if n < 0:
        return default
    return min(n, max_v)


def _bool_env(name: str, *, default: bool = False) -> bool:
    """Parse ``1`` / ``true`` / ``yes`` / ``on`` as True (case-insensitive); unset → ``default``."""
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _float_env(name: str, default: float, *, min_v: float = 0.0, max_v: float = 3600.0) -> float:
    """Parse a float from the environment with bounds; invalid values fall back to ``default``."""
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        v = float(str(raw).strip())
    except ValueError:
        return default
    return max(min_v, min(v, max_v))


# Total attempts per LLM call, including the first. The historical Gemini path clamped
# ``GEMINI_TRANSIENT_MAX_ATTEMPTS`` to 1..12; that clamp is preserved for every provider.
_LLM_MAX_ATTEMPTS_DEFAULT = 5
_LLM_MAX_ATTEMPTS_CEILING = 12

# Gemini model used when nothing is configured — the reranker's historical default.
_GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"


def _llm_api_key_from_env(provider: str) -> str:
    """API key for ``provider``: the generic variable first, then that provider's own.

    The per-provider fallback is what lets an existing Gemini deployment keep working
    untouched — it already sets ``GEMINI_API_KEY`` and never hears of
    ``RERANKER_LLM_API_KEY``.
    """
    generic = (os.environ.get("RERANKER_LLM_API_KEY") or "").strip()
    if generic:
        return generic
    for name in PROVIDER_API_KEY_ENV_VARS.get(provider, ()):
        v = (os.environ.get(name) or "").strip()
        if v:
            return v
    return ""


def _llm_base_url_from_env(provider: str) -> str:
    """API endpoint for ``provider``: the generic variable first, then that provider's own.

    Only Ollama has one (``OLLAMA_BASE_URL`` / ``OLLAMA_HOST``), which is what lets a
    host that already runs Ollama point the reranker at it without a reranker-specific
    variable. ``OLLAMA_HOST`` is conventionally scheme-less (``127.0.0.1:11434``), so a
    scheme is added here; ``RERANKER_LLM_BASE_URL`` stays strict and is rejected by the
    client when it has none.
    """
    generic = (os.environ.get("RERANKER_LLM_BASE_URL") or "").strip()
    if generic:
        return generic.rstrip("/")
    for name in PROVIDER_BASE_URL_ENV_VARS.get(provider, ()):
        v = (os.environ.get(name) or "").strip()
        if v:
            if not v.startswith(("http://", "https://")):
                v = f"http://{v}"
            return v.rstrip("/")
    return ""


def _llm_model_from_env(provider: str) -> str:
    """Model id for ``provider``.

    ``RERANKER_LLM_MODEL`` is the provider-neutral name. Gemini additionally honours the
    pre-existing ``GEMINI_MODEL`` / ``VERTEX_MODEL`` chain and its baked-in default, so
    switching to the new config path changes no Gemini deployment's model. Other
    providers have no default: a missing model id is a configuration error raised by the
    client at startup, not a silent fallback to somebody else's model.
    """
    explicit = (os.environ.get("RERANKER_LLM_MODEL") or "").strip()
    if explicit:
        return explicit
    if provider == "gemini":
        return (
            os.environ.get("GEMINI_MODEL")
            or os.environ.get("VERTEX_MODEL")
            or _GEMINI_DEFAULT_MODEL
        ).strip()
    return ""


def _llm_config_from_env() -> LlmConfig:
    """Resolve the deployment's model selection into a frozen :class:`LlmConfig`.

    Selection order:

    1. ``RERANKER_LLM_CLIENT_CLASS`` — an explicit dotted path always wins, and may name
       a class outside this repo.
    2. ``RERANKER_LLM_PROVIDER`` — a shipped alias, mapped through
       :data:`app.llm.base.PROVIDER_CLASS_PATHS`.
    3. Neither set — Gemini, so an existing deployment's behaviour is unchanged.

    Setting both is legal and useful: the class path selects the implementation while the
    provider alias still supplies the API-key fallback (e.g. a hand-rolled Anthropic
    client that should read ``ANTHROPIC_API_KEY``).

    An unknown provider alias raises :class:`~app.llm.errors.LlmConfigurationError` here,
    at config time. Everything else that can be wrong (unimportable class, missing model
    id or key) is validated when the client is built, which the batch CLI and the API
    both do before processing any job.
    """
    class_path_raw = (os.environ.get("RERANKER_LLM_CLIENT_CLASS") or "").strip()
    provider_raw = (os.environ.get("RERANKER_LLM_PROVIDER") or "").strip()

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

    return LlmConfig(
        provider=provider,
        client_class=client_class,
        model=_llm_model_from_env(provider),
        api_key=_llm_api_key_from_env(provider),
        base_url=_llm_base_url_from_env(provider),
        temperature=_float_env("RERANKER_LLM_TEMPERATURE", DEFAULT_TEMPERATURE, max_v=2.0),
        max_output_tokens=_int_env_min0("RERANKER_LLM_MAX_OUTPUT_TOKENS", 0, max_v=1_000_000),
        timeout_s=_float_env("RERANKER_LLM_TIMEOUT_S", 0.0),
        max_attempts=_int_env(
            "RERANKER_LLM_MAX_ATTEMPTS",
            _int_env(
                "GEMINI_TRANSIENT_MAX_ATTEMPTS",
                _LLM_MAX_ATTEMPTS_DEFAULT,
                max_v=_LLM_MAX_ATTEMPTS_CEILING,
            ),
            max_v=_LLM_MAX_ATTEMPTS_CEILING,
        ),
        options_json=(os.environ.get("RERANKER_LLM_OPTIONS") or "").strip(),
    )


def _llm3_menu_signal_filter_from_env() -> str:
    """
    ``RERANKER_LLM3_MENU_SIGNAL_FILTER``: ``off`` (default), ``high``, ``medium``, or ``low``.

    When not ``off``, the LLM3 menu keeps only occupation–skill rows whose relations CSV
    ``SIGNALLINGVALUELABEL`` is at least that tier; rows with an empty label are excluded.
    """
    raw = (os.environ.get("RERANKER_LLM3_MENU_SIGNAL_FILTER") or "off").strip().lower()
    if raw in ("high", "medium", "low"):
        return raw
    return "off"


def _occupation_skills_signal_include_from_env() -> frozenset[str] | None:
    """
    ``RERANKER_OCCUPATION_SKILLS_SIGNAL_INCLUDE``: comma-separated subset of
    ``high``, ``medium``, ``low``, ``unlabelled``.

    Unset / ``all`` / ``*`` → ``None`` (keep every taxonomy skill on stored documents).
    Applied **after** taxonomy resolution and **independent of** ``RERANKER_ENABLE_LLM3``,
    so filtering applies even when LLM3 is off.
    """
    raw = (os.environ.get("RERANKER_OCCUPATION_SKILLS_SIGNAL_INCLUDE") or "").strip().lower()
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


def _classified_linked_entity_min_similarity_from_env() -> float | None:
    """
    ``CLASSIFIED_LINKED_ENTITY_MIN_SIMILARITY`` (0.0–1.0): drop classifier ``linked_entities``
    rows whose ``similarity_score`` is below this value.

    Aliases: ``CLASSIFIER_LINKED_ENTITY_MIN_SIMILARITY``. Unset → no threshold (keep all scored/unscored
    subject to URI dedupe). With ``CLASSIFIED_LINKED_ENTITY_MIN_SIMILARITY_STRICT=1``, rows with no
    score are excluded when a minimum is set.
    """
    raw = (
        os.environ.get("CLASSIFIED_LINKED_ENTITY_MIN_SIMILARITY")
        or os.environ.get("CLASSIFIER_LINKED_ENTITY_MIN_SIMILARITY")
        or ""
    ).strip()
    if not raw:
        return None
    try:
        v = float(raw)
    except ValueError:
        _logger.warning("CLASSIFIED_LINKED_ENTITY_MIN_SIMILARITY: invalid %r, ignoring", raw)
        return None
    if v < 0.0 or v > 1.0:
        _logger.warning("CLASSIFIED_LINKED_ENTITY_MIN_SIMILARITY: %s not in [0,1], ignoring", v)
        return None
    return v


def _classified_linked_entity_label_dedupe_types_from_env() -> frozenset[str] | None:
    """
    ``CLASSIFIED_LINKED_ENTITY_LABEL_DEDUPE_TYPES``: comma-separated ``entity_type`` values for which
    keep a single linked hit per **normalized label** (highest ``similarity_score``; ties → smallest ``uri``).

    Tokens: ``skill``, ``occupation`` (plural forms normalized like ``CLASSIFIED_LINKED_ENTITY_TYPES``).
    Unset / empty → no label dedupe (URI dedupe only). Unrelated to min-similarity filtering.
    """
    raw = (os.environ.get("CLASSIFIED_LINKED_ENTITY_LABEL_DEDUPE_TYPES") or "").strip()
    if not raw:
        return None
    allowed = frozenset({"skill", "occupation"})
    seen: set[str] = set()
    chosen: list[str] = []
    for part in raw.split(","):
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


def _occupation_skill_relation_occupation_types_from_env() -> frozenset[str] | None:
    """
    ``OCCUPATION_SKILL_RELATION_OCCUPATION_TYPES``: comma-separated ``OCCUPATIONTYPE`` cell values
    (case-insensitive), e.g. ``escooccupation`` or ``escooccupation,localoccupation``.

    - Unset / empty → ``None`` (load all relation rows — backward compatible).
    - ``*``, ``all``, or ``any`` → ``None`` (all types).
    """
    raw = (os.environ.get("OCCUPATION_SKILL_RELATION_OCCUPATION_TYPES") or "").strip()
    if not raw:
        return None
    low = raw.lower()
    if low in ("*", "all", "any"):
        return None
    parts = frozenset(x.strip().lower() for x in raw.split(",") if x.strip())
    return parts or None


def _target_country_slug_from_env() -> str:
    """``TARGET_COUNTRY``: batch state key + reranker_log ``country`` field. Invalid → ``zambia``."""
    allowed = frozenset({"zambia", "kenya", "south_africa", "argentina"})
    t = (os.environ.get("TARGET_COUNTRY") or "zambia").strip().lower()
    return t if t in allowed else "zambia"


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


def _occupation_relation_lookup_from_env() -> tuple[str, bool]:
    """
    ``OCCUPATION_RELATION_LOOKUP``: ``tabiya_bridge`` (default / alias ``local``) or
    ``esco_uuid_bridge`` (alias ``esco``). Unknown values fall back to ``tabiya_bridge``.

    ``OCCUPATION_RELATION_ESCO_FALLBACK`` (default ``1``): when ``esco_uuid_bridge`` finds no
    bridge hit from ESCO keys alone, fall back to the full ``tabiya_bridge`` key scan.
    """
    raw = (os.environ.get("OCCUPATION_RELATION_LOOKUP") or "tabiya_bridge").strip().lower()
    if raw in ("local", "tabiya", "tabiya_bridge"):
        mode = "tabiya_bridge"
    elif raw in ("esco", "esco_uuid", "esco_uuid_bridge", "esco_bridge"):
        mode = "esco_uuid_bridge"
    else:
        mode = "tabiya_bridge"
    return mode, _bool_env("OCCUPATION_RELATION_ESCO_FALLBACK", default=True)


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
    repo_root: Path,
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
        under_root = (repo_root / p).resolve()
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
    repo_root: Path,
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
    return _resolve_language_resource_path(repo_root, resources_dir, language, raw, default_name)


def _resolve_occupation_to_skill_relations_path(
    repo_root: Path,
    resources_dir: Path,
    raw: str | None,
    language: str = DEFAULT_LANGUAGE,
) -> Path:
    """
    Path to occupation→skill relations CSV (see ``OCCUPATION_TO_SKILL_RELATIONS_CSV``).

    Default: ``resources/<language>/occupation_to_skill_relations.csv``.
    """
    return _resolve_language_resource_path(
        repo_root, resources_dir, language, raw, "occupation_to_skill_relations.csv"
    )


def _resolve_occupations_csv_path(
    repo_root: Path,
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
        repo_root, resources_dir, language, raw, "occupations.csv"
    )


def _resolve_skills_csv_path(
    repo_root: Path,
    resources_dir: Path,
    raw: str | None,
    language: str = DEFAULT_LANGUAGE,
) -> Path:
    """
    Tabiya ``skills.csv``: ``ID`` + ``PREFERREDLABEL`` + ``UUIDHISTORY`` / ``ORIGINURI``.

    ``SKILLS_CSV``; default ``resources/<language>/skills.csv``.
    """
    return _resolve_language_resource_path(repo_root, resources_dir, language, raw, "skills.csv")


def _resolve_concat_whitening_path(
    repo_root: Path,
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
        repo_root, resources_dir, language, raw, "concat_whitening_gemini.npz"
    )


def _resolve_skill_hierarchy_csv_path(
    repo_root: Path,
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
        repo_root.parent
        / "shp-matching-algorithm"
        / "backend"
        / "resources"
        / "skill_taxonomy"
        / "skill_hierarchy.csv"
    )
    return _resolve_language_resource_path(
        repo_root,
        resources_dir,
        language,
        raw,
        "skill_hierarchy.csv",
        extra_fallbacks=(sibling,),
    )


def _resources_dir(repo_root: Path) -> Path:
    """
    Versioned prompts and schema live under ``resources/`` at the repo root.

    Override with ``RESOURCES_DIR`` or ``LLM_RERANKER_RESOURCES_DIR`` (absolute
    path or path relative to the repo root), e.g. for a private checkout or overlay.
    """
    raw = os.environ.get("RESOURCES_DIR") or os.environ.get("LLM_RERANKER_RESOURCES_DIR")
    if raw and str(raw).strip():
        p = Path(str(raw).strip())
        return p.resolve() if p.is_absolute() else (repo_root / p).resolve()
    return (repo_root / "resources").resolve()


@lru_cache(maxsize=1)
def _gcp_project_from_default_credentials() -> str | None:
    """Project from Application Default Credentials (e.g. ``gcloud auth application-default login``)."""
    try:
        import google.auth

        _, project = google.auth.default()
        if project is None:
            return None
        s = str(project).strip()
        return s or None
    except Exception:
        return None


@dataclass(frozen=True)
class Settings:
    """Runtime configuration: Mongo connection and classified job field / key names."""

    mongodb_uri: str
    mongodb_database: str
    mongodb_source_collection: str
    mongodb_target_collection: str
    mongodb_tls_ca_file: str | None
    mongodb_tls_insecure: bool
    resources_dir: Path

    # Resolved path to job attributes JSON (see ``JOB_ATTRIBUTES_SCHEMA``).
    job_attributes_schema_path: Path

    # ESCO occupation→skill relations CSV (see ``OCCUPATION_TO_SKILL_RELATIONS_CSV``).
    occupation_to_skill_relations_path: Path

    # ``occupations.csv``: classifier → Tabiya occupation ``ID`` (see ``OCCUPATIONS_CSV``).
    occupations_csv_path: Path
    # LLM2: label column for occupation line (``OCCUPATIONS_CSV_LABEL_COLUMN``, default PREFERREDLABEL).
    occupations_csv_label_column: str
    # LLM2: include definition text from ``OCCUPATIONS_CSV_DESCRIPTION_COLUMN`` when True.
    occupations_csv_include_descriptions: bool
    occupations_csv_description_column: str

    # ``skills.csv``: Tabiya skill ``ID`` / ESCO uuid → ``PREFERREDLABEL`` (see ``SKILLS_CSV``).
    skills_csv_path: Path
    # Hierarchy CSV for skill->skillgroup mapping (see ``SKILL_HIERARCHY_CSV``).
    skill_hierarchy_csv_path: Path

    # If True, LLM1 skill menu includes descriptions from ``SKILLS_CSV_DESCRIPTION_COLUMN`` (see ``SKILLS_CSV_INCLUDE_DESCRIPTIONS``).
    skills_csv_include_descriptions: bool
    # CSV column name for definition text when include is True (``SKILLS_CSV_DESCRIPTION_COLUMN``).
    skills_csv_description_column: str

    # Mongo filter keys (document field names used in find_one)
    mongo_field_job_id: str
    mongo_field_job_fingerprint: str

    # Comma-separated lists: first non-empty match wins (top-level only unless noted)
    keys_description: tuple[str, ...]
    keys_requirements: tuple[str, ...]
    keys_salary: tuple[str, ...]
    keys_location: tuple[str, ...]
    keys_county: tuple[str, ...]
    keys_employment: tuple[str, ...]
    keys_fingerprint: tuple[str, ...]

    # Single field names (used top-level, then same name under metadata where applicable)
    field_title: str
    field_employer: str
    field_category: str
    field_application_url: str
    field_posted_date: str
    field_closing_date: str
    field_source_platform: str

    # The reranker's backing LLM: which implementation to load and how to call it.
    # This — not the ``gemini_*`` fields below — is what the three grading/enrichment
    # passes use. See ``app.llm`` and ``RERANKER_LLM_*`` in ``.env.example``.
    llm: LlmConfig

    # Gemini specifics. ``gemini_api_key`` is still read directly by the job-embedding
    # step (``utils/gemini_embeddings.py``), which is pinned to Gemini because
    # match_v4's vector space and the whitening artifact are built from that model.
    # ``gemini_model`` / ``vertex_model`` remain the source of the default for
    # ``llm.model`` when the provider is Gemini; read ``settings.llm.model`` for
    # "which model actually ran".
    gemini_api_key: str
    gemini_model: str
    gcp_project_id: str
    gcp_location: str
    vertex_model: str
    google_application_credentials_resolved: str | None

    # Target collection: flush Mongo bulk_write after this many upserts (see ``BULK_WRITE_BATCH_SIZE``).
    bulk_write_batch_size: int

    # LLM3 occupation-skills menu size / latency (``RERANKER_LLM3_MAX_MENU_SKILLS``,
    # ``RERANKER_LLM3_CLASSIFIER_INTERSECT``, ``RERANKER_LLM3_MENU_SIGNAL_FILTER``).
    # When capped, CSV signalling columns prioritize rows; signal filter keeps tiered labels only.
    # Stored taxonomy (``RERANKER_OCCUPATION_SKILLS_SIGNAL_INCLUDE``): independent combos of
    # high / medium / low / unlabelled; ESCO vs local relations use
    # ``OCCUPATION_SKILL_RELATION_OCCUPATION_TYPES``.
    reranker_llm3_max_menu_skills: int
    reranker_llm3_classifier_intersect: bool
    reranker_llm3_menu_signal_filter: str
    reranker_occupation_skills_signal_include: frozenset[str] | None

    # How classifier occupation entities map to ``occupation_to_skill_relations`` keys (Tabiya ids).
    # ``tabiya_bridge``: first match from all bridge keys (preferred id, URI, UUID history, …).
    # ``esco_uuid_bridge``: try ESCO uuid from URI / originUUID first, then optional full scan.
    occupation_relation_lookup: str
    occupation_relation_esco_fallback: bool

    # ``occupation_to_skill_relations.csv`` rows: restrict by ``OCCUPATIONTYPE`` (``None`` = all rows).
    occupation_skill_relation_occupation_types: frozenset[str] | None

    # Batch CLI: process up to N jobs concurrently (``main.py --limit`` required; ``1`` = sequential).
    reranker_parallel_workers: int
    # Per-job overlap of LLM1 and LLM2 Gemini calls (~max latency not sum; watch rate limits with parallel workers).
    reranker_overlap_llm1_llm2: bool

    # Gemini concat job embedding (match_v4 stage-1 space): ``llm_classified_skills`` labels → job_embedding.
    reranker_enable_job_embedding: bool
    reranker_fail_on_job_embedding_error: bool
    reranker_job_embedding_model: str
    reranker_job_embedding_dim: int
    reranker_job_embedding_task_type: str
    reranker_job_embedding_l2_normalize: bool
    reranker_job_embedding_batch_size: int
    reranker_job_embedding_sleep_s: float
    reranker_job_embedding_source: str | None

    # Concat whitening (v4 parity): offline transform before writing job_embedding.
    reranker_concat_whitening_enabled: bool
    reranker_concat_whitening_path: Path
    reranker_concat_whitening_required: bool

    # Classifier ``classification.entities[].linked_entities`` with taxonomy ``esco`` (fixed).
    classified_linked_entity_types: tuple[str, ...]
    # Optional filter on ``linked_entities.similarity_score`` (applies to skills + occupations).
    classified_linked_entity_min_similarity: float | None
    classified_linked_entity_min_similarity_strict: bool
    classified_linked_entity_label_dedupe_types: frozenset[str] | None

    # ``TARGET_COUNTRY`` slug and resolved country name for prompts / opening lines.
    target_country: str
    country_name: str

    # Language of the job ads being ranked: selects the ``resources/<language>/`` pack
    # (taxonomy labels, prompts, job-attributes schema) and the language the prompts ask
    # the model to answer in. See ``app.languages``.
    language: str = DEFAULT_LANGUAGE
    # Human-readable name inserted into prompts as ``{language}``.
    language_name: str = "English"
    # Resolved ``resources/<language>/`` for this run (files may fall back to English).
    language_resources_dir: Path = Path("resources") / DEFAULT_LANGUAGE

    @staticmethod
    def from_env(language: str | None = None) -> Settings:
        root = _repo_root()
        resources = _resources_dir(root)
        # Resolved before any resource path: every pack lookup depends on it.
        tc_for_language = _target_country_slug_from_env()
        lang = (
            normalise_language(language)
            if language is not None
            else language_from_env(tc_for_language)
        )
        lang_cfg = get_language_config(lang)
        creds_resolved = _resolve_google_credentials_path(
            root,
            os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            or os.environ.get("VERTEX_SERVICE_ACCOUNT_JSON"),
        )
        gcp_pid = (
            os.environ.get("GCP_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT") or ""
        ).strip()
        if not gcp_pid and creds_resolved:
            gcp_pid = _read_gcp_project_id_from_service_account(Path(creds_resolved)) or ""
        if not gcp_pid:
            gcp_pid = _gcp_project_from_default_credentials() or ""
        source_coll = (
            os.environ.get("MONGODB_SOURCE_COLLECTION")
            or os.environ.get("MONGODB_CLASSIFIED_COLLECTION")
            or "classified_jobs"
        ).strip()
        # Default target must not mirror source — writing ranked output back into classified_jobs is wrong.
        target_coll = (os.environ.get("MONGODB_TARGET_COLLECTION") or "ranked_jobs").strip()
        ja_schema_raw = os.environ.get("JOB_ATTRIBUTES_SCHEMA") or os.environ.get(
            "JOB_ATTRIBUTES_SCHEMA_FILE"
        )
        job_attributes_schema_path = _resolve_job_attributes_schema_path(
            root, resources, ja_schema_raw, lang
        )
        occ_skill_raw = os.environ.get("OCCUPATION_TO_SKILL_RELATIONS_CSV")
        occupation_to_skill_relations_path = _resolve_occupation_to_skill_relations_path(
            root, resources, occ_skill_raw, lang
        )
        occ_csv_raw = os.environ.get("OCCUPATIONS_CSV")
        occupations_csv_path = _resolve_occupations_csv_path(root, resources, occ_csv_raw, lang)
        occ_label_col = (os.environ.get("OCCUPATIONS_CSV_LABEL_COLUMN") or "PREFERREDLABEL").strip()
        occ_inc_desc = _bool_env("OCCUPATIONS_CSV_INCLUDE_DESCRIPTIONS", default=False)
        occ_desc_col = (os.environ.get("OCCUPATIONS_CSV_DESCRIPTION_COLUMN") or "").strip()
        skills_csv_raw = os.environ.get("SKILLS_CSV")
        skills_csv_path = _resolve_skills_csv_path(root, resources, skills_csv_raw, lang)
        skill_hierarchy_raw = os.environ.get("SKILL_HIERARCHY_CSV")
        skill_hierarchy_csv_path = _resolve_skill_hierarchy_csv_path(
            root, resources, skill_hierarchy_raw, lang
        )
        concat_whitening_raw = os.environ.get("RERANKER_CONCAT_WHITENING_PATH")
        concat_whitening_path = _resolve_concat_whitening_path(
            root, resources, concat_whitening_raw, lang
        )
        skills_include_desc = _bool_env("SKILLS_CSV_INCLUDE_DESCRIPTIONS", default=False)
        skills_desc_col = (os.environ.get("SKILLS_CSV_DESCRIPTION_COLUMN") or "").strip()
        occ_rel_lookup, occ_rel_esco_fb = _occupation_relation_lookup_from_env()
        occ_skill_relation_types = _occupation_skill_relation_occupation_types_from_env()
        tc = tc_for_language
        c_name = _country_name_from_slug(tc)
        return Settings(
            mongodb_uri=os.environ.get("MONGODB_URI", "mongodb://localhost:27017"),
            mongodb_database=os.environ.get("MONGODB_DATABASE", "ZambiaJobs"),
            mongodb_source_collection=source_coll,
            mongodb_target_collection=target_coll,
            mongodb_tls_ca_file=os.environ.get("MONGODB_TLS_CA_FILE") or None,
            mongodb_tls_insecure=os.environ.get("MONGODB_TLS_INSECURE", "").lower()
            in ("1", "true", "yes"),
            resources_dir=resources,
            job_attributes_schema_path=job_attributes_schema_path,
            occupation_to_skill_relations_path=occupation_to_skill_relations_path,
            occupations_csv_path=occupations_csv_path,
            occupations_csv_label_column=occ_label_col or "PREFERREDLABEL",
            occupations_csv_include_descriptions=occ_inc_desc,
            occupations_csv_description_column=occ_desc_col,
            skills_csv_path=skills_csv_path,
            skill_hierarchy_csv_path=skill_hierarchy_csv_path,
            skills_csv_include_descriptions=skills_include_desc,
            skills_csv_description_column=skills_desc_col,
            mongo_field_job_id=os.environ.get("CLASSIFIED_MONGO_FIELD_JOB_ID", "job_id"),
            mongo_field_job_fingerprint=os.environ.get(
                "CLASSIFIED_MONGO_FIELD_JOB_FINGERPRINT", "job_fingerprint"
            ),
            keys_description=_unique_key_order(
                ("description", "job_description"),
                _comma_keys(
                    os.environ.get("CLASSIFIED_KEYS_DESCRIPTION"),
                    "body,text,content,summary,details,snippet,html",
                ),
            ),
            keys_requirements=_unique_key_order(
                ("requirements",),
                _comma_keys(
                    os.environ.get("CLASSIFIED_KEYS_REQUIREMENTS"),
                    "job_requirements,qualifications",
                ),
            ),
            keys_salary=_comma_keys(
                os.environ.get("CLASSIFIED_KEYS_SALARY"),
                "salary_text,salary,compensation,pay",
            ),
            keys_location=_comma_keys(
                os.environ.get("CLASSIFIED_KEYS_LOCATION"),
                "city,location,town",
            ),
            keys_county=_comma_keys(
                os.environ.get("CLASSIFIED_KEYS_COUNTY"),
                "province,county,district,region",
            ),
            keys_employment=_comma_keys(
                os.environ.get("CLASSIFIED_KEYS_EMPLOYMENT"),
                "employment_type,employmentType,contract_type",
            ),
            keys_fingerprint=_comma_keys(
                os.environ.get("CLASSIFIED_KEYS_FINGERPRINT"),
                "job_fingerprint,jobFingerprint",
            ),
            field_title=os.environ.get("CLASSIFIED_FIELD_TITLE", "title"),
            field_employer=os.environ.get("CLASSIFIED_FIELD_EMPLOYER", "employer"),
            field_category=os.environ.get("CLASSIFIED_FIELD_CATEGORY", "category"),
            field_application_url=os.environ.get(
                "CLASSIFIED_FIELD_APPLICATION_URL", "application_url"
            ),
            field_posted_date=os.environ.get("CLASSIFIED_FIELD_POSTED_DATE", "posted_date"),
            field_closing_date=os.environ.get("CLASSIFIED_FIELD_CLOSING_DATE", "closing_date"),
            field_source_platform=os.environ.get(
                "CLASSIFIED_FIELD_SOURCE_PLATFORM", "source_platform"
            ),
            llm=_llm_config_from_env(),
            gemini_api_key=(
                os.environ.get("GEMINI_API_KEY") or os.environ.get("API_KEY") or ""
            ).strip(),
            gemini_model=(
                os.environ.get("GEMINI_MODEL")
                or os.environ.get("VERTEX_MODEL")
                or "gemini-2.5-flash"
            ).strip(),
            gcp_project_id=gcp_pid,
            gcp_location=os.environ.get("GCP_LOCATION", "us-central1"),
            vertex_model=os.environ.get("VERTEX_MODEL", "gemini-2.5-flash"),
            google_application_credentials_resolved=creds_resolved,
            bulk_write_batch_size=_int_env("BULK_WRITE_BATCH_SIZE", 100),
            reranker_llm3_max_menu_skills=_int_env_min0("RERANKER_LLM3_MAX_MENU_SKILLS", 0),
            reranker_llm3_classifier_intersect=_bool_env(
                "RERANKER_LLM3_CLASSIFIER_INTERSECT", default=False
            ),
            reranker_llm3_menu_signal_filter=_llm3_menu_signal_filter_from_env(),
            reranker_occupation_skills_signal_include=_occupation_skills_signal_include_from_env(),
            occupation_relation_lookup=occ_rel_lookup,
            occupation_relation_esco_fallback=occ_rel_esco_fb,
            occupation_skill_relation_occupation_types=occ_skill_relation_types,
            reranker_parallel_workers=_int_env("RERANKER_PARALLEL_WORKERS", 1, min_v=1, max_v=32),
            reranker_overlap_llm1_llm2=_bool_env("RERANKER_OVERLAP_LLM1_LLM2", default=False),
            reranker_enable_job_embedding=_bool_env("RERANKER_ENABLE_JOB_EMBEDDING", default=True),
            reranker_fail_on_job_embedding_error=_bool_env(
                "RERANKER_FAIL_ON_JOB_EMBEDDING_ERROR", default=False
            ),
            reranker_job_embedding_model=(
                os.environ.get("RERANKER_JOB_EMBEDDING_MODEL") or "gemini-embedding-001"
            ).strip(),
            reranker_job_embedding_dim=_int_env("RERANKER_JOB_EMBEDDING_DIM", 3072),
            reranker_job_embedding_task_type=(
                os.environ.get("RERANKER_JOB_EMBEDDING_TASK_TYPE") or "SEMANTIC_SIMILARITY"
            ).strip(),
            reranker_job_embedding_l2_normalize=_bool_env(
                "RERANKER_JOB_EMBEDDING_L2_NORMALIZE", default=True
            ),
            reranker_job_embedding_batch_size=_int_env("RERANKER_JOB_EMBEDDING_BATCH_SIZE", 100),
            reranker_job_embedding_sleep_s=_float_env("RERANKER_JOB_EMBEDDING_SLEEP_S", 0.12),
            reranker_job_embedding_source=(
                (os.environ.get("RERANKER_JOB_EMBEDDING_SOURCE") or "llm_classified_skills").strip()
                or None
            ),
            reranker_concat_whitening_enabled=_bool_env(
                "RERANKER_CONCAT_WHITENING_ENABLED", default=False
            ),
            reranker_concat_whitening_path=concat_whitening_path,
            reranker_concat_whitening_required=_bool_env(
                "RERANKER_CONCAT_WHITENING_REQUIRED", default=True
            ),
            classified_linked_entity_types=_entity_types_tuple(
                os.environ.get("CLASSIFIED_LINKED_ENTITY_TYPES")
            ),
            classified_linked_entity_min_similarity=_classified_linked_entity_min_similarity_from_env(),
            classified_linked_entity_min_similarity_strict=_bool_env(
                "CLASSIFIED_LINKED_ENTITY_MIN_SIMILARITY_STRICT", default=False
            ),
            classified_linked_entity_label_dedupe_types=_classified_linked_entity_label_dedupe_types_from_env(),
            target_country=tc,
            country_name=c_name,
            language=lang,
            language_name=str(lang_cfg.get("prompt_language") or lang_cfg.get("name") or lang),
            language_resources_dir=language_resource_dirs(resources, lang)[0],
        )


def _read_gcp_project_id_from_service_account(json_path: Path) -> str | None:
    """Use ``project_id`` inside a GCP service account JSON when ``GCP_PROJECT_ID`` is not set."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        pid = data.get("project_id")
        if pid is None:
            return None
        s = str(pid).strip()
        return s or None
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _resolve_google_credentials_path(repo_root: Path, raw: str | None) -> str | None:
    if not raw or not raw.strip():
        return None
    p = Path(raw.strip())
    if not p.is_absolute():
        p = repo_root / p
    try:
        return str(p.resolve()) if p.is_file() else None
    except OSError:
        return None


def vertex_project_unset_message() -> str:
    """Message when Vertex is invoked but no GCP project id could be resolved."""
    return (
        "Vertex needs a GCP project id. Set one of:\n"
        "  GCP_PROJECT_ID or GOOGLE_CLOUD_PROJECT\n"
        "  Or point VERTEX_SERVICE_ACCOUNT_JSON / GOOGLE_APPLICATION_CREDENTIALS "
        "to a service account JSON (field project_id)\n"
        "  Or: gcloud auth application-default login (uses Application Default Credentials)"
    )


def reranker_enable_llm3_from_env() -> bool:
    """``RERANKER_ENABLE_LLM3`` (default on): run LLM3 occupation–skills relevance when the menu is non-empty."""
    return _bool_env("RERANKER_ENABLE_LLM3", default=True)


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
    enable_llm3 = reranker_enable_llm3_from_env()
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

    wa_run = _bool_env("RERANKER_RUN_WA_ENRICHMENT", default=True)
    wa_fail = _bool_env("RERANKER_FAIL_ON_WA_ENRICHMENT_ERROR", default=False)
    sync_is_active = _bool_env("RERANKER_SYNC_IS_ACTIVE_FROM_CLOSING_DATE", default=True)
    key_tail = settings.llm.api_key_tail()
    loc = f"{settings.target_country} ({settings.country_name})"

    log.info(
        "Ranker run configuration | invocation=%s | TARGET_COUNTRY=%s | "
        "RERANKER_LANGUAGE=%s | resources=%s | "
        "RERANKER_LLM_PROVIDER=%s | RERANKER_LLM_MODEL=%s | RERANKER_LLM_CLIENT_CLASS=%s | "
        "llm_api_key=...%s | "
        "MONGODB_SOURCE_COLLECTION=%s | MONGODB_TARGET_COLLECTION=%s | "
        "RERANKER_ENABLE_LLM3=%s | RERANKER_OCCUPATION_SKILLS_SIGNAL_INCLUDE=%s | "
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
        int(enable_llm3),
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
        int(wa_run),
        int(wa_fail),
        int(sync_is_active),
        int(settings.reranker_concat_whitening_enabled),
        settings.reranker_concat_whitening_path,
        int(settings.reranker_concat_whitening_required),
        cli_limit if invocation == "batch" else 0,
        int(log_prompt_debug),
    )


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
