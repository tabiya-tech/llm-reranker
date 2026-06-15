from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


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


def _resolve_job_attributes_schema_path(
    repo_root: Path,
    resources_dir: Path,
    raw: str | None,
) -> Path:
    """
    Path to the job-attributes JSON (same structure for any number of attributes).

    ``JOB_ATTRIBUTES_SCHEMA`` / ``JOB_ATTRIBUTES_SCHEMA_FILE``:
    absolute path, or relative to ``resources_dir`` (preferred), or relative to repo root.
    Default basename: ``job_attributes_reduced.json`` under ``resources_dir``.
    """
    default_name = "job_attributes_reduced.json"
    if raw is None or not str(raw).strip():
        return (resources_dir / default_name).resolve()
    s = str(raw).strip()
    p = Path(s)
    if p.is_absolute():
        return p.resolve()
    under_res = (resources_dir / p).resolve()
    if under_res.is_file():
        return under_res
    under_root = (repo_root / p).resolve()
    return under_root if under_root.is_file() else under_res


def _resolve_occupation_to_skill_relations_path(
    repo_root: Path,
    resources_dir: Path,
    raw: str | None,
) -> Path:
    """
    Path to occupation→skill relations CSV (see ``OCCUPATION_TO_SKILL_RELATIONS_CSV``).

    Default: ``resources/occupation_to_skill_relations.csv`` (file may be created later).
    """
    default_name = "occupation_to_skill_relations.csv"
    if raw is None or not str(raw).strip():
        return (resources_dir / default_name).resolve()
    s = str(raw).strip()
    p = Path(s)
    if p.is_absolute():
        return p.resolve()
    under_res = (resources_dir / p).resolve()
    if under_res.is_file():
        return under_res
    under_root = (repo_root / p).resolve()
    return under_root if under_root.is_file() else under_res


def _resolve_occupations_csv_path(
    repo_root: Path,
    resources_dir: Path,
    raw: str | None,
) -> Path:
    """
    Tabiya ``occupations.csv``: ``ID`` + ``UUIDHISTORY`` / ``ORIGINURI``.

    Row ``ID`` must match ``OCCUPATIONID`` in ``occupation_to_skill_relations.csv``.

    ``OCCUPATIONS_CSV``; default ``resources/occupations.csv``.
    """
    default_name = "occupations.csv"
    if raw is None or not str(raw).strip():
        return (resources_dir / default_name).resolve()
    s = str(raw).strip()
    p = Path(s)
    if p.is_absolute():
        return p.resolve()
    under_res = (resources_dir / p).resolve()
    if under_res.is_file():
        return under_res
    under_root = (repo_root / p).resolve()
    return under_root if under_root.is_file() else under_res


def _resolve_skills_csv_path(
    repo_root: Path,
    resources_dir: Path,
    raw: str | None,
) -> Path:
    """
    Tabiya ``skills.csv``: ``ID`` + ``PREFERREDLABEL`` + ``UUIDHISTORY`` / ``ORIGINURI``.

    ``SKILLS_CSV``; default ``resources/skills.csv``.
    """
    default_name = "skills.csv"
    if raw is None or not str(raw).strip():
        return (resources_dir / default_name).resolve()
    s = str(raw).strip()
    p = Path(s)
    if p.is_absolute():
        return p.resolve()
    under_res = (resources_dir / p).resolve()
    if under_res.is_file():
        return under_res
    under_root = (repo_root / p).resolve()
    return under_root if under_root.is_file() else under_res


def _resolve_skill_hierarchy_csv_path(
    repo_root: Path,
    resources_dir: Path,
    raw: str | None,
) -> Path:
    """
    Path to ``skill_hierarchy.csv`` used for skill -> skill-group derivation.

    Resolution order:
      1) explicit env path (absolute or relative)
      2) ``resources/skill_hierarchy.csv``
      3) sibling repo fallback:
         ``../shp-matching-algorithm/backend/resources/skill_taxonomy/skill_hierarchy.csv``
    """
    default_name = "skill_hierarchy.csv"
    if raw is not None and str(raw).strip():
        s = str(raw).strip()
        p = Path(s)
        if p.is_absolute():
            return p.resolve()
        under_res = (resources_dir / p).resolve()
        if under_res.is_file():
            return under_res
        under_root = (repo_root / p).resolve()
        return under_root if under_root.is_file() else under_res

    under_res = (resources_dir / default_name).resolve()
    if under_res.is_file():
        return under_res

    sibling = (
        repo_root.parent
        / "shp-matching-algorithm"
        / "backend"
        / "resources"
        / "skill_taxonomy"
        / "skill_hierarchy.csv"
    ).resolve()
    if sibling.is_file():
        return sibling
    return under_res


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

    # Gemini (GenAI API key or Vertex AI fallback)
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

    # Classifier ``classification.entities[].linked_entities`` with taxonomy ``esco`` (fixed).
    classified_linked_entity_types: tuple[str, ...]
    # Optional filter on ``linked_entities.similarity_score`` (applies to skills + occupations).
    classified_linked_entity_min_similarity: float | None
    classified_linked_entity_min_similarity_strict: bool
    classified_linked_entity_label_dedupe_types: frozenset[str] | None

    # ``TARGET_COUNTRY`` slug and resolved country name for prompts / opening lines.
    target_country: str
    country_name: str

    @staticmethod
    def from_env() -> Settings:
        root = _repo_root()
        resources = _resources_dir(root)
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
            root, resources, ja_schema_raw
        )
        occ_skill_raw = os.environ.get("OCCUPATION_TO_SKILL_RELATIONS_CSV")
        occupation_to_skill_relations_path = _resolve_occupation_to_skill_relations_path(
            root, resources, occ_skill_raw
        )
        occ_csv_raw = os.environ.get("OCCUPATIONS_CSV")
        occupations_csv_path = _resolve_occupations_csv_path(root, resources, occ_csv_raw)
        occ_label_col = (os.environ.get("OCCUPATIONS_CSV_LABEL_COLUMN") or "PREFERREDLABEL").strip()
        occ_inc_desc = _bool_env("OCCUPATIONS_CSV_INCLUDE_DESCRIPTIONS", default=False)
        occ_desc_col = (os.environ.get("OCCUPATIONS_CSV_DESCRIPTION_COLUMN") or "").strip()
        skills_csv_raw = os.environ.get("SKILLS_CSV")
        skills_csv_path = _resolve_skills_csv_path(root, resources, skills_csv_raw)
        skill_hierarchy_raw = os.environ.get("SKILL_HIERARCHY_CSV")
        skill_hierarchy_csv_path = _resolve_skill_hierarchy_csv_path(
            root, resources, skill_hierarchy_raw
        )
        skills_include_desc = _bool_env("SKILLS_CSV_INCLUDE_DESCRIPTIONS", default=False)
        skills_desc_col = (os.environ.get("SKILLS_CSV_DESCRIPTION_COLUMN") or "").strip()
        occ_rel_lookup, occ_rel_esco_fb = _occupation_relation_lookup_from_env()
        occ_skill_relation_types = _occupation_skill_relation_occupation_types_from_env()
        tc = _target_country_slug_from_env()
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
            reranker_enable_job_embedding=_bool_env(
                "RERANKER_ENABLE_JOB_EMBEDDING", default=True
            ),
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
            reranker_job_embedding_batch_size=_int_env(
                "RERANKER_JOB_EMBEDDING_BATCH_SIZE", 100
            ),
            reranker_job_embedding_sleep_s=_float_env(
                "RERANKER_JOB_EMBEDDING_SLEEP_S", 0.12
            ),
            reranker_job_embedding_source=(
                (os.environ.get("RERANKER_JOB_EMBEDDING_SOURCE") or "llm_classified_skills").strip()
                or None
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
    key_tail = settings.gemini_api_key[-4:] if settings.gemini_api_key else "?"
    loc = f"{settings.target_country} ({settings.country_name})"

    log.info(
        "Ranker run configuration | invocation=%s | TARGET_COUNTRY=%s | GEMINI_MODEL=%s | GEMINI_API_KEY=...%s | "
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
        "cli_limit=%s | log_prompt_debug=%s",
        invocation,
        loc,
        settings.gemini_model,
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
        cli_limit if invocation == "batch" else 0,
        int(log_prompt_debug),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
