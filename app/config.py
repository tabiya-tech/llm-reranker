from __future__ import annotations

import json
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


load_dotenv(_repo_root() / ".env", override=False)


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


def _resources_dir(repo_root: Path) -> Path:
    """
    Versioned prompts and schema live under ``resources/`` at the repo root.

    Override with ``RESOURCES_DIR`` or ``LLM_RERANKER_RESOURCES_DIR`` (absolute
    path or path relative to the repo root), e.g. for a private checkout or overlay.
    """
    raw = os.environ.get("RESOURCES_DIR") or os.environ.get("LLM_RERANKER_RESOURCES_DIR")
    if raw and str(raw).strip():
        p = Path(str(raw).strip())
        return (p.resolve() if p.is_absolute() else (repo_root / p).resolve())
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
    """Runtime configuration: Mongo connection and ClassifiedJobs field / key names."""

    mongodb_uri: str
    mongodb_database: str
    mongodb_source_collection: str
    mongodb_target_collection: str
    mongodb_tls_ca_file: str | None
    mongodb_tls_insecure: bool
    resources_dir: Path

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

    # Vertex AI (Gemini); resolved SA path or None (then rely on existing GOOGLE_APPLICATION_CREDENTIALS / gcloud ADC)
    gcp_project_id: str
    gcp_location: str
    vertex_model: str
    google_application_credentials_resolved: str | None

    # Target collection: flush Mongo bulk_write after this many upserts (see ``BULK_WRITE_BATCH_SIZE``).
    bulk_write_batch_size: int

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
            os.environ.get("GCP_PROJECT_ID")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or ""
        ).strip()
        if not gcp_pid and creds_resolved:
            gcp_pid = _read_gcp_project_id_from_service_account(Path(creds_resolved)) or ""
        if not gcp_pid:
            gcp_pid = _gcp_project_from_default_credentials() or ""
        source_coll = (
            os.environ.get("MONGODB_SOURCE_COLLECTION")
            or os.environ.get("MONGODB_CLASSIFIED_COLLECTION")
            or "ClassifiedJobs"
        ).strip()
        target_coll = (os.environ.get("MONGODB_TARGET_COLLECTION") or source_coll).strip()
        return Settings(
            mongodb_uri=os.environ.get("MONGODB_URI", "mongodb://localhost:27017"),
            mongodb_database=os.environ.get("MONGODB_DATABASE", "ZambiaJobs"),
            mongodb_source_collection=source_coll,
            mongodb_target_collection=target_coll,
            mongodb_tls_ca_file=os.environ.get("MONGODB_TLS_CA_FILE") or None,
            mongodb_tls_insecure=os.environ.get("MONGODB_TLS_INSECURE", "").lower()
            in ("1", "true", "yes"),
            resources_dir=resources,
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
                "location,city,town",
            ),
            keys_county=_comma_keys(
                os.environ.get("CLASSIFIED_KEYS_COUNTY"),
                "county,district,region",
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
            gcp_project_id=gcp_pid,
            gcp_location=os.environ.get("GCP_LOCATION", "us-central1"),
            vertex_model=os.environ.get("VERTEX_MODEL", "gemini-2.0-flash-001"),
            google_application_credentials_resolved=creds_resolved,
            bulk_write_batch_size=_int_env("BULK_WRITE_BATCH_SIZE", 100),
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


def get_settings() -> Settings:
    return Settings.from_env()
