from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Iterator

import certifi
from bson import ObjectId
from bson.errors import InvalidId
from pymongo import MongoClient
from pymongo.operations import UpdateOne
from pymongo.collection import Collection
from pymongo.database import Database

from app.config import Settings, get_settings
from app.job_context import JobDescriptionContext
from app.logging_config import sanitize_mongodb_uri
from app.schemas import JobAttributesResult

if TYPE_CHECKING:
    from utils.classifier_linked_entities import EscoSkillCandidate, MappedEscoSkills
    from utils.uuid_to_tabiya_id import SkillsUuidTabiyaMap

logger = logging.getLogger(__name__)
_mongo_client_connection_logged = False


def _mongo_client_kwargs(uri: str, settings: Settings) -> dict[str, Any]:
    """TLS options for Atlas / macOS Python (cert chain) and optional dev overrides."""
    if settings.mongodb_tls_insecure:
        return {"tlsAllowInvalidCertificates": True}
    if settings.mongodb_tls_ca_file:
        return {"tlsCAFile": settings.mongodb_tls_ca_file}
    if "mongodb+srv://" in uri or "tls=true" in uri.lower() or "ssl=true" in uri.lower():
        return {"tlsCAFile": certifi.where()}
    return {}


def get_mongo_client(uri: str | None = None) -> MongoClient:
    global _mongo_client_connection_logged
    settings = get_settings()
    resolved = uri or settings.mongodb_uri
    kwargs = _mongo_client_kwargs(resolved, settings)
    client = MongoClient(resolved, **kwargs)
    if not _mongo_client_connection_logged:
        logger.info(
            "MongoDB client ready (uri=%s, database=%s, read_collection=%s, write_collection=%s)",
            sanitize_mongodb_uri(resolved),
            settings.mongodb_database,
            settings.mongodb_source_collection,
            settings.mongodb_target_collection,
        )
        _mongo_client_connection_logged = True
    return client


def get_database(
    client: MongoClient | None = None,
    *,
    settings: Settings | None = None,
) -> Database:
    settings = settings or get_settings()
    c = client if client is not None else get_mongo_client(settings.mongodb_uri)
    return c[settings.mongodb_database]


def classified_jobs_collection(
    db: Database | None = None,
    *,
    settings: Settings | None = None,
) -> Collection:
    settings = settings or get_settings()
    database = db if db is not None else get_database(settings=settings)
    return database[settings.mongodb_source_collection]


def target_jobs_collection(
    db: Database | None = None,
    *,
    settings: Settings | None = None,
) -> Collection:
    """Collection where LLM pipeline outputs are upserted (``MONGODB_TARGET_COLLECTION``)."""
    settings = settings or get_settings()
    database = db if db is not None else get_database(settings=settings)
    return database[settings.mongodb_target_collection]


def _upsert_filter_for_llm_results(
    classified_doc: dict[str, Any],
    ctx: JobDescriptionContext,
    settings: Settings,
) -> dict[str, Any]:
    if ctx.job_id is not None:
        return {settings.mongo_field_job_id: ctx.job_id}
    if ctx.job_fingerprint:
        return {settings.mongo_field_job_fingerprint: ctx.job_fingerprint}
    oid = classified_doc.get("_id")
    if oid is not None:
        return {"source_classified_id": oid}
    raise ValueError(
        "Cannot upsert LLM results: document has no job id, fingerprint, or source _id"
    )


def _llm_pipeline_set_document(
    classified_doc: dict[str, Any],
    ctx: JobDescriptionContext,
    settings: Settings,
    ja: JobAttributesResult | None,
    skill_ranker: tuple[list[str], list[str], "MappedEscoSkills", str] | None,
    occupation_to_skills: list[dict[str, Any]] | None = None,
    skills_uuid_tabiya_map: SkillsUuidTabiyaMap | None = None,
    occupations_bridge: dict[str, str] | None = None,
    occupations_all_ids: frozenset[str] | None = None,
    ranker_candidates: list["EscoSkillCandidate"] | None = None,
    classifier_skill_candidates: list["EscoSkillCandidate"] | None = None,
) -> dict[str, Any] | None:
    """``$set`` body for target upsert (see ``app.ranked_job_schema``); ``None`` when nothing to write."""
    from app.ranked_job_schema import build_ranked_job_document

    return build_ranked_job_document(
        classified_doc,
        ctx,
        settings=settings,
        ja=ja,
        skill_ranker=skill_ranker,
        occupation_to_skills=occupation_to_skills,
        skills_uuid_tabiya_map=skills_uuid_tabiya_map,
        occupations_bridge=occupations_bridge,
        occupations_all_ids=occupations_all_ids,
        ranker_candidates=ranker_candidates,
        classifier_skill_candidates=classifier_skill_candidates,
    )


def llm_pipeline_upsert_update_one(
    classified_doc: dict[str, Any],
    ctx: JobDescriptionContext,
    *,
    settings: Settings | None = None,
    ja: JobAttributesResult | None = None,
    skill_ranker: tuple[list[str], list[str], "MappedEscoSkills", str] | None = None,
    occupation_to_skills: list[dict[str, Any]] | None = None,
    skills_uuid_tabiya_map: SkillsUuidTabiyaMap | None = None,
    occupations_bridge: dict[str, str] | None = None,
    occupations_all_ids: frozenset[str] | None = None,
    ranker_candidates: list["EscoSkillCandidate"] | None = None,
    classifier_skill_candidates: list["EscoSkillCandidate"] | None = None,
) -> UpdateOne | None:
    """
    Build one ``UpdateOne`` for the target collection, or ``None`` if there is nothing to write.

    May raise ``ValueError`` if the classified document cannot be matched (see ``_upsert_filter_for_llm_results``).
    """
    settings = settings or get_settings()
    payload = _llm_pipeline_set_document(
        classified_doc,
        ctx,
        settings,
        ja,
        skill_ranker,
        occupation_to_skills=occupation_to_skills,
        skills_uuid_tabiya_map=skills_uuid_tabiya_map,
        occupations_bridge=occupations_bridge,
        occupations_all_ids=occupations_all_ids,
        ranker_candidates=ranker_candidates,
        classifier_skill_candidates=classifier_skill_candidates,
    )
    if payload is None:
        return None
    flt = _upsert_filter_for_llm_results(classified_doc, ctx, settings)
    # Drop legacy nested raw-response blobs (schema v2+).
    return UpdateOne(
        flt,
        {
            "$set": payload,
            "$unset": {
                "llm_skill_ranker": "",
                "classified_linked_by_type": "",
                "occupation_skills": "",
                "occupation_skills_uuids": "",
                "classified_skills_uuids": "",
                "classified_occupations_uuids": "",
                "essential_skills": "",
                "essential_skills_uuids": "",
                "optional_skills": "",
                "optional_skills_uuids": "",
                "job_attributes": "",
                "work_activity_ids": "",
                "work_activity_names": "",
                "title": "",
            },
        },
        upsert=True,
    )


def bulk_write_llm_upserts(
    operations: list[UpdateOne],
    *,
    settings: Settings | None = None,
    client: MongoClient | None = None,
) -> None:
    """Run ``bulk_write`` (unordered) for LLM pipeline upserts."""
    if not operations:
        return
    settings = settings or get_settings()
    coll = target_jobs_collection(settings=settings, db=get_database(client, settings=settings))
    result = coll.bulk_write(operations, ordered=False)
    logger.info(
        "Bulk upsert LLM results → %s.%s: ops=%s matched=%s modified=%s upserted=%s",
        settings.mongodb_database,
        settings.mongodb_target_collection,
        len(operations),
        result.matched_count,
        result.modified_count,
        result.upserted_count,
    )


def iter_classified_jobs(
    limit: int = 100,
    skip: int = 0,
    *,
    sort_field: str = "_id",
    sort_direction: int = -1,
    settings: Settings | None = None,
    client: MongoClient | None = None,
) -> Iterator[dict[str, Any]]:
    """
    Stream documents from the configured source collection, newest first by default.

    ``limit`` <= 0 means no limit (entire collection). Results are streamed one document at a time.
    """
    settings = settings or get_settings()
    coll = classified_jobs_collection(settings=settings, db=get_database(client, settings=settings))
    cur = coll.find({}).sort(sort_field, sort_direction).skip(skip)
    if limit > 0:
        cur = cur.limit(limit)
    yield from cur


def _metadata(doc: dict[str, Any]) -> dict[str, Any]:
    m = doc.get("metadata")
    return m if isinstance(m, dict) else {}


def _scalar_to_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        try:
            s = bytes(value).decode("utf-8").strip()
            return s or None
        except UnicodeDecodeError:
            return None
    return None


def _first_str(d: dict[str, Any], *keys: str) -> str | None:
    for k in keys:
        s = _scalar_to_str(d.get(k))
        if s is not None:
            return s
    return None


def _field_or_metadata(
    classified_doc: dict[str, Any],
    key: str,
    *,
    meta_key: str | None = None,
) -> str | None:
    """Prefer top-level ``key``, then ``metadata[meta_key or key]``."""
    v = _first_str(classified_doc, key)
    if v is not None:
        return v
    return _first_str(_metadata(classified_doc), meta_key or key)


def _get_dotted(doc: dict[str, Any], path: str) -> str | None:
    """Resolve ``a.b.c`` style paths; returns None if any segment is missing or not a dict."""
    if "." not in path:
        return _scalar_to_str(doc.get(path))
    cur: Any = doc
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return _scalar_to_str(cur)


def _multi_key_top_or_metadata(doc: dict[str, Any], *keys: str) -> str | None:
    """Try flat keys on top-level, then ``metadata``, then dotted paths on each."""
    flat = [k for k in keys if "." not in k]
    dotted = [k for k in keys if "." in k]
    v = _first_str(doc, *flat) if flat else None
    if v is not None:
        return v
    meta = _metadata(doc)
    v = _first_str(meta, *flat) if flat else None
    if v is not None:
        return v
    for path in dotted:
        v = _get_dotted(doc, path)
        if v is not None:
            return v
        v = _get_dotted(meta, path)
        if v is not None:
            return v
    return None


def _parse_object_id(value: str | ObjectId) -> ObjectId:
    if isinstance(value, ObjectId):
        return value
    try:
        return ObjectId(str(value).strip())
    except InvalidId as e:
        raise ValueError(f"Invalid ObjectId value: {value!r}") from e


def build_job_description_context(
    classified_doc: dict[str, Any],
    *,
    settings: Settings | None = None,
) -> JobDescriptionContext:
    """
    Extract listing fields from a ClassifiedJobs document.

    All Mongo field names are configured via ``Settings`` / environment variables.
    """
    settings = settings or get_settings()
    fp = _first_str(classified_doc, *settings.keys_fingerprint)
    job_id = classified_doc.get(settings.mongo_field_job_id)

    title = _field_or_metadata(classified_doc, settings.field_title)
    employer = _field_or_metadata(classified_doc, settings.field_employer)
    category = _field_or_metadata(classified_doc, settings.field_category)
    application_url = _field_or_metadata(classified_doc, settings.field_application_url)
    posted_date = _field_or_metadata(classified_doc, settings.field_posted_date)
    closing_date = _field_or_metadata(classified_doc, settings.field_closing_date)
    source_platform = _field_or_metadata(classified_doc, settings.field_source_platform)

    raw_jd = _multi_key_top_or_metadata(classified_doc, *settings.keys_description)
    req_text = _multi_key_top_or_metadata(classified_doc, *settings.keys_requirements)
    salary = _multi_key_top_or_metadata(classified_doc, *settings.keys_salary)
    city = _multi_key_top_or_metadata(classified_doc, *settings.keys_location)
    county = _multi_key_top_or_metadata(classified_doc, *settings.keys_county)
    employment = _multi_key_top_or_metadata(classified_doc, *settings.keys_employment)

    return JobDescriptionContext(
        job_id=job_id,
        job_fingerprint=fp,
        title=title,
        employer=employer,
        category=category,
        application_url=application_url,
        posted_date=posted_date,
        closing_date=closing_date,
        source_platform=source_platform,
        raw_job_description=raw_jd,
        requirements_text=req_text,
        salary=salary,
        city=city,
        county=county,
        employment_type=employment,
        classified=dict(classified_doc),
    )


def get_job_context_by_job_id(
    job_id: str | ObjectId,
    *,
    settings: Settings | None = None,
    client: MongoClient | None = None,
) -> JobDescriptionContext | None:
    """
    Load ClassifiedJobs by the configured job id field (see ``CLASSIFIED_MONGO_FIELD_JOB_ID``).

    Returns ``None`` if no classified document exists for that id.
    """
    settings = settings or get_settings()
    oid = _parse_object_id(job_id)
    coll = classified_jobs_collection(settings=settings, db=get_database(client, settings=settings))
    classified = coll.find_one({settings.mongo_field_job_id: oid})
    if not classified:
        return None
    return build_job_description_context(classified, settings=settings)


def get_job_context_by_fingerprint(
    job_fingerprint: str,
    *,
    settings: Settings | None = None,
    client: MongoClient | None = None,
) -> JobDescriptionContext | None:
    """
    Load ClassifiedJobs by the configured fingerprint field
    (see ``CLASSIFIED_MONGO_FIELD_JOB_FINGERPRINT``).

    Returns ``None`` if no classified document exists for that fingerprint.
    """
    settings = settings or get_settings()
    coll = classified_jobs_collection(settings=settings, db=get_database(client, settings=settings))
    classified = coll.find_one({settings.mongo_field_job_fingerprint: job_fingerprint})
    if not classified:
        return None
    return build_job_description_context(classified, settings=settings)
