from __future__ import annotations

import itertools
import logging
import os
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
        logger.warning(
            "MongoDB upsert skipped: ranked job payload is empty (job attributes / LLM2 result missing)."
        )
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


RANKING_LOGS_COLLECTION_DEFAULT = "reranker-logs"


def _ranking_logs_collection(
    settings: Settings | None = None,
    client: MongoClient | None = None,
) -> Collection:
    settings = settings or get_settings()
    db = get_database(client, settings=settings)
    coll_name = (os.getenv("RERANKER_LOGS_COLLECTION") or RANKING_LOGS_COLLECTION_DEFAULT).strip()
    return db[coll_name]


def insert_ranking_run_log(
    *,
    settings: Settings | None = None,
    limit: int,
    model: str,
    source_collection: str,
    target_collection: str,
) -> ObjectId:
    """Insert a new run document with status ``running`` and return its ``_id``."""
    doc = {
        "started_at": datetime.utcnow(),
        "completed_at": None,
        "status": "running",
        "success": None,
        "error_message": None,
        "config": {
            "model": model,
            "limit": limit,
            "source_collection": source_collection,
            "target_collection": target_collection,
        },
        "stats": {
            "jobs_processed": 0,
            "upserts_queued": 0,
            "llm1_calls": 0,
            "llm2_calls": 0,
            "llm3_calls": 0,
        },
        "elapsed_s": None,
        "updated_at": datetime.utcnow(),
    }
    coll = _ranking_logs_collection(settings=settings)
    result = coll.insert_one(doc)
    logger.info("ranking-logs | run started | _id=%s", result.inserted_id)
    return result.inserted_id


def update_ranking_run_progress(
    run_id: ObjectId,
    *,
    jobs_processed: int,
    upserts_queued: int,
    llm1_calls: int,
    llm2_calls: int,
    llm3_calls: int,
    settings: Settings | None = None,
) -> None:
    """Update stats and ``updated_at`` on the run document (called periodically)."""
    coll = _ranking_logs_collection(settings=settings)
    coll.update_one(
        {"_id": run_id},
        {
            "$set": {
                "stats.jobs_processed": jobs_processed,
                "stats.upserts_queued": upserts_queued,
                "stats.llm1_calls": llm1_calls,
                "stats.llm2_calls": llm2_calls,
                "stats.llm3_calls": llm3_calls,
                "updated_at": datetime.utcnow(),
            }
        },
    )


def complete_ranking_run_log(
    run_id: ObjectId,
    *,
    success: bool,
    jobs_processed: int,
    upserts_queued: int,
    llm1_calls: int,
    llm2_calls: int,
    llm3_calls: int,
    elapsed_s: float,
    error_message: str | None = None,
    settings: Settings | None = None,
) -> None:
    """Mark the run as ``completed`` or ``failed`` with final stats."""
    now = datetime.utcnow()
    coll = _ranking_logs_collection(settings=settings)
    coll.update_one(
        {"_id": run_id},
        {
            "$set": {
                "completed_at": now,
                "status": "completed" if success else "failed",
                "success": success,
                "error_message": error_message,
                "stats.jobs_processed": jobs_processed,
                "stats.upserts_queued": upserts_queued,
                "stats.llm1_calls": llm1_calls,
                "stats.llm2_calls": llm2_calls,
                "stats.llm3_calls": llm3_calls,
                "elapsed_s": round(elapsed_s, 3),
                "updated_at": now,
            }
        },
    )
    logger.info(
        "ranking-logs | run %s | _id=%s | jobs=%s upserts=%s elapsed=%.3fs",
        "completed" if success else "failed",
        run_id,
        jobs_processed,
        upserts_queued,
        elapsed_s,
    )


def iter_classified_jobs(
    limit: int = 100,
    skip: int = 0,
    *,
    sort_field: str = "_id",
    sort_direction: int = -1,
    settings: Settings | None = None,
    client: MongoClient | None = None,
    query: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """
    Stream documents from the configured source collection, newest first by default.

    ``limit`` <= 0 means no limit (entire collection). Results are streamed one document at a time.
    """
    settings = settings or get_settings()
    coll = classified_jobs_collection(settings=settings, db=get_database(client, settings=settings))
    cur = coll.find(query or {}).sort(sort_field, sort_direction).skip(skip)
    if limit > 0:
        cur = cur.limit(limit)
    yield from cur


def iter_classified_jobs_chunks(
    chunk_size: int,
    *,
    limit: int = 0,
    skip: int = 0,
    sort_field: str = "_id",
    sort_direction: int = -1,
    query: dict[str, Any] | None = None,
    settings: Settings | None = None,
    client: MongoClient | None = None,
) -> Iterator[list[dict[str, Any]]]:
    """
    Yield lists of documents from the source collection (same order as ``iter_classified_jobs``).

    Use for parallel batch processing without loading unlimited collections into memory.
    ``chunk_size`` is typically ``>= RERANKER_PARALLEL_WORKERS``.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    it = iter_classified_jobs(
        limit=limit,
        skip=skip,
        sort_field=sort_field,
        sort_direction=sort_direction,
        settings=settings,
        client=client,
        query=query,
    )
    while True:
        chunk = list(itertools.islice(it, chunk_size))
        if not chunk:
            break
        yield chunk


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
