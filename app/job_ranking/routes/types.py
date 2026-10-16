from __future__ import annotations

from datetime import datetime
from typing import Any, Union

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from app.job_ranking.types import HealthStatus, JobResult, Llm3Outcome

JsonValue = Union[str, int, float, bool, None, dict[str, "JsonValue"], list["JsonValue"]]


def normalize_mongo_extended_json(obj: JsonValue) -> JsonValue:
    """Convert MongoDB Extended JSON (``$oid``, ``$date``) to plain Python types.

    Accepts documents copy-pasted from MongoDB Compass / Atlas / mongoexport.
    """
    if isinstance(obj, dict):
        if len(obj) == 1:
            if "$oid" in obj:
                return str(obj["$oid"])
            if "$date" in obj:
                return str(obj["$date"])
            if "$numberInt" in obj:
                return int(str(obj["$numberInt"]))
            if "$numberLong" in obj:
                return int(str(obj["$numberLong"]))
            if "$numberDouble" in obj:
                return float(str(obj["$numberDouble"]))
        return {k: normalize_mongo_extended_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_mongo_extended_json(item) for item in obj]
    return obj


class _PassThrough(BaseModel):
    """Typed where the pipeline depends on the shape; any other key is kept as sent."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class LinkedEntity(_PassThrough):
    """One taxonomy concept the classifier linked an extracted entity to."""

    taxonomy: str | None = Field(
        None,
        description="Taxonomy of the linked concept. Only `esco` links are used.",
        examples=["esco"],
    )
    label: str | None = Field(
        None,
        description="Preferred label of the concept.",
        examples=["Python (computer programming)"],
    )
    uri: str | None = Field(
        None,
        description="Concept URI. The ESCO UUID is parsed from its last path segment.",
        examples=["http://data.europa.eu/esco/skill/5247a6ce-6ed0-4fb3-b121-758921dcc44e"],
    )
    similarity_score: float | None = Field(
        None,
        description="Classifier similarity. Filtered by `CLASSIFIED_LINKED_ENTITY_MIN_SIMILARITY` "
        "when set; the best-scoring link wins duplicates.",
        examples=[0.91],
    )


class ClassifiedEntity(_PassThrough):
    """One entity the classifier extracted from the job ad, with its taxonomy links."""

    entity_type: str | None = Field(
        None,
        description="Entity kind. `skill` links are ranked by LLM1; `occupation` links select "
        "the occupation-to-skill taxonomy menu. Other kinds are ignored unless listed in "
        "`CLASSIFIED_LINKED_ENTITY_TYPES`.",
        examples=["skill"],
    )
    linked_entities: list[LinkedEntity] = Field(
        default_factory=list, description="Taxonomy concepts linked to this entity."
    )


class Classification(_PassThrough):
    """Classifier output carried on the job document."""

    entities: list[ClassifiedEntity] = Field(
        default_factory=list, description="Extracted entities, each with its taxonomy links."
    )


_RANK_REQUEST_EXAMPLE: dict[str, Any] = {
    "_id": {"$oid": "6650c1f2a1b2c3d4e5f60718"},
    "job_id": "jid-1",
    "job_fingerprint": "fp-1",
    "title": "Software Engineer",
    "employer": "ACME",
    "description": "Build and maintain backend services in Python.",
    "requirements": "3 years of experience.",
    "city": "Nairobi",
    "classification": {
        "entities": [
            {
                "entity_type": "skill",
                "linked_entities": [
                    {
                        "taxonomy": "esco",
                        "label": "Python (computer programming)",
                        "uri": "http://data.europa.eu/esco/skill/5247a6ce-6ed0-4fb3-b121-758921dcc44e",
                        "similarity_score": 0.91,
                    }
                ],
            }
        ]
    },
}


class RankRequest(_PassThrough):
    """A classified job document, as stored in the classifier's `classified_jobs` collection.

    The fields below are the defaults the pipeline reads; deployments can point it at other
    field names (`CLASSIFIED_FIELD_*`, `CLASSIFIED_KEYS_*`), and it also falls back to
    `metadata.<field>`. Every other field (salary, location, dates, `translation`, scraper
    enrichment, …) is accepted and passed through unchanged.
    """

    model_config = ConfigDict(json_schema_extra={"examples": [_RANK_REQUEST_EXAMPLE]})

    id: str | int | None = Field(
        None,
        alias="_id",
        description="Source document id. Used as `source_classified_id`, and as the upsert key "
        "when the document has neither a job id nor a fingerprint. `{\"$oid\": …}` is accepted.",
    )
    job_id: str | int | None = Field(None, description="Job id (`CLASSIFIED_MONGO_FIELD_JOB_ID`).")
    job_fingerprint: str | None = Field(
        None, description="Listing fingerprint (`CLASSIFIED_KEYS_FINGERPRINT`)."
    )
    title: str | None = Field(None, description="Listing title.")
    employer: str | None = Field(None, description="Employer name.")
    description: str | None = Field(None, description="Full job description text.")
    requirements: str | None = Field(None, description="Requirements text.")
    language: str | None = Field(
        None,
        description="Language pack to rank against (`en`, `es`, or a locale such as `AR-es`). "
        "The `?language=` query parameter wins over this field.",
        examples=["en"],
    )
    classification: Classification | None = Field(
        None,
        description="Classifier output. Without it the job is still ranked, but LLM1 has no "
        "skills to rank and there is no occupation-skill menu.",
    )

    @model_validator(mode="before")
    @classmethod
    def _unwrap_extended_json(cls, data: JsonValue) -> JsonValue:
        return normalize_mongo_extended_json(data)

    def to_document(self) -> dict[str, Any]:
        """The document as the pipeline reads it: sent keys only, original names (``_id``)."""
        return self.model_dump(by_alias=True, exclude_unset=True)


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class ClassifiedSkill(_PassThrough):
    label: str = Field(description="Classifier label.")
    uri: str = Field(description="ESCO skill URI.")
    esco_uuid: str | None = Field(description="ESCO skill UUID, lowercased.")
    similarity_score: float | None = Field(description="Classifier similarity score.")
    tabiya_skill_id: str | None = Field(description="Tabiya skill id, when `skills.csv` maps it.")


class ClassifiedOccupation(_PassThrough):
    label: str = Field(description="Classifier label.")
    uri: str = Field(description="ESCO occupation URI.")
    esco_uuid: str | None = Field(description="ESCO occupation UUID, lowercased.")
    similarity_score: float | None = Field(description="Classifier similarity score.")
    tabiya_occupation_id: str | None = Field(
        description="Tabiya occupation id, when `occupations.csv` maps it."
    )


class SkillCounts(_PassThrough):
    total: int = Field(description="Taxonomy skills listed.")
    essential: int = Field(description="Skills with relation type `essential`.")
    optional: int = Field(description="Skills with relation type `optional`.")
    unspecified: int = Field(description="Skills with no relation type.")
    relevant_total: int | None = Field(
        None,
        description="Skills the LLM3 relevance call kept. Present on an occupation block only in "
        "`RERANKER_LLM3_MODE=call`; in `occupation_to_skills.meta` always present, and 0 "
        "when no call ran.",
    )


class OccupationSkill(_PassThrough):
    skill_id: str = Field(description="ESCO skill UUID.")
    tabiya_skill_id: str = Field(description="Tabiya skill id from the relations CSV.")
    label: str = Field(description="Skill label in the request's language pack.")
    uri: str = Field(description="ESCO skill URI.")
    relation_type: str = Field(
        description="`essential`, `optional`, or `unspecified` — the taxonomy relation."
    )
    signalling_value: str | None = Field(
        None, description="Relations-CSV signalling value, when the CSV has one."
    )
    signalling_label: str | None = Field(
        None, description="Relations-CSV signalling tier (`high`, `medium`, `low`), when present."
    )
    llm_is_relevant: bool | None = Field(
        None,
        description="LLM3 relevance verdict. Present only in `RERANKER_LLM3_MODE=call`.",
    )


class OccupationSkillsBlock(_PassThrough):
    occupation_label: str = Field(description="Classifier occupation label.")
    occupation_esco_uuid: str | None = Field(description="ESCO occupation UUID.")
    occupation_uri: str = Field(description="ESCO occupation URI.")
    tabiya_occupation_id: str = Field(description="Tabiya occupation id the skills come from.")
    skill_counts: SkillCounts = Field(description="Counts over this block's `skills`.")
    skills: list[OccupationSkill] = Field(
        description="Taxonomy skills for the occupation, essential first, after "
        "`RERANKER_OCCUPATION_SKILLS_SIGNAL_INCLUDE`."
    )


class OccupationToSkillsMeta(_PassThrough):
    occupation_count: int = Field(description="Number of occupation blocks.")
    skill_counts: SkillCounts = Field(description="Counts summed over every block.")


class OccupationToSkills(_PassThrough):
    meta: OccupationToSkillsMeta = Field(description="Totals across all blocks.")
    items: list[OccupationSkillsBlock] = Field(
        description="One block per classifier occupation found in the taxonomy."
    )


class WorkActivity(_PassThrough):
    id: str = Field(description="ONET work activity element id.", examples=["4.A.3.b.1"])
    name: str = Field(description="Work activity name (simplified label when available).")


class WorkActivities(_PassThrough):
    items: list[WorkActivity] = Field(description="Work activities LLM2 picked for the job.")


class RankedSkill(_PassThrough):
    label: str = Field(description="Skill label as offered to LLM1.")
    uri: str | None = Field(description="ESCO skill URI.")
    esco_uuid: str | None = Field(description="ESCO skill UUID, lowercased.")
    tabiya_skill_id: str | None = Field(description="Tabiya skill id, when `skills.csv` maps it.")
    similarity_score: float | None = Field(
        None, description="Classifier similarity score. Only on `discarded` rows that have one."
    )


class LlmClassifiedSkills(_PassThrough):
    essential: list[RankedSkill] = Field(description="Skills LLM1 ranked essential.")
    optional: list[RankedSkill] = Field(description="Skills LLM1 ranked optional.")
    discarded: list[RankedSkill] = Field(
        description="Classifier skills LLM1 ranked neither essential nor optional."
    )
    unmatched_essential: list[str] = Field(
        description="Essential labels LLM1 returned that match no candidate."
    )
    unmatched_optional: list[str] = Field(
        description="Optional labels LLM1 returned that match no candidate."
    )


class AttributeEstimateOut(_PassThrough):
    attribute_name: str = Field(
        description="Attribute from the job-attribute schema.", examples=["earnings_per_month"]
    )
    selected_level_id: str = Field(
        description="Level id LLM2 chose, from that attribute's levels.", examples=["earn_15k"]
    )
    confidence: str = Field(
        description="LLM2's confidence: `low`, `medium`, or `high`.", examples=["medium"]
    )
    reasoning: str = Field(description="LLM2's reasoning for the level.")


class LlmJobAttributes(_PassThrough):
    summary: str = Field(description="LLM2's one-paragraph summary of the job.")
    estimates: list[AttributeEstimateOut] = Field(
        description="One estimate per attribute in the language pack's job-attribute schema."
    )
    attributes: dict[str, str | bool] = Field(
        description="`attribute_name` → selected level id (or flag), for direct lookup."
    )


class ClassifierMetadata(_PassThrough):
    """Listing fields read from the source document (top level, then ``metadata``).

    Scraper enrichment fields (``province``, ``sector``, ``isco_major_group``, ``zqf_min``, …)
    are added as extra keys when the source document has them.
    """

    title: str | None = Field(None, description="Listing title (translated when preferred).")
    employer: str | None = Field(None, description="Employer name.")
    category: str | None = Field(None, description="Listing category.")
    application_url: str | None = Field(None, description="Where to apply.")
    posted_date: str | None = Field(None, description="Posting date, as the source wrote it.")
    closing_date: str | None = Field(
        None, description="Closing date, as the source wrote it. Drives `is_active` in the batch."
    )
    source_platform: str | None = Field(None, description="Job board the listing came from.")
    city: str | None = Field(None, description="City; a list in the source is joined with `, `.")
    salary: str | None = Field(None, description="Salary text.")
    employment_type: str | None = Field(None, description="Employment type text.")
    job_description: str | None = Field(None, description="Description text given to the LLMs.")
    requirements: str | None = Field(None, description="Requirements text given to the LLMs.")


class LlmRerankerMeta(_PassThrough):
    schema_version: str = Field(description="Ranked-document schema version.", examples=["32"])
    llm_provider: str = Field(description="Provider alias that served the LLM calls.")
    skill_ranker_model: str = Field(description="Model behind LLM1.")
    occupation_skills_relevance_model: str = Field(
        description="Model configured for LLM3. Recorded whether or not the call ran — see "
        "`llm3_outcome` on the response."
    )
    job_attributes_model: str = Field(description="Model behind LLM2.")
    classified_linked_entity_types: list[str] = Field(
        description="Entity types read from `classification.entities`."
    )


class RankedJobDocument(_PassThrough):
    """The ranked job — the document the batch upserts into the target collection,
    minus the job embedding, which only the batch computes."""

    job_id: str | int | None = Field(description="Job id from the source document.")
    job_fingerprint: str | None = Field(description="Listing fingerprint from the source document.")
    is_active: bool = Field(description="Always true on ranking; the batch later syncs it.")
    processed_at: datetime = Field(description="When the job was ranked (UTC).")
    source_classified_id: str | int | None = Field(description="`_id` of the source document.")
    classified_skills: list[ClassifiedSkill] = Field(
        description="Classifier ESCO skill links, deduplicated."
    )
    classified_occupations: list[ClassifiedOccupation] = Field(
        description="Classifier ESCO occupation links, deduplicated."
    )
    occupation_to_skills: OccupationToSkills = Field(
        description="Taxonomy skills of each classified occupation."
    )
    work_activity: WorkActivities = Field(description="ONET work activities from LLM2.")
    skill_groups_origin_uuids: list[str] = Field(
        description="Skill-group ids derived from the essential and optional skills."
    )
    llm_classified_skills: LlmClassifiedSkills = Field(description="LLM1's skill ranking.")
    llm_job_attributes: LlmJobAttributes = Field(description="LLM2's job attributes.")
    classifier_metadata: ClassifierMetadata = Field(
        description="Listing fields copied from the source document."
    )
    llm_reranker_meta: LlmRerankerMeta = Field(
        description="Schema version and the models that produced the document."
    )

    @field_serializer("processed_at", when_used="json")
    def _processed_at_iso(self, value: datetime) -> str:
        # ``+00:00`` as the dict response always wrote it, not pydantic's ``Z``.
        return value.isoformat()


class RankResponse(BaseModel):
    ranked_document: RankedJobDocument | None = Field(
        description="The ranked job, or null if LLM2 produced no job attributes (nothing to rank)."
    )
    timings: dict[str, float] = Field(
        description="Wall-clock seconds per step that ran: `taxonomy_s`, `llm1_s`, `llm2_s`, "
        "`llm3_s` (only when the LLM3 call ran), and `total_s`.",
        examples=[{"taxonomy_s": 0.004, "llm1_s": 2.1, "llm2_s": 3.4, "total_s": 5.6}],
    )
    ran_llm1: bool = Field(
        description="LLM1 (skill ranker) was called — false when the job has no ESCO skill links."
    )
    ran_llm2: bool = Field(description="LLM2 (job attributes) was called.")
    ran_llm3: bool = Field(
        description="The standalone LLM3 relevance call was made. Only possible with "
        "`RERANKER_LLM3_MODE=call`; in the default `fold` mode the menu goes to LLM1 instead."
    )
    llm3_outcome: Llm3Outcome = Field(
        description="What the LLM3 step did: `disabled` (`RERANKER_ENABLE_LLM3=0`), `no_menu` "
        "(no occupation skills), `menu_filtered_out` (`RERANKER_LLM3_*` filters emptied it), "
        "`folded` (menu added to LLM1's candidates), or `called` (LLM3 call ran)."
    )

    @classmethod
    def from_job_result(cls, result: JobResult) -> RankResponse:
        return cls(
            ranked_document=(
                RankedJobDocument.model_validate(result.ranked_document)
                if result.ranked_document is not None
                else None
            ),
            timings=result.timings,
            ran_llm1=result.ran_llm1,
            ran_llm2=result.ran_llm2,
            ran_llm3=result.ran_llm3,
            llm3_outcome=result.llm3_outcome,
        )


class HealthResponse(BaseModel):
    status: str = Field(description="Always `ok` once the process answers.", examples=["ok"])
    taxonomy_loaded: bool = Field(description="The default language's taxonomy is loaded.")
    default_language: str = Field(description="Language used when a request names none.")
    supported_languages: list[str] = Field(description="Language packs this image ships.")
    loaded_languages: list[str] = Field(
        description="Language packs loaded so far; others load on their first request."
    )

    @classmethod
    def from_status(cls, status: HealthStatus) -> HealthResponse:
        return cls(
            status="ok",
            taxonomy_loaded=status.taxonomy_loaded,
            default_language=status.default_language,
            supported_languages=status.supported_languages,
            loaded_languages=status.loaded_languages,
        )
