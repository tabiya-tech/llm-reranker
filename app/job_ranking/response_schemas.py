"""JSON Schemas for the three LLM passes, sent as the provider's structured-output schema.

Each prompt already spells out its JSON shape in prose; these schemas say the same thing
in a form the model is decoded against (Gemini's ``response_json_schema``), so the reply
always parses and every id/label it carries is one the prompt offered. The pass parsers
(``app.job_ranking.*.parser``) still validate, because a
provider without structured output gets only the prose contract.

Each reply shape is a Pydantic model; :func:`_json_schema` turns it into the dict sent to
the provider. Menus are the models' type parameters: a ``Literal`` over the menu the
prompt shows, built per call by :func:`_menu`. Above :data:`MAX_ENUM_VALUES` a menu is a
plain ``str``: very large enums make the schema too big for the API to accept, and the
parsers already discard values outside the menu.
"""

from __future__ import annotations

from typing import Any, Generic, Iterable, Literal, Sequence, TypeVar, cast

from pydantic import BaseModel, Field
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import core_schema

from app.job_ranking.job_attributes.types import JobAttributesSchema, OnetWorkActivity
from app.job_ranking.routes.types import OccupationSkillsBlock

# Largest menu sent as an ``enum``; beyond it the field is a plain string.
MAX_ENUM_VALUES = 500

Label = TypeVar("Label")
AttributeName = TypeVar("AttributeName")
LevelId = TypeVar("LevelId")
WorkActivityId = TypeVar("WorkActivityId")
SkillId = TypeVar("SkillId")
OccupationId = TypeVar("OccupationId")


class SkillRankerResponse(BaseModel, Generic[Label]):
    """LLM1: ``essential_skills`` / ``optional_skills``, each a list of menu labels."""

    essential_skills: list[Label]
    optional_skills: list[Label]


class ExperienceSkillRankerResponse(BaseModel, Generic[SkillId]):
    """``POST /rank/skills``: the candidate ids the experience shows, best first."""

    ranked_skill_ids: list[SkillId]


class AttributeEstimate(BaseModel, Generic[AttributeName, LevelId]):
    attribute_name: AttributeName
    selected_level_id: LevelId
    confidence: Literal["high", "medium", "low"]
    reasoning: str


class JobAttributesResponse(BaseModel, Generic[AttributeName, LevelId, WorkActivityId]):
    """LLM2: one estimate per attribute, a summary and 3–10 ONET ids."""

    attributes: list[AttributeEstimate[AttributeName, LevelId]]
    summary: str
    work_activities: list[WorkActivityId] = Field(min_length=3, max_length=10)


class RelevanceDecision(BaseModel, Generic[SkillId, OccupationId]):
    skill_id: SkillId
    tabiya_occupation_id: OccupationId
    is_relevant: bool


class OccupationSkillsRelevanceResponse(BaseModel, Generic[SkillId, OccupationId]):
    """LLM3: one ``{skill_id, tabiya_occupation_id, is_relevant}`` per judged skill."""

    decisions: list[RelevanceDecision[SkillId, OccupationId]]


def _menu(values: Iterable[str]) -> object:
    """``Literal`` over the distinct non-empty values, in order (``str`` if none or too many)."""
    distinct = tuple(dict.fromkeys(v for v in values if v))
    if not distinct or len(distinct) > MAX_ENUM_VALUES:
        return str
    return Literal[distinct]


class _ProviderJsonSchema(GenerateJsonSchema):
    """Pydantic's JSON Schema minus titles, with every ``Literal`` as an ``enum``."""

    def field_title_should_be_set(self, schema: object) -> bool:
        return False

    def model_schema(self, schema: core_schema.ModelSchema) -> JsonSchemaValue:
        out = super().model_schema(schema)
        out.pop("title", None)
        return out

    def literal_schema(self, schema: core_schema.LiteralSchema) -> JsonSchemaValue:
        out = super().literal_schema(schema)
        if "const" in out:  # a one-value menu
            out["enum"] = [out.pop("const")]
        return out


def _inline_refs(node: object, defs: dict[str, Any]) -> object:
    """Replace each ``{"$ref": "#/$defs/X"}`` with ``defs["X"]``, recursively."""
    if isinstance(node, dict):
        if "$ref" in node:
            return _inline_refs(defs[node["$ref"].rsplit("/", 1)[-1]], defs)
        return {k: _inline_refs(v, defs) for k, v in node.items()}
    if isinstance(node, list):
        return [_inline_refs(v, defs) for v in node]
    return node


def _json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """``model``'s JSON Schema with nested models inlined (no ``$defs`` / ``$ref``)."""
    schema = model.model_json_schema(schema_generator=_ProviderJsonSchema)
    defs = schema.pop("$defs", {})
    return cast(dict[str, Any], _inline_refs(schema, defs))


def skill_ranker_response_schema(skill_labels: Iterable[str]) -> dict[str, Any]:
    """LLM1: see :class:`SkillRankerResponse`."""
    return _json_schema(SkillRankerResponse[_menu(skill_labels)])


def experience_skill_ranker_response_schema(skill_ids: Iterable[str]) -> dict[str, Any]:
    """``POST /rank/skills``: see :class:`ExperienceSkillRankerResponse`."""
    return _json_schema(ExperienceSkillRankerResponse[_menu(skill_ids)])


def job_attributes_response_schema(
    schema: JobAttributesSchema,
    onet: Iterable[OnetWorkActivity],
    *,
    extra_fields: Sequence[str] = (),
) -> dict[str, Any]:
    """LLM2: see :class:`JobAttributesResponse`, plus one required boolean per name in
    ``extra_fields`` (the deployment's :attr:`Settings.job_attributes_extra_fields`).

    Level ids are the union across attributes; whether a level belongs to *its*
    attribute is left to :func:`app.job_ranking.job_attributes.parser.parse_job_attributes_response`, which keeps
    the schema small and free of per-attribute branches.
    """
    names = _menu(a.name for a in schema.attributes)
    level_ids = _menu(lev.id for a in schema.attributes for lev in a.levels)
    wa_ids = _menu(x.wa_element_id for x in onet)
    out = _json_schema(JobAttributesResponse[names, level_ids, wa_ids])
    for name in extra_fields:
        out["properties"][name] = {"type": "boolean"}
        out["required"].append(name)
    # One estimate per attribute: the count is per call, so it is set on the schema.
    n_attrs = len(schema.attributes)
    if n_attrs:
        out["properties"]["attributes"].update(minItems=n_attrs, maxItems=n_attrs)
    return out


def occupation_skills_relevance_response_schema(
    occupation_to_skills_items: list[OccupationSkillsBlock],
) -> dict[str, Any]:
    """LLM3: see :class:`OccupationSkillsRelevanceResponse`; ids are the judged skills'."""
    occupation_ids: list[str] = []
    skill_ids: list[str] = []
    for block in occupation_to_skills_items:
        tid = (block.tabiya_occupation_id or "").strip()
        if not tid:
            continue
        for sk in block.skills:
            if sk.relation_type not in ("essential", "optional"):
                continue
            sid = sk.skill_id.strip()
            if sid:
                occupation_ids.append(tid)
                skill_ids.append(sid)
    return _json_schema(OccupationSkillsRelevanceResponse[_menu(skill_ids), _menu(occupation_ids)])
