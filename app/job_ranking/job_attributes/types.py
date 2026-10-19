"""Job-attributes schema, ONET work activities and the LLM 2 result."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.domain_base import DomainModel


class AttributeLevel(DomainModel):
    """One selectable level of a job attribute (e.g. an earnings bracket)."""

    id: str = Field(description="Level id referenced by LLM2's `selected_level_id`.")
    label: str = Field(description="Human-readable label for this level.")
    value: float | None = Field(
        default=None, description="Numeric value for this level, when ordinal."
    )


class AttributeDefinition(DomainModel):
    """One job attribute from the language pack's job-attributes schema."""

    name: str = Field(description="Attribute key, matched against LLM2's `attribute_name`.")
    label: str = Field(description="Human-readable attribute name.")
    group: str = Field(description="Schema grouping this attribute belongs to.")
    type: str = Field(description="Attribute value type, as declared by the schema.")
    coding: str = Field(description="Schema coding scheme for this attribute's levels.")
    levels: tuple[AttributeLevel, ...] = Field(description="Selectable levels for this attribute.")
    base_level_id: str | None = Field(
        default=None, description="Default/baseline level id, when defined."
    )


class JobAttributesSchema(DomainModel):
    """The full job-attributes schema loaded from `Settings.job_attributes_schema_path`."""

    attributes: tuple[AttributeDefinition, ...] = Field(
        description="Every attribute LLM2 is asked to estimate."
    )
    metadata: dict[str, Any] | None = Field(
        default=None, description="Free-form schema metadata, passed through unvalidated."
    )


class OnetWorkActivity(DomainModel):
    """One ONET work activity descriptor from the language pack's work-activity list."""

    wa_element_id: str = Field(description="ONET work activity element id.")
    wa_element_name: str = Field(description="Work activity name.")
    wa_element_name_simplified: str = Field(
        description="Simplified display label (falls back to the full name)."
    )
    wa_element_name_swahili: str | None = Field(
        default=None, description="Swahili translation, when the language pack provides one."
    )


class AttributeEstimate(DomainModel):
    """One attribute row from LLM2 JSON (name, level, confidence, reasoning)."""

    attribute_name: str = Field(description="Attribute from the job-attribute schema.")
    selected_level_id: str = Field(description="Level id LLM2 chose, from that attribute's levels.")
    confidence: str = Field(description="LLM2's confidence: `low`, `medium`, or `high`.")
    reasoning: str = Field(description="LLM2's reasoning for the level.")


class JobAttributesResult(DomainModel):
    """Parsed output of LLM call 2: structured estimates + summary + ONET work activities."""

    attribute_estimates: tuple[AttributeEstimate, ...] = Field(
        description="One estimate per attribute LLM2 returned a valid level for."
    )
    summary: str = Field(description="LLM2's one-paragraph summary of the job.")
    attributes: dict[str, str | bool] = Field(
        description="`attribute_name` -> selected level id (or flag), for direct lookup."
    )
    work_activity_ids: tuple[str, ...] = Field(
        description="ONET work activity ids LLM2 selected, filtered to the allowed set."
    )
    raw_response: str = Field(description="LLM2's raw response text, fence-stripped.")
