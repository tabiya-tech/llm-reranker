"""Row models for the occupation-taxonomy CSVs (``occupations.csv``, ``occupation_to_skill_relations.csv``).

Column names vary across deployments (``OCCUPATIONID`` vs ``occupation_uuid`` vs a positional
2-column file, …), so header inference stays a separate pre-pass over the CSV's fieldnames
(see ``utils.occupation_to_skills_mapping``'s ``_infer_*_column`` functions) that maps each
file's actual headers to these models' canonical field names; only the per-row dict, already
keyed by canonical name, is validated into one of these models.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.domain_base import SourceRecord
from utils.classifier_linked_entities import LinkedEntityCandidate


class OccupationLinkedEntity(BaseModel):
    """One classifier-linked ESCO occupation hit, shaped for ``occupations.csv`` lookup helpers.

    Mirrors the classifier's own ``linked_entities`` row shape (``uri``, ``label``, ``taxonomy``,
    ``similarity_score``, ``originUUID``) so existing lookup helpers that read this shape by key
    (:func:`utils.classifier_linked_entities.linked_entity_preferred_taxonomy_id` and friends)
    need no change — only its three independent construction call sites do.
    """

    model_config = ConfigDict(populate_by_name=True)

    uri: str = Field(description="ESCO occupation concept URI.")
    label: str = Field(description="Classifier label for the occupation.")
    taxonomy: str = Field(default="esco", description="Always `esco` for linked occupation hits.")
    similarity_score: float | None = Field(
        default=None, description="Classifier similarity score, when present."
    )
    origin_uuid: str | None = Field(
        default=None,
        alias="originUUID",
        description="Concept UUID, when resolved from the linked entity.",
    )


def occupation_linked_entity_from_candidate(c: LinkedEntityCandidate) -> OccupationLinkedEntity:
    """Build the shared occupation-lookup row from one classifier linked-entity candidate."""
    return OccupationLinkedEntity.model_validate(
        {
            "uri": c.uri,
            "label": c.label,
            "similarity_score": c.similarity_score,
            "originUUID": c.concept_uuid,
        }
    )


class OccupationCsvRow(SourceRecord):
    """One row of ``occupations.csv``, after column-name inference maps it to canonical keys."""

    id: str = Field(description="Tabiya occupation id (the row's `ID` column).")
    uuid_history: str | None = Field(
        default=None, description="Raw `UUIDHISTORY` cell (newline-separated ESCO uuids/URIs)."
    )
    origin_uri: str | None = Field(default=None, description="`ORIGINURI` cell, if present.")


class OccupationSkillRelationCsvRow(SourceRecord):
    """One row of ``occupation_to_skill_relations.csv``, after column-name inference."""

    occupation_id: str = Field(description="`OCCUPATIONID` cell (Tabiya id, ESCO uuid, or URI).")
    skill_id: str = Field(description="`SKILLID` cell (Tabiya id, ESCO uuid, or URI).")
    relation_type: str | None = Field(
        default=None, description="`RELATIONTYPE` cell, when the column is present."
    )
    occupation_type: str | None = Field(
        default=None, description="`OCCUPATIONTYPE` cell, for the type filter."
    )
    skill_label: str | None = Field(
        default=None, description="Skill display-label column, when one is present."
    )
    signalling_value: str | None = Field(
        default=None, description="`SIGNALLINGVALUE` cell, when present."
    )
    signalling_label: str | None = Field(
        default=None, description="`SIGNALLINGVALUELABEL` cell, when present."
    )


class SkillsCsvRow(SourceRecord):
    """One row of ``skills.csv``, after column-name inference maps it to canonical keys."""

    id: str = Field(description="Tabiya skill id (the row's `ID` column).")
    preferred_label: str = Field(description="The row's `PREFERREDLABEL` column.")
    uuid_history: str | None = Field(
        default=None, description="Raw `UUIDHISTORY` cell (newline-separated ESCO uuids/URIs)."
    )
    origin_uri: str | None = Field(default=None, description="`ORIGINURI` cell, if present.")
