"""The per-language lookup data every ranking step reads."""

from __future__ import annotations

from pydantic import Field

from app.domain_base import DomainModel
from utils.uuid_to_tabiya_id import SkillsUuidTabiyaMap


class TaxonomyContext(DomainModel):
    """Pre-loaded CSV lookups for one language pack, shared read-only across all jobs.

    Built by :class:`app.taxonomy.repository.TaxonomyRepository`.
    """

    occ_map: dict[str, frozenset[str]] = Field(
        description="Occupation id -> set of related skill ids (union, for the reranker)."
    )
    csv_skill_labels: dict[str, str] | None = Field(
        description="Optional per-skill display labels from the relations CSV."
    )
    occ_skill_relation: dict[str, dict[str, str]] | None = Field(
        description="Occupation id -> skill id -> relation type (essential/optional/…)."
    )
    occ_skill_signalling: dict[str, dict[str, dict[str, str]]] | None = Field(
        description="Occupation id -> skill id -> signalling value/label, when the CSV has them."
    )
    occ_bridge: dict[str, str] = Field(
        description="Lookup key (ESCO uuid, Tabiya id, …) -> occupations.csv row id."
    )
    occ_all_ids: frozenset[str] = Field(
        description="Every distinct occupation id in occupations.csv."
    )
    skills_uuid_tabiya_map: SkillsUuidTabiyaMap = Field(
        description="ESCO uuid <-> Tabiya skill id maps from skills.csv."
    )
    skill_desc_by_esco: dict[str, str] = Field(
        description="Lowercased ESCO skill uuid -> description, when configured."
    )
    occ_label_by_esco: dict[str, str] = Field(
        description="Lowercased ESCO occupation uuid -> label, for LLM2 prompts."
    )
    occ_desc_by_esco: dict[str, str] = Field(
        description="Lowercased ESCO occupation uuid -> description, when configured."
    )
    skill_groups_by_skill: dict[str, frozenset[str]] = Field(
        default_factory=dict,
        description="`skill_hierarchy.csv`: Tabiya skill ID -> parent skill-group ids.",
    )
