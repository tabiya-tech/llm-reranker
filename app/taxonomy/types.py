"""The per-language lookup data every ranking step reads."""

from __future__ import annotations

from dataclasses import dataclass, field

from utils.uuid_to_tabiya_id import SkillsUuidTabiyaMap


@dataclass
class TaxonomyContext:
    """Pre-loaded CSV lookups for one language pack, shared read-only across all jobs.

    Built by :class:`app.taxonomy.repository.TaxonomyRepository`.
    """

    occ_map: dict[str, frozenset[str]]
    csv_skill_labels: dict[str, str] | None
    occ_skill_relation: dict[str, dict[str, str]] | None
    occ_skill_signalling: dict[str, dict[str, dict[str, str]]] | None
    occ_bridge: dict[str, str]
    occ_all_ids: frozenset[str]
    skills_uuid_tabiya_map: SkillsUuidTabiyaMap
    skill_desc_by_esco: dict[str, str]
    occ_label_by_esco: dict[str, str]
    occ_desc_by_esco: dict[str, str]
    # ``skill_hierarchy.csv``: Tabiya skill ``ID`` → parent skill-group ids.
    skill_groups_by_skill: dict[str, frozenset[str]] = field(default_factory=dict)
