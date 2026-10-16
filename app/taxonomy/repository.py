"""Builds a language pack's :class:`TaxonomyContext` from its CSV files."""

from __future__ import annotations

from app.config import Settings
from app.taxonomy.csv_loaders import (
    load_occupation_labels_and_descriptions_by_esco,
    load_occupation_skill_relations,
    load_occupations_lookup,
    load_skill_description_by_esco,
    load_skill_to_groups_map,
    load_skills_uuid_tabiya_map,
)
from app.taxonomy.types import TaxonomyContext


class TaxonomyRepository:
    """Reads the taxonomy CSVs of a language pack. Every call re-reads the files.

    Caching is the caller's choice: the batch loads one context per run, the API keeps one per
    language in :class:`app.taxonomy.service.TaxonomyService`.
    """

    def load(self, settings: Settings) -> TaxonomyContext:
        occ_map, csv_skill_labels, occ_skill_relation, occ_skill_signalling = (
            load_occupation_skill_relations(
                str(settings.occupation_to_skill_relations_path.resolve()),
                accepted_occupation_types=settings.occupation_skill_relation_occupation_types,
            )
        )
        occ_bridge, occ_all_ids = load_occupations_lookup(
            str(settings.occupations_csv_path.resolve())
        )
        skills_uuid_tabiya_map = load_skills_uuid_tabiya_map(
            str(settings.skills_csv_path.resolve())
        )
        skill_desc_by_esco: dict[str, str] = {}
        if (
            settings.skills_csv_include_descriptions
            and (settings.skills_csv_description_column or "").strip()
        ):
            skill_desc_by_esco = load_skill_description_by_esco(
                str(settings.skills_csv_path.resolve()),
                settings.skills_csv_description_column.strip(),
            )
        occ_label_by_esco, occ_desc_by_esco = load_occupation_labels_and_descriptions_by_esco(
            str(settings.occupations_csv_path.resolve()),
            label_column=settings.occupations_csv_label_column,
            include_descriptions=settings.occupations_csv_include_descriptions,
            description_column=(settings.occupations_csv_description_column or "").strip() or None,
        )
        return TaxonomyContext(
            occ_map=occ_map,
            csv_skill_labels=csv_skill_labels,
            occ_skill_relation=occ_skill_relation,
            occ_skill_signalling=occ_skill_signalling,
            occ_bridge=occ_bridge,
            occ_all_ids=occ_all_ids,
            skills_uuid_tabiya_map=skills_uuid_tabiya_map,
            skill_desc_by_esco=skill_desc_by_esco,
            occ_label_by_esco=occ_label_by_esco,
            occ_desc_by_esco=occ_desc_by_esco,
            skill_groups_by_skill=load_skill_to_groups_map(
                str(settings.skill_hierarchy_csv_path.resolve())
            ),
        )
