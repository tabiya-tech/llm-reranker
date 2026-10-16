"""Classifier occupations → occupation-skill relations, as ``occupation_to_skills`` blocks."""

from __future__ import annotations

from typing import Any

from app.config import Settings
from app.job_ranking.occupation_skills.menu import (
    filter_occupation_skills_nested_by_signal_include,
    occupation_skills_relevance_menu_size,
)
from app.job_ranking.occupation_skills.types import (
    OccupationTaxonomyOutcome,
)
from app.taxonomy.types import TaxonomyContext
from utils.classifier_linked_entities import (
    EscoSkillCandidate,
)
from utils.occupation_to_skills_mapping import (
    resolve_occupation_taxonomy_for_classified_job,
)


class OccupationTaxonomyStep:
    """Classifier occupations → occupation-skill relations, as ``occupation_to_skills`` blocks."""

    def __init__(self, settings: Settings, taxonomy: TaxonomyContext):
        self._settings = settings
        self._taxonomy = taxonomy

    def run(
        self, classified_doc: dict[str, Any], classifier_skills: list[EscoSkillCandidate]
    ) -> OccupationTaxonomyOutcome:
        tx = self._taxonomy
        relation_uuids, stats, nested = resolve_occupation_taxonomy_for_classified_job(
            classified_doc,
            classifier_skills,
            tx.occ_map,
            tx.csv_skill_labels,
            tx.occ_skill_relation,
            tx.skills_uuid_tabiya_map,
            occupation_skill_signalling=tx.occ_skill_signalling,
            occupations_bridge=tx.occ_bridge,
            occupations_all_ids=tx.occ_all_ids,
            settings=self._settings,
        )
        nested = filter_occupation_skills_nested_by_signal_include(
            nested,
            include_tiers=self._settings.reranker_occupation_skills_signal_include,
        )
        return OccupationTaxonomyOutcome(
            relation_skill_uuids=relation_uuids,
            stats=stats,
            occupation_to_skills=nested,
            menu_size=occupation_skills_relevance_menu_size(nested),
        )
