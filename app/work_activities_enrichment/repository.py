"""Reads ``combined_occupation_database_with_wa.json`` (O*NET work activities per occupation)."""

from __future__ import annotations

import json
from pathlib import Path

from app.work_activities_enrichment.rules import build_work_activity_taxonomy
from app.work_activities_enrichment.types import WorkActivityTaxonomy


class WorkActivityTaxonomyRepository:
    def load(self, path: Path) -> WorkActivityTaxonomy:
        if not path.is_file():
            raise FileNotFoundError(f"Occupation WA taxonomy JSON not found: {path}")
        return build_work_activity_taxonomy(json.loads(path.read_text(encoding="utf-8")))
