"""Work-activities enrichment data and counters."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.domain_base import DomainModel


class WorkActivityTaxonomy(DomainModel):
    """``combined_occupation_database_with_wa.json``, indexed."""

    occ_lookup: dict[str, dict[str, dict[str, float]]] = Field(
        description="Lowercase occupation label -> WA code -> `{importance, level}`."
    )
    wa_averages: dict[str, dict[str, float]] = Field(
        description="WA code -> cross-occupation average `{importance, level}`."
    )


class WorkActivitiesEnrichmentStats(BaseModel):
    """Per-run counters for the work-activities enrichment post-step.

    Mutated field-by-field as documents are processed, so this is a plain ``BaseModel``
    with assignment validation, not the frozen :class:`~app.domain_base.DomainModel`.
    """

    model_config = ConfigDict(strict=True, extra="forbid", validate_assignment=True)

    scanned: int = Field(default=0, description="Documents scanned.")
    with_occ_match: int = Field(default=0, description="Documents matched by occupation label.")
    with_avg_match: int = Field(
        default=0, description="Documents matched by cross-occupation average."
    )
    with_default: int = Field(default=0, description="Documents given the default fallback.")
    no_data: int = Field(default=0, description="Documents with no usable match.")
    queued: int = Field(default=0, description="Documents queued for a bulk write.")
    applied: int = Field(default=0, description="Documents actually written to MongoDB.")
