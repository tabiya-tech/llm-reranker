"""Shared base classes for the reranker's domain models.

Two tiers, matching how a value's origin determines how strict its model can be:

- :class:`DomainModel` — anything derived or computed by the reranker itself (LLM
  parse results, taxonomy rows, the ranked document and everything nested in it).
  Strict, closed to extra keys, and immutable (mutate via ``model_copy(update=...)``).
- :class:`SourceRecord` — anything shaped by an upstream system the reranker doesn't
  control (the classifier's ``classified_jobs`` documents, LLM JSON before it's been
  shape-checked). Permissive on unknown keys, since the exact shape isn't contractual.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class DomainModel(BaseModel):
    """Base for every domain model: strict, closed, immutable values."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class SourceRecord(BaseModel):
    """Base for models of documents produced by other systems (classifier, LLM JSON)."""

    model_config = ConfigDict(extra="ignore")
