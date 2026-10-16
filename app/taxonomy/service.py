"""Per-language taxonomy contexts, loaded on first use."""

from __future__ import annotations

import logging
import threading

from app.config import Settings
from app.taxonomy.repository import TaxonomyRepository
from app.taxonomy.types import TaxonomyContext

logger = logging.getLogger(__name__)


class TaxonomyService:
    """Holds one :class:`TaxonomyContext` per language, loading each language's CSVs once.

    The CSV label packs differ per language, so a Spanish request cannot be served from the
    English lookups. Thread-safe: concurrent first requests for a language load it once.
    """

    def __init__(self, repository: TaxonomyRepository):
        self._repository = repository
        self._contexts: dict[str, TaxonomyContext] = {}
        self._lock = threading.Lock()

    def get(self, settings: Settings) -> TaxonomyContext:
        """The context for ``settings.language``, loading it on first use."""
        cached = self._contexts.get(settings.language)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._contexts.get(settings.language)
            if cached is not None:
                return cached
            logger.info("Loading taxonomy data for language=%s (CSV lookups)…", settings.language)
            context = self._repository.load(settings)
            self._contexts[settings.language] = context
            logger.info(
                "Taxonomy ready | language=%s | resources=%s | occupations=%s",
                settings.language,
                settings.language_resources_dir.name,
                len(context.occ_map),
            )
            return context

    def is_loaded(self, language: str) -> bool:
        return language in self._contexts

    @property
    def loaded_languages(self) -> list[str]:
        return sorted(self._contexts)

    def clear(self) -> None:
        with self._lock:
            self._contexts.clear()
