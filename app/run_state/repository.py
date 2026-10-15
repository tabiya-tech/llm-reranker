from __future__ import annotations

from abc import ABC, abstractmethod


class IRunStateRepository(ABC):
    """
    Interface for the batch run state repository (one document per country).
    Allows to mock the repository in tests.
    """

    @abstractmethod
    def get_last_cutoff_date(self, country: str) -> str | None:
        """The ``last_cutoff_date`` saved by the country's last successful run, as ``YYYY-MM-DD``.

        ``None`` if the country has no state yet.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, country: str, *, jobs_processed: int) -> None:
        """Record a successful run: today becomes the country's next cutoff date."""
        raise NotImplementedError
