"""Single owner of the MongoDB client: creation and shutdown.

The batch (``main.py``) opens the client once at the start of a run and closes it at the end; the
CLIs under ``utils/`` and ``scripts/`` open it through ``standalone_db()``. Repositories receive the
database by injection and never create a client. Nothing else in the process constructs a
``MongoClient``. See ``app/server_dependencies/README.md``.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import certifi
from pymongo import MongoClient
from pymongo.database import Database

from app.config import Settings, get_settings
from app.logging_config import sanitize_mongodb_uri

SERVER_SELECTION_TIMEOUT_MS = 30000


def build_client_kwargs(settings: Settings) -> dict[str, Any]:
    """Timeout and TLS options for ``MongoClient``.

    Atlas (mongodb+srv) uses TLS, and on some macOS/Python installs the default CA store is empty
    (``SSL: CERTIFICATE_VERIFY_FAILED``). Precedence:

    * ``MONGODB_TLS_INSECURE=1`` — skip certificate verification (dev only).
    * ``MONGODB_TLS_CA_FILE=/path.pem`` — explicit CA bundle.
    * Otherwise, if the URI looks TLS, use ``certifi.where()``.
    """
    kwargs: dict[str, Any] = {"serverSelectionTimeoutMS": SERVER_SELECTION_TIMEOUT_MS}
    uri = settings.mongodb_uri
    if settings.mongodb_tls_insecure:
        kwargs["tlsAllowInvalidCertificates"] = True
    elif settings.mongodb_tls_ca_file:
        kwargs["tlsCAFile"] = settings.mongodb_tls_ca_file
    elif "mongodb+srv://" in uri or "tls=true" in uri.lower() or "ssl=true" in uri.lower():
        kwargs["tlsCAFile"] = certifi.where()
    return kwargs


def _create_client(settings: Settings) -> MongoClient[dict[str, Any]]:
    """
    Decouples the client creation from the provider.
    This allows to mock the client creation in tests, instead of mocking the provider.
    """
    return MongoClient(settings.mongodb_uri, **build_client_kwargs(settings))


class RerankerDBProvider:
    """
    Provides the reranker database. Holds the only MongoDB client in the process.

    ``MongoClient`` is thread-safe and pools connections, so the batch's worker threads share it.
    """

    _client: MongoClient[dict[str, Any]] | None = None
    _lock = threading.Lock()
    _logger = logging.getLogger(__qualname__)

    @classmethod
    def get_client(cls, settings: Settings | None = None) -> MongoClient[dict[str, Any]]:
        """Return the client, creating it on first call.

        Creating the client does no I/O (pymongo connects lazily), so this never blocks on Mongo.
        """
        if cls._client is None:  # Check if the client has been created
            with cls._lock:  # Ensure that only one thread is creating the client
                if cls._client is None:  # Double-check after acquiring the lock
                    # Settings are read on first use, not at import time, so importing the app
                    # never needs a MONGODB_URI.
                    settings = settings or get_settings()
                    cls._client = _create_client(settings)
                    cls._logger.info(
                        "Created MongoDB client (uri=%s, database=%s)",
                        sanitize_mongodb_uri(settings.mongodb_uri),
                        settings.mongodb_database,
                    )
        return cls._client

    @classmethod
    def get_db(cls, settings: Settings | None = None) -> Database[dict[str, Any]]:
        """Return the configured database (``MONGODB_DATABASE``) on the shared client."""
        settings = settings or get_settings()
        return cls.get_client(settings)[settings.mongodb_database]

    @classmethod
    def close(cls) -> None:
        """Close the client and forget it, so a later ``get_db`` opens a fresh one."""
        with cls._lock:
            if cls._client is not None:
                cls._client.close()
                cls._client = None
                cls._logger.info("Closed MongoDB client")

    @classmethod
    def clear_cache(cls) -> None:
        """
        Forget the cached client without closing it.

        This is useful for testing purposes to ensure that the client is re-created.
        """
        cls._client = None


@contextmanager
def standalone_db(settings: Settings | None = None) -> Iterator[Database[dict[str, Any]]]:
    """Database for code running outside the batch run (CLI scripts under ``utils/`` and ``scripts/``).

    Opens the provider's client for the duration of the block and closes it on exit. Never use this
    inside the batch run — ``main.py`` owns the client there and passes the database down.
    """
    try:
        yield RerankerDBProvider.get_db(settings)
    finally:
        RerankerDBProvider.close()
