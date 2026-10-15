"""Tests for the MongoDB client lifecycle: provider, client options, and the standalone context."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import certifi
import pytest

from app.server_dependencies.database_collections import Collections
from app.server_dependencies.db_dependencies import (
    RerankerDBProvider,
    build_client_kwargs,
    standalone_db,
)

_CREATE_CLIENT = "app.server_dependencies.db_dependencies._create_client"


@pytest.fixture(autouse=True)
def _fresh_provider():
    RerankerDBProvider.clear_cache()
    yield
    RerankerDBProvider.clear_cache()


class TestRerankerDBProvider:
    def test_creates_the_client_once_across_threads(self, make_settings):
        s = make_settings(MONGODB_DATABASE="KenyaJobs")
        client = MagicMock()
        results: list[object] = []
        with patch(_CREATE_CLIENT, return_value=client) as factory:
            threads = [
                threading.Thread(target=lambda: results.append(RerankerDBProvider.get_db(s)))
                for _ in range(8)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        factory.assert_called_once()
        assert len(results) == 8
        assert all(r is client["KenyaJobs"] for r in results)

    def test_get_db_selects_the_configured_database(self, make_settings):
        s = make_settings(MONGODB_DATABASE="ArgentinaJobs")
        client = MagicMock()
        with patch(_CREATE_CLIENT, return_value=client):
            RerankerDBProvider.get_db(s)
        client.__getitem__.assert_called_with("ArgentinaJobs")

    def test_close_closes_the_client_and_a_later_call_opens_a_new_one(self, make_settings):
        s = make_settings()
        client_1, client_2 = MagicMock(), MagicMock()
        with patch(_CREATE_CLIENT, side_effect=[client_1, client_2]) as factory:
            assert RerankerDBProvider.get_client(s) is client_1
            RerankerDBProvider.close()
            client_1.close.assert_called_once()
            assert RerankerDBProvider.get_client(s) is client_2
        assert factory.call_count == 2

    def test_close_without_a_client_is_a_no_op(self):
        RerankerDBProvider.close()


class TestStandaloneDb:
    def test_closes_the_client_on_exit(self, make_settings):
        s = make_settings()
        client = MagicMock()
        with patch(_CREATE_CLIENT, return_value=client):
            with standalone_db(s) as db:
                assert db is client[s.mongodb_database]
        client.close.assert_called_once()
        assert RerankerDBProvider._client is None

    def test_closes_the_client_when_the_block_raises(self, make_settings):
        s = make_settings()
        client = MagicMock()
        with patch(_CREATE_CLIENT, return_value=client):
            with pytest.raises(RuntimeError):
                with standalone_db(s):
                    raise RuntimeError("boom")
        client.close.assert_called_once()


class TestBuildClientKwargs:
    def test_plain_uri_has_no_tls_options(self, make_settings):
        s = make_settings(MONGODB_URI="mongodb://localhost:27017")
        kwargs = build_client_kwargs(s)
        assert "tlsCAFile" not in kwargs
        assert "tlsAllowInvalidCertificates" not in kwargs
        assert kwargs["serverSelectionTimeoutMS"] > 0

    def test_srv_uri_uses_certifi(self, make_settings):
        s = make_settings(MONGODB_URI="mongodb+srv://u:p@cluster.example.net")
        assert build_client_kwargs(s)["tlsCAFile"] == certifi.where()

    def test_explicit_ca_file_wins_over_certifi(self, make_settings):
        s = make_settings(
            MONGODB_URI="mongodb+srv://u:p@cluster.example.net",
            MONGODB_TLS_CA_FILE="/etc/ca.pem",
        )
        assert build_client_kwargs(s)["tlsCAFile"] == "/etc/ca.pem"

    def test_insecure_wins_over_everything(self, make_settings):
        s = make_settings(
            MONGODB_URI="mongodb+srv://u:p@cluster.example.net",
            MONGODB_TLS_CA_FILE="/etc/ca.pem",
            MONGODB_TLS_INSECURE="1",
        )
        kwargs = build_client_kwargs(s)
        assert kwargs["tlsAllowInvalidCertificates"] is True
        assert "tlsCAFile" not in kwargs


class TestCollections:
    def test_defaults(self, make_settings):
        c = Collections.from_settings(make_settings())
        assert c == Collections(
            SOURCE="classified_jobs",
            TARGET="ranked_jobs",
            LOGS="reranker_logs",
            STATE="reranker_state",
        )

    def test_env_overrides(self, make_settings):
        c = Collections.from_settings(
            make_settings(
                MONGODB_SOURCE_COLLECTION="src",
                MONGODB_TARGET_COLLECTION="dst",
                RERANKER_LOGS_COLLECTION="logs",
                RERANKER_STATE_COLLECTION="state",
            )
        )
        assert (c.SOURCE, c.TARGET, c.LOGS, c.STATE) == ("src", "dst", "logs", "state")
