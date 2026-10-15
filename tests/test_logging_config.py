"""``configure_logging`` applies ``LOG_LEVEL`` without ever building ``Settings``.

It must run before the first ``Settings`` is built: ``main.py`` exports
``RERANKER_LANGUAGE`` from ``--language`` only after calling it, and ``get_settings()``
is cached, so building ``Settings`` here first would freeze the wrong language for the
rest of the process.
"""

from __future__ import annotations

import logging

import pytest

from app.logging_config import configure_logging

pytestmark = pytest.mark.unit


def test_applies_log_level_env_var(isolated_env, monkeypatch):
    isolated_env.setenv("LOG_LEVEL", "DEBUG")
    # pytest's logging plugin keeps a handler on the root logger, which would otherwise
    # always trip configure_logging's "already configured" guard.
    monkeypatch.setattr(logging.getLogger(), "handlers", [])
    captured: dict[str, object] = {}
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: captured.update(kwargs))

    configure_logging()

    assert captured.get("level") == logging.DEBUG


def test_warn_is_accepted_as_an_alias_for_warning(isolated_env, monkeypatch):
    """Must match ``Settings.log_level``'s own ``WARN`` → ``WARNING`` alias (Python's
    ``logging`` module treats them as the same level), or the two readers disagree."""
    isolated_env.setenv("LOG_LEVEL", "warn")
    monkeypatch.setattr(logging.getLogger(), "handlers", [])
    captured: dict[str, object] = {}
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: captured.update(kwargs))

    configure_logging()

    assert captured.get("level") == logging.WARNING


def test_does_not_reconfigure_an_already_configured_root_logger(monkeypatch):
    monkeypatch.setattr(logging.getLogger(), "handlers", [logging.NullHandler()])
    called = False

    def _fail_if_called(**_kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(logging, "basicConfig", _fail_if_called)

    configure_logging()

    assert called is False


def test_does_not_build_settings_before_language_is_exported(isolated_env, monkeypatch):
    """Regression: configuring logging must not cache a Settings built before
    --language's RERANKER_LANGUAGE export, or the language stays wrong all run."""
    from app import config as app_config

    monkeypatch.setattr(root := logging.getLogger(), "handlers", [])
    monkeypatch.setattr(logging, "basicConfig", lambda **_kwargs: None)

    configure_logging()
    assert app_config.get_settings.cache_info().currsize == 0

    isolated_env.setenv("RERANKER_LANGUAGE", "es")
    assert app_config.get_settings().language == "es"
    root.handlers.clear()
