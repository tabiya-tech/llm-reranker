"""Guards for the module layout described in README.md, "Architecture".

Code is grouped into one module per use case (like compass). Inside a module, each file has a
role given by its name, and dependencies between roles point one way:
routes → services / steps → repositories, with the pure files usable from every role.
These tests read each file's imports statically (nothing is executed).
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parent.parent
_APP_FILES = sorted(p for p in (_ROOT / "app").rglob("*.py") if p.name != "__init__.py")


def _module(path: Path) -> str:
    return ".".join(path.relative_to(_ROOT).with_suffix("").parts)


def _is_pure(path: Path) -> bool:
    """Files with no IO: types, prompt building, response parsing, rules, document assembly."""
    if "routes" in path.parts:
        return False
    return path.name in {
        "types.py",
        "prompt.py",
        "parser.py",
        "menu.py",
        "rules.py",
        "is_active.py",
        "ranked_document.py",
        "templates.py",
        "job_listing.py",
        "whitening.py",
        "upsert.py",
        "response_schemas.py",
    }


def _is_repository(path: Path) -> bool:
    return path.name.endswith("repository.py") or path.name == "csv_loaders.py"


def _is_provider(path: Path) -> bool:
    return path.name.startswith("get_")


def _is_service(path: Path) -> bool:
    return not _is_provider(path) and (
        path.name.endswith(("service.py", "step.py"))
        or path.name in {"post_steps.py", "job_ranker.py"}
    )


def _is_route(path: Path) -> bool:
    return "routes" in path.parts or path.name == "routes.py"


_BY_MODULE = {_module(p): p for p in _APP_FILES}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module)
    return out


def _violations(
    files: list[Path],
    forbidden_role: Callable[[Path], bool] | None = None,
    forbidden_prefixes: tuple[str, ...] = (),
    allowed: tuple[str, ...] = (),
) -> list[str]:
    found = []
    for path in files:
        for name in sorted(_imports(path)):
            if name in allowed:
                continue
            target = _BY_MODULE.get(name)
            bad_role = forbidden_role is not None and target is not None and forbidden_role(target)
            bad_prefix = any(name == f or name.startswith(f + ".") for f in forbidden_prefixes)
            if bad_role or bad_prefix:
                found.append(f"{path.relative_to(_ROOT)} imports {name}")
    return found


_IO_LIBRARIES = ("pymongo", "csv", "httpx", "google", "app.server_dependencies")
_ENTRYPOINTS = ("api", "main")


def test_pure_files_do_no_io():
    pure = [p for p in _APP_FILES if _is_pure(p)]
    assert pure, "no pure files found — the role rules need updating"
    assert (
        _violations(
            pure,
            lambda t: _is_repository(t) or _is_service(t) or _is_provider(t) or _is_route(t),
            (*_IO_LIBRARIES, "app.llm", *_ENTRYPOINTS),
            # Exception types only: parsers raise ``LlmParseError`` without calling a model.
            allowed=("app.llm.errors",),
        )
        == []
    )


def test_repositories_never_reach_up():
    repositories = [p for p in _APP_FILES if _is_repository(p)]
    assert (
        _violations(
            repositories,
            lambda t: _is_service(t) or _is_provider(t) or _is_route(t),
            _ENTRYPOINTS,
        )
        == []
    )


def test_services_receive_their_dependencies():
    """Services never import routes or the providers that build them."""
    services = [p for p in _APP_FILES if _is_service(p)]
    assert _violations(services, lambda t: _is_route(t) or _is_provider(t), _ENTRYPOINTS) == []


def test_services_leave_the_database_to_repositories():
    """Queries and write operations are built inside the repositories, not passed down to them."""
    services = [p for p in _APP_FILES if _is_service(p)]
    assert _violations(services, forbidden_prefixes=("pymongo", "bson")) == []


def test_routes_talk_to_services_only():
    routes = [p for p in _APP_FILES if _is_route(p)]
    assert _violations(routes, _is_repository, ("pymongo", "utils", *_ENTRYPOINTS)) == []


def test_utils_stay_below_every_module():
    utils = sorted((_ROOT / "utils").rglob("*.py"))
    assert (
        _violations(
            utils,
            lambda t: _is_repository(t) or _is_service(t) or _is_provider(t) or _is_route(t),
            _ENTRYPOINTS,
        )
        == []
    )


@pytest.mark.parametrize("entrypoint", ["api.py", "main.py"])
def test_entrypoints_hold_no_ranking_or_data_access(entrypoint: str):
    assert (
        _violations(
            [_ROOT / entrypoint],
            # Option and result types are fine; prompts, parsers, rules and repositories are not.
            lambda t: (_is_pure(t) and t.name != "types.py") or _is_repository(t),
            ("pymongo", "utils"),
        )
        == []
    )


def test_model_clients_are_built_only_by_providers():
    """Services receive clients; only the ``get_*`` providers construct them."""
    calls = (
        "build_llm_client(",
        "build_traced_llm_client(",
        "GeminiEmbeddingClient(",
        "GeminiEmbeddingClient.from_settings(",
    )
    defining = {
        _ROOT / "app" / "llm" / "factory.py",
        _ROOT / "app" / "job_embedding" / "gemini_client.py",
    }
    offenders = [
        f"{path.relative_to(_ROOT)}: {call}"
        for path in [*_APP_FILES, _ROOT / "api.py", _ROOT / "main.py"]
        if path not in defining and not _is_provider(path)
        for call in calls
        if call in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_process_settings_are_read_only_where_the_app_is_assembled():
    """Everything else takes ``settings`` from its caller, so a ``?language=`` request's pack
    reaches every step instead of the process default."""
    allowed_dirs = {_ROOT / "app" / "server_dependencies"}
    offenders = [
        str(path.relative_to(_ROOT))
        for path in [*_APP_FILES, *sorted((_ROOT / "utils").rglob("*.py"))]
        if path != _ROOT / "app" / "config.py"
        and not _is_provider(path)
        and not path.name.endswith("cli.py")
        and path.parent not in allowed_dirs
        and "get_settings()" in path.read_text(encoding="utf-8").replace("``get_settings()``", "")
    ]
    assert offenders == []


def test_no_module_level_llm_client_getter():
    import app.llm

    assert not hasattr(app.llm, "get_llm_client")
