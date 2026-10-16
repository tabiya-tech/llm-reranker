from __future__ import annotations

import importlib
import inspect
import logging

from app.llm.base import (
    CUSTOM_PROVIDER,
    PROVIDER_CLASS_PATHS,
    LlmClient,
    LlmConfig,
)
from app.llm.errors import LlmConfigurationError
from app.llm.traced import TracedLlmClient

logger = logging.getLogger(__name__)


def _split_class_path(dotted: str) -> tuple[str, str]:
    """Split ``module:Class`` (preferred) or ``module.Class`` into its two halves."""
    raw = (dotted or "").strip()
    if not raw:
        raise LlmConfigurationError(
            "No LLM client class configured. Set RERANKER_LLM_PROVIDER to one of "
            f"{', '.join(sorted(PROVIDER_CLASS_PATHS))}, or RERANKER_LLM_CLIENT_CLASS "
            "to a dotted path such as app.llm.providers.gemini:GeminiLlmClient."
        )
    if ":" in raw:
        module_name, _, attr = raw.partition(":")
    elif "." in raw:
        module_name, _, attr = raw.rpartition(".")
    else:
        raise LlmConfigurationError(
            f"RERANKER_LLM_CLIENT_CLASS={raw!r} is not a class path. Use "
            f"'module:Class' (e.g. app.llm.providers.gemini:GeminiLlmClient)."
        )
    if not module_name.strip() or not attr.strip():
        raise LlmConfigurationError(
            f"RERANKER_LLM_CLIENT_CLASS={raw!r} is incomplete — expected "
            f"'module:Class', got module={module_name!r} class={attr!r}."
        )
    return module_name.strip(), attr.strip()


def load_llm_client_class(dotted: str) -> type[LlmClient]:
    """Import a dotted class path and check it really is a usable ``LlmClient``.

    Every failure mode gets its own message: the import failed, the attribute is
    missing, it is not a class, it is not an ``LlmClient``, or it is still abstract.
    Guessing which of those happened from an ``ImportError`` traceback is exactly
    the debugging tax this indirection would otherwise impose.
    """
    module_name, attr = _split_class_path(dotted)
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise LlmConfigurationError(
            f"Cannot import module {module_name!r} for LLM client {dotted!r}: {e}. "
            f"Check the module path, and that the provider's SDK is installed "
            f"(see requirements.txt)."
        ) from e
    try:
        obj = getattr(module, attr)
    except AttributeError as e:
        exported = [
            n
            for n, v in vars(module).items()
            if inspect.isclass(v) and issubclass(v, LlmClient) and v is not LlmClient
        ]
        hint = (
            f" Available LlmClient classes there: {', '.join(sorted(exported))}."
            if exported
            else ""
        )
        raise LlmConfigurationError(
            f"Module {module_name!r} has no attribute {attr!r} (from {dotted!r}).{hint}"
        ) from e
    if not inspect.isclass(obj):
        raise LlmConfigurationError(
            f"{dotted!r} resolved to a {type(obj).__name__}, not a class. "
            f"RERANKER_LLM_CLIENT_CLASS must point at an app.llm.base.LlmClient subclass."
        )
    if not issubclass(obj, LlmClient):
        raise LlmConfigurationError(
            f"{dotted!r} ({obj.__module__}.{obj.__qualname__}) is not a subclass of "
            f"app.llm.base.LlmClient. Implement that interface — a single "
            f"generate_json(prompt, *, purpose) -> str method — and retry."
        )
    if inspect.isabstract(obj):
        missing = ", ".join(sorted(getattr(obj, "__abstractmethods__", ()) or ()))
        raise LlmConfigurationError(
            f"{dotted!r} is abstract and cannot be instantiated — unimplemented: {missing}."
        )
    return obj


def build_llm_client(config: LlmConfig) -> LlmClient:
    """Construct a fresh client for ``config`` (no caching).

    Validation the provider performs in ``__init__`` (missing key, missing model)
    surfaces here, which is why startup calls this before touching MongoDB.
    """
    cls = load_llm_client_class(config.client_class)
    try:
        client = cls(config)
    except LlmConfigurationError:
        raise
    except Exception as e:
        raise LlmConfigurationError(
            f"Failed to construct LLM client {config.client_class!r} for "
            f"{config.describe()}: {type(e).__name__}: {e}"
        ) from e
    logger.debug(
        "LLM client ready | provider=%s | model=%s | class=%s",
        config.provider,
        config.model,
        config.client_class,
    )
    return client


def describe_available_providers() -> str:
    """Human-readable provider list for error messages and ``--help`` style output."""
    rows = [f"  {name:<18} {path}" for name, path in sorted(PROVIDER_CLASS_PATHS.items())]
    rows.append(f"  {CUSTOM_PROVIDER:<18} any dotted path in RERANKER_LLM_CLIENT_CLASS")
    return "\n".join(rows)


def build_traced_llm_client(config: LlmConfig) -> TracedLlmClient:
    """The configured model, traced. Raises :class:`LlmConfigurationError` on a bad selection."""
    return TracedLlmClient(build_llm_client(config))


def describe_llm_configuration_error(config: LlmConfig | None, error: Exception) -> str:
    """Operator-facing message for a model selection that cannot be built."""
    selection = (
        f"Configured selection: provider={config.provider} "
        f"model={config.model or '(unset)'} class={config.client_class}\n\n"
        if config is not None
        else ""
    )
    return (
        f"LLM configuration error: {error}\n\n"
        f"{selection}"
        f"Available providers (RERANKER_LLM_PROVIDER):\n"
        f"{describe_available_providers()}"
    )
