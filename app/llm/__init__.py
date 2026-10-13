from __future__ import annotations

from app.llm.base import (
    CUSTOM_PROVIDER,
    DEFAULT_TEMPERATURE,
    GEMINI_3_TEMPERATURE,
    PROVIDER_API_KEY_ENV_VARS,
    PROVIDER_BASE_URL_ENV_VARS,
    PROVIDER_CLASS_PATHS,
    PROVIDER_DEFAULT_MODELS,
    THINKING_LEVELS,
    LlmClient,
    LlmConfig,
    default_temperature,
    normalise_provider,
    resolve_provider_class_path,
)
from app.llm.errors import LlmConfigurationError, LlmError, LlmInvocationError
from app.llm.factory import (
    build_llm_client,
    build_traced_llm_client,
    describe_available_providers,
    describe_llm_configuration_error,
    load_llm_client_class,
)
from app.llm.traced import TracedLlmClient

__all__ = [
    "TracedLlmClient",
    "CUSTOM_PROVIDER",
    "DEFAULT_TEMPERATURE",
    "GEMINI_3_TEMPERATURE",
    "LlmClient",
    "LlmConfig",
    "LlmConfigurationError",
    "LlmError",
    "LlmInvocationError",
    "PROVIDER_API_KEY_ENV_VARS",
    "PROVIDER_BASE_URL_ENV_VARS",
    "PROVIDER_CLASS_PATHS",
    "PROVIDER_DEFAULT_MODELS",
    "THINKING_LEVELS",
    "build_llm_client",
    "build_traced_llm_client",
    "describe_llm_configuration_error",
    "default_temperature",
    "describe_available_providers",
    "load_llm_client_class",
    "normalise_provider",
    "resolve_provider_class_path",
]
