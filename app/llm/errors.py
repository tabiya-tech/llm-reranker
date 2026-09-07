from __future__ import annotations


class LlmError(RuntimeError):
    """Base class for every failure raised by ``app.llm``."""


class LlmConfigurationError(LlmError):
    """The configured model / provider selection is unusable (fail fast)."""


class LlmInvocationError(LlmError):
    """A configured provider was called and the call failed."""
