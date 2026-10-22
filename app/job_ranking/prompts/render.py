"""Fills a prompt template's slots. No IO."""

from __future__ import annotations

from app.config import Settings
from app.job_ranking.prompts.templates import PromptTemplates
from app.languages import DEFAULT_LANGUAGE


def render_prompt(template: str, **slots: str) -> str:
    """``template`` with every ``{slot}`` filled from ``slots``.

    :raises KeyError: the template uses a slot that was not supplied, i.e. a typo in a
        template or a slot the builder does not know. Values are inserted as-is, so braces
        in job text reach the model unchanged.
    """
    return template.format_map(slots)


def answer_language_slot(templates: PromptTemplates, settings: Settings) -> str:
    """The ``{answer_language}`` block: empty for English, else a paragraph of its own."""
    if settings.language == DEFAULT_LANGUAGE:
        return ""
    return "\n\n" + render_prompt(templates.answer_language, language=settings.language_name)
