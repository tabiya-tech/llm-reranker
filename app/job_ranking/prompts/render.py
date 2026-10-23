"""Fills a prompt template's slots. No IO."""

from __future__ import annotations


def render_prompt(template: str, **slots: str) -> str:
    """``template`` with every ``{slot}`` filled from ``slots``.

    :raises KeyError: the template uses a slot that was not supplied, i.e. a typo in a
        template or a slot the builder does not know. Values are inserted as-is, so braces
        in job text reach the model unchanged.
    """
    return template.format_map(slots)
