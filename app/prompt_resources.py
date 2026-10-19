"""
Load prompt templates from the run's language pack under ``Settings.resources_dir``.

Every prompt the reranker sends is a file in ``resources/<language>/prompts/`` (see
``resources/README.md`` for the list, the placeholders each one takes, and how to add a
country variant). Each file is looked up in ``resources/<language>/prompts/`` →
``resources/en/prompts/`` → ``resources/prompts/`` (see
:func:`app.config.language_resource_dirs`), so a language only has to ship the prompts it
actually translates.

Templates use ``str.format`` syntax: ``{name}`` is a placeholder, ``{{`` / ``}}`` a literal
brace. ``{country}`` and ``{language}`` are available in every template.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from app.config import Settings, get_settings, language_resource_dirs

PROMPTS_SUBDIR = "prompts"

SKILL_RANKER = "skill_ranker.md"
JOB_ATTRIBUTES_SYSTEM = "job_attributes_system.md"
JOB_ATTRIBUTES_RESPONSE = "job_attributes_response.md"
OCCUPATION_SKILLS_RELEVANCE_SYSTEM = "occupation_skills_relevance_system.md"
OCCUPATION_SKILLS_RELEVANCE_USER = "occupation_skills_relevance_user.md"
PROMPT_TEMPLATES = (
    SKILL_RANKER,
    JOB_ATTRIBUTES_SYSTEM,
    JOB_ATTRIBUTES_RESPONSE,
    OCCUPATION_SKILLS_RELEVANCE_SYSTEM,
    OCCUPATION_SKILLS_RELEVANCE_USER,
)

VERSION_FILE = "VERSION"


def country_field_template(country: str, field: str) -> str:
    """Template holding the LLM2 instructions for one of a country's extra response fields."""
    return f"countries/{country}/{field}.md"


def prompt_templates_for(settings: Settings) -> tuple[str, ...]:
    """Every template this run's prompts are assembled from, country blocks included."""
    country = tuple(
        country_field_template(settings.target_country, f)
        for f in settings.job_attributes_country_fields
    )
    return PROMPT_TEMPLATES + country


def resolve_prompt_file(settings: Settings, name: str) -> Path:
    """First existing copy of ``prompts/<name>`` across the language pack, English, ``resources/``.

    :raises FileNotFoundError: no pack ships it — a broken deploy, not a per-job problem.
    """
    dirs = language_resource_dirs(settings.resources_dir, settings.language)
    for directory in dirs:
        candidate = directory / PROMPTS_SUBDIR / name
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(d / PROMPTS_SUBDIR) for d in dirs)
    raise FileNotFoundError(f"Prompt template {name!r} not found in: {searched}")


def load_prompt_template(settings: Settings, name: str) -> str:
    """Raw template text, without surrounding whitespace."""
    return resolve_prompt_file(settings, name).read_text(encoding="utf-8").strip()


class _KeepMissing(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def format_prompt_template(body: str, values: dict[str, str]) -> str:
    """Substitute ``values`` into ``body``.

    A placeholder the template uses but we do not supply is left as-is rather than
    raising: these files are hand-edited, and a stray brace should not fail a batch.
    """
    try:
        return body.format_map(_KeepMissing(values))
    except (IndexError, ValueError):
        for key, value in values.items():
            body = body.replace("{" + key + "}", value)
        return body


def render_prompt(name: str, settings: Settings | None = None, **values: str) -> str:
    """Template ``name`` with ``{country}``, ``{language}`` and ``values`` substituted.

    ``{language}`` is the language the model must answer in — the same language the skill
    menu and occupation labels are written in (:attr:`Settings.language_name`).
    """
    s = settings or get_settings()
    body = load_prompt_template(s, name)
    return format_prompt_template(
        body, {"country": s.country_name, "language": s.language_name, **values}
    )


def prompt_provenance(settings: Settings | None = None) -> dict[str, str]:
    """``prompt_version`` and ``prompt_digest`` for ``llm_reranker_meta``.

    ``prompt_version`` is ``<language>/<VERSION>``, bumped by hand when a pack's prompts
    change. ``prompt_digest`` hashes the template files this run actually resolved
    (fallbacks and country blocks included), so an edit nobody versioned still shows up.
    """
    s = settings or get_settings()
    digest = hashlib.sha256()
    for name in prompt_templates_for(s):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(resolve_prompt_file(s, name).read_bytes() + b"\0")
    version = load_prompt_template(s, VERSION_FILE)
    return {
        "prompt_version": f"{s.language}/{version}",
        "prompt_digest": digest.hexdigest()[:12],
    }
