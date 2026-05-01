"""
Load prompt text from ``Settings.resources_dir`` (default: repo ``resources/``).

Relative paths:

- ``Skills_reranker_prompts.md``
- ``prompts.py`` (``JOB_ATTRIBUTES_SYSTEM_PROMPT`` for LLM call 2, ``OCCUPATION_SKILLS_RELEVANCE_SYSTEM_PROMPT`` for LLM call 3)
"""

from __future__ import annotations

import re
from pathlib import Path

from app.config import Settings, get_settings

_SKILLS_REL = Path("Skills_reranker_prompts.md")
_PREF_PROMPTS_REL = Path("prompts.py")


def country_prompt_format_kwargs(settings: Settings) -> dict[str, str]:
    """``{country}`` for ``resources/prompts.py`` (from :attr:`Settings.country_name`)."""
    return {"country": settings.country_name}


def format_country_prompt_template(body: str, settings: Settings) -> str:
    """Substitute ``{country}`` from ``settings``."""
    if "{" not in body:
        return body
    return body.format(**country_prompt_format_kwargs(settings))


# Verbatim fallback if files under ``resources/`` are absent.
_FALLBACK_SKILLS_TEMPLATE = """You are an expert technical recruiter with deep knowledge of the job market. Your task is to analyze a job opportunity and categorize relevant skills from a provided list into specific buckets.

**Opportunity Details:**
---
{full_details}
---

**List of Potential Skills to Choose From (each with description):**
{skills_text}

**Instructions:**
1.  **Analyze the Role**: Carefully read the opportunity details to evaluate what the employer requires versus what they merely prefer.
2.  **Categorize Skills (The "Menu" Approach)**: Treat the provided list of potential skills as a menu. Many of these skills will NOT be relevant to the job. Evaluate the list and classify them as follows:
    * **Essential Skills (Non-negotiable)**: Skills the employer **must** see to consider hiring the candidate at the time of application. Strictly exclude skills that the job ad implies will be taught after hiring.
    * **Optional Skills (Value-add)**: Skills that make an applicant stronger or more competitive, but are **not strictly required**. Order these from most to least important.
    * **Irrelevant Skills (Discard)**: If a skill is neither strictly essential nor distinctly valuable for this specific role, **ignore it completely**. Do not force every skill into the JSON.
3.  **Mutual Exclusivity**: A skill can be essential, optional, or discarded. It cannot appear in more than one category.
4.  **Use Provided List Only**: Do not invent or infer skills outside the provided list. Strictly exclude non-skills (benefits, conditions, years of experience counts, employer names, and traits).
5.  **Verbatim Matching**: Every skill string you return must be an **exact, verbatim** copy of a skill label from the list above.
6.  **Strict JSON Output**: Format your response as a single, valid JSON object. Do not include any explanatory text, markdown formatting (like ```json), or conversational filler outside the JSON.

**Output JSON shape:**
{{
  "essential_skills": ["...", "..."],
  "optional_skills": ["...", "...", "..."]
}}"""

_FALLBACK_JOB_ATTRIBUTES_SYSTEM = """You are an expert labor market analyst specializing in the economy and job characteristics of {country}.
Your task is to estimate job attributes for occupations in a specific region of {country}.

You have deep knowledge of:
- The economic structure and regional labor markets of {country}
- Wage structures across formal and informal sectors
- Working conditions in different industries
- Career progression patterns in {country}
- Regional differences in job characteristics

When estimating attributes, consider:
1. EARNINGS: Base estimates on formal sector data when available, adjust for informal sector prevalence.
   Major urban centers typically have higher wages relative to rural areas in {country}.
2. TASK CONTENT: Consider the nature of work - routine vs problem-solving vs creative.
3. PHYSICAL DEMAND: Assess physical requirements and occupational safety risks.
4. FLEXIBILITY: Consider typical supervision levels and schedule flexibility.
5. SOCIAL INTERACTION: Evaluate whether work is primarily solitary or collaborative.
6. CAREER GROWTH: Assess learning opportunities and advancement potential.
7. SOCIAL MEANING: Consider community contribution and social impact.

Confidence levels:
- high: Strong evidence from data or well-established patterns
- medium: Reasonable inference based on similar occupations
- low: Limited data, estimate based on general patterns"""

_FALLBACK_OCCUPATION_SKILLS_RELEVANCE_SYSTEM = """You are an expert recruiter familiar with the job market in {country}. Your task is to judge whether each occupation-linked taxonomy skill is relevant to a specific job advertisement.

The skills you see are already tagged as essential or optional **for the occupation** (from a standard taxonomy). They may still be irrelevant **for this particular job posting** (wrong seniority, different industry focus, not mentioned, or clearly not required).

For every skill in the list the user provides:
- Set is_relevant to true if the skill plausibly applies to the role described in the job ad.
- Set is_relevant to false if the skill is not a reasonable requirement or context for this posting.

Rules:
- Prioritize domain-specific, technical, or role-specific skills over general soft skills.
- Use only the skills listed; do not invent skill_ids.
- Copy skill_id and tabiya_occupation_id exactly as given (strings).
- Output nothing but the JSON object specified in the user message: a decisions array only, no summary or reasons."""


def _skills_md_path(settings: Settings) -> Path:
    return settings.resources_dir / _SKILLS_REL


def _pref_prompts_path(settings: Settings) -> Path:
    return settings.resources_dir / _PREF_PROMPTS_REL


def load_skills_ranker_prompt_template(settings: Settings | None = None) -> str:
    """
    Body of the skill-ranker prompt from ``Skills_reranker_prompts.md`` (inside ``return f\"\"\"...\"\"\"``).

    Use ``str.format(full_details=..., skills_text=...)`` (JSON braces in template are doubled).
    """
    s = settings or get_settings()
    path = _skills_md_path(s)
    if not path.is_file():
        return _FALLBACK_SKILLS_TEMPLATE
    raw = path.read_text(encoding="utf-8").strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = lines[1:]
        while lines and lines[-1].strip() in ("```", ""):
            lines.pop()
        raw = "\n".join(lines)
    key = 'return f"""'
    i = raw.find(key)
    if i == -1:
        return _FALLBACK_SKILLS_TEMPLATE
    start = i + len(key)
    end = raw.find('}}"""', start)
    if end == -1:
        return _FALLBACK_SKILLS_TEMPLATE
    return raw[start : end + 2].strip()


def load_job_attributes_system_prompt(settings: Settings | None = None) -> str:
    """Load ``JOB_ATTRIBUTES_SYSTEM_PROMPT`` from ``resources/prompts.py`` and apply country placeholders."""
    s = settings or get_settings()
    path = _pref_prompts_path(s)
    if not path.is_file():
        return format_country_prompt_template(_FALLBACK_JOB_ATTRIBUTES_SYSTEM, s)
    text = path.read_text(encoding="utf-8")
    m = re.search(
        r"JOB_ATTRIBUTES_SYSTEM_PROMPT\s*=\s*\"\"\"(.*?)\"\"\"",
        text,
        re.DOTALL,
    )
    if not m:
        return format_country_prompt_template(_FALLBACK_JOB_ATTRIBUTES_SYSTEM, s)
    return format_country_prompt_template(m.group(1).strip(), s)


def load_occupation_skills_relevance_system_prompt(settings: Settings | None = None) -> str:
    """Load ``OCCUPATION_SKILLS_RELEVANCE_SYSTEM_PROMPT`` from ``resources/prompts.py`` and apply country placeholders."""
    s = settings or get_settings()
    path = _pref_prompts_path(s)
    if not path.is_file():
        return format_country_prompt_template(_FALLBACK_OCCUPATION_SKILLS_RELEVANCE_SYSTEM, s)
    text = path.read_text(encoding="utf-8")
    m = re.search(
        r"OCCUPATION_SKILLS_RELEVANCE_SYSTEM_PROMPT\s*=\s*\"\"\"(.*?)\"\"\"",
        text,
        re.DOTALL,
    )
    if not m:
        return format_country_prompt_template(_FALLBACK_OCCUPATION_SKILLS_RELEVANCE_SYSTEM, s)
    return format_country_prompt_template(m.group(1).strip(), s)
