"""Snapshot of every assembled LLM prompt, to catch unintended prompt changes.

Each case renders LLM1, LLM2 and LLM3 for one country/language/job combination and
compares each prompt byte-for-byte with its syrupy snapshot, one ``.txt`` file per prompt
under ``tests/__snapshots__/test_prompt_snapshots/``.

A failing case means the text sent to the model changed. If that was the intent (a
template or label edit), regenerate, review the diff of the ``.txt`` files, and bump
``VERSION`` in the affected ``resources/<lang>/prompts/``::

    python -m pytest tests/test_prompt_snapshots.py --snapshot-update

Only the public prompt builders are called, so the inputs here are the same for any
refactor of how the prompts are assembled.
"""

from __future__ import annotations

import pytest
from syrupy.extensions.single_file import SingleFileSnapshotExtension, WriteMode

pytestmark = pytest.mark.unit


class TextSnapshotExtension(SingleFileSnapshotExtension):
    """One plain-text file per prompt, so a prompt change reads as a normal diff."""

    _write_mode = WriteMode.TEXT
    file_extension = "txt"


# (country, language): every shipped pairing, plus Kenya in Spanish to cover a country
# block resolved through the English fallback.
LOCALES = [
    ("kenya", "en"),
    ("zambia", "en"),
    ("south_africa", "en"),
    ("argentina", "es"),
    ("kenya", "es"),
]
JOBS = ("full", "minimal", "city_is_country")


def _context(job: str, country_name: str):
    from app.job_context import JobDescriptionContext

    full = job == "full"
    return JobDescriptionContext(
        job_id="J1" if full else None,
        job_fingerprint=None,
        title="Line Cook" if job != "minimal" else None,
        employer="ACME Hotels" if full else None,
        category="Hospitality" if full else None,
        application_url="https://example.com/apply" if full else None,
        posted_date="2026-01-01" if full else None,
        closing_date="2026-02-01" if full else None,
        source_platform="BrighterMonday" if full else None,
        # Braces in job text must reach the model unescaped.
        raw_job_description=(
            "Prepare meals {daily} for 200 guests.\n\nWork shifts." if full else None
        ),
        requirements_text="Diploma in culinary arts. KCSE C+." if full else None,
        salary="KES 30,000" if full else None,
        city=country_name if job == "city_is_country" else ("Nairobi" if full else None),
        county="Nairobi County" if full else None,
        employment_type="Full-time" if full else None,
        classified={},
    )


def _occupation(job: str):
    from app.prompts import JobAttributesLlm2Extras, OccupationPromptEnrichment

    if job == "minimal":
        return OccupationPromptEnrichment("Unknown role", None, None, None, None), None
    occupation = OccupationPromptEnrichment(
        occupation_label="cook",
        csv_definition="Cooks prepare and present food.",
        tabiya_occupation_id="T-5120",
        esco_uuid="uuid-cook",
        classifier_label="chef",
    )
    if job != "full":
        return occupation, None
    extras = JobAttributesLlm2Extras(
        alt_labels=("chef de partie", "kitchen cook"),
        context_str="Median monthly earnings: KES 28,000.",
        included_occupations="cooks; grill cooks",
        included_tasks="preparing ingredients; plating dishes",
        excluded_occupations="head chefs",
        notes="Excludes fast-food preparers.",
    )
    return occupation, extras


def _skills(job: str) -> list[dict[str, str]]:
    if job == "minimal":
        return []
    return [
        {"label": "prepare meals", "description": "Cook dishes from raw ingredients."},
        {"label": "food safety", "description": ""},
        {"label": "teamwork"},
    ]


def _occupation_skills(job: str) -> list[dict]:
    if job == "minimal":
        return []
    return [
        {
            "tabiya_occupation_id": "T-5120",
            "occupation_label": "cook" if job == "full" else "",
            "skills": [
                {"relation_type": "essential", "skill_id": "S1", "label": "prepare meals"},
                {"relation_type": "Optional", "skill_id": "S2", "label": ""},
                {"relation_type": "other", "skill_id": "S3", "label": "ignored"},
                {"relation_type": "essential", "skill_id": "", "label": "no id"},
                "not a dict",
            ],
        }
    ]


def _render(settings, job: str) -> dict[str, str]:
    from app.llm_service import build_job_attributes_prompt_for_job
    from app.prompts import build_occupation_skills_relevance_prompt, build_skill_ranker_prompt
    from app.schemas import load_job_attributes_schema, load_onet_work_activities
    from utils.classifier_linked_entities import full_details_from_context

    ctx = _context(job, settings.country_name)
    occupation, extras = _occupation(job)
    onet = load_onet_work_activities(settings=settings)
    llm1 = build_skill_ranker_prompt(
        full_details_from_context(ctx), _skills(job), settings=settings
    )
    llm2 = build_job_attributes_prompt_for_job(
        ctx,
        occupation=occupation,
        extras=extras,
        schema=load_job_attributes_schema(settings=settings),
        # A handful of rows, including the last, keeps the files readable.
        onet=(*onet[:4], onet[-1]),
        settings=settings,
    )
    llm3 = build_occupation_skills_relevance_prompt(ctx, _occupation_skills(job), settings=settings)
    return {"llm1": llm1, "llm2": llm2, "llm3": llm3}


@pytest.mark.parametrize("job", JOBS)
@pytest.mark.parametrize(("country", "language"), LOCALES)
def test_prompts_match_snapshot(make_settings, snapshot, country, language, job):
    settings = make_settings(TARGET_COUNTRY=country, RERANKER_LANGUAGE=language)
    text = snapshot.with_defaults(extension_class=TextSnapshotExtension)
    for name, prompt in _render(settings, job).items():
        assert prompt == text(name=name)
