"""English language config — the reranker's original behaviour, unchanged.

Resource pack: ``resources/en/``.
"""

LANGUAGE_CONFIG = {
    "language": "en",
    "name": "English",
    "locales": ("en", "eng", "english", "en-gb", "en-us", "en-ke", "en-zm", "en-za"),
    # Sub-directory of `resources/` holding this language's pack. Files absent from it
    # fall back to `resources/` (the pre-language layout).
    "resources_subdir": "en",
    # LOCALE column of the pack's model_info.csv, when it ships one.
    "taxonomy_locale": "en",
    # Inserted into prompts as {language}: the language the model must answer in, and
    # the language the skill menu / occupation labels are written in.
    "prompt_language": "English",
    # Default job-attributes schema basename inside the pack (JOB_ATTRIBUTES_SCHEMA wins).
    "job_attributes_schema": "job_attributes_reduced.json",
    # Prefer `translation.text_en` / `fields_en` on classified_jobs for prompt text.
    # Irrelevant for English documents, which carry no `translation` field.
    "prefer_translation": True,
    # Human-facing scaffolding of the LLM2 user block (app/job_ranking/job_attributes/prompt.py). These are the
    # English strings the prompts have always used — changing them changes English
    # prompts. The JSON response contract stays in English for every language: its keys
    # and level ids are English identifiers the model must copy verbatim.
    "prompt_labels": {
        "opening_with_region": (
            "Estimate job attributes for the following occupation in {region}, {country}:"
        ),
        "opening_country_only": (
            "Estimate job attributes for the following occupation in {country}:"
        ),
        "opening_no_region": (
            "Estimate job attributes for the following occupation in {country} "
            "(region not specified):"
        ),
        "occupation": "Occupation",
        "code": "KESCO Code",
        "alt_names": "Alternative Names",
        "description": "Description",
        "included_occupations": "Included Occupations",
        "excluded_occupations": "Excluded Occupations",
        "included_tasks": "Included Tasks",
        "notes": "Notes",
        "job_listing_block": (
            "Job listing — description and requirements (use together with the occupation above):"
        ),
        "listing_metadata_block": "Additional listing metadata (may be incomplete):",
        "attributes_heading": "## Attributes to Estimate",
        "attributes_instruction": (
            "For each attribute, select the most appropriate level_id and provide brief reasoning."
        ),
    },
}
