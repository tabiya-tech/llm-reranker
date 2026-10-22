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
}
