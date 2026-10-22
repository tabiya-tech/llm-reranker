"""Spanish language config (Argentina / Empujar — taxonomy locale ``AR-es``).

Resource pack: ``resources/es/`` — the Argentine Spanish taxonomy (Tabiya ESCO 1.1.1,
locale AR-es). The ESCO relation graph is the same as the English pack's; what differs is
the label text LLM1's skill menu and LLM2's occupation line are built from. The prompts
themselves are the shared English templates in ``resources/prompts/``, which tell the model
to answer in Spanish and copy those labels verbatim.

Files the pack does not ship fall back to ``resources/en/`` and then ``resources/``.
"""

LANGUAGE_CONFIG = {
    "language": "es",
    "name": "Spanish",
    "locales": (
        "es",
        "spa",
        "spanish",
        "espanol",
        "ar-es",  # taxonomy export locale (model_info.csv LOCALE)
        "es-ar",
        "es-419",
        "es-es",
        "es-mx",
        "argentina",  # TARGET_COUNTRY spelling
        "argentine",
    ),
    "resources_subdir": "es",
    "taxonomy_locale": "AR-es",
    "prompt_language": "Spanish",
    # Spanish labels, and earnings levels in ARS rather than KES.
    "job_attributes_schema": "job_attributes_reduced.json",
    # False: with a Spanish taxonomy pack and answers in Spanish, the model reads the
    # original Spanish ad. Spanish labels over machine-translated English text would mix
    # the two. Set RERANKER_PREFER_TRANSLATION=1 to force the MT path back on.
    "prefer_translation": False,
}
