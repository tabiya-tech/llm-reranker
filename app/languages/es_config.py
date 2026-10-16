"""Spanish language config (Argentina / Empujar — taxonomy locale ``AR-es``).

Resource pack: ``resources/es/`` — the Argentine Spanish taxonomy (Tabiya ESCO 1.1.1,
locale AR-es) plus Spanish prompt files. The ESCO relation graph is the same as the
English pack's; what differs is the label text LLM1's skill menu and LLM2's occupation
line are built from, and the language the prompts are written in.

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
    # False: with a Spanish taxonomy pack and Spanish prompts, the model reads the
    # original Spanish ad. Prompting in Spanish over machine-translated English text
    # would mix the two. Set RERANKER_PREFER_TRANSLATION=1 to force the MT path back on.
    "prefer_translation": False,
    # Human-facing scaffolding of the LLM2 user block (app/job_ranking/job_attributes/prompt.py). The JSON response
    # contract is deliberately NOT translated: its keys (`attribute_name`,
    # `selected_level_id`, …) and the level ids are identifiers the model must copy
    # verbatim, and a translated contract invites translated keys back.
    "prompt_labels": {
        "opening_with_region": (
            "Estimá los atributos del puesto para la siguiente ocupación en {region}, {country}:"
        ),
        "opening_country_only": (
            "Estimá los atributos del puesto para la siguiente ocupación en {country}:"
        ),
        "opening_no_region": (
            "Estimá los atributos del puesto para la siguiente ocupación en {country} "
            "(región no especificada):"
        ),
        "occupation": "Ocupación",
        # The AR-es taxonomy carries ESCO/Tabiya codes, not Kenya's KESCO codes.
        "code": "Código de la taxonomía",
        "alt_names": "Nombres alternativos",
        "description": "Descripción",
        "included_occupations": "Ocupaciones incluidas",
        "excluded_occupations": "Ocupaciones excluidas",
        "included_tasks": "Tareas incluidas",
        "notes": "Notas",
        "job_listing_block": (
            "Aviso de trabajo — descripción y requisitos (usalos junto con la ocupación anterior):"
        ),
        "listing_metadata_block": "Datos adicionales del aviso (pueden estar incompletos):",
        "attributes_heading": "## Atributos a estimar",
        "attributes_instruction": (
            "Para cada atributo, elegí el level_id más apropiado y explicá brevemente por qué. "
            "Copiá los valores de level_id exactamente como figuran."
        ),
    },
}
