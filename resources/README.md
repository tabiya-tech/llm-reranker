# Prompt templates

Every prompt the reranker sends to the model lives in a language pack, under
`resources/<language>/prompts/`. `app/prompts.py` only assembles them: it fills the
placeholders with job data and joins the pieces. If you want to change what the model is
told, edit these files, not Python.

```
resources/<language>/prompts/
  VERSION                                   prompt version of this pack
  skill_ranker.md                           LLM1 — skill ranking (whole prompt)
  job_attributes_system.md                  LLM2 — system part
  job_attributes_response.md                LLM2 — ONET menu + JSON response contract
  occupation_skills_relevance_system.md     LLM3 — system part
  occupation_skills_relevance_user.md       LLM3 — job ad, skill menu, JSON response contract
  countries/<country>/<field>.md            LLM2 — one country-specific response field
```

## How a prompt is resolved

Each file is looked up in `resources/<language>/prompts/`, then `resources/en/prompts/`,
then `resources/prompts/`. A language only ships the files it translates. Everything else
comes from English, so `resources/en/prompts/` must always be complete. A template no pack
ships raises `FileNotFoundError` at startup of the batch and the API, before any job runs.

The `es` pack currently translates `skill_ranker.md` and the two system prompts. Its
response contracts and LLM3 user message come from English on purpose: the JSON keys,
attribute `name`s, `level_id`s, `skill_id`s and `tabiya_occupation_id`s are identifiers
the model must copy verbatim, and a translated contract invites translated keys back.

## Placeholders

Templates use `str.format` syntax. `{name}` is a placeholder, and a literal brace must be
doubled: `{{` / `}}`. That matters in the JSON examples. A placeholder the code does not
supply is left in the text as-is instead of failing the batch. To catch a mistyped
placeholder, read an assembled prompt (`main.py --log-llm1-prompt`, `--log-llm2-prompt`,
`--log-llm3-prompt`).

Every template gets:

| Placeholder | Value |
|---|---|
| `{country}` | Country name from `TARGET_COUNTRY` (`Kenya`, `Argentina`, …) |
| `{language}` | Language the model must answer in (`prompt_language` in `app/languages/<lang>_config.py`) |

And per template:

| Template | Placeholders |
|---|---|
| `skill_ranker.md` | `{full_details}` job ad text · `{skills_text}` numbered skill menu |
| `job_attributes_system.md` | — |
| `job_attributes_response.md` | `{work_activities_menu}` numbered ONET list · `{attributes_heading}` the heading the LLM2 user block used · `{country_field_instructions}` the country's field blocks (empty for most countries; place it right after the last response-field bullet) · `{country_field_json}` the country's keys in the JSON example (place it after the last key) |
| `occupation_skills_relevance_system.md` | — |
| `occupation_skills_relevance_user.md` | `{job_description}` ad description + requirements · `{listing_metadata}` structured listing fields · `{skills_menu}` occupation skills to judge |
| `countries/<country>/<field>.md` | — |

The short labels assembled around these templates are not files. They live under
`prompt_labels` in `app/languages/<lang>_config.py`: the LLM2 user block headings, the
metadata field names, menu captions, and empty-menu placeholders. A key a language does not
define falls back to English.

## Versioning

Every LLM-produced document records the prompts that produced it in `llm_reranker_meta`,
next to the model fields:

- `prompt_version`: `<language>/<contents of VERSION>`, e.g. `en/1`. **Bump `VERSION` in
  the same commit as any change to that pack's prompt files.**
- `prompt_digest`: the first 12 hex chars of a SHA-256 over every template this run resolved,
  including English fallbacks and the country's blocks. It changes on any edit, so a
  forgotten bump still shows up, and it tells Kenya's prompts apart from Zambia's under the
  same version.

Every pack ships its own `VERSION`. Otherwise it would report English's version for
its own text, and `tests/test_prompt_resources.py` checks this.

## Adding a country variant

A country can ask LLM2 for extra yes/no fields on top of the shared contract. Kenya asks for
`requires_post_secondary`, for example. Each field is a boolean at the top level of the model's
JSON, stored on `llm_job_attributes.attributes.<field>` (`false` when missing or unparseable).

1. Add the country and its fields to `_JOB_ATTRIBUTES_COUNTRY_FIELDS` in `app/config.py`:

   ```python
   _JOB_ATTRIBUTES_COUNTRY_FIELDS = {
       "kenya": ("requires_post_secondary",),
       "zambia": ("requires_post_secondary",),
   }
   ```

   Settings expose this as `settings.job_attributes_country_fields`. The prompt builder and
   the response parser both read it, and neither branches on the country.

2. Write the instructions at `resources/en/prompts/countries/<country>/<field>.md`. Format
   it as a bullet in the response-field list, continuing lines with two spaces, as in
   `countries/kenya/requires_post_secondary.md`. Use the country's own terms (Zambia's
   Grade 12 certificate, not Kenya's KCSE). Add a translated copy under another language's
   pack only if that language runs for the country.

3. Bump `VERSION` in every pack whose prompts changed, then run
   `python -m pytest tests/test_prompt_resources.py`. It fails if any configured field lacks
   a block in some language.

To *change* a country's wording, edit its block and bump `VERSION`; no code changes.

## Adding a language's prompts

Copy the files you want to translate from `resources/en/prompts/` into
`resources/<language>/prompts/`, add a `VERSION`, and translate the prose. Keep every
placeholder and every doubled brace, and leave identifiers (JSON keys, `level_id`,
`skill_id`, `WA_Element_ID`) untranslated. The skill and occupation labels in the menus come
from that language's taxonomy CSVs. The prompt must tell the model to return them verbatim
in that language. See `resources/es/prompts/skill_ranker.md`.
