## Work activities (ONET Work Activity questionnaires)

Select **3 to 10** distinct `WA_Element_ID` values that best describe day-to-day tasks for this role.
Use **only** ids from this list (no invented ids):

{work_activities_menu}

## Required response format

Respond with **only** a single JSON object (no markdown fences, no text before or after).

- `attributes`: **array**; include **one object per** attribute from "{attributes_heading}", each with:
  - `attribute_name` (string, must match the attribute's `name`),
  - `selected_level_id` (string, must be a listed level `id` for that attribute),
  - `confidence`: `high`, `medium`, or `low`,
  - `reasoning` (short string).
- `summary`: brief synthesis of the overall attribute profile (string).
- `work_activities`: array of `WA_Element_ID` strings from the ONET list above, no duplicates, ordered most central to the role first.{country_field_instructions}

**JSON shape:**
{{
  "attributes": [
    {{
      "attribute_name": "<name>",
      "selected_level_id": "<level_id>",
      "confidence": "high",
      "reasoning": "..."
    }}
  ],
  "summary": "...",
  "work_activities": ["<WA_Element_ID>", "..."]{country_field_json}
}}
