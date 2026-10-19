## Job listing (description and requirements)

{job_description}

## Listing metadata

{listing_metadata}

## Occupation-linked skills to judge

Each line below is one taxonomy skill tied to an occupation classified on this posting.
Decide whether it is **relevant to this specific job advertisement** (not only whether it belongs to the occupation in general).

{skills_menu}

## Required response format

Respond with **only** a single JSON object (no markdown fences, no other text).

- `decisions`: array with **exactly one object per skill** listed under "Occupation-linked skills" above that has `taxonomy_role` **essential** or **optional**. Each object must have:
  - `skill_id` (string, must match exactly)
  - `tabiya_occupation_id` (string, must match exactly)
  - `is_relevant` (boolean)

Do not include `summary`, `reason`, or any other top-level keys.

**JSON shape:**
{{
  "decisions": [
    {{"skill_id": "<uuid>", "tabiya_occupation_id": "<id>", "is_relevant": true}}
  ]
}}
