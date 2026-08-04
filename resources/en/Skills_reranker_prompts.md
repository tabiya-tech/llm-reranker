```python
from typing import List, Dict

def create_skills_prompt_combined(
    full_details: str,
    potential_skills: List[Dict[str, str]]
) -> str:
    """
    Combined skills prompt: Extracts both essential and optional skills 
    while explicitly filtering out irrelevant skills from the potential list.
    """
    skills_text = _format_labeled_list(potential_skills)

    return f"""You are an expert technical recruiter with deep knowledge of the job market. Your task is to analyze a job opportunity and categorize relevant skills from a provided list into specific buckets.

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

```