"""``POST /rank/skills`` response: prompt ids mapped back onto the caller's ids. No IO."""

from __future__ import annotations

from app.job_ranking.experience_skills.types import ExperienceSkillsResult
from app.llm.errors import LlmParseError
from app.llm_models import parse_llm_json_response


class ExperienceSkillRankerResponseParser:
    """``{"ranked_skill_ids": [...]}`` → caller ids, best first.

    Ids the prompt did not offer and repeats are dropped, and the list is cut at
    ``max_skills``: a provider without structured output gets only the prose contract.
    """

    def parse(
        self, raw: str, caller_id_by_prompt_id: dict[str, str], max_skills: int
    ) -> ExperienceSkillsResult:
        data = parse_llm_json_response(raw)
        ranked_raw = data.get("ranked_skill_ids")
        if not isinstance(ranked_raw, list):
            raise LlmParseError("ranked_skill_ids must be a list")
        ranked: list[str] = []
        for prompt_id in ranked_raw:
            if not isinstance(prompt_id, str):
                continue
            caller_id = caller_id_by_prompt_id.get(prompt_id.strip())
            if caller_id is not None and caller_id not in ranked:
                ranked.append(caller_id)
        ranked = ranked[:max_skills]
        kept = set(ranked)
        discarded = [cid for cid in caller_id_by_prompt_id.values() if cid not in kept]
        return ExperienceSkillsResult(ranked_ids=ranked, discarded_ids=discarded, raw_response=raw)
