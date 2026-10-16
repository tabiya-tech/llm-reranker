"""Job-ad text blocks shared by the LLM 2 and LLM 3 prompts. No IO."""

from __future__ import annotations

from app.job_context import JobDescriptionContext


def build_job_attributes_jd_text(ctx: JobDescriptionContext) -> str:
    """Raw job description + requirements for the attribute classifier."""
    parts: list[str] = []
    if ctx.raw_job_description:
        parts.append(ctx.raw_job_description.strip())
    if ctx.requirements_text:
        parts.append("Requirements:\n" + ctx.requirements_text.strip())
    return "\n\n".join(parts) if parts else "(no job description text provided)"


def build_job_attributes_metadata_block(ctx: JobDescriptionContext) -> str:
    """Structured listing fields (salary, location, etc.) for LLM call 2."""
    rows: list[str] = []
    for label, val in (
        ("Title", ctx.title),
        ("Employer", ctx.employer),
        ("Category", ctx.category),
        ("Salary", ctx.salary),
        ("Location / city", ctx.city),
        ("Region", ctx.county),
        ("Employment type", ctx.employment_type),
        ("Posted", ctx.posted_date),
        ("Closing", ctx.closing_date),
        ("Source platform", ctx.source_platform),
        ("Application URL", ctx.application_url),
    ):
        if val is not None and str(val).strip():
            rows.append(f"- {label}: {val}")
    return "\n".join(rows) if rows else "(no structured metadata fields available)"
