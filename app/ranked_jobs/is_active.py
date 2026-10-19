"""``is_active`` from ``classifier_metadata.closing_date``: open until the closing day. No IO."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import Field

from app.domain_base import DomainModel

MissingPolicy = Literal["keep", "active", "inactive"]


def parse_closing_date(raw: object) -> date | None:
    """Best-effort parser for common closing_date formats."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None

    # Try ISO first (supports "YYYY-MM-DD" and many datetime variants).
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).date()
    except ValueError:
        pass

    # Common fallback formats seen in mixed datasets.
    patterns = (
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y/%m/%d",
        "%d %b %Y",
        "%d %B %Y",
    )
    for p in patterns:
        try:
            return datetime.strptime(s, p).date()
        except ValueError:
            continue
    return None


class IsActiveSyncStats(DomainModel):
    """Per-run counters for ``sync_is_active_from_closing_date``."""

    total: int = Field(description="Documents scanned.")
    parsed: int = Field(description="Documents with a parseable `closing_date`.")
    missing_or_bad: int = Field(description="Documents with no or unparseable `closing_date`.")
    will_set_true: int = Field(description="Documents whose `is_active` is set to true.")
    will_set_false: int = Field(description="Documents whose `is_active` is set to false.")
    unchanged: int = Field(description="Documents whose `is_active` already matched the target.")
    modified: int = Field(description="Documents actually written (0 in `dry_run`).")


def target_is_active(
    current: bool,
    closing_raw: object,
    *,
    today: date,
    missing_policy: MissingPolicy,
) -> tuple[bool, bool]:
    """``(is_active, closing_date_parsed)`` for one ranked job.

    A job is active while its closing date is today or later. ``missing_policy`` decides a job
    whose closing date is missing or unparseable: keep its current flag, or force it.
    """
    d = parse_closing_date(closing_raw)
    if d is None:
        if missing_policy == "keep":
            return current, False
        return missing_policy == "active", False
    return d >= today, True
