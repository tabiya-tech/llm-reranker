"""Tiered, deterministic sampling for LLM traces.

A batch run ranks thousands of jobs at three LLM calls and one embedding call each, so
sampling is a cost control from day one. The decision is a hash of a stable key (the
job id) rather than a random draw, which means:

- re-ranking the same job gets the same decision, across processes and restarts;
- a job is always traced whole or not at all, never half-recorded;
- the behaviour is testable without patching the random module.
"""

from __future__ import annotations

import hashlib
from enum import Enum

from app.observability.config import TracingConfig

_HASH_BYTES = 8
_HASH_SPACE = 2 ** (_HASH_BYTES * 8)


class SamplingTier(Enum):
    """The kind of work being sampled. Each tier has its own rate in ``TracingConfig``."""

    # A job ranked by the batch CLI (the Cloud Run Job).
    BATCH = "batch"
    # A job ranked by ``POST /rank`` (the Cloud Run Service).
    API = "api"


def rate_for_tier(tier: SamplingTier, config: TracingConfig) -> float:
    """The configured sample rate for a tier, between 0.0 and 1.0."""
    if tier == SamplingTier.API:
        return config.api_sample_rate
    return config.batch_sample_rate


def hash_to_unit_interval(key: str) -> float:
    """Map an arbitrary key onto [0.0, 1.0) deterministically and uniformly."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:_HASH_BYTES], "big") / _HASH_SPACE


def should_sample(*, tier: SamplingTier, key: str, config: TracingConfig) -> bool:
    """Whether the unit of work identified by ``key`` should be traced."""
    rate = rate_for_tier(tier, config)
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    # Salt with the tier so the batch and API decisions for one job are independent.
    return hash_to_unit_interval(f"{tier.value}:{key}") < rate
