from __future__ import annotations

import logging
import random
import time
from typing import Callable, TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Cap on the exponential backoff sleep, in seconds.
MAX_BACKOFF_S = 60.0

# Substrings that identify a reset/aborted connection in a message we cannot type-match.
_TRANSIENT_MESSAGE_NEEDLES = (
    "connection reset",
    "connection aborted",
    "broken pipe",
    "econnreset",
    "readerror",
    "temporarily unavailable",
)

_TRANSIENT_EXCEPTION_TYPES = (
    httpx.ReadError,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.TimeoutException,
    ConnectionResetError,
    BrokenPipeError,
    TimeoutError,
)


def is_transient_network_error(exc: BaseException) -> bool:
    """True when retrying the identical request may still succeed.

    Covers httpx transport errors, OS-level resets, and SDK exceptions that merely
    wrap one of those (``__cause__`` is followed).
    """
    if isinstance(exc, _TRANSIENT_EXCEPTION_TYPES):
        return True
    msg = str(exc).lower()
    for needle in _TRANSIENT_MESSAGE_NEEDLES:
        if needle in msg:
            return True
    if getattr(exc, "errno", None) == 54:  # macOS ECONNRESET / "connection reset by peer"
        return True
    cause = getattr(exc, "__cause__", None)
    if cause is not None and cause is not exc:
        return is_transient_network_error(cause)
    return False


def backoff_delay_s(attempt: int) -> float:
    """Jittered exponential backoff for a zero-based attempt index."""
    return min(MAX_BACKOFF_S, (2**attempt) + random.uniform(0.0, 1.5))


def call_with_transient_retry(
    call: Callable[[], T],
    *,
    max_attempts: int,
    label: str,
    is_transient: Callable[[BaseException], bool] = is_transient_network_error,
    on_retry: Callable[[], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run ``call``, retrying transient failures with backoff.

    ``on_retry`` runs before each retry — providers use it to drop a stale HTTP
    client so the next attempt opens a fresh pool. Non-transient exceptions, and
    the failure on the final attempt, propagate untouched so the provider can
    translate them into its own error type.
    """
    attempts = max(1, int(max_attempts))
    for attempt in range(attempts):
        try:
            return call()
        except Exception as e:
            if attempt + 1 >= attempts or not is_transient(e):
                raise
            delay = backoff_delay_s(attempt)
            logger.warning(
                "%s transient network error (attempt %s/%s): %s — retry in %.1fs; "
                "recreating API client",
                label,
                attempt + 1,
                attempts,
                e,
                delay,
            )
            sleep(delay)
            if on_retry is not None:
                on_retry()
    # Unreachable: the loop either returns or raises on its final attempt.
    raise AssertionError(f"{label}: retry loop exited without a result")
