"""Exponential backoff + a generic retry/reconnect helper.

Used for per-device reconnection: instead of a fixed ``time.sleep(2)`` retry loop, wait
``base * factor**attempt`` (capped, with jitter) so a flapping device doesn't hammer the
transport. The retry helper is stop-aware and injectable for headless testing.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class ExponentialBackoff:
    base: float = 0.5        # first delay (seconds)
    factor: float = 2.0      # growth per attempt
    max_delay: float = 30.0  # cap
    jitter: float = 0.1      # +/- fraction of randomization

    def delay(self, attempt: int) -> float:
        """Delay before retry ``attempt`` (attempt 0 = first retry)."""
        raw = self.base * (self.factor ** max(0, attempt))
        capped = min(self.max_delay, raw)
        if self.jitter:
            capped *= 1.0 + random.uniform(-self.jitter, self.jitter)
        return max(0.0, capped)


def retry_with_backoff(
    operation: Callable[[], bool],
    *,
    max_attempts: int = 5,
    backoff: Optional[ExponentialBackoff] = None,
    should_stop: Callable[[], bool] = lambda: False,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Optional[Callable[[int, float], None]] = None,
) -> bool:
    """Call ``operation`` until it returns truthy or attempts/stop are exhausted.

    Returns True on success. ``should_stop`` is checked before each attempt and before
    each sleep so shutdown is responsive. ``sleep``/``backoff`` are injectable for tests.
    """
    backoff = backoff or ExponentialBackoff()
    for attempt in range(max_attempts):
        if should_stop():
            return False
        try:
            if operation():
                return True
        except Exception:
            pass  # treated as a failed attempt; caller logs as needed
        if attempt < max_attempts - 1:
            delay = backoff.delay(attempt)
            if on_retry is not None:
                on_retry(attempt + 1, delay)
            if should_stop():
                return False
            sleep(delay)
    return False
