"""Unified session clock.

All streamers in a capture session should timestamp samples against a single reference
so multi-device data shares one timebase. ``SessionClock`` wraps the best available
monotonic source (``pylsl.local_clock`` when present, else ``time.monotonic``) and
exposes a fixed session ``epoch``.
"""

from __future__ import annotations

import time
from typing import Callable, Optional


def default_time_fn() -> Callable[[], float]:
    """Return the best monotonic time source available (pylsl if importable)."""
    try:
        from pylsl import local_clock  # lazy
        return local_clock
    except Exception:
        return time.monotonic


class SessionClock:
    def __init__(self, epoch: Optional[float] = None, time_fn: Optional[Callable[[], float]] = None):
        self._time_fn = time_fn or default_time_fn()
        self.epoch = epoch if epoch is not None else self._time_fn()

    def now(self) -> float:
        """Absolute reading on the underlying clock."""
        return self._time_fn()

    def elapsed(self) -> float:
        """Seconds since the session epoch."""
        return self._time_fn() - self.epoch

    def stamp(self, t: Optional[float] = None) -> float:
        """Convert an absolute reading (or now) to seconds-since-epoch."""
        return (self._time_fn() if t is None else t) - self.epoch
