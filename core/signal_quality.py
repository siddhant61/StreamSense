"""A real (if basic) signal-quality index computed from recent samples.

This replaces the fabricated constants (92/87/85) the legacy UI emitted. It is a simple,
honest SQI in 0..1 derived from observable properties of a rolling sample buffer:

- **finite**: fraction of finite (non-NaN/inf) values — drops/garbage lower this.
- **liveness**: penalizes flatline (variance ~ 0 → sensor not in contact / disconnected).
- **headroom**: penalizes railing/saturation against a known amplitude range.
- **rate**: ratio of observed to expected sample rate (dropouts), when both are known.

It is deliberately not a clinical metric; insufficient data yields ``None`` (unknown)
rather than a guessed number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

MIN_SAMPLES = 8


@dataclass
class QualityScore:
    value: Optional[float]            # 0..1, or None when unknown
    label: str                        # "good" | "fair" | "poor" | "unknown"
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "label": self.label, "detail": self.detail}


def _label(value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    if value >= 0.8:
        return "good"
    if value >= 0.5:
        return "fair"
    return "poor"


def _as_channels(samples: Sequence[Any]) -> List[List[float]]:
    """Normalize samples to a list of per-channel float lists."""
    if not samples:
        return []
    first = samples[0]
    if isinstance(first, (list, tuple)):
        n = len(first)
        channels: List[List[float]] = [[] for _ in range(n)]
        for s in samples:
            for i in range(n):
                channels[i].append(_to_float(s[i]))
        return channels
    return [[_to_float(s) for s in samples]]


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _mean_std(values: List[float]) -> tuple:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float("nan"), 0.0
    mean = sum(finite) / len(finite)
    var = sum((v - mean) ** 2 for v in finite) / len(finite)
    return mean, math.sqrt(var)


def _channel_quality(values: List[float], amplitude_range: Optional[tuple]) -> float:
    n = len(values)
    finite = [v for v in values if math.isfinite(v)]
    finite_frac = len(finite) / n if n else 0.0

    _, std = _mean_std(values)
    # Liveness: any real biosignal varies; flatline -> ~0. Scale is heuristic.
    liveness = 1.0 if std > 1e-6 else 0.0

    headroom = 1.0
    if amplitude_range is not None and finite:
        lo, hi = amplitude_range
        railed = sum(1 for v in finite if v <= lo or v >= hi)
        headroom = 1.0 - railed / len(finite)

    return max(0.0, min(1.0, finite_frac * headroom * liveness))


def assess(
    samples: Sequence[Any],
    *,
    expected_rate: Optional[float] = None,
    actual_rate: Optional[float] = None,
    amplitude_range: Optional[tuple] = None,
) -> QualityScore:
    """Score a rolling buffer of samples. Too few samples -> unknown (None)."""
    if samples is None or len(samples) < MIN_SAMPLES:
        return QualityScore(None, "unknown", {"reason": "insufficient samples",
                                              "n": 0 if not samples else len(samples)})

    channels = _as_channels(samples)
    per_channel = [_channel_quality(ch, amplitude_range) for ch in channels]
    quality = sum(per_channel) / len(per_channel) if per_channel else 0.0

    rate_ratio = None
    if expected_rate and actual_rate is not None and expected_rate > 0:
        rate_ratio = max(0.0, min(1.0, actual_rate / expected_rate))
        quality *= rate_ratio

    quality = max(0.0, min(1.0, quality))
    return QualityScore(
        value=quality,
        label=_label(quality),
        detail={
            "n": len(samples),
            "channels": len(channels),
            "per_channel": [round(q, 3) for q in per_channel],
            "rate_ratio": None if rate_ratio is None else round(rate_ratio, 3),
        },
    )
