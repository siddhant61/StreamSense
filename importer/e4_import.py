"""Empatica E4 offline importer.

Empatica withdrew the E4 real-time streaming server, so the E4 can't join the live
synchronized capture. Instead we import an **E4 Connect session archive** post-hoc and
align it to the rest of a session by absolute (UTC) timestamps.

An E4 Connect session is a folder (or `.zip`) of CSVs:

- ``ACC.csv`` / ``BVP.csv`` / ``EDA.csv`` / ``HR.csv`` / ``TEMP.csv`` — row 0 is the
  start time (UTC unix seconds, repeated per channel), row 1 is the sample rate (Hz,
  repeated per channel), rows 2+ are samples (ACC has 3 channels x/y/z).
- ``IBI.csv`` — row 0 is ``start_time, "IBI"``; rows are ``seconds_since_start, ibi``.
- ``tags.csv`` — one UTC unix timestamp per line (button presses).

The parsing core works on already-split rows so it is trivially unit-tested; thin loaders
read those rows from a directory or zip.
"""

from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SIGNAL_FILES = ["ACC", "BVP", "EDA", "HR", "TEMP"]
Row = Sequence[str]


@dataclass
class E4Signal:
    name: str
    start_time: float          # UTC unix seconds
    sample_rate: float         # Hz
    samples: List[Any]         # list[float], or list[[x,y,z]] for ACC
    channels: int = 1

    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def duration(self) -> float:
        return self.n / self.sample_rate if self.sample_rate else 0.0

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration

    def timestamps(self) -> List[float]:
        """Absolute UTC timestamp per sample."""
        if not self.sample_rate:
            return [self.start_time] * self.n
        step = 1.0 / self.sample_rate
        return [self.start_time + i * step for i in range(self.n)]


@dataclass
class E4Session:
    signals: Dict[str, E4Signal] = field(default_factory=dict)
    ibi: List[Tuple[float, float]] = field(default_factory=list)  # (abs_time, ibi_seconds)
    tags: List[float] = field(default_factory=list)               # abs UTC times
    source: str = ""

    def start_time(self) -> Optional[float]:
        starts = [s.start_time for s in self.signals.values()]
        return min(starts) if starts else None

    def summary(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "start_time": self.start_time(),
            "signals": {
                name: {"sample_rate": s.sample_rate, "channels": s.channels,
                       "n": s.n, "duration": round(s.duration, 3)}
                for name, s in self.signals.items()
            },
            "ibi_count": len(self.ibi),
            "tags": self.tags,
        }


# --------------------------------------------------------------------------- #
# Parsing core (row-based — no file IO)
# --------------------------------------------------------------------------- #
def _to_float(value: str) -> float:
    return float(str(value).strip())


def parse_signal_rows(name: str, rows: Sequence[Row]) -> E4Signal:
    """Parse standard E4 signal rows (ACC/BVP/EDA/HR/TEMP)."""
    if len(rows) < 2:
        raise ValueError(f"{name}: expected at least a start-time and sample-rate row")
    channels = max(1, len(rows[0]))
    start_time = _to_float(rows[0][0])
    sample_rate = _to_float(rows[1][0])
    samples: List[Any] = []
    for row in rows[2:]:
        if not row or all(str(c).strip() == "" for c in row):
            continue
        if channels > 1:
            if len(row) < channels:
                raise ValueError(
                    f"{name}: sample row has {len(row)} columns, expected {channels}")
            samples.append([_to_float(c) for c in row[:channels]])
        else:
            samples.append(_to_float(row[0]))
    return E4Signal(name=name, start_time=start_time, sample_rate=sample_rate,
                    samples=samples, channels=channels)


def parse_ibi_rows(rows: Sequence[Row]) -> List[Tuple[float, float]]:
    """Parse IBI rows -> list of (absolute_time, ibi_seconds)."""
    if not rows:
        return []
    start_time = _to_float(rows[0][0])
    out: List[Tuple[float, float]] = []
    for row in rows[1:]:
        if len(row) < 2 or str(row[0]).strip() == "":
            continue
        out.append((start_time + _to_float(row[0]), _to_float(row[1])))
    return out


def parse_tags_rows(rows: Sequence[Row]) -> List[float]:
    """Parse tags rows -> list of absolute UTC timestamps."""
    out: List[float] = []
    for row in rows:
        if not row or str(row[0]).strip() == "":
            continue
        out.append(_to_float(row[0]))
    return out


# --------------------------------------------------------------------------- #
# Loaders (directory or zip)
# --------------------------------------------------------------------------- #
def _read_rows(text: str) -> List[Row]:
    return [row for row in csv.reader(io.StringIO(text))]


def _gather_csvs(path: str) -> Dict[str, str]:
    """Return {UPPER_BASENAME_WITHOUT_EXT: text} for every CSV in a dir or zip."""
    p = Path(path)
    out: Dict[str, str] = {}
    if p.is_file() and p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as zf:
            for member in zf.namelist():
                mp = Path(member)
                if mp.suffix.lower() == ".csv":
                    out[mp.stem.upper()] = zf.read(member).decode("utf-8", "replace")
    elif p.is_dir():
        for f in p.glob("*.csv"):
            out[f.stem.upper()] = f.read_text(encoding="utf-8", errors="replace")
    else:
        raise FileNotFoundError(f"not an E4 session dir or zip: {path}")
    return out


def load_session(path: str) -> E4Session:
    """Load an E4 Connect session from a directory or `.zip`."""
    csvs = _gather_csvs(path)
    session = E4Session(source=str(path))
    for name in SIGNAL_FILES:
        if name in csvs:
            session.signals[name] = parse_signal_rows(name, _read_rows(csvs[name]))
    if "IBI" in csvs:
        session.ibi = parse_ibi_rows(_read_rows(csvs["IBI"]))
    if "TAGS" in csvs:
        session.tags = parse_tags_rows(_read_rows(csvs["TAGS"]))
    return session
