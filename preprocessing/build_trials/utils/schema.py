from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrialRecord:
    trial_id: str
    subject_id: str
    label: int
    label_name: str
    start_sec: float
    duration_sec: float
    source: str
    extra: dict[str, Any]


@dataclass
class Recording:
    raw: Any
    records: list[TrialRecord]
    source: Path
    unit_mode: str
