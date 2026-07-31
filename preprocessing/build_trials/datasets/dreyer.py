from __future__ import annotations

from pathlib import Path

from profiles.channels import rename_raw_channels
from utils.schema import Recording, TrialRecord


RUNS = ["R3_onlineT", "R4_onlineT", "R5_onlineT", "R6_onlineT"]
LABELS = {769: (0, "left"), 770: (1, "right")}


def require_real_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        raise ValueError(f"placeholder file has not been replaced: {path}")


def raw_root(base: Path) -> Path:
    return base / "Dreyer_MI_25" / "DATA B"


def subjects(base: Path, dataset_id: str) -> list[str]:
    root = raw_root(base)
    return sorted(path.name for path in root.iterdir() if path.is_dir() and path.name.startswith("B"))


def load_recordings(base: Path, dataset_id: str, subject_id: str, mne) -> list[Recording]:
    out = []
    for run in RUNS:
        source = raw_root(base) / subject_id / f"{subject_id}_{run}.gdf"
        require_real_file(source)
        raw = mne.io.read_raw_gdf(source, preload=False, verbose=False)
        drop = [ch for ch in raw.ch_names if ch in {"EOG1", "EOG2", "EOG3", "EMGg", "EMGd"}]
        if drop:
            raw.drop_channels(drop)
        rename_raw_channels(raw)
        raw.set_montage("standard_1005", on_missing="ignore", verbose=False)
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        reverse = cue_code_map(event_id)
        records = []
        for seq, (sample, _, mne_code) in enumerate(events, start=1):
            code = reverse.get(int(mne_code))
            if code not in LABELS:
                continue
            label, label_name = LABELS[code]
            start = float(sample) / float(raw.info["sfreq"])
            trial_id = f"dreyer_{subject_id}_{run}_{seq:03d}"
            records.append(TrialRecord(trial_id, subject_id, label, label_name, start, 5.0, str(source), {"event_code": code}))
        out.append(Recording(raw=raw, records=records, source=source, unit_mode="mne_uv"))
    return out


def cue_code_map(event_id: dict) -> dict[int, int]:
    out = {}
    for name, code in event_id.items():
        digits = "".join(ch for ch in str(name) if ch.isdigit())
        if digits:
            original = int(digits)
            if original in LABELS:
                out[int(code)] = original
    return out
