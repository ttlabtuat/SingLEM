from __future__ import annotations

import warnings
from pathlib import Path

from profiles.channels import rename_raw_channels
from utils.schema import Recording, TrialRecord


TASKS = {
    "atten_nback": {
        "prefix": "nback",
        "task_id": "nback",
        "start_codes": (112, 128, 144),
        "task_codes": {16, 48, 64, 80, 96},
        "duration": 10.0,
        "trials_per_block": 2,
        "rest_offset": 46.0,
    },
    "atten_dsr": {
        "prefix": "gonogo",
        "task_id": "dsr",
        "start_codes": (48,),
        "task_codes": {16, 32},
        "duration": 10.0,
        "trials_per_block": 2,
        "rest_offset": 46.0,
    },
    "atten_word": {
        "prefix": "word",
        "task_id": "word",
        "task_code": 16,
        "rest_code": 32,
        "duration": 10.0,
    },
}


def require_real_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        raise ValueError(f"placeholder file has not been replaced: {path}")


def raw_root(base: Path) -> Path:
    return base / "ATTEN_28"


def subjects(base: Path, dataset_id: str) -> list[str]:
    root = raw_root(base)
    return sorted(path.name for path in root.iterdir() if path.is_dir() and path.name.startswith("VP"))


def load_recordings(base: Path, dataset_id: str, subject_id: str, mne) -> list[Recording]:
    spec = TASKS[dataset_id]
    out = []
    for vhdr in sorted((raw_root(base) / subject_id).glob(f"{spec['prefix']}*.vhdr")):
        require_real_file(vhdr)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose=False)
        drop = [ch for ch in raw.ch_names if ch.upper() in {"HEOG", "VEOG"}]
        if drop:
            raw.drop_channels(drop)
        rename_raw_channels(raw)
        raw.set_montage("standard_1005", on_missing="ignore", verbose=False)
        events, _ = mne.events_from_annotations(raw, verbose=False)
        records = word_records(dataset_id, subject_id, vhdr, raw.info["sfreq"], events) if dataset_id == "atten_word" else block_records(dataset_id, subject_id, vhdr, raw.info["sfreq"], events)
        out.append(Recording(raw=raw, records=records, source=vhdr, unit_mode="mne_uv"))
    return out


def event_list(events) -> list[tuple[int, int]]:
    return [(int(sample), int(code)) for sample, _, code in events if int(code) < 10000]


def block_records(dataset_id: str, subject_id: str, source: Path, sfreq: float, events) -> list[TrialRecord]:
    spec = TASKS[dataset_id]
    rows = []
    ev = event_list(events)
    seq = 0
    stride = spec["duration"]
    for start_code in spec["start_codes"]:
        for idx, (_sample, code) in enumerate(ev):
            if code != start_code or idx + 1 >= len(ev):
                continue
            task_sample, task_code = ev[idx + 1]
            if task_code not in spec["task_codes"]:
                continue
            seq += 1
            t0 = task_sample / sfreq
            for seg in range(spec["trials_per_block"]):
                rows.append(record(dataset_id, subject_id, source, seq, seg, 1, "task", t0 + seg * stride, spec["duration"], task_code))
            rest0 = t0 + spec["rest_offset"]
            for seg in range(spec["trials_per_block"]):
                rows.append(record(dataset_id, subject_id, source, seq, seg, 0, "rest", rest0 + seg * stride, spec["duration"], task_code))
    return rows


def word_records(dataset_id: str, subject_id: str, source: Path, sfreq: float, events) -> list[TrialRecord]:
    spec = TASKS[dataset_id]
    rows = []
    seq = {"task": 0, "rest": 0}
    for sample, code in event_list(events):
        if code == spec["task_code"]:
            label, name = 1, "task"
        elif code == spec["rest_code"]:
            label, name = 0, "rest"
        else:
            continue
        seq[name] += 1
        rows.append(record(dataset_id, subject_id, source, seq[name], 0, label, name, sample / sfreq, spec["duration"], code))
    return rows


def record(dataset_id: str, subject_id: str, source: Path, seq: int, seg: int, label: int, label_name: str, start: float, duration: float, code: int | None) -> TrialRecord:
    trial_id = f"{dataset_id}_{subject_id}_{source.stem}_{label_name}_{seq:03d}_{seg:02d}"
    return TrialRecord(trial_id, subject_id, label, label_name, start, duration, str(source), {"event_code": code})
