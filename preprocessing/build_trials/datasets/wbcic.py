from __future__ import annotations

import re
from pathlib import Path

from profiles.channels import rename_raw_channels
from utils.schema import Recording, TrialRecord


CONFIG = {
    "wbcic_2c": {
        "raw_subdir": "2C dataset",
        "subjects": [f"sub-{idx:03d}" for idx in range(1, 52)],
        "labels": {"1": (0, "left_hand"), "2": (1, "right_hand")},
    },
    "wbcic_3c": {
        "raw_subdir": "3C dataset",
        "subjects": [f"sub-{idx:03d}" for idx in range(1, 12)],
        "labels": {"1": (0, "left_hand"), "2": (1, "right_hand"), "3": (2, "foot")},
    },
}


def require_real_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        raise ValueError(f"placeholder file has not been replaced: {path}")


def raw_root(base: Path) -> Path:
    return base / "WBCIC_MI_23" / "sourcedata"


def subjects(base: Path, dataset_id: str) -> list[str]:
    return CONFIG[dataset_id]["subjects"]


def eeg_dir(base: Path, dataset_id: str, subject_id: str) -> Path:
    root = raw_root(base) / CONFIG[dataset_id]["raw_subdir"] / subject_id
    if dataset_id == "wbcic_2c":
        return root / "ses-01" / "eeg"
    return root / f"{subject_id}_ses-01_task-motorimagery_eeg"


def load_recordings(base: Path, dataset_id: str, subject_id: str, mne) -> list[Recording]:
    folder = eeg_dir(base, dataset_id, subject_id)
    data_bdf = folder / "data.bdf"
    evt_bdf = folder / "evt.bdf"
    require_real_file(data_bdf)
    require_real_file(evt_bdf)
    raw = mne.io.read_raw_bdf(data_bdf, preload=False, verbose=False)
    drop = [ch for ch in raw.ch_names if ch in {"ECG", "HEOR", "HEOL", "VEOU", "VEOL"}]
    if drop:
        raw.drop_channels(drop)
    rename_raw_channels(raw)
    raw.set_montage("standard_1005", on_missing="ignore", verbose=False)
    labels = CONFIG[dataset_id]["labels"]
    records = []
    for seq, event in enumerate(task_events(evt_bdf, labels), start=1):
        label, label_name = labels[event["event_code"]]
        trial_id = f"{dataset_id}_{subject_id}_{seq:03d}"
        records.append(TrialRecord(trial_id, subject_id, label, label_name, event["onset_sec"], 4.0, str(data_bdf), {"event_code": event["event_code"]}))
    return [Recording(raw=raw, records=records, source=data_bdf, unit_mode="native")]


def task_events(evt_bdf: Path, labels: dict[str, tuple[int, str]]) -> list[dict]:
    events = [event for event in parse_bdf_tal_annotations(evt_bdf) if event["event_code"] in labels]
    return sorted(events, key=lambda x: x["onset_sec"])


def parse_bdf_tal_annotations(path: Path) -> list[dict]:
    data = path.read_bytes()
    header_len = int(data[184:192].decode("ascii").strip())
    n_signals = int(data[252:256].decode("ascii").strip())
    pos = 256
    fields = []
    for width in (16, 80, 8, 8, 8, 8, 8, 80, 8, 32):
        values = []
        for _ in range(n_signals):
            values.append(data[pos:pos + width].decode("latin1"))
            pos += width
        fields.append(values)
    labels = fields[0]
    samples_per_record = [int(value.strip()) for value in fields[8]]
    ann_idx = next(idx for idx, label in enumerate(labels) if "annotation" in label.lower())
    offsets, offset = [], 0
    for n_samples in samples_per_record:
        offsets.append(offset)
        offset += n_samples * 3
    bytes_per_record = sum(samples_per_record) * 3
    ann_bytes = samples_per_record[ann_idx] * 3
    n_records = (len(data) - header_len) // bytes_per_record
    events = []
    for record in range(n_records):
        start = header_len + record * bytes_per_record + offsets[ann_idx]
        for part in data[start:start + ann_bytes].split(b"\x00"):
            if not part or part[:1] not in {b"+", b"-"}:
                continue
            text = part.decode("latin1", errors="ignore")
            match = re.match(r"([+-]\d+(?:\.\d+)?)(?:\x15([^\x14]*))?\x14(.*)", text)
            if not match:
                continue
            for desc in [item for item in match.group(3).split("\x14") if item]:
                events.append({"onset_sec": float(match.group(1)), "event_code": desc})
    return events
