from __future__ import annotations

from typing import Any

import numpy as np

from profiles.channels import positions
from profiles.profiles import ModelProfile
from utils.schema import TrialRecord


def preprocess_raw(raw: Any, profile: ModelProfile, notch: float, n_jobs: int) -> Any:
    raw = raw.copy().load_data()
    nyq = float(raw.info["sfreq"]) / 2.0
    if notch and notch < nyq:
        raw.notch_filter([notch], n_jobs=n_jobs, verbose=False)
    low, high = profile.bandpass
    high = min(float(high), nyq - 0.5)
    raw.filter(float(low), high, n_jobs=n_jobs, verbose=False)
    if abs(float(raw.info["sfreq"]) - float(profile.sfreq)) > 1e-6:
        raw.resample(float(profile.sfreq), n_jobs=n_jobs, verbose=False)
    return raw


def raw_array(raw: Any, unit_mode: str) -> np.ndarray:
    if unit_mode == "mne_uv":
        return raw.get_data(units="uV").astype("float32")
    return raw.get_data().astype("float32")


def extract_trials(raw: Any, records: list[TrialRecord], unit_mode: str, profile: ModelProfile) -> tuple[np.ndarray, list[int], list[str], list[str]]:
    sfreq = float(raw.info["sfreq"])
    continuous = raw_array(raw, unit_mode)
    if profile.normalization == "recording_zscore_clip15":
        mean = continuous.mean(axis=1, keepdims=True)
        std = continuous.std(axis=1, keepdims=True) + 1e-6
        continuous = np.clip((continuous - mean) / std, -15.0, 15.0)

    data, labels, trial_ids, label_names = [], [], [], []
    for record in records:
        start = int(round(record.start_sec * sfreq))
        stop = start + int(round(record.duration_sec * sfreq))
        if start < 0 or stop > continuous.shape[1]:
            continue
        data.append(continuous[:, start:stop])
        labels.append(int(record.label))
        trial_ids.append(record.trial_id)
        label_names.append(record.label_name)
    if not data:
        return np.zeros((0, len(raw.ch_names), 0), dtype="float32"), [], [], []
    return np.stack(data).astype("float32"), labels, trial_ids, label_names


def subject_transform(data: np.ndarray, channel_names: list[str], profile: ModelProfile) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    report: dict[str, Any] = {"channel_policy": profile.channel_policy, "normalization": profile.normalization}
    if profile.scale is not None:
        data = data * float(profile.scale)
        report["scale"] = float(profile.scale)
    if profile.normalization == "trial_zscore":
        mean = data.mean(axis=2, keepdims=True)
        std = data.std(axis=2, keepdims=True) + 1e-6
        data = (data - mean) / std
    return data.astype("float32"), channel_names, report


def channel_metadata(channel_names: list[str]) -> dict[str, Any]:
    pos = positions(channel_names)
    return {
        "channel_names": channel_names,
        "channel_positions": {name: pos[name].tolist() for name in channel_names},
    }
