from __future__ import annotations

from typing import Any

import numpy as np


ALIASES = {
    "FP1": "Fp1", "FP2": "Fp2", "FPZ": "Fpz",
    "FZ": "Fz", "FCZ": "FCz", "CZ": "Cz", "CPZ": "CPz", "PZ": "Pz",
    "POZ": "POz", "OZ": "Oz", "IZ": "Iz",
    "AFF5": "AF5", "AFF6": "AF6",
}

PRETRAINED_19 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
    "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2",
]

STANDARD_ALIASES = {
    "T3": ["T7"],
    "T4": ["T8"],
    "T5": ["P7"],
    "T6": ["P8"],
}

ATTEN_PRETRAINED_SUBSTITUTIONS = {
    "F7": ["AF5"],
    "F3": ["F1"],
    "Fz": ["AFz"],
    "F4": ["F2"],
    "F8": ["AF6"],
}

BIOT_PAIRS = [
    ("Fp1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1"),
    ("Fp2", "F8"), ("F8", "T8"), ("T8", "P8"), ("P8", "O2"),
    ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
]

MIREPNET = [
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
]

PRETRAINED_19_MODELS = {"cbramod", "csbrain", "codebrain"}


def clean_name(name: str) -> str:
    key = str(name).strip().replace(".", "").replace(" ", "").upper()
    return ALIASES.get(key, str(name).strip().replace(".", ""))


def key(name: str) -> str:
    return clean_name(name).upper()


def source_lookup(channel_names: list[str]) -> dict[str, int]:
    return {key(name): i for i, name in enumerate(channel_names)}


def candidate_keys(target: str, substitutions: dict[str, list[str]] | None = None) -> list[str]:
    names = [target] + STANDARD_ALIASES.get(target, [])
    if substitutions:
        names += substitutions.get(target, [])
    return [key(name) for name in names]


def select_intersection(
    data: np.ndarray,
    channel_names: list[str],
    targets: list[str],
    substitutions: dict[str, list[str]] | None = None,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    lookup = source_lookup(channel_names)
    indices, selected, exact, missing, aliases, substituted = [], [], [], [], {}, {}
    for target in targets:
        matched = None
        for cand in candidate_keys(target, substitutions):
            if cand in lookup:
                matched = cand
                break
        if matched is None:
            missing.append(target)
            continue
        indices.append(lookup[matched])
        selected.append(target)
        standard_keys = {key(name) for name in STANDARD_ALIASES.get(target, [])}
        if matched == key(target) or matched in standard_keys:
            exact.append(target)
        else:
            substituted[target] = channel_names[lookup[matched]]
        if matched != key(target):
            aliases[target] = channel_names[lookup[matched]]
    out = data[:, indices, :] if indices else data[:, :0, :]
    return out.astype("float32", copy=False), selected, {
        "selected_channels": selected,
        "exact_channels": exact,
        "substituted_channels": substituted,
        "missing_dropped": missing,
        "aliases": aliases,
    }


def map_or_zero(data: np.ndarray, channel_names: list[str], targets: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    lookup = source_lookup(channel_names)
    out = np.zeros((data.shape[0], len(targets), data.shape[2]), dtype="float32")
    copied, zero_filled, aliases = [], [], {}
    for target_idx, target in enumerate(targets):
        matched = None
        for cand in candidate_keys(target):
            if cand in lookup:
                matched = cand
                break
        if matched is None:
            zero_filled.append(target)
            continue
        out[:, target_idx] = data[:, lookup[matched]]
        copied.append(target)
        if matched != key(target):
            aliases[target] = channel_names[lookup[matched]]
    return out, {"copied_channels": copied, "zero_filled_channels": zero_filled, "aliases": aliases}


def prepare_bendr(data: np.ndarray, channel_names: list[str]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    x19, report = map_or_zero(data, channel_names, PRETRAINED_19)
    amp = np.maximum(np.max(np.abs(x19), axis=(1, 2), keepdims=True), 1e-6)
    scaled = np.clip(x19 / amp, -1.0, 1.0)
    rel = np.repeat(np.log10(amp + 1.0), scaled.shape[2], axis=2)
    out = np.concatenate([scaled, rel], axis=1).astype("float32")
    names = PRETRAINED_19 + ["relative_amplitude"]
    report.update({"selected_channels": names, "relative_amplitude": "log10(max_abs_template_channel + 1), repeated over time"})
    return out, names, report


def prepare_biot(data: np.ndarray, channel_names: list[str]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    lookup = source_lookup(channel_names)
    out = np.zeros((data.shape[0], 18, data.shape[2]), dtype="float32")
    names = [f"{a}-{b}" for a, b in BIOT_PAIRS] + ["C3", "C4"]
    built, zero_filled, aliases = [], [], {}

    def find(name: str):
        for cand in candidate_keys(name):
            if cand in lookup:
                return lookup[cand], cand
        return None, None

    for i, (a, b) in enumerate(BIOT_PAIRS):
        ia, ka = find(a)
        ib, kb = find(b)
        if ia is None or ib is None:
            zero_filled.append(names[i])
            continue
        out[:, i] = data[:, ia] - data[:, ib]
        built.append(names[i])
        if ka != key(a):
            aliases[a] = channel_names[ia]
        if kb != key(b):
            aliases[b] = channel_names[ib]
    for j, name in enumerate(["C3", "C4"], start=16):
        idx, matched = find(name)
        if idx is None:
            zero_filled.append(name)
            continue
        out[:, j] = data[:, idx]
        built.append(name)
        if matched != key(name):
            aliases[name] = channel_names[idx]

    denom = np.percentile(np.abs(out), 95, axis=2, keepdims=True)
    out = out / np.maximum(denom, 1e-6)
    return out.astype("float32"), names, {"selected_channels": names, "built_channels": built, "zero_filled_channels": zero_filled, "aliases": aliases, "normalization": "per-trial channel p95 absolute amplitude"}


def positions(names: list[str]) -> dict[str, np.ndarray]:
    import mne

    montage = mne.channels.make_standard_montage("standard_1005")
    pos = {key(k): np.asarray(v, dtype="float32") for k, v in montage.get_positions()["ch_pos"].items()}
    return {name: pos.get(key(name), np.full(3, np.nan, dtype="float32")) for name in names}


def interpolate_to_template(data: np.ndarray, source_names: list[str], target_names: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    lookup = source_lookup(source_names)
    source_pos = positions(source_names)
    target_pos = positions(target_names)
    out = np.zeros((data.shape[0], len(target_names), data.shape[2]), dtype="float32")
    report = {"copied_channels": [], "interpolated_channels": [], "missing_channels": []}
    valid = [(i, source_pos[ch]) for i, ch in enumerate(source_names) if not np.isnan(source_pos[ch]).any()]
    for target_idx, target in enumerate(target_names):
        source_idx = lookup.get(key(target))
        if source_idx is not None:
            out[:, target_idx] = data[:, source_idx]
            report["copied_channels"].append(target)
            continue
        xyz = target_pos[target]
        if not valid or np.isnan(xyz).any():
            report["missing_channels"].append(target)
            continue
        dist = np.asarray([np.linalg.norm(src_xyz - xyz) for _, src_xyz in valid], dtype="float64")
        nearest = np.argsort(dist)[:4]
        denom = np.maximum(dist[nearest], 1e-6)
        weights = (1.0 / denom) / np.sum(1.0 / denom)
        for weight, nearest_idx in zip(weights, nearest):
            out[:, target_idx] += float(weight) * data[:, valid[int(nearest_idx)][0]]
        report["interpolated_channels"].append(target)
    return out, report


def euclidean_alignment(data: np.ndarray) -> np.ndarray:
    cov = np.zeros((data.shape[1], data.shape[1]), dtype="float64")
    for trial in data:
        cov += trial @ trial.T / max(trial.shape[1] - 1, 1)
    cov /= max(data.shape[0], 1)
    eigval, eigvec = np.linalg.eigh(cov + np.eye(cov.shape[0]) * 1e-6)
    inv_sqrt = eigvec @ np.diag(1.0 / np.sqrt(np.maximum(eigval, 1e-6))) @ eigvec.T
    return np.asarray([inv_sqrt @ trial for trial in data], dtype="float32")


def prepare_mirepnet(data: np.ndarray, channel_names: list[str]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    out, report = interpolate_to_template(data, channel_names, MIREPNET)
    out = euclidean_alignment(out)
    report.update({"selected_channels": MIREPNET, "normalization": "Euclidean Alignment after 45-channel template interpolation"})
    return out.astype("float32"), MIREPNET, report


def apply_policy(
    dataset: str,
    model: str,
    policy: str,
    data: np.ndarray,
    channel_names: list[str],
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    data = data.astype("float32", copy=False)
    report: dict[str, Any] = {"model": model, "channel_policy": policy, "input_channels": list(channel_names)}
    if model == "bendr":
        out, names, details = prepare_bendr(data, channel_names)
    elif model == "biot":
        out, names, details = prepare_biot(data, channel_names)
    elif model == "mirepnet":
        out, names, details = prepare_mirepnet(data, channel_names)
    elif policy == "pretrained_matched" and model in PRETRAINED_19_MODELS:
        substitutions = ATTEN_PRETRAINED_SUBSTITUTIONS if dataset.startswith("atten_") else None
        out, names, details = select_intersection(data, channel_names, PRETRAINED_19, substitutions)
    else:
        out, names, details = data, list(channel_names), {"selected_channels": list(channel_names)}
    report.update(details)
    report["output_shape"] = list(out.shape)
    return out, names, report
