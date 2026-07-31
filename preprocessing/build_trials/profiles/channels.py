from __future__ import annotations

import numpy as np


ALIASES = {
    "FP1": "Fp1", "FP2": "Fp2", "FPZ": "Fpz",
    "FZ": "Fz", "FCZ": "FCz", "CZ": "Cz", "CPZ": "CPz", "PZ": "Pz",
    "POZ": "POz", "OZ": "Oz", "IZ": "Iz",
    "AFF5": "AF5", "AFF6": "AF6",
    "S1": "Fp1", "S2": "Fp2", "S3": "C3", "S4": "C4", "S5": "Fz",
    "TRG": "TRG", "EOG1": "EOG1", "EOG2": "EOG2", "HEOG": "HEOG", "VEOG": "VEOG",
}

def clean_name(name: str) -> str:
    key = name.strip().replace(".", "").replace(" ", "").upper()
    return ALIASES.get(key, name.strip().replace(".", ""))


def rename_raw_channels(raw) -> None:
    mapping = {ch: clean_name(ch) for ch in raw.ch_names}
    raw.rename_channels(mapping)


def channel_key(name: str) -> str:
    return clean_name(name).upper()


def positions(names: list[str]) -> dict[str, np.ndarray]:
    import mne

    montage = mne.channels.make_standard_montage("standard_1005")
    pos = {channel_key(k): np.asarray(v, dtype="float32") for k, v in montage.get_positions()["ch_pos"].items()}
    return {name: pos.get(channel_key(name), np.full(3, np.nan, dtype="float32")) for name in names}
