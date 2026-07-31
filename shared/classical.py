from __future__ import annotations

import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler


def bandpass(
    x: np.ndarray, sfreq: float, low: float, high: float
) -> np.ndarray:
    sos = signal.butter(
        4, [low, high], btype="bandpass", fs=sfreq, output="sos"
    )
    return signal.sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def welch(
    x: np.ndarray, sfreq: float, low: float, high: float
) -> np.ndarray:
    freqs, psd = signal.welch(
        x,
        fs=sfreq,
        nperseg=min(int(sfreq * 2), x.shape[-1]),
        axis=-1,
    )
    keep = (freqs >= low) & (freqs <= high)
    return np.log(psd[..., keep] + 1e-8).reshape(len(x), -1).astype(
        np.float32
    )


def raw_features(
    method: str,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_apply: np.ndarray,
    sfreq: float,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "welch_psd":
        low, high = config["welch_band"]
        return welch(x_fit, sfreq, low, high), welch(
            x_apply, sfreq, low, high
        )
    if method != "csp":
        raise ValueError(f"unknown classical method: {method}")
    import mne
    from mne.decoding import CSP

    mne.set_log_level("WARNING")
    low, high = config["csp_bandpass"]
    fit_filtered = bandpass(x_fit, sfreq, low, high).astype(
        np.float64, copy=False
    )
    apply_filtered = bandpass(x_apply, sfreq, low, high).astype(
        np.float64, copy=False
    )
    transform = CSP(
        n_components=min(config["csp_components"], x_fit.shape[1]),
        reg=config["csp_regularization"],
        log=True,
        norm_trace=False,
    )
    transform.fit(fit_filtered, y_fit)
    return (
        transform.transform(fit_filtered).astype(np.float32),
        transform.transform(apply_filtered).astype(np.float32),
    )


def fit_transform(
    method: str,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_apply: np.ndarray,
    sfreq: float,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    fit_features, apply_features = raw_features(
        method, x_fit, y_fit, x_apply, sfreq, config
    )
    scaler = StandardScaler().fit(fit_features)
    return (
        scaler.transform(fit_features).astype(np.float32),
        scaler.transform(apply_features).astype(np.float32),
    )
