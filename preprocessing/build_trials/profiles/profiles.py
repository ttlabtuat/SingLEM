from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    name: str
    sfreq: float
    bandpass: tuple[float, float]
    scale: float | None = None
    normalization: str = "none"
    channel_policy: str = "available"
    mi_only: bool = False


PROFILES: dict[str, ModelProfile] = {
    "singlem": ModelProfile("singlem", 128, (0.5, 50.0), scale=0.01),
    "labram": ModelProfile("labram", 200, (0.1, 75.0), scale=0.01),
    "cbramod": ModelProfile("cbramod", 200, (0.3, 75.0), scale=0.01),
    "csbrain": ModelProfile("csbrain", 200, (0.1, 75.0), scale=0.01),
    "codebrain": ModelProfile("codebrain", 200, (0.3, 75.0), scale=0.01),
    "luna_large": ModelProfile("luna_large", 256, (0.1, 75.0), normalization="trial_zscore"),
    "bendr": ModelProfile("bendr", 256, (0.5, 50.0)),
    "biot": ModelProfile("biot", 200, (0.5, 50.0)),
    "mirepnet": ModelProfile("mirepnet", 250, (8.0, 30.0), mi_only=True),
}

MODEL_ORDER = list(PROFILES)
MI_DATASETS = {"dreyer", "wbcic_2c", "wbcic_3c"}


def dataset_notch(dataset_id: str) -> float:
    return 50.0


def selected_models(value: str) -> list[str]:
    if value == "all":
        return MODEL_ORDER
    models = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(models) - set(PROFILES))
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}")
    return models
