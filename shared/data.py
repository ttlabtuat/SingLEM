from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"
RESULT_ROOT = PROJECT_ROOT / "results"
METRICS = [
    "accuracy",
    "f1_macro",
    "precision_macro",
    "recall_macro",
    "kappa",
]


@dataclass
class SubjectData:
    data: np.ndarray
    labels: np.ndarray
    trial_ids: list[str]


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def read_csv(path: str | Path, restart: bool = False) -> list[dict]:
    path = Path(path)
    if restart or not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def load_configs(config_dir: str | Path = CONFIG_ROOT) -> dict:
    root = Path(config_dir)
    project_root = root.parent if root.name == "configs" else PROJECT_ROOT
    config = {}
    for name in ["datasets", "models", "experiments"]:
        config.update(read_json(root / f"{name}.json"))
    local_paths = root / "paths.local.json"
    if local_paths.exists():
        config.update(read_json(local_paths))
    else:
        config.update(
            {
                "trials_root": str(project_root / "datasets" / "trials"),
                "features_root": str(project_root / "datasets" / "features"),
                "existing_results_root": str(project_root / "results"),
            }
        )
    return config


def trial_dir(config: dict, dataset: str) -> Path:
    return Path(config["trials_root"]) / dataset / "singlem"


def feature_dir(config: dict, dataset: str, model: str) -> Path:
    if model == "singlem":
        return singlem_variant_feature_dir(
            config,
            dataset,
            config.get("singlem_default_variant", "downstream_excluded"),
        )
    return Path(config["features_root"]) / dataset / model


def singlem_variant_feature_dir(
    config: dict, dataset: str, variant: str
) -> Path:
    """Return the feature directory for one SingLEM checkpoint variant.

    SingLEM features are stored below
    ``datasets/features/<dataset>/singlem/<variant>/`` so the primary
    downstream-excluded checkpoint and ablation checkpoints can coexist without
    overwriting each other. The special value ``default`` resolves to
    ``singlem_default_variant`` from ``configs/models.json``.
    """
    if variant == "default":
        variant = config.get("singlem_default_variant", "downstream_excluded")
    return Path(config["features_root"]) / dataset / "singlem" / variant


def _load_pickle(path: str | Path) -> dict:
    path = Path(path)
    if path.exists() and path.stat().st_size == 0:
        raise ValueError(f"placeholder file has not been replaced: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def _path_from_parts(parts: list[str]) -> Path:
    if parts and parts[0] == "":
        return Path("/") / Path(*parts[1:])
    return Path(*parts)


def _sibling_trial_candidate(feature_path: Path) -> Path | None:
    parts = list(feature_path.parts)
    if "extracted_features" in parts:
        index = parts.index("extracted_features")
        if len(parts) >= index + 4 and parts[index + 1] == "pretrained_matched":
            prefix = parts[:index]
            dataset = parts[index + 2]
            model = parts[index + 3]
            return _path_from_parts(
                prefix
                + ["preprocessed_trials_by_model", dataset, model, feature_path.name]
            )
    if "features" in parts:
        index = parts.index("features")
        if len(parts) >= index + 3:
            prefix = parts[:index]
            dataset = parts[index + 1]
            model = parts[index + 2]
            return _path_from_parts(
                prefix + ["trials", dataset, model, feature_path.name]
            )
    return None


def _source_candidates(source: Path, feature_path: Path) -> list[Path]:
    text = str(source)
    values = []
    if text:
        values.append(source)
        if not source.is_absolute():
            values.append(PROJECT_ROOT / source)
    if "preprocessed_trials_by_model" in text:
        parts = list(source.parts)
        index = parts.index("preprocessed_trials_by_model")
        if len(parts) >= index + 4:
            dataset = parts[index + 1]
            model = parts[index + 2]
            values.append(
                PROJECT_ROOT
                / "datasets"
                / "trials"
                / dataset
                / model
                / feature_path.name
            )
    sibling = _sibling_trial_candidate(feature_path)
    if sibling is not None:
        values.append(sibling)
    unique = []
    seen = set()
    for value in values:
        if str(value) not in seen:
            unique.append(value)
            seen.add(str(value))
    return unique


def load_subject(path: str | Path, flatten: bool = False) -> SubjectData:
    path = Path(path)
    obj = _load_pickle(path)
    data = np.asarray(obj["data"], dtype=np.float32)
    labels = np.asarray(obj["label"], dtype=np.int64)
    trial_ids = obj.get("trial_id")
    if trial_ids is None:
        source = Path(obj.get("metadata", {}).get("source", ""))
        for candidate in _source_candidates(source, path):
            if candidate.exists():
                source = candidate
                break
        else:
            raise ValueError(f"missing trial IDs and source file: {path}")
        source_obj = _load_pickle(source)
        trial_ids = source_obj.get("trial_id")
        if trial_ids is None or not np.array_equal(
            labels, np.asarray(source_obj["label"], dtype=np.int64)
        ):
            raise ValueError(f"feature/source trial mismatch: {path}")
    trial_ids = [str(value) for value in trial_ids]
    if len(data) != len(labels) or len(labels) != len(trial_ids):
        raise ValueError(f"data, labels, and trial IDs differ: {path}")
    if flatten:
        data = data.reshape(data.shape[0], -1)
    return SubjectData(data, labels, trial_ids)


def load_subjects(
    data_dir: str | Path, flatten: bool = False
) -> dict[str, SubjectData]:
    return {
        path.stem: load_subject(path, flatten)
        for path in sorted(Path(data_dir).glob("*.pkl"))
    }


def filtered_subject(subject: SubjectData, labels: list[int]) -> SubjectData:
    keep = np.isin(subject.labels, labels)
    indices = np.flatnonzero(keep)
    mapping = {int(label): index for index, label in enumerate(labels)}
    encoded = np.asarray(
        [mapping[int(value)] for value in subject.labels[indices]],
        dtype=np.int64,
    )
    return SubjectData(
        subject.data[indices],
        encoded,
        [subject.trial_ids[index] for index in indices],
    )


def stack_subjects(
    subjects: dict[str, SubjectData],
    subject_ids: list[str],
    labels: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    selected = [
        filtered_subject(subjects[subject_id], labels)
        for subject_id in subject_ids
    ]
    return (
        np.concatenate([subject.data for subject in selected]),
        np.concatenate([subject.labels for subject in selected]),
    )


def original_source_split(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.RandomState,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    permutation = np.arange(len(y))
    rng.shuffle(permutation)
    x = x[permutation]
    y = y[permutation]
    train_idx, val_idx = train_test_split(
        np.arange(len(y)),
        test_size=0.2,
        stratify=y,
        random_state=seed,
    )
    return x, y, np.asarray(train_idx), np.asarray(val_idx)


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def fraction_key(fraction: float) -> str:
    return f"{float(fraction):.2f}"


def calibration_count(class_count: int, fraction: float) -> int:
    count = math.floor(class_count * fraction + 0.5)
    return min(class_count - 1, max(1, count))


def manifest_hash(
    datasets: list[str], fraction: float, seeds: list[int]
) -> str:
    value = {
        "datasets": sorted(datasets),
        "fraction": fraction_key(fraction),
        "seeds": seeds,
    }
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def build_calibration_manifest(
    config: dict,
    datasets: list[str],
    fraction: float,
    seeds: list[int],
) -> dict:
    manifest = {
        "version": 1,
        "selection": "class_balanced_half_up_rounding",
        "fraction": fraction,
        "seeds": seeds,
        "datasets": {},
    }
    for dataset in datasets:
        entry = config["datasets"][dataset]
        subjects = load_subjects(trial_dir(config, dataset))
        dataset_value = {"labels": entry["labels"], "subjects": {}}
        for subject_id, raw_subject in sorted(subjects.items()):
            subject = filtered_subject(raw_subject, entry["labels"])
            by_seed = {}
            for seed in seeds:
                rng = np.random.default_rng(seed)
                calibration_indices = []
                per_class = {}
                for label in range(len(entry["labels"])):
                    class_indices = np.flatnonzero(subject.labels == label)
                    count = calibration_count(len(class_indices), fraction)
                    calibration_indices.extend(
                        rng.permutation(class_indices)[:count].tolist()
                    )
                    per_class[str(entry["labels"][label])] = count
                selected = set(calibration_indices)
                by_seed[str(seed)] = {
                    "calibration_per_class": per_class,
                    "calibration_trial_ids": [
                        trial_id
                        for index, trial_id in enumerate(subject.trial_ids)
                        if index in selected
                    ],
                    "test_trial_ids": [
                        trial_id
                        for index, trial_id in enumerate(subject.trial_ids)
                        if index not in selected
                    ],
                }
            dataset_value["subjects"][subject_id] = {
                "n_trials": len(subject.labels),
                "splits": {fraction_key(fraction): by_seed},
            }
        manifest["datasets"][dataset] = dataset_value
    return manifest


def ensure_manifest(path: str | Path, expected: dict) -> None:
    path = Path(path)
    if path.exists():
        if read_json(path) != expected:
            raise ValueError(f"existing calibration manifest differs: {path}")
        return
    write_json(path, expected)


def split_indices(
    subject: SubjectData, split: dict
) -> tuple[np.ndarray, np.ndarray]:
    index = {
        trial_id: position
        for position, trial_id in enumerate(subject.trial_ids)
    }
    calibration_ids = split["calibration_trial_ids"]
    test_ids = split["test_trial_ids"]
    if set(index) != set(calibration_ids) | set(test_ids):
        raise ValueError("manifest and subject trial IDs differ")
    if set(calibration_ids) & set(test_ids):
        raise ValueError("calibration and test trial IDs overlap")
    return (
        np.asarray([index[value] for value in calibration_ids]),
        np.asarray([index[value] for value in test_ids]),
    )


def adaptation_key(row: dict) -> tuple:
    return (
        row["subject"],
        fraction_key(float(row["requested_fraction"])),
        int(row["seed"]),
        row["method"],
    )


def split_metadata(
    subject: SubjectData,
    split: dict,
    fraction: float,
    calibration_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict:
    labels = sorted(np.unique(subject.labels).tolist())
    calibration_counts = {
        str(label): int(np.sum(subject.labels[calibration_idx] == label))
        for label in labels
    }
    test_counts = {
        str(label): int(np.sum(subject.labels[test_idx] == label))
        for label in labels
    }
    valid = all(calibration_counts.values()) and all(test_counts.values())
    return {
        "requested_fraction": fraction_key(fraction),
        "actual_fraction": len(calibration_idx) / len(subject.labels),
        "calibration_count": len(calibration_idx),
        "test_count": len(test_idx),
        "calibration_counts": json.dumps(calibration_counts, sort_keys=True),
        "test_counts": json.dumps(test_counts, sort_keys=True),
        "calibration_trial_ids": json.dumps(split["calibration_trial_ids"]),
        "test_trial_ids": json.dumps(split["test_trial_ids"]),
        "valid": valid,
        "invalid_reason": "" if valid else "class_missing",
    }
