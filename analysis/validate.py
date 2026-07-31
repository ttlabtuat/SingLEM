#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT_ROOT))

from shared.data import (
    feature_dir,
    filtered_subject,
    load_configs,
    load_subjects,
    read_json,
    trial_dir,
)


FORBIDDEN_DIRS = [
    "__pycache__",
    ".ipynb_checkpoints",
    "outputs",
    "runs",
    ".pytest_cache",
]
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}
SINGLEM_CHECKPOINTS = {
    "SingLEM/checkpoints/singlem_downstream_excluded.pt": "cb4c1b4f4a6fae99a984b883177e09b8c2fe44ed0d4c9a0ded3928dc425584bb",
    "SingLEM/checkpoints/singlem_downstream_included.pt": "e8649be17699d3d8d21c0a5b78fa166703b42c848e652b63386ba4f777f28651",
    "SingLEM/checkpoints/singlem_no_feature_embedding.pt": "5b2de666085bbcc54571308778a1631bda395bf6d7585c834a5bff77c709ecda",
}
SINGLEM_GPU_ABLATION_VARIANTS = ["downstream_included", "no_feature_embedding"]
ORIGINAL_COUNTS = {
    "dreyer": 21,
    "wbcic_3c": 11,
    "wbcic_2c": 51,
    "atten_nback": 26,
    "atten_dsr": 26,
    "atten_word": 26,
}
CHANNEL_COUNTS = {
    "dreyer": 567,
    "wbcic_3c": 649,
    "wbcic_2c": 3009,
    "atten_nback": 728,
    "atten_dsr": 728,
    "atten_word": 728,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package_root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config_dir", type=Path)
    parser.add_argument("--portable", action="store_true")
    parser.add_argument("--raw_package", action="store_true")
    parser.add_argument(
        "--reference_root",
        type=Path,
        help="Optional local SingLEM_update root for byte-level result comparison.",
    )
    return parser.parse_args()


def contains_absolute_path(value) -> bool:
    if isinstance(value, dict):
        return any(contains_absolute_path(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_absolute_path(item) for item in value)
    if isinstance(value, str):
        return Path(value).is_absolute()
    return False


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_no_forbidden_dirs(package_root: Path) -> list[str]:
    failures = []
    for name in FORBIDDEN_DIRS:
        for path in package_root.rglob(name):
            if path.is_dir():
                failures.append(f"forbidden directory copied: {path}")
    return failures


def validate_configs(config_dir: Path, portable: bool) -> list[str]:
    failures = []
    if portable and (config_dir / "paths.local.json").exists():
        failures.append("portable package contains configs/paths.local.json")
    for path in sorted(config_dir.glob("*.json")):
        value = read_json(path)
        if portable and contains_absolute_path(value):
            failures.append(f"absolute path in config: {path}")
    return failures


def validate_raw_package(config: dict, package_root: Path) -> list[str]:
    failures = []
    if (package_root / "foundation_models").exists():
        failures.append(
            "raw public package must not include foundation_models/; "
            "it is only a local staging folder for downloaded upstream repositories"
        )
    for path in [
        package_root / "raw_datasets" / "README.md",
        package_root / "datasets" / "trials",
        package_root / "datasets" / "features",
        package_root / "models" / "foundation" / "manifest.json",
        package_root / "SingLEM" / "model.py",
        package_root / "SingLEM" / "model_no_feature_embedding.py",
        package_root / "SingLEM" / "checkpoints",
        package_root / "results",
    ]:
        if not path.exists():
            failures.append(f"missing package path: {path}")
    for value, expected in SINGLEM_CHECKPOINTS.items():
        path = package_root / value
        if not path.exists():
            failures.append(f"missing SingLEM checkpoint: {path}")
        elif sha256(path) != expected:
            failures.append(f"SingLEM checkpoint checksum mismatch: {path}")
    if (package_root / "SingLEM" / "checkpoints" / "singlem_pretrained.pt").exists():
        failures.append("legacy singlem_pretrained.pt must not be published on main")
    return failures


def foundation_source_paths(entry: dict) -> list[str]:
    return list(entry.get("source_paths", [entry["source_path"]]))


def foundation_checkpoint_paths(entry: dict) -> list[str]:
    paths = list(entry.get("checkpoint_paths", []))
    if "checkpoint_path" in entry:
        paths.append(entry["checkpoint_path"])
    return paths


def relative_to_model_dir(model_name: str, value: str) -> str:
    path = Path(value)
    prefix = Path("models") / "foundation" / model_name
    try:
        return str(path.relative_to(prefix))
    except ValueError:
        return path.name


def foundation_candidates(
    package_root: Path,
    model_root: Path,
    model_name: str,
    entry: dict,
    kind: str,
    value: str,
) -> list[Path]:
    candidates = [package_root / value]
    configured = (
        entry.get("install", {})
        .get(f"{kind}_candidates", {})
        .get(value)
    )
    if not configured:
        configured = [relative_to_model_dir(model_name, value)]
    candidates.extend(model_root / model_name / candidate for candidate in configured)
    unique = []
    seen = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def has_ready_candidate(candidates: list[Path]) -> bool:
    return any(path.exists() and path.stat().st_size > 0 for path in candidates)


def validate_foundation_artifacts(
    package_root: Path, publication_checkout: bool
) -> list[str]:
    failures = []
    model_root = package_root / "models" / "foundation"
    manifest = json.loads((model_root / "manifest.json").read_text(encoding="utf-8"))["models"]
    for name, entry in manifest.items():
        for value in foundation_source_paths(entry):
            candidates = foundation_candidates(
                package_root, model_root, name, entry, "source", value
            )
            path = candidates[0]
            if not path.exists():
                failures.append(f"missing model source or placeholder: {path}")
                continue
            is_placeholder = path.stat().st_size == 0
            expected_placeholder = entry["source_distribution"] == "placeholder"
            if publication_checkout and expected_placeholder != is_placeholder:
                failures.append(f"incorrect source placeholder state: {path}")
            if publication_checkout and expected_placeholder:
                for alternative in candidates[1:]:
                    if alternative.exists() and alternative.stat().st_size > 0:
                        failures.append(f"external model source present in raw package: {alternative}")
            if not publication_checkout and not has_ready_candidate(candidates):
                failures.append(f"model source is still a placeholder: {path}")
        for value in foundation_checkpoint_paths(entry):
            candidates = foundation_candidates(
                package_root, model_root, name, entry, "checkpoint", value
            )
            path = candidates[0]
            if not path.exists():
                failures.append(f"missing checkpoint or placeholder: {path}")
            elif name == "singlem" and path.stat().st_size == 0:
                failures.append("SingLEM checkpoint is empty")
            elif publication_checkout and name != "singlem" and path.stat().st_size != 0:
                failures.append(f"competing checkpoint must remain a placeholder: {path}")
            elif publication_checkout and name != "singlem":
                for alternative in candidates[1:]:
                    if alternative.exists() and alternative.stat().st_size > 0:
                        failures.append(f"external checkpoint present in raw package: {alternative}")
            elif not publication_checkout and not has_ready_candidate(candidates):
                failures.append(f"checkpoint is still a placeholder: {path}")
    for path in model_root.rglob("*"):
        if path.is_file() and path.suffix in CHECKPOINT_SUFFIXES:
            if path.stat().st_size > 100_000_000:
                failures.append(f"oversized checkpoint: {path}")
    return failures


def validate_results(package_root: Path) -> list[str]:
    """Validate public result artifacts without requiring raw data.

    The public main branch publishes the revised-manuscript result families
    only. Old submitted CPU sklearn archives and adapted GPU-SVM outputs are
    intentionally kept out of this tree.
    """
    failures = []
    result_root = package_root / "results"
    if (result_root / "original_sklearn").exists():
        failures.append("old CPU sklearn archive must not be under results/ on main")
    if (package_root / "experiments" / "original_sklearn").exists():
        failures.append("old CPU sklearn scripts must not be under experiments/ on main")
    if (result_root / "splits").exists():
        failures.append("stale results/splits must not be published; use results/manifests")
    if (result_root / "adapted_30" / "svm").exists():
        failures.append("adapted-SVM results must not be published")
    if (result_root / "single_channel" / "single_channel_accuracy.csv").exists():
        failures.append("legacy single_channel_accuracy.csv must not be published")
    for path in result_root.glob("ablation/**/manuscript_summary.csv"):
        failures.append(f"manual ablation manuscript summary must not be published: {path}")
    for path in (result_root / "strict" / "svm").glob("**/*"):
        if path.name in {"optuna_results.pkl", "original_results.txt"}:
            failures.append(f"archived sklearn artifact must not be in strict GPU-SVM tree: {path}")
    metric_files = sorted(
        list((result_root / "strict").glob("**/per_subject_metrics.csv"))
        + list((result_root / "adapted_30").glob("**/per_subject_metrics.csv"))
    )
    for metrics in metric_files:
        for filename in ["summary.csv", "run_metadata.json"]:
            if not (metrics.parent / filename).exists():
                failures.append(f"missing {filename}: {metrics.parent}")
    if len(metric_files) != 225:
        failures.append(f"expected 225 canonical result files, found {len(metric_files)}")
    svm_required_keys = {
        "subject", "best_params", "validation_f1_macro", "accuracy",
        "f1_macro", "precision_macro", "recall_macro", "kappa",
    }
    ablation_root = result_root / "ablation" / "gpu_svm" / "singlem"
    for variant in SINGLEM_GPU_ABLATION_VARIANTS:
        for dataset, expected in ORIGINAL_COUNTS.items():
            result_dir = ablation_root / variant / dataset
            metrics = result_dir / "per_subject_metrics.csv"
            summary = result_dir / "summary.csv"
            metadata = result_dir / "run_metadata.json"
            if not metrics.exists() or not summary.exists() or not metadata.exists():
                failures.append(f"incomplete GPU-SVM ablation result set: {variant}/{dataset}")
                continue
            rows = [row for row in read_csv(metrics) if row.get("valid", "True") == "True"]
            if len(rows) != expected or any(not svm_required_keys <= set(row) for row in rows):
                failures.append(f"invalid GPU-SVM ablation metrics: {metrics}")
            info = read_json(metadata)
            if (
                info.get("backend") != "cuml"
                or info.get("model") != "singlem"
                or info.get("singlem_variant") != variant
                or info.get("dataset") != dataset
            ):
                failures.append(f"invalid GPU-SVM ablation metadata: {metadata}")
    required_keys = {
        "subject", "accuracy", "f1_macro", "precision_macro",
        "recall_macro", "kappa",
    }
    for dataset, expected in CHANNEL_COUNTS.items():
        result_dir = (
            result_root / "single_channel" / "gpu_svm" / "singlem"
            / "downstream_excluded" / dataset
        )
        metrics = result_dir / "per_channel_subject_metrics.csv"
        summary = result_dir / "channel_summary.csv"
        metadata = result_dir / "run_metadata.json"
        if not metrics.exists() or not summary.exists() or not metadata.exists():
            failures.append(f"incomplete GPU-SVM single-channel result set: {dataset}")
            continue
        rows = [row for row in read_csv(metrics) if row.get("valid", "True") == "True"]
        if len(rows) != expected or any("channel" not in row or not required_keys <= set(row) for row in rows):
            failures.append(f"invalid GPU-SVM single-channel metrics: {metrics}")
        info = read_json(metadata)
        if (
            info.get("backend") != "cuml"
            or info.get("model") != "singlem"
            or info.get("singlem_variant") != "downstream_excluded"
            or info.get("dataset") != dataset
        ):
            failures.append(f"invalid GPU-SVM single-channel metadata: {metadata}")
    if not (
        result_root / "single_channel" / "gpu_svm" / "singlem"
        / "downstream_excluded" / "all_channel_summary.csv"
    ).exists():
        failures.append("missing all-channel single-channel summary")
    return failures


def validate_reference_results(
    package_root: Path, reference_root: Path
) -> list[str]:
    """Compare included public metric files against a local reference tree."""
    failures = []
    result_root = package_root / "results"
    reference_results = reference_root / "results"
    patterns = [
        "strict/**/per_subject_metrics.csv",
        "adapted_30/mlp/**/per_subject_metrics.csv",
        "adapted_30/neural/**/per_subject_metrics.csv",
        "ablation/gpu_svm/singlem/**/per_subject_metrics.csv",
        (
            "single_channel/gpu_svm/singlem/downstream_excluded/"
            "*/per_channel_subject_metrics.csv"
        ),
    ]
    for pattern in patterns:
        for reference in sorted(reference_results.glob(pattern)):
            rel = reference.relative_to(reference_results)
            current = result_root / rel
            if not current.exists():
                failures.append(f"missing reference-matched result: {current}")
            elif sha256(current) != sha256(reference):
                failures.append(f"result differs from reference: {current}")
    manifest = "manifests/calibration_0.30_31d05bd14f4b.json"
    current_manifest = result_root / manifest
    reference_manifest = reference_results / manifest
    if not current_manifest.exists():
        failures.append(f"missing reference calibration manifest: {current_manifest}")
    elif reference_manifest.exists() and read_json(current_manifest) != read_json(reference_manifest):
        failures.append(f"calibration manifest differs from reference: {current_manifest}")
    return failures


def validate_pickles(
    config: dict, package_root: Path, portable: bool
) -> tuple[int, list[str]]:
    failures = []
    checked = 0
    for dataset, entry in config["datasets"].items():
        canonical = load_subjects(trial_dir(config, dataset))
        expected_subjects = set(canonical)
        for model in config["foundation_models"]:
            if (
                model in config["mi_only_models"]
                and entry["task_type"] != "mi"
            ):
                continue
            features = load_subjects(
                feature_dir(config, dataset, model), flatten=True
            )
            if set(features) != expected_subjects:
                failures.append(f"{model}/{dataset}: subject IDs differ")
                continue
            for subject_id in sorted(canonical):
                trial_path = (
                    Path(config["trials_root"])
                    / dataset
                    / model
                    / f"{subject_id}.pkl"
                )
                feature_path = feature_dir(
                    config, dataset, model
                ) / f"{subject_id}.pkl"
                trials = filtered_subject(
                    canonical[subject_id], entry["labels"]
                )
                values = filtered_subject(
                    features[subject_id], entry["labels"]
                )
                if trials.trial_ids != values.trial_ids:
                    failures.append(
                        f"{model}/{dataset}/{subject_id}: trial IDs differ"
                    )
                if not np.array_equal(trials.labels, values.labels):
                    failures.append(
                        f"{model}/{dataset}/{subject_id}: labels differ"
                    )
                if portable:
                    feature_obj = load_pickle(feature_path)
                    if "trial_id" not in feature_obj:
                        failures.append(f"feature lacks trial_id: {feature_path}")
                    if contains_absolute_path(feature_obj):
                        failures.append(
                            f"absolute path in feature: {feature_path}"
                        )
                    trial_obj = load_pickle(trial_path)
                    if contains_absolute_path(trial_obj):
                        failures.append(
                            f"absolute path in trial: {trial_path}"
                        )
                checked += 1
    return checked, failures


def main() -> None:
    args = parse_args()
    package_root = args.package_root.resolve()
    config_dir = args.config_dir or package_root / "configs"
    portable = args.portable or not (config_dir / "paths.local.json").exists()
    config = load_configs(config_dir)
    failures = []
    failures.extend(validate_configs(config_dir, portable))
    failures.extend(
        validate_foundation_artifacts(package_root, args.raw_package)
    )
    failures.extend(validate_results(package_root))
    if args.reference_root:
        failures.extend(
            validate_reference_results(package_root, args.reference_root.resolve())
        )
    if portable:
        failures.extend(validate_no_forbidden_dirs(package_root))
    if args.raw_package:
        failures.extend(validate_raw_package(config, package_root))
        checked = 0
        pickle_failures = []
    else:
        checked, pickle_failures = validate_pickles(
            config, package_root, portable
        )
    failures.extend(pickle_failures)
    print(f"checked_subject_model_pairs={checked}")
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        raise SystemExit(f"{len(failures)} validation failures")
    print("all configured inputs are aligned")
    if portable:
        print("portable package checks passed")


if __name__ == "__main__":
    main()
