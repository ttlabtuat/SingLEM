#!/usr/bin/env python3
"""Reproduce paired subject-accuracy tests for manuscript Tables II--V.

Every comparison uses one accuracy per held-out subject, explicitly aligned by
subject ID. Two-sided paired Wilcoxon tests use the validated tie/zero policy;
Holm correction is independent within each evaluation-setting/dataset family.
Final outputs are replaced only after every validation succeeds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import PermutationMethod, rankdata, wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PROJECT_ROOT / "results"
DEFAULT_OUTPUT = DEFAULT_RESULTS / "statistical_significance"
ALPHA = 0.05
ROUND_DECIMALS = 12
MEAN_TOLERANCE_PERCENT = 0.0050001
STRICT_REGRESSION_SHA256 = "eb2c179bfc90b96fa685359c6d91ecb743234ecf7a9e1fe3f368bad7c80b8107"

DATASETS = {
    "dreyer": ("motor_imagery", 21, "Dreyer-MI-2C"),
    "wbcic_3c": ("motor_imagery", 11, "WBCIC-MI-3C"),
    "wbcic_2c": ("motor_imagery", 51, "WBCIC-MI-2C"),
    "atten_nback": ("cognitive", 26, "N-back-2C"),
    "atten_dsr": ("cognitive", 26, "DSR-2C"),
    "atten_word": ("cognitive", 26, "WG-2C"),
}
BASELINES = (
    ("BENDR", "foundation", "bendr", "all"),
    ("BIOT", "foundation", "biot", "all"),
    ("LaBraM", "foundation", "labram", "all"),
    ("CBraMod", "foundation", "cbramod", "all"),
    ("CodeBrain", "foundation", "codebrain", "all"),
    ("CSBrain", "foundation", "csbrain", "all"),
    ("LUNA", "foundation", "luna_large", "all"),
    ("MIRepNet", "foundation", "mirepnet", "motor_imagery"),
    ("CSP", "classical", "csp", "all"),
    ("Welch PSD", "classical", "welch_psd", "all"),
)
NEURAL = (
    ("EEGNet", "neural", "eegnet", "all"),
    ("EEGConformer", "neural", "eegconformer", "all"),
    ("IFNetV2", "neural", "ifnetv2", "all"),
)
SETTINGS = {
    "strict_svm": ("Tables II-III", False, False),
    "strict_mlp_neural": ("Table IV", False, True),
    "subject_adapted": ("Table V", True, True),
}

# Submitted-table mean accuracies (%) in DATASETS order. These validate inputs;
# they never enter the tests or corrections.
PUBLISHED = {
    "strict_svm": {
        "SingLEM": [74.58, 68.14, 79.68, 84.15, 85.68, 70.26],
        "BENDR": [52.23, 35.50, 51.09, 62.75, 59.62, 50.00],
        "BIOT": [52.83, 35.95, 50.83, 60.47, 61.70, 57.18],
        "LaBraM": [55.00, 39.89, 57.24, 62.25, 66.72, 61.54],
        "CBraMod": [71.16, 60.20, 78.14, 78.03, 79.38, 69.94],
        "CodeBrain": [65.09, 51.53, 74.17, 80.13, 82.00, 66.35],
        "CSBrain": [68.96, 63.05, 79.29, 78.13, 76.44, 68.01],
        "LUNA": [59.58, 45.35, 62.27, 69.37, 70.73, 58.59],
        "MIRepNet": [72.68, 48.28, 61.26, None, None, None],
        "CSP": [62.56, 34.77, 49.71, 58.37, 56.41, 53.53],
        "Welch PSD": [55.98, 37.65, 54.34, 64.78, 62.29, 61.47],
    },
    "strict_mlp_neural": {
        "SingLEM": [73.63, 67.44, 79.83, 82.80, 83.87, 68.78],
        "BENDR": [49.49, 35.43, 50.82, 56.87, 54.43, 51.15],
        "BIOT": [50.18, 35.62, 49.82, 58.12, 59.56, 53.53],
        "LaBraM": [54.67, 37.56, 51.59, 59.69, 60.63, 61.09],
        "CBraMod": [70.57, 60.81, 78.49, 75.85, 77.78, 70.45],
        "CodeBrain": [65.74, 53.68, 74.25, 78.31, 81.89, 67.18],
        "CSBrain": [68.39, 64.20, 79.36, 77.56, 75.80, 65.38],
        "LUNA": [57.23, 41.80, 59.56, 63.28, 64.64, 59.68],
        "MIRepNet": [73.04, 48.71, 61.73, None, None, None],
        "CSP": [64.49, 35.01, 50.12, 57.94, 57.64, 54.74],
        "Welch PSD": [56.19, 38.28, 54.78, 66.38, 64.74, 61.22],
        "EEGNet": [73.10, 60.53, 73.01, 73.61, 78.04, 68.33],
        "EEGConformer": [70.18, 51.99, 71.91, 77.71, 77.62, 60.38],
        "IFNetV2": [68.45, 54.38, 73.94, 72.04, 73.66, 67.56],
    },
    "subject_adapted": {
        "SingLEM": [73.94, 70.59, 80.56, 84.41, 83.69, 69.87],
        "BENDR": [50.09, 35.43, 50.25, 56.73, 54.15, 50.92],
        "BIOT": [50.34, 36.99, 49.72, 60.48, 59.62, 54.76],
        "LaBraM": [54.63, 39.02, 52.34, 62.10, 60.54, 60.62],
        "CBraMod": [71.17, 65.61, 79.45, 81.07, 80.54, 72.62],
        "CodeBrain": [66.41, 59.51, 76.44, 83.91, 82.31, 69.87],
        "CSBrain": [70.79, 67.43, 79.86, 79.25, 76.69, 69.41],
        "LUNA": [57.95, 43.53, 60.66, 64.63, 64.69, 59.34],
        "MIRepNet": [73.51, 48.29, 61.64, None, None, None],
        "CSP": [65.05, 36.16, 50.56, 60.37, 58.69, 57.05],
        "Welch PSD": [59.31, 42.53, 55.39, 77.02, 69.77, 71.43],
        "EEGNet": [72.66, 62.02, 76.24, 74.34, 77.77, 68.86],
        "EEGConformer": [70.54, 52.41, 71.55, 78.24, 77.62, 60.35],
        "IFNetV2": [70.92, 60.59, 77.54, 79.00, 77.69, 74.18],
    },
}


def parse_args() -> argparse.Namespace:
    """Parse input/output roots and optional manuscript regression checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--verify-manuscript",
        action="store_true",
        help=(
            "Require input means and the strict-SVM p-value digest to match "
            "the revised manuscript. Omit this option for independent reruns."
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_std(values: np.ndarray) -> float:
    """Return sample standard deviation, or zero for one value."""
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def tie_diagnostics(differences: np.ndarray) -> dict[str, int]:
    """Describe zeros and absolute-difference ties used for signed ranks."""
    nonzero_abs = np.abs(differences[differences != 0.0])
    counts = Counter(nonzero_abs.tolist())
    tied = [count for count in counts.values() if count > 1]
    return {
        "n_zero": int(np.sum(differences == 0.0)),
        "n_positive": int(np.sum(differences > 0.0)),
        "n_negative": int(np.sum(differences < 0.0)),
        "n_nonzero": int(len(nonzero_abs)),
        "n_unique_abs_nonzero": len(counts),
        "n_tied_abs_groups": len(tied),
        "n_nonzero_in_tied_abs_groups": sum(tied),
        "largest_abs_tie_group": max(tied, default=1 if len(nonzero_abs) else 0),
    }


def wilcoxon_method(diagnostics: dict[str, int]):
    """Preserve the validated exact/permutation/asymptotic rule."""
    n = diagnostics["n_zero"] + diagnostics["n_nonzero"]
    ties_or_zeros = diagnostics["n_zero"] > 0 or diagnostics["n_tied_abs_groups"] > 0
    if not ties_or_zeros and n <= 50:
        return "exact", "exact"
    if ties_or_zeros and n <= 13:
        return PermutationMethod(n_resamples=np.inf), "exhaustive_permutation"
    return "asymptotic", "asymptotic"


def compare(primary: np.ndarray, baseline: np.ndarray) -> tuple[dict, np.ndarray]:
    """Calculate paired descriptives, Wilcoxon results, and effect sizes."""
    raw_differences = primary - baseline
    differences = np.round(raw_differences, ROUND_DECIMALS)
    diagnostics = tie_diagnostics(differences)
    nonzero = differences[differences != 0.0]
    positive_rank_sum = negative_rank_sum = rank_biserial = 0.0
    if len(nonzero):
        ranks = rankdata(np.abs(nonzero), method="average")
        positive_rank_sum = float(np.sum(ranks[nonzero > 0.0]))
        negative_rank_sum = float(np.sum(ranks[nonzero < 0.0]))
        rank_total = positive_rank_sum + negative_rank_sum
        rank_biserial = (positive_rank_sum - negative_rank_sum) / rank_total
    if diagnostics["n_nonzero"] == 0:
        statistic, p_value, z_statistic, method_name = 0.0, 1.0, 0.0, "all_zero_no_test"
    else:
        method, method_name = wilcoxon_method(diagnostics)
        result = wilcoxon(differences, zero_method="wilcox", correction=False,
                          alternative="two-sided", method=method)
        statistic, p_value = float(result.statistic), float(result.pvalue)
        z_value = getattr(result, "zstatistic", None)
        z_statistic = float(z_value) if z_value is not None else math.nan
    difference_std = sample_std(differences)
    directional_n = diagnostics["n_positive"] + diagnostics["n_negative"]
    return {
        "n_subjects": len(differences),
        "primary_mean_accuracy": float(np.mean(primary)),
        "primary_sd_accuracy": sample_std(primary),
        "baseline_mean_accuracy": float(np.mean(baseline)),
        "baseline_sd_accuracy": sample_std(baseline),
        "mean_difference": float(np.mean(differences)),
        "sd_difference": difference_std,
        "median_difference": float(np.median(differences)),
        "q1_difference": float(np.quantile(differences, 0.25)),
        "q3_difference": float(np.quantile(differences, 0.75)),
        "min_difference": float(np.min(differences)),
        "max_difference": float(np.max(differences)),
        "n_exact_zero_before_rounding": int(np.sum(raw_differences == 0.0)),
        "n_created_zero_by_rounding": int(np.sum((raw_differences != 0.0) & (differences == 0.0))),
        **diagnostics,
        "positive_rank_sum": positive_rank_sum,
        "negative_rank_sum": negative_rank_sum,
        "wilcoxon_statistic": statistic,
        "z_statistic": z_statistic,
        "p_value_raw": p_value,
        "p_value_method": method_name,
        "rank_biserial_effect": rank_biserial,
        "cohen_dz": float(np.mean(differences) / difference_std) if difference_std else 0.0,
        "common_language_singlem_win_rate_excluding_zeros": diagnostics["n_positive"] / directional_n if directional_n else 0.5,
    }, differences


def apply_holm(rows: list[dict]) -> None:
    """Apply Holm step-down adjustment to one complete family."""
    ordered = sorted(range(len(rows)), key=lambda index: rows[index]["p_value_raw"])
    running_max = 0.0
    for rank, index in enumerate(ordered, start=1):
        multiplier = len(rows) - rank + 1
        running_max = max(running_max, min(1.0, multiplier * rows[index]["p_value_raw"]))
        rows[index].update(holm_rank=rank, holm_multiplier=multiplier,
                           p_value_holm=running_max)
    for row in rows:
        row["holm_family_size"] = len(rows)
        row["significant_holm_0_05"] = row["p_value_holm"] < ALPHA
        if not row["significant_holm_0_05"]:
            row["direction"] = "not_significant"
        elif row["positive_rank_sum"] > row["negative_rank_sum"]:
            row["direction"] = "singlem_higher"
        elif row["negative_rank_sum"] > row["positive_rank_sum"]:
            row["direction"] = "baseline_higher"
        else:
            raise ValueError("significant comparison has equal signed-rank sums")


def applicable_baselines(setting: str, task: str) -> list[tuple[str, str, str, str]]:
    """Return the ordered, applicable baseline family."""
    candidates = BASELINES + (NEURAL if SETTINGS[setting][2] else ())
    return [item for item in candidates if item[3] in {"all", task}]


def source_spec(setting: str, category: str, model: str) -> tuple[Path, str | None]:
    """Resolve a result folder and optional adapted-method selector."""
    if setting == "strict_svm":
        return Path("strict/svm") / category / model, None
    if setting == "strict_mlp_neural":
        root = Path("strict/neural") if category == "neural" else Path("strict/mlp") / category
        return root / model, None
    if category == "neural":
        return Path("adapted_30/neural") / model, "existing_head_adaptation"
    return Path("adapted_30/mlp") / category / model, "pooled_refit"


def trial_ids(row: dict, field: str, path: Path) -> tuple[str, ...]:
    """Parse and validate an adapted calibration/test trial-ID list."""
    try:
        values = json.loads(row[field])
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: invalid {field} for {row.get('subject')}") from error
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"{path}: invalid {field} for {row.get('subject')}")
    if len(values) != len(set(values)):
        raise ValueError(f"{path}: duplicate IDs in {field} for {row.get('subject')}")
    return tuple(values)


def read_rows(path: Path, expected_count: int, method: str | None) -> tuple[dict[str, dict], int]:
    """Select exactly one finite accuracy row per expected subject."""
    if not path.is_file():
        raise FileNotFoundError(f"missing subject-level result: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"subject", "accuracy"}
        if method:
            required.update({"seed", "method", "valid", "calibration_trial_ids", "test_trial_ids"})
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: missing columns {sorted(required)}")
        source_rows = list(reader)
    selected = {}
    for line_number, row in enumerate(source_rows, 2):
        if method and not (row["method"] == method and row["seed"] == "2023" and row["valid"] == "True"):
            continue
        subject = row["subject"]
        if not subject or subject != subject.strip():
            raise ValueError(f"{path}:{line_number}: invalid subject ID {subject!r}")
        if subject in selected:
            raise ValueError(f"{path}:{line_number}: duplicate selected subject {subject!r}")
        try:
            accuracy = float(row["accuracy"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}:{line_number}: invalid accuracy") from error
        if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError(f"{path}:{line_number}: accuracy outside [0, 1]")
        if method:
            calibration = trial_ids(row, "calibration_trial_ids", path)
            test = trial_ids(row, "test_trial_ids", path)
            if set(calibration) & set(test):
                raise ValueError(f"{path}:{line_number}: calibration/test overlap")
        selected[subject] = {**row, "accuracy_value": accuracy}
    if len(selected) != expected_count:
        raise ValueError(f"{path}: expected {expected_count} selected subjects, found {len(selected)}")
    return selected, len(source_rows)


def mean_error(
    setting: str,
    dataset: str,
    model: str,
    mean: float,
    verify_manuscript: bool,
) -> float:
    """Measure manuscript-mean error and optionally enforce its tolerance."""
    expected = PUBLISHED[setting][model][list(DATASETS).index(dataset)]
    if expected is None:
        raise ValueError(f"no published mean for {setting}/{dataset}/{model}")
    error = abs(100 * mean - expected)
    if verify_manuscript and error > MEAN_TOLERANCE_PERCENT:
        raise ValueError(f"{setting}/{dataset}/{model}: calculated {100*mean:.8f}% != published {expected:.2f}%")
    return error


def inventory_row(root: Path, path: Path, setting: str, dataset: str, role: str,
                  model: str, method: str | None, rows: dict, source_count: int) -> dict:
    """Return one input-provenance row after validation."""
    return {
        "evaluation_setting": setting, "manuscript_table": SETTINGS[setting][0],
        "dataset": dataset, "role": role, "model": model,
        "selected_method": method or "one_row_per_subject",
        "input_file": str(path.relative_to(root)), "sha256": sha256(path),
        "source_rows": source_count, "selected_rows": len(rows),
        "unique_subjects": len(rows), "subject_ids": "|".join(sorted(rows)),
    }


def analyze_setting(
    setting: str,
    root: Path,
    verify_manuscript: bool,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Validate and analyze all six datasets for one evaluation setting."""
    inventory, aligned, comparisons, summaries = [], [], [], []
    primary_fragment, primary_method = source_spec(setting, "foundation", "singlem")
    for dataset, (task, expected_count, published_name) in DATASETS.items():
        primary_path = root / primary_fragment / dataset / "per_subject_metrics.csv"
        primary, count = read_rows(primary_path, expected_count, primary_method)
        primary_mean = float(np.mean([row["accuracy_value"] for row in primary.values()]))
        primary_error = mean_error(
            setting, dataset, "SingLEM", primary_mean, verify_manuscript
        )
        inventory.append(inventory_row(root, primary_path, setting, dataset, "primary",
                                       "SingLEM", primary_method, primary, count))
        family = []
        for display, category, model, _scope in applicable_baselines(setting, task):
            fragment, method = source_spec(setting, category, model)
            path = root / fragment / dataset / "per_subject_metrics.csv"
            baseline, count = read_rows(path, expected_count, method)
            if set(primary) != set(baseline):
                missing, extra = sorted(set(primary)-set(baseline)), sorted(set(baseline)-set(primary))
                raise ValueError(f"{setting}/{dataset}/{display}: subjects mismatch; missing={missing}, extra={extra}")
            inventory.append(inventory_row(root, path, setting, dataset, "baseline",
                                           display, method, baseline, count))
            subjects = sorted(primary)
            if SETTINGS[setting][1]:
                for subject in subjects:
                    for field in ("calibration_trial_ids", "test_trial_ids"):
                        if trial_ids(primary[subject], field, primary_path) != trial_ids(baseline[subject], field, path):
                            raise ValueError(f"{setting}/{dataset}/{display}/{subject}: different {field}")
            primary_array = np.asarray([primary[subject]["accuracy_value"] for subject in subjects])
            baseline_array = np.asarray([baseline[subject]["accuracy_value"] for subject in subjects])
            baseline_error = mean_error(
                setting,
                dataset,
                display,
                float(np.mean(baseline_array)),
                verify_manuscript,
            )
            statistics, differences = compare(primary_array, baseline_array)
            family.append({
                "evaluation_setting": setting, "manuscript_table": SETTINGS[setting][0],
                "dataset": dataset, "published_dataset": published_name, "task": task,
                "primary": "SingLEM", "baseline": display, "baseline_family": category,
                "baseline_model": model, "selected_method": method or "one_row_per_subject",
                "primary_published_mean_abs_error_percent": primary_error,
                "baseline_published_mean_abs_error_percent": baseline_error, **statistics,
            })
            for subject, p_value, b_value, difference in zip(subjects, primary_array, baseline_array, differences, strict=True):
                aligned_row = {
                    "evaluation_setting": setting, "manuscript_table": SETTINGS[setting][0],
                    "dataset": dataset, "subject": subject, "primary": "SingLEM",
                    "baseline": display, "primary_accuracy": float(p_value),
                    "baseline_accuracy": float(b_value),
                    "difference_primary_minus_baseline": float(difference),
                    "calibration_trial_ids_sha256": "", "test_trial_ids_sha256": "",
                }
                if SETTINGS[setting][1]:
                    for field in ("calibration_trial_ids", "test_trial_ids"):
                        payload = "\n".join(trial_ids(primary[subject], field, primary_path)).encode()
                        aligned_row[f"{field}_sha256"] = hashlib.sha256(payload).hexdigest()
                aligned.append(aligned_row)
        expected_family = (10 if task == "motor_imagery" else 9) if setting == "strict_svm" else (13 if task == "motor_imagery" else 12)
        if len(family) != expected_family:
            raise ValueError(f"{setting}/{dataset}: expected {expected_family} comparisons, found {len(family)}")
        apply_holm(family)
        comparisons.extend(family)
        summaries.append({
            "evaluation_setting": setting, "manuscript_table": SETTINGS[setting][0],
            "dataset": dataset, "published_dataset": published_name, "task": task,
            "n_subjects": expected_count, "holm_family_size": expected_family,
            "n_significant_holm_0_05": sum(row["significant_holm_0_05"] for row in family),
            "n_singlem_higher": sum(row["direction"] == "singlem_higher" for row in family),
            "n_baseline_higher": sum(row["direction"] == "baseline_higher" for row in family),
        })
    return inventory, aligned, comparisons, summaries


def strict_digest(rows: list[dict]) -> str:
    """Hash Tables II--III raw/adjusted p-values in validated order."""
    payload = [(row["dataset"], row["baseline"], format(row["p_value_raw"], ".17g"),
                format(row["p_value_holm"], ".17g"))
               for row in rows if row["evaluation_setting"] == "strict_svm"]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def verify_holm(rows: list[dict]) -> None:
    """Independently recalculate each family adjustment and decision."""
    families = {}
    for row in rows:
        families.setdefault((row["evaluation_setting"], row["dataset"]), []).append(row)
    for key, family in families.items():
        running = 0.0
        for rank, row in enumerate(sorted(family, key=lambda item: item["p_value_raw"]), 1):
            running = max(running, min(1.0, (len(family)-rank+1)*row["p_value_raw"]))
            if not math.isclose(running, row["p_value_holm"], rel_tol=0, abs_tol=1e-15):
                raise ValueError(f"independent Holm check failed: {key}/{row['baseline']}")
            if (running < ALPHA) != row["significant_holm_0_05"]:
                raise ValueError(f"significance check failed: {key}/{row['baseline']}")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    """Write deterministic UTF-8 CSV."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields or list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict) -> None:
    """Write stable, readable JSON."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def markdown_report(rows: list[dict]) -> str:
    """Create raw and Holm-adjusted p-value matrices for every setting."""
    lines = ["# Paired Wilcoxon–Holm results", "",
             "`*` denotes Holm-adjusted p < 0.05. Differences are SingLEM minus baseline.", ""]
    for setting, (table, _adapted, _neural) in SETTINGS.items():
        subset = [row for row in rows if row["evaluation_setting"] == setting]
        baselines = [item[0] for item in applicable_baselines(setting, "motor_imagery")]
        for field, title in (("p_value_raw", "Raw p-values"), ("p_value_holm", "Holm-adjusted p-values")):
            lines += [f"## {table}: {title}", "", "| Baseline | " + " | ".join(DATASETS) + " |",
                      "|---|" + "---:|" * len(DATASETS)]
            for baseline in baselines:
                cells = []
                for dataset in DATASETS:
                    match = next((row for row in subset if row["dataset"] == dataset and row["baseline"] == baseline), None)
                    if not match:
                        cells.append("—")
                    else:
                        marker = "*" if field == "p_value_holm" and match["significant_holm_0_05"] else ""
                        cells.append(f"{match[field]:.8g}{marker}")
                lines.append(f"| {baseline} | " + " | ".join(cells) + " |")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def results_readme() -> str:
    """Describe the complete public statistical result family."""
    return """# Statistical Significance Results

These files compare SingLEM (primary) with every applicable non-SingLEM
baseline in manuscript Tables II--V. Each observation is one held-out-subject
accuracy, paired by exact subject ID. Tests are two-sided paired Wilcoxon
signed-rank tests. Holm correction is applied independently within each
evaluation-setting and dataset family, and significance is defined as
Holm-adjusted `p < 0.05`.

The manuscript and root README use `*` as a compact marker for a baseline whose
accuracy is significantly lower than SingLEM. Full raw and adjusted p-values,
effect sizes, difference summaries, tie diagnostics, and test methods are in
`paired_comparisons.csv`; `p_value_report.md` provides readable p-value tables.
`aligned_subject_accuracies.csv` records every paired value, and
`input_inventory.csv` records source paths, hashes, and selected subject IDs.
`dataset_summary.csv`, `analysis_config.json`, and `validation_report.json`
provide family-level summaries, analysis settings, and integrity checks.

Regenerate and verify the committed manuscript results from the repository root:

```bash
python analysis/statistical_significance.py --verify-manuscript
python analysis/build_result_tables.py
```

For independently reproduced result files, omit `--verify-manuscript`; the same
statistical procedure will run without requiring exact equality to the committed
manuscript means.
"""


def portable_path(path: Path) -> str:
    """Represent a path without recording machine-specific absolute prefixes."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return f"external:{path.name}"


def replace_directory(staged: Path, final: Path) -> None:
    """Atomically publish a complete directory with rollback on rename failure."""
    backup = final.with_name(f".{final.name}.backup-{os.getpid()}")
    if backup.exists():
        raise FileExistsError(f"stale backup exists: {backup}")
    if final.exists():
        final.rename(backup)
    try:
        staged.rename(final)
    except BaseException:
        if backup.exists() and not final.exists():
            backup.rename(final)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def main() -> None:
    """Analyze, validate, stage, and publish all reproducible outputs."""
    args = parse_args()
    root, output = args.results_dir.resolve(), args.output_dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"results directory missing: {root}")
    staged = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if staged.exists():
        raise SystemExit(f"stale staging directory exists: {staged}")
    inventory, aligned, comparisons, summaries = [], [], [], []
    for setting in SETTINGS:
        parts = analyze_setting(setting, root, args.verify_manuscript)
        inventory.extend(parts[0]); aligned.extend(parts[1]); comparisons.extend(parts[2]); summaries.extend(parts[3])
    if len(comparisons) != 207:
        raise ValueError(f"expected 207 comparisons, found {len(comparisons)}")
    regression = strict_digest(comparisons)
    if args.verify_manuscript and regression != STRICT_REGRESSION_SHA256:
        raise ValueError(f"Tables II-III p-value regression failed: {regression}")
    verify_holm(comparisons)
    published_means_match = all(
        row["primary_published_mean_abs_error_percent"] <= MEAN_TOLERANCE_PERCENT
        and row["baseline_published_mean_abs_error_percent"] <= MEAN_TOLERANCE_PERCENT
        for row in comparisons
    )
    timestamp = datetime.now(timezone.utc).isoformat()
    command = "python analysis/statistical_significance.py"
    if args.verify_manuscript:
        command += " --verify-manuscript"
    staged.mkdir(parents=True)
    try:
        write_csv(staged / "input_inventory.csv", inventory)
        write_csv(staged / "aligned_subject_accuracies.csv", aligned)
        write_csv(staged / "paired_comparisons.csv", comparisons)
        significant = [row for row in comparisons if row["significant_holm_0_05"]]
        write_csv(staged / "significant_comparisons.csv", significant, list(comparisons[0]))
        write_csv(staged / "dataset_summary.csv", summaries)
        write_json(staged / "analysis_config.json", {
            "analysis": "paired_held_out_subject_accuracy_wilcoxon_holm_tables_ii_v",
            "command": command, "execution_timestamp_utc": timestamp,
            "results_dir": portable_path(root), "output_dir": portable_path(output),
            "manuscript_verification_requested": args.verify_manuscript,
            "manuscript_mean_check_passed": published_means_match,
            "metric": "held-out-subject accuracy", "accuracy_unit": "proportion",
            "observation_unit": "one explicitly subject-ID-aligned held-out-subject accuracy",
            "difference_direction": "SingLEM minus baseline", "alternative": "two-sided",
            "zero_method": "wilcox", "continuity_correction": False,
            "difference_round_decimals": ROUND_DECIMALS,
            "p_value_method_rule": {"no_ties_or_zeros_n_le_50": "exact", "ties_or_zeros_n_le_13": "exhaustive sign permutation", "otherwise": "tie-adjusted asymptotic"},
            "multiple_testing": "Holm separately within evaluation setting and dataset",
            "significance_rule": "Holm-adjusted p-value < 0.05",
            "published_mean_tolerance_percentage_points": MEAN_TOLERANCE_PERCENT,
            "settings": SETTINGS, "datasets": DATASETS,
            "input_sha256": {f"{row['evaluation_setting']}:{row['input_file']}": row["sha256"] for row in inventory},
            "script": "analysis/statistical_significance.py", "script_sha256": sha256(Path(__file__)),
            "strict_svm_regression_sha256": regression,
            "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        })
        (staged / "p_value_report.md").write_text(markdown_report(comparisons), encoding="utf-8")
        write_json(staged / "validation_report.json", {
            "status": "passed", "paired_comparisons": len(comparisons),
            "comparisons_by_setting": {setting: sum(row["evaluation_setting"] == setting for row in comparisons) for setting in SETTINGS},
            "aligned_subject_rows": len(aligned), "input_file_uses": len(inventory),
            "all_subject_ids_unique_and_expected_counts_met": True,
            "all_subject_sets_exactly_aligned": True,
            "all_arrays_built_from_explicitly_sorted_subject_ids": True,
            "observation_unit_is_one_held_out_subject_accuracy": True,
            "manuscript_verification_requested": args.verify_manuscript,
            "all_published_accuracy_means_reproduced": published_means_match,
            "all_adapted_calibration_and_test_trial_ids_identical": True,
            "holm_independently_cross_checked": True,
            "strict_svm_57_comparison_regression_unchanged": regression == STRICT_REGRESSION_SHA256,
            "strict_svm_regression_sha256": regression,
            "family_sizes": {f"{row['evaluation_setting']}:{row['dataset']}": row["holm_family_size"] for row in summaries},
        })
        (staged / "README.md").write_text(results_readme(), encoding="utf-8")
        replace_directory(staged, output)
    except BaseException:
        if staged.exists():
            shutil.rmtree(staged)
        raise
    print(f"comparisons={len(comparisons)} significant={len(significant)} output={output}")


if __name__ == "__main__":
    main()
