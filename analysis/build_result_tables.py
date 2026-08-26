#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path

from result_utils import read_csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results"
README_PATH = PROJECT_ROOT / "README.md"
README_START = "<!-- RESULTS_TABLES_START -->"
README_END = "<!-- RESULTS_TABLES_END -->"
SIGNIFICANCE_ROOT = RESULT_ROOT / "statistical_significance"

DATASETS = ["dreyer", "wbcic_3c", "wbcic_2c", "atten_nback", "atten_dsr", "atten_word"]
DISPLAY_DATASETS = {
    "dreyer": "Dreyer-2C",
    "wbcic_3c": "WBCIC-3C",
    "wbcic_2c": "WBCIC-2C",
    "atten_nback": "N-back-2C",
    "atten_dsr": "DSR-2C",
    "atten_word": "WG-2C",
}
DISPLAY_MODELS = {
    "singlem": "SingLEM (primary)",
    "singlem_all_71": "SingLEM (all 71 datasets)",
    "singlem_wo_feature": "SingLEM (w/o feature emb.)",
    "bendr": "BENDR",
    "biot": "BIOT",
    "labram": "LaBraM",
    "cbramod": "CBraMod",
    "codebrain": "CodeBrain",
    "csbrain": "CSBrain",
    "luna_large": "LUNA",
    "mirepnet": "MIRepNet",
    "csp": "CSP",
    "welch_psd": "Welch PSD",
    "eegnet": "EEGNet",
    "eegconformer": "EEGConformer",
    "ifnetv2": "IFNetV2",
}
METRICS = ["accuracy", "f1_macro", "kappa"]
STRICT_SVM_MODELS = [
    "bendr", "biot", "labram", "cbramod", "codebrain", "csbrain",
    "luna_large", "mirepnet", "csp", "welch_psd", "singlem",
]
STRICT_SVM_WITH_ABLATIONS = STRICT_SVM_MODELS + [
    "singlem_all_71",
    "singlem_wo_feature",
]
MLP_NEURAL_MODELS = [
    "singlem", "bendr", "biot", "labram", "cbramod", "codebrain",
    "csbrain", "luna_large", "mirepnet", "csp", "welch_psd",
    "eegnet", "eegconformer", "ifnetv2",
]


def read_summary(path: Path) -> dict:
    row = read_csv(path)[0]
    return {
        key: int(value) if key == "n_subjects" else float(value)
        for key, value in row.items()
        if value != ""
    }


def revised_summary(protocol: str, family: str, model: str, dataset: str) -> Path:
    return RESULT_ROOT / protocol / family / model / dataset / "summary.csv"


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for stale-analysis detection."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_significance() -> set[tuple[str, str, str]]:
    """Load verified Holm-significant SingLEM-higher comparisons."""
    inventory_path = SIGNIFICANCE_ROOT / "input_inventory.csv"
    comparisons_path = SIGNIFICANCE_ROOT / "paired_comparisons.csv"
    if not inventory_path.is_file() or not comparisons_path.is_file():
        raise FileNotFoundError(
            "statistical results are missing; run "
            "python analysis/statistical_significance.py first"
        )
    for row in read_csv(inventory_path):
        source = RESULT_ROOT / row["input_file"]
        if not source.is_file() or sha256(source) != row["sha256"]:
            raise RuntimeError(
                f"statistical results are stale for {source}; rerun "
                "python analysis/statistical_significance.py"
            )
    comparisons = read_csv(comparisons_path)
    if len(comparisons) != 207:
        raise RuntimeError(
            f"expected 207 statistical comparisons, found {len(comparisons)}"
        )
    return {
        (row["evaluation_setting"], row["dataset"], row["baseline_model"])
        for row in comparisons
        if row["significant_holm_0_05"] == "True"
        and row["direction"] == "singlem_higher"
    }


def collect_gpu_svm() -> list[dict]:
    """Collect strict LOSO SVM summaries from the public GPU/cuML result tree."""
    rows = []
    for model in STRICT_SVM_MODELS:
        for dataset in DATASETS:
            if model == "mirepnet" and dataset.startswith("atten_"):
                continue
            group = "classical" if model in {"csp", "welch_psd"} else "foundation"
            path = revised_summary("strict", f"svm/{group}", model, dataset)
            if path.exists():
                rows.append({"model": model, "dataset": dataset, **read_summary(path)})
    ablations = {
        "singlem_all_71": "downstream_included",
        "singlem_wo_feature": "no_feature_embedding",
    }
    for model, variant in ablations.items():
        for dataset in DATASETS:
            path = (
                RESULT_ROOT / "ablation" / "gpu_svm" / "singlem"
                / variant / dataset / "summary.csv"
            )
            if path.exists():
                rows.append({"model": model, "dataset": dataset, **read_summary(path)})
    return rows


def collect_mlp_neural(protocol: str) -> list[dict]:
    rows = []
    for model in MLP_NEURAL_MODELS:
        for dataset in DATASETS:
            if model == "mirepnet" and dataset.startswith("atten_"):
                continue
            if model in {"eegnet", "eegconformer", "ifnetv2"}:
                family = "neural"
            elif model in {"csp", "welch_psd"}:
                family = "mlp/classical"
            else:
                family = "mlp/foundation"
            path = revised_summary(protocol, family, model, dataset)
            if path.exists():
                rows.append({"model": model, "dataset": dataset, **read_summary(path)})
    return rows


def result_table(
    rows: list[dict],
    models: list[str],
    datasets: list[str],
    evaluation_setting: str,
    significance: set[tuple[str, str, str]],
    rank_models: list[str] | None = None,
) -> str:
    lookup = {(row["model"], row["dataset"]): row for row in rows}
    rank_models = rank_models or models
    best = {}
    for dataset in datasets:
        available = [
            row
            for row in rows
            if row["dataset"] == dataset and row["model"] in rank_models
        ]
        for metric in METRICS:
            if available:
                best[(dataset, metric)] = max(row[f"{metric}_mean"] for row in available)
    lines = [
        "| Model | " + " | ".join(DISPLAY_DATASETS[name] for name in datasets) + " |",
        "|---|" + "---|" * len(datasets),
    ]
    for model in models:
        cells = []
        for dataset in datasets:
            row = lookup.get((model, dataset))
            if row is None:
                cells.append("--")
                continue
            values = []
            for metric, label in [("accuracy", "Acc"), ("f1_macro", "F1"), ("kappa", "κ")]:
                digits = 3 if metric == "kappa" else 2
                text = f"{label} {row[f'{metric}_mean']:.{digits}f}±{row[f'{metric}_std']:.{digits}f}"
                if (
                    metric == "accuracy"
                    and (evaluation_setting, dataset, model) in significance
                ):
                    text += "*"
                if (
                    model in rank_models
                    and abs(row[f"{metric}_mean"] - best[(dataset, metric)]) < 1e-12
                ):
                    text = f"**{text}**"
                values.append(text)
            cells.append(" / ".join(values))
        lines.append(f"| {DISPLAY_MODELS[model]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def update_readme(sections: list[tuple[str, str]]) -> None:
    generated = "\n\n".join(f"### {title}\n\n{table}" for title, table in sections)
    readme = README_PATH.read_text(encoding="utf-8")
    if README_START not in readme or README_END not in readme:
        raise RuntimeError("README result-table markers are missing")
    before, rest = readme.split(README_START, 1)
    _, after = rest.split(README_END, 1)
    README_PATH.write_text(
        f"{before}{README_START}\n\n{generated}\n\n{README_END}{after}",
        encoding="utf-8",
    )


def main() -> None:
    significance = load_significance()
    gpu_svm = collect_gpu_svm()
    strict_mlp = collect_mlp_neural("strict")
    adapted = collect_mlp_neural("adapted_30")
    cognitive_models = [model for model in STRICT_SVM_MODELS if model != "mirepnet"]
    cognitive_with_ablations = cognitive_models + [
        "singlem_all_71",
        "singlem_wo_feature",
    ]
    update_readme([
        (
            "Strict LOSO GPU/cuML SVM Results on MI Tasks",
            result_table(
                gpu_svm,
                STRICT_SVM_WITH_ABLATIONS,
                DATASETS[:3],
                "strict_svm",
                significance,
                rank_models=STRICT_SVM_MODELS,
            ),
        ),
        (
            "Strict LOSO GPU/cuML SVM Results on Cognitive Tasks",
            result_table(
                gpu_svm,
                cognitive_with_ablations,
                DATASETS[3:],
                "strict_svm",
                significance,
                rank_models=cognitive_models,
            ),
        ),
        (
            "Strict LOSO MLP and Neural Results",
            result_table(
                strict_mlp,
                MLP_NEURAL_MODELS,
                DATASETS,
                "strict_mlp_neural",
                significance,
            ),
        ),
        (
            "30% Subject-Adapted MLP and Neural Results",
            result_table(
                adapted,
                MLP_NEURAL_MODELS,
                DATASETS,
                "subject_adapted",
                significance,
            ),
        ),
    ])
    print(
        f"gpu_svm={len(gpu_svm)} strict_mlp_neural={len(strict_mlp)} "
        f"adapted_mlp_neural={len(adapted)}"
    )


if __name__ == "__main__":
    main()
