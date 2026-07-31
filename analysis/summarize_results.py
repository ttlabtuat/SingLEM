#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from result_utils import aggregate_rows, read_csv, valid_rows, write_csv, write_json


DATASETS = {
    "dreyer",
    "wbcic_3c",
    "wbcic_2c",
    "atten_nback",
    "atten_dsr",
    "atten_word",
}
METRICS = ["accuracy", "f1_macro", "precision_macro", "recall_macro", "kappa"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate public result summaries and portable metadata from "
            "per-subject or per-channel metric CSV files."
        )
    )
    parser.add_argument("--result_root", type=Path, default=Path("results"))
    parser.add_argument(
        "--source_label",
        default="SingLEM_update",
        help="Portable label recorded as the source of synchronized metrics.",
    )
    return parser.parse_args()


def method_filter(parts: tuple[str, ...]) -> str | None:
    """Return the adapted-result method used for manuscript summaries."""
    if len(parts) >= 2 and parts[0] == "adapted_30" and parts[1] == "mlp":
        return "pooled_refit"
    if len(parts) >= 2 and parts[0] == "adapted_30" and parts[1] == "neural":
        return "existing_head_adaptation"
    return None


def metadata_for(rel_dir: Path, summary: dict) -> dict:
    """Build public-safe metadata from the canonical result path."""
    parts = rel_dir.parts
    metadata = {
        "source_run": "final_revised_manuscript",
        "summary": summary,
    }
    if parts[0] == "strict":
        metadata.update(
            {
                "protocol": "strict",
                "backend": "cuml" if parts[1] == "svm" else "pytorch",
                "family": "/".join(parts[1:-2]),
                "model": parts[-2],
                "dataset": parts[-1],
            }
        )
    elif parts[0] == "adapted_30":
        metadata.update(
            {
                "protocol": "adapted_30",
                "backend": "pytorch",
                "family": "/".join(parts[1:-2]),
                "model": parts[-2],
                "dataset": parts[-1],
            }
        )
    elif parts[:3] == ("ablation", "gpu_svm", "singlem"):
        metadata.update(
            {
                "protocol": "strict",
                "backend": "cuml",
                "family": "gpu_svm/singlem_ablation",
                "model": "singlem",
                "singlem_variant": parts[-2],
                "dataset": parts[-1],
            }
        )
    else:
        metadata.update(
            {
                "protocol": parts[0],
                "family": "/".join(parts[1:-2]),
                "model": parts[-2] if len(parts) >= 2 else "",
                "dataset": parts[-1],
            }
        )
    return metadata


def summarize_subject_metrics(result_root: Path, source_label: str) -> int:
    """Summarize every included ``per_subject_metrics.csv`` file."""
    count = 0
    for metrics in sorted(result_root.glob("**/per_subject_metrics.csv")):
        rel_dir = metrics.parent.relative_to(result_root)
        if rel_dir.parts[:2] == ("adapted_30", "svm"):
            continue
        rows = read_csv(metrics)
        method = method_filter(rel_dir.parts)
        selected = valid_rows(rows, method)
        summary = aggregate_rows(selected)
        write_csv(metrics.parent / "summary.csv", [summary])
        metadata = metadata_for(rel_dir, summary)
        metadata["source_metrics"] = (
            f"{source_label}/results/{metrics.relative_to(result_root)}"
        )
        write_json(metrics.parent / "run_metadata.json", metadata)
        count += 1
    return count


def channel_summary_rows(rows: list[dict]) -> list[dict]:
    """Aggregate single-channel metrics over subjects for each channel."""
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if str(row.get("valid", "True")).lower() in {"false", "0"}:
            continue
        grouped.setdefault(row["channel"], []).append(row)
    out = []
    for channel, channel_rows in grouped.items():
        item = {"channel": channel, "n_subjects": len(channel_rows)}
        for metric in METRICS:
            values = [float(row[metric]) for row in channel_rows]
            scale = 1.0 if metric == "kappa" else 100.0
            summary = aggregate_rows(channel_rows, metrics=[metric])
            item[f"{metric}_mean"] = summary[f"{metric}_mean"]
            item[f"{metric}_std"] = summary[f"{metric}_std"]
            if scale == 1.0 and not values:
                item[f"{metric}_mean"] = ""
        out.append(item)
    return out


def summarize_single_channel(result_root: Path, source_label: str) -> int:
    """Summarize SingLEM per-channel strict GPU-SVM metrics."""
    root = (
        result_root
        / "single_channel"
        / "gpu_svm"
        / "singlem"
        / "downstream_excluded"
    )
    if not root.exists():
        return 0
    all_rows = []
    count = 0
    for metrics in sorted(root.glob("*/per_channel_subject_metrics.csv")):
        dataset = metrics.parent.name
        rows = read_csv(metrics)
        summary_rows = channel_summary_rows(rows)
        write_csv(metrics.parent / "channel_summary.csv", summary_rows)
        metadata = {
            "source_run": "final_revised_manuscript",
            "source_metrics": (
                f"{source_label}/results/{metrics.relative_to(result_root)}"
            ),
            "protocol": "strict",
            "backend": "cuml",
            "family": "single_channel/gpu_svm",
            "model": "singlem",
            "singlem_variant": "downstream_excluded",
            "dataset": dataset,
            "n_channel_subject_rows": len(rows),
        }
        write_json(metrics.parent / "run_metadata.json", metadata)
        for row in summary_rows:
            all_rows.append(
                {
                    "protocol": "strict_loso_gpu_svm_single_channel",
                    "family": "single_channel",
                    "model": "singlem",
                    "variant": "downstream_excluded",
                    "dataset": dataset,
                    **row,
                }
            )
        count += 1
    if all_rows:
        write_csv(root / "all_channel_summary.csv", all_rows)
    return count


def main() -> None:
    args = parse_args()
    subject_count = summarize_subject_metrics(
        args.result_root, args.source_label
    )
    channel_count = summarize_single_channel(
        args.result_root, args.source_label
    )
    print(
        f"summarized_subject_results={subject_count} "
        f"summarized_single_channel_datasets={channel_count}"
    )


if __name__ == "__main__":
    main()
