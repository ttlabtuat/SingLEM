#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENTS = [
    "strict_svm_foundation",
    "strict_svm_classical",
    "strict_svm_singlem_ablation",
    "strict_svm_single_channel",
    "strict_mlp",
    "strict_mlp_classical",
    "strict_neural",
    "adapted_mlp_foundation",
    "adapted_mlp_classical",
    "adapted_neural",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--config_dir", type=Path, default=PROJECT_ROOT / "configs"
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--cpu_threads", type=int, default=8)
    parser.add_argument("--max_subjects", type=int)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    from shared.data import load_configs
    from shared.protocols import dispatch
    from analysis.result_utils import read_csv, write_summary

    if (
        args.experiment == "adapted_mlp_foundation"
        or args.experiment == "adapted_mlp_classical"
        or args.experiment == "adapted_neural"
    ) and not args.manifest:
        raise SystemExit("--manifest is required for adaptation experiments")
    config = load_configs(args.config_dir)
    if args.dataset not in config["datasets"]:
        raise SystemExit(f"unknown dataset: {args.dataset}")
    if args.experiment == "strict_svm_singlem_ablation":
        allowed = config["singlem_ablation_variants"]
    elif args.experiment == "strict_svm_single_channel":
        allowed = ["singlem"]
    elif args.experiment.endswith("_classical"):
        allowed = config["classical_methods"]
    elif (
        args.experiment.endswith("_neural")
        or args.experiment == "strict_neural"
    ):
        allowed = config["neural_models"]
    else:
        allowed = config["foundation_models"]
    if args.model not in allowed:
        raise SystemExit(
            f"{args.model} is not valid for {args.experiment}"
        )
    if (
        args.model in config["mi_only_models"]
        and args.dataset not in [
            name
            for name, entry in config["datasets"].items()
            if entry["task_type"] == "mi"
        ]
    ):
        raise SystemExit(f"{args.model} is MI-only")
    if "svm" not in args.experiment:
        import torch

        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(1)
    dispatch(args, config)
    if not args.dry_run and args.experiment != "strict_svm_single_channel":
        method = None
        if args.experiment in {"adapted_mlp_foundation", "adapted_mlp_classical"}:
            method = "pooled_refit"
        elif args.experiment == "adapted_neural":
            method = "existing_head_adaptation"
        write_summary(
            args.output_dir / "summary.csv",
            read_csv(args.output_dir / "per_subject_metrics.csv"),
            method=method,
        )


if __name__ == "__main__":
    main()
