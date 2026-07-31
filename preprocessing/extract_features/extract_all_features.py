#!/usr/bin/env python3
"""Run frozen feature extraction for all selected datasets and foundation models."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(THIS_DIR))

from extract_foundation_features import DATASET_INFO, MODELS, SINGLEM_VARIANTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="all", help="Comma-separated dataset ids or all.")
    parser.add_argument("--models", default="all", help="Comma-separated model ids or all.")
    parser.add_argument(
        "--singlem_variants",
        default="downstream_excluded",
        help="Comma-separated SingLEM variants or all.",
    )
    parser.add_argument("--channel_policy", default="pretrained_matched", choices=["pretrained_matched", "current_compat"])
    parser.add_argument("--input_root", type=Path, default=PROJECT_ROOT / "datasets" / "trials")
    parser.add_argument("--output_root", type=Path, default=PROJECT_ROOT / "datasets" / "features")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_subjects", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--keep_going", action="store_true")
    return parser.parse_args()


def selected(value: str, allowed: list[str] | tuple[str, ...]) -> list[str]:
    allowed = list(allowed)
    if value == "all":
        return allowed
    items = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(items) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown value(s): {unknown}")
    return items


def command_for(args: argparse.Namespace, dataset: str, model: str, variant: str | None) -> list[str]:
    command = [
        sys.executable,
        str(THIS_DIR / "extract_foundation_features.py"),
        "--dataset",
        dataset,
        "--model",
        model,
        "--channel_policy",
        args.channel_policy,
        "--input_root",
        str(args.input_root),
        "--output_root",
        str(args.output_root),
        "--gpu",
        str(args.gpu),
        "--batch_size",
        str(args.batch_size),
    ]
    if model == "singlem" and variant is not None:
        command += ["--singlem_variant", variant]
    if args.max_subjects:
        command += ["--max_subjects", str(args.max_subjects)]
    if args.overwrite:
        command.append("--overwrite")
    if args.dry_run:
        command.append("--dry_run")
    return command


def main() -> None:
    args = parse_args()
    datasets = selected(args.datasets, tuple(DATASET_INFO))
    models = selected(args.models, MODELS)
    singlem_variants = selected(args.singlem_variants, SINGLEM_VARIANTS)

    jobs: list[tuple[str, str, str | None, list[str]]] = []
    skipped = []
    for dataset in datasets:
        for model in models:
            if model == "mirepnet" and DATASET_INFO[dataset]["task_type"] != "mi":
                skipped.append((dataset, model, "mi_only"))
                continue
            variants = singlem_variants if model == "singlem" else [None]
            for variant in variants:
                jobs.append((dataset, model, variant, command_for(args, dataset, model, variant)))

    print(f"jobs={len(jobs)} skipped={len(skipped)}")
    for dataset, model, reason in skipped:
        print(f"skip {dataset}/{model}: {reason}")
    for index, (dataset, model, variant, command) in enumerate(jobs, start=1):
        label = f"{dataset}/{model}" if variant is None else f"{dataset}/{model}/{variant}"
        print(f"[{index}/{len(jobs)}] {label}")
        print(shlex.join(command))
        if args.dry_run:
            continue
        code = subprocess.run(command).returncode
        if code:
            message = f"failed {label}: exit code {code}"
            if args.keep_going:
                print(message)
                continue
            raise SystemExit(message)


if __name__ == "__main__":
    main()
