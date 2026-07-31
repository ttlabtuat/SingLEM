#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Lock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from shared.data import (
    RESULT_ROOT,
    build_calibration_manifest,
    ensure_manifest,
    fraction_key,
    load_configs,
    manifest_hash,
)


EXPERIMENTS = [
    "strict_svm_foundation",
    "strict_svm_classical",
    "strict_svm_singlem_ablation",
    "strict_svm_single_channel",
    "strict_mlp",
    "strict_mlp_classical",
    "adapted_mlp_foundation",
    "adapted_mlp_classical",
    "strict_neural",
    "adapted_neural",
]
SVM_EXPERIMENTS = {
    "strict_svm_foundation",
    "strict_svm_classical",
    "strict_svm_singlem_ablation",
    "strict_svm_single_channel",
}
ADAPTED_EXPERIMENTS = {
    "adapted_mlp_foundation",
    "adapted_mlp_classical",
    "adapted_neural",
}
OUTPUT_PARTS = {
    "strict_svm_foundation": ("strict", "svm", "foundation"),
    "strict_svm_classical": ("strict", "svm", "classical"),
    "strict_svm_singlem_ablation": ("ablation", "gpu_svm", "singlem"),
    "strict_svm_single_channel": (
        "single_channel",
        "gpu_svm",
        "singlem",
        "downstream_excluded",
    ),
    "strict_mlp": ("strict", "mlp", "foundation"),
    "strict_mlp_classical": ("strict", "mlp", "classical"),
    "strict_neural": ("strict", "neural"),
    "adapted_mlp_foundation": (
        "adapted_30",
        "mlp",
        "foundation",
    ),
    "adapted_mlp_classical": (
        "adapted_30",
        "mlp",
        "classical",
    ),
    "adapted_neural": ("adapted_30", "neural"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_dir", type=Path, default=PROJECT_ROOT / "configs"
    )
    parser.add_argument(
        "--experiments", nargs="+", choices=EXPERIMENTS
    )
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--cpu_threads", type=int, default=8)
    parser.add_argument("--max_subjects", type=int)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--keep_going", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument(
        "--result_root", type=Path, default=RESULT_ROOT,
        help="Result root; use a separate directory for reruns.",
    )
    parser.add_argument(
        "--pytorch_python",
        default=os.environ.get("PYTORCH_PYTHON", sys.executable),
    )
    parser.add_argument(
        "--rapids_python",
        default=os.environ.get("RAPIDS_PYTHON", "python"),
    )
    return parser.parse_args()


def models_for(config: dict, experiment: str) -> list[str]:
    """Return valid model identifiers for one public experiment group.

    The SingLEM GPU-SVM ablation group uses checkpoint variant names as the
    command-line ``--model`` values because each variant maps to a separate
    feature directory under ``datasets/features/<dataset>/singlem/``.
    """
    if experiment == "strict_svm_singlem_ablation":
        return config["singlem_ablation_variants"]
    if experiment == "strict_svm_single_channel":
        return ["singlem"]
    if experiment.endswith("_classical"):
        return config["classical_methods"]
    if (
        experiment.endswith("_neural")
        or experiment == "strict_neural"
    ):
        return config["neural_models"]
    return config["foundation_models"]


def build_jobs(
    args, config: dict, manifest_path: Path | None
) -> tuple[list[dict], list]:
    experiments = args.experiments or EXPERIMENTS
    datasets = args.datasets or list(config["datasets"])
    jobs, skipped = [], []
    for experiment in experiments:
        selected = [
            model
            for model in (args.models or models_for(config, experiment))
            if model in models_for(config, experiment)
        ]
        for model in selected:
            for dataset in datasets:
                if (
                    model in config["mi_only_models"]
                    and config["datasets"][dataset]["task_type"] != "mi"
                ):
                    skipped.append(
                        (experiment, model, dataset, "mi_only")
                    )
                    continue
                if experiment == "strict_svm_single_channel":
                    output_dir = args.result_root.joinpath(
                        *OUTPUT_PARTS[experiment], dataset
                    )
                else:
                    output_dir = args.result_root.joinpath(
                        *OUTPUT_PARTS[experiment], model, dataset
                    )
                complete_file = (
                    output_dir / "channel_summary.csv"
                    if experiment == "strict_svm_single_channel"
                    else output_dir / "summary.csv"
                )
                if not args.restart and complete_file.exists():
                    skipped.append(
                        (experiment, model, dataset, "complete")
                    )
                    continue
                python = (
                    args.rapids_python
                    if experiment in SVM_EXPERIMENTS
                    else args.pytorch_python
                )
                command = [
                    python,
                    str(Path(__file__).with_name("run_experiment.py")),
                    "--experiment",
                    experiment,
                    "--dataset",
                    dataset,
                    "--model",
                    model,
                    "--config_dir",
                    str(args.config_dir),
                    "--output_dir",
                    str(output_dir),
                    "--gpu",
                    "0",
                    "--cpu_threads",
                    str(args.cpu_threads),
                ]
                if experiment in ADAPTED_EXPERIMENTS:
                    if manifest_path is None:
                        raise ValueError("adaptation manifest was not prepared")
                    command += ["--manifest", str(manifest_path)]
                if args.max_subjects:
                    command += ["--max_subjects", str(args.max_subjects)]
                if args.restart:
                    command.append("--restart")
                jobs.append(
                    {
                        "experiment": experiment,
                        "model": model,
                        "dataset": dataset,
                        "output_dir": output_dir,
                        "command": command,
                    }
                )
    return jobs, skipped


def worker(
    gpu: str,
    queue: Queue,
    log_dir: Path,
    cpu_threads: int,
    keep_going: bool,
    lock: Lock,
) -> list[tuple]:
    failures = []
    while True:
        try:
            job = queue.get_nowait()
        except Empty:
            break
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PYTHONUNBUFFERED"] = "1"
        for name in [
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ]:
            env[name] = str(cpu_threads)
        log_path = log_dir / (
            f"{job['experiment']}__{job['model']}__"
            f"{job['dataset']}__gpu{gpu}.log"
        )
        with lock:
            print(
                f"GPU {gpu}: {job['experiment']} "
                f"{job['model']}/{job['dataset']}"
            )
        with log_path.open("w", encoding="utf-8") as log:
            code = subprocess.run(
                job["command"],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            ).returncode
        queue.task_done()
        if code:
            failures.append(
                (
                    job["experiment"],
                    job["model"],
                    job["dataset"],
                    code,
                )
            )
            if not keep_going:
                break
    return failures


def main() -> None:
    args = parse_args()
    if len(args.gpus) != len(set(args.gpus)):
        raise SystemExit("--gpus contains duplicate device IDs")
    config = load_configs(args.config_dir)
    selected_experiments = args.experiments or EXPERIMENTS
    fraction = config["adaptation"]["fraction"]
    seeds = config["adaptation"]["seeds"]
    datasets = args.datasets or list(config["datasets"])
    manifest = None
    manifest_path = None
    if any(value in ADAPTED_EXPERIMENTS for value in selected_experiments):
        if args.dry_run:
            manifest_path = (
                args.result_root
                / "manifests"
                / (
                    f"calibration_{fraction_key(fraction)}_"
                    f"{manifest_hash(datasets, fraction, seeds)}.json"
                )
            )
        else:
            manifest = build_calibration_manifest(
                config, datasets, fraction, seeds
            )
            manifest_path = (
                args.result_root
                / "manifests"
                / (
                    f"calibration_{fraction_key(fraction)}_"
                    f"{manifest_hash(datasets, fraction, seeds)}.json"
                )
            )
    jobs, skipped = build_jobs(args, config, manifest_path)
    print(
        f"jobs={len(jobs)} skipped={len(skipped)} "
        f"fraction={fraction_key(fraction)}"
    )
    for index, job in enumerate(jobs, 1):
        print(f"[{index}/{len(jobs)}] {shlex.join(job['command'])}")
    for row in skipped:
        print("skip " + "/".join(row))
    if args.dry_run:
        return

    if manifest_path is not None and manifest is not None:
        ensure_manifest(manifest_path, manifest)
    run_dir = (
        args.result_root
        / "_logs"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    queue = Queue()
    for job in jobs:
        queue.put(job)
    lock = Lock()
    failures = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [
            pool.submit(
                worker,
                gpu,
                queue,
                run_dir,
                args.cpu_threads,
                args.keep_going,
                lock,
            )
            for gpu in args.gpus
        ]
        for future in as_completed(futures):
            failures.extend(future.result())
    if failures:
        for row in failures:
            print("failed " + "/".join(map(str, row)))
        raise SystemExit(f"{len(failures)} jobs failed")


if __name__ == "__main__":
    main()
