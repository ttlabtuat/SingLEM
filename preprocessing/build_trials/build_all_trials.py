#!/usr/bin/env python3
"""Build raw-first, all-channel preprocessed subject .pkl inputs for the SingLEM revision."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(THIS_DIR))

from datasets import atten, dreyer, wbcic
from profiles.profiles import MI_DATASETS, PROFILES, dataset_notch, selected_models
from profiles.transforms import channel_metadata, extract_trials, preprocess_raw, subject_transform


DATASETS = {
    "dreyer": dreyer,
    "wbcic_2c": wbcic,
    "wbcic_3c": wbcic,
    "atten_nback": atten,
    "atten_dsr": atten,
    "atten_word": atten,
}


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_root", type=Path, default=PROJECT_ROOT / "raw_datasets")
    parser.add_argument("--output_root", type=Path, default=PROJECT_ROOT / "datasets" / "trials")
    parser.add_argument("--datasets", default="all", help="Comma-separated dataset ids or all.")
    parser.add_argument("--models", default="all", help="Comma-separated model ids or all.")
    parser.add_argument("--subjects", default="all", help="Comma-separated subject ids or all.")
    parser.add_argument("--max_subjects", type=int, default=None)
    parser.add_argument("--n_jobs", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--mne_log_level", default="WARNING")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import mne

    mne.set_log_level(args.mne_log_level)
    datasets = list(DATASETS) if args.datasets == "all" else [x.strip() for x in args.datasets.split(",") if x.strip()]
    models = selected_models(args.models)
    for dataset_id in datasets:
        if dataset_id not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_id}")
        try:
            subject_ids = DATASETS[dataset_id].subjects(
                args.raw_root, dataset_id
            )
        except FileNotFoundError:
            if not args.dry_run:
                raise
            subject_ids = []
        if args.subjects != "all":
            wanted = {x.strip() for x in args.subjects.split(",") if x.strip()}
            subject_ids = [s for s in subject_ids if s in wanted]
        if args.max_subjects:
            subject_ids = subject_ids[: args.max_subjects]
        for model_id in models:
            profile = PROFILES[model_id]
            if profile.mi_only and dataset_id not in MI_DATASETS:
                print(f"skip {dataset_id}/{model_id}: MI-only model")
                continue
            if args.dry_run:
                print(f"dry-run {dataset_id}/{model_id}: {len(subject_ids)} subjects -> {args.output_root / dataset_id / model_id}")
                continue
            build_dataset_model(args, mne, dataset_id, model_id, subject_ids)


def build_dataset_model(args: argparse.Namespace, mne, dataset_id: str, model_id: str, subject_ids: list[str]) -> None:
    profile = PROFILES[model_id]
    out_dir = args.output_root / dataset_id / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    manifest = {
        "dataset": dataset_id,
        "model": model_id,
        "raw_root": portable_path(args.raw_root),
        "output_dir": portable_path(out_dir),
        "sfreq": profile.sfreq,
        "bandpass": list(profile.bandpass),
        "notch": dataset_notch(dataset_id),
        "scale": profile.scale,
        "normalization": profile.normalization,
        "channel_policy": profile.channel_policy,
        "subjects": {},
        "created_seconds": None,
    }
    for subject_id in subject_ids:
        out_file = out_dir / f"{subject_id}.pkl"
        if out_file.exists() and out_file.stat().st_size > 0 and not args.overwrite:
            manifest["subjects"][subject_id] = {"status": "skipped_existing", "path": portable_path(out_file)}
            print(f"skip existing {dataset_id}/{model_id}/{subject_id}")
            continue
        try:
            subject_payload = build_subject(args, mne, dataset_id, model_id, subject_id)
        except Exception as exc:
            manifest["subjects"][subject_id] = {"status": "failed", "error": str(exc)}
            print(f"failed {dataset_id}/{model_id}/{subject_id}: {exc}")
            continue
        with out_file.open("wb") as f:
            pickle.dump(subject_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        labels = Counter(int(x) for x in subject_payload["label"])
        manifest["subjects"][subject_id] = {
            "status": "written",
            "path": portable_path(out_file),
            "n_trials": int(len(subject_payload["label"])),
            "shape": list(subject_payload["data"].shape),
            "labels": dict(sorted(labels.items())),
        }
        print(f"wrote {dataset_id}/{model_id}/{subject_id}: {subject_payload['data'].shape}")
    manifest["created_seconds"] = round(time.time() - started, 3)
    (out_dir / "preprocessing_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def build_subject(args: argparse.Namespace, mne, dataset_id: str, model_id: str, subject_id: str) -> dict:
    profile = PROFILES[model_id]
    arrays, labels, trial_ids, label_names, sources = [], [], [], [], []
    last_channel_names = []
    recordings = DATASETS[dataset_id].load_recordings(args.raw_root, dataset_id, subject_id, mne)
    for recording in recordings:
        raw = preprocess_raw(recording.raw, profile, dataset_notch(dataset_id), args.n_jobs)
        data, y, ids, names = extract_trials(raw, recording.records, recording.unit_mode, profile)
        if len(y):
            arrays.append(data)
            labels.extend(y)
            trial_ids.extend(ids)
            label_names.extend(names)
            sources.append(portable_path(recording.source))
            last_channel_names = list(raw.ch_names)
        raw.close()
        recording.raw.close()
    if arrays:
        data = np.concatenate(arrays, axis=0)
    else:
        data = np.zeros((0, len(last_channel_names), 0), dtype="float32")
    data, channel_names, transform_report = subject_transform(data, last_channel_names, profile)
    meta = channel_metadata(channel_names)
    meta.update(
        {
            "dataset": dataset_id,
            "model": model_id,
            "subject_id": subject_id,
            "sfreq": profile.sfreq,
            "sources": sources,
            "transform": transform_report,
            "unit_contract": unit_contract(model_id),
        }
    )
    return {
        "data": data,
        "label": np.asarray(labels, dtype="int64"),
        "trial_id": trial_ids,
        "label_name": label_names,
        "channel_names": channel_names,
        "sfreq": float(profile.sfreq),
        "metadata": meta,
    }


def unit_contract(model_id: str) -> str:
    if model_id in {"singlem", "labram", "cbramod", "csbrain", "codebrain", "eegmamba"}:
        return "all real EEG channels after filtering/resampling, stored after multiply-by-0.01 model scaling"
    if model_id == "luna_large":
        return "all real EEG channels after filtering/resampling, per-trial per-channel z-score"
    if model_id == "reve_base":
        return "all real EEG channels after filtering/resampling, recording-level z-score, clipped to +/-15 SD"
    if model_id == "bendr":
        return "all real EEG channels after BENDR filtering/resampling; BENDR 20-channel construction happens during feature extraction"
    if model_id == "biot":
        return "all real EEG channels after BIOT filtering/resampling; BIOT 18-channel montage construction happens during feature extraction"
    if model_id == "mirepnet":
        return "all real EEG channels after MIRepNet 8-30 Hz filtering/resampling; template interpolation and Euclidean Alignment happen during feature extraction"
    return "model-ready EEG"


if __name__ == "__main__":
    main()
