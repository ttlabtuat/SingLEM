#!/usr/bin/env python3
"""Extract frozen foundation-model features from preprocessed all-channel trials."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))

from adapters.policies import apply_policy
from adapters.common import require_model_artifacts


MODELS = ["singlem", "bendr", "biot", "cbramod", "labram", "csbrain", "codebrain", "luna_large", "mirepnet"]
SINGLEM_VARIANTS = ["downstream_excluded", "downstream_included", "no_feature_embedding"]
DATASET_INFO = {
    "dreyer": {"num_classes": 2, "task_type": "mi"},
    "wbcic_2c": {"num_classes": 2, "task_type": "mi"},
    "wbcic_3c": {"num_classes": 3, "task_type": "mi"},
    "atten_nback": {"num_classes": 2, "task_type": "cognitive"},
    "atten_dsr": {"num_classes": 2, "task_type": "cognitive"},
    "atten_word": {"num_classes": 2, "task_type": "cognitive"},
}


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_INFO))
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument(
        "--singlem_variant",
        default="downstream_excluded",
        choices=SINGLEM_VARIANTS,
        help="SingLEM checkpoint variant; used only when --model singlem.",
    )
    parser.add_argument("--channel_policy", default="pretrained_matched", choices=["pretrained_matched", "current_compat"])
    parser.add_argument("--input_root", type=Path, default=PROJECT_ROOT / "datasets" / "trials")
    parser.add_argument("--data_dir", type=Path)
    parser.add_argument("--output_root", type=Path, default=PROJECT_ROOT / "datasets" / "features")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--max_subjects", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def subject_files(data_dir: Path) -> list[Path]:
    return sorted(path for path in data_dir.glob("*.pkl") if path.name != "preprocessing_manifest.json")


def load_subject(path: Path) -> dict:
    if path.exists() and path.stat().st_size == 0:
        raise ValueError(f"placeholder file has not been replaced: {path}")
    with path.open("rb") as f:
        obj = pickle.load(f)
    return {
        "data": np.asarray(obj["data"], dtype="float32"),
        "label": np.asarray(obj["label"], dtype="int64"),
        "channel_names": list(obj.get("channel_names") or obj.get("metadata", {}).get("channel_names") or []),
        "sfreq": float(obj.get("sfreq") or obj.get("metadata", {}).get("sfreq") or 0.0),
    }


def write_subject(path: Path, features: np.ndarray, labels: np.ndarray, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump({"data": features, "label": labels, "metadata": metadata}, f, protocol=pickle.HIGHEST_PROTOCOL)


def prepare(path: Path, dataset: str, model: str, policy: str) -> tuple[dict, np.ndarray, list[str], dict]:
    obj = load_subject(path)
    if not obj["channel_names"]:
        raise ValueError(f"{path} has no channel_names")
    data, channel_names, report = apply_policy(dataset, model, policy, obj["data"], obj["channel_names"])
    if data.shape[1] == 0:
        raise ValueError(f"{model}/{policy} selected zero channels for {path.name}")
    return obj, data, channel_names, report


def main() -> None:
    args = parse_args()
    if args.model == "mirepnet" and DATASET_INFO[args.dataset]["task_type"] != "mi":
        raise SystemExit(f"MIRepNet is MI-only; skip {args.dataset}")
    if not args.dry_run:
        require_model_artifacts(args.model)

    data_dir = args.data_dir or args.input_root / args.dataset / args.model
    out_dir = args.output_dir or args.output_root / args.dataset / args.model
    if args.model == "singlem" and args.output_dir is None:
        out_dir /= args.singlem_variant
    files = subject_files(data_dir)
    if args.max_subjects:
        files = files[: args.max_subjects]
    if not files:
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "dataset": args.dataset,
                        "model": args.model,
                        "channel_policy": args.channel_policy,
                        "data_dir": portable_path(data_dir),
                        "output_dir": portable_path(out_dir),
                        "subjects": 0,
                        "status": "waiting_for_generated_trials",
                    },
                    indent=2,
                )
            )
            return
        raise SystemExit(f"No subject .pkl files found in {data_dir}")

    dry_payload = {
        "dataset": args.dataset,
        "model": args.model,
        "channel_policy": args.channel_policy,
        "singlem_variant": (
            args.singlem_variant if args.model == "singlem" else None
        ),
        "data_dir": str(data_dir),
        "output_dir": str(out_dir),
        "subjects": len(files),
        "first_subject": files[0].name,
    }
    if args.dry_run and files[0].stat().st_size == 0:
        dry_payload.update(
            {
                "status": "waiting_for_real_trials",
                "first_subject_is_placeholder": True,
            }
        )
        print(json.dumps(dry_payload, indent=2))
        return

    first_obj, first_x, first_channels, first_policy = prepare(files[0], args.dataset, args.model, args.channel_policy)
    dry_payload.update(
        {
            "first_input_shape": list(first_obj["data"].shape),
            "first_model_input_shape": list(first_x.shape),
            "first_selected_channels": first_channels,
            "first_channel_policy_report": first_policy,
        }
    )
    if args.dry_run:
        print(json.dumps(dry_payload, indent=2))
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.gpu == "cpu" else str(args.gpu)
    import torch
    from adapters import build_adapter

    device = torch.device("cuda" if args.gpu != "cpu" and torch.cuda.is_available() else "cpu")
    context = {
        **DATASET_INFO[args.dataset],
        "dataset": args.dataset,
        "model": args.model,
        "channel_policy": args.channel_policy,
        "channel_names": first_channels,
        "sample_shape": first_x.shape,
        "sfreq": first_obj["sfreq"],
        "singlem_variant": args.singlem_variant,
    }
    adapter = build_adapter(args.model, context, device)
    adapter_info = adapter.info()

    started = time.time()
    rows = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(files, 1):
        out_path = out_dir / path.name
        if out_path.exists() and out_path.stat().st_size > 0 and not args.overwrite:
            rows.append({"subject": path.stem, "status": "skipped_existing", "path": str(out_path)})
            print(f"[{i}/{len(files)}] skip {path.stem}")
            continue
        obj, x_model, selected_channels, policy_report = (first_obj, first_x, first_channels, first_policy) if path == files[0] else prepare(path, args.dataset, args.model, args.channel_policy)
        features = adapter.extract(x_model, args.batch_size)
        metadata = {
            "source": portable_path(path),
            "selected_channels": selected_channels,
            "channel_policy": policy_report,
        }
        if args.model == "singlem":
            metadata.update({
                "singlem_variant": args.singlem_variant,
                "checkpoint_sha256": adapter_info["checkpoint_sha256"],
            })
        write_subject(out_path, features, obj["label"], metadata)
        rows.append({"subject": path.stem, "status": "written", "path": str(out_path), "feature_shape": list(features.shape), "model_input_shape": list(x_model.shape), "selected_channels": selected_channels})
        print(f"[{i}/{len(files)}] wrote {path.stem}: {features.shape}")

    manifest = {
        **dry_payload,
        "gpu": args.gpu,
        "device": str(device),
        "batch_size": args.batch_size,
        "seconds": round(time.time() - started, 3),
        "adapter": adapter_info,
        "subjects": rows,
    }
    (out_dir / "feature_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
