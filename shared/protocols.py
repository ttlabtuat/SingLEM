from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from shared.classical import fit_transform as classical_fit_transform
from shared.classical import raw_features as classical_raw_features
from shared.data import (
    adaptation_key,
    classification_metrics,
    feature_dir,
    filtered_subject,
    fraction_key,
    load_subjects,
    original_source_split,
    read_csv,
    read_json,
    singlem_variant_feature_dir,
    split_indices,
    split_metadata,
    stack_subjects,
    trial_dir,
    write_csv,
    write_json,
)
from shared.svm import fit_predict, tune_svm


STRICT_SVM = {"strict_svm_foundation", "strict_svm_classical"}
STRICT_SVM_ABLATION = {"strict_svm_singlem_ablation"}
STRICT_SVM_SINGLE_CHANNEL = {"strict_svm_single_channel"}
STRICT_MLP = {
    "strict_mlp",
    "strict_mlp_classical",
}
STRICT_MLP_FOUNDATION = {
    "strict_mlp",
}
MLP_DROPOUT = 0.1
ADAPTED_MLP = {"adapted_mlp_foundation", "adapted_mlp_classical"}


def input_dir(config: dict, experiment: str, dataset: str, model: str) -> Path:
    """Return the input feature/trial directory for one experiment job.

    Strict SingLEM SVM ablations pass the checkpoint variant name as
    ``model``. Those jobs must read from
    ``datasets/features/<dataset>/singlem/<variant>/`` while ordinary
    foundation-model jobs continue to use their model directory.
    """
    if experiment in STRICT_SVM_ABLATION:
        return singlem_variant_feature_dir(config, dataset, model)
    if experiment in STRICT_SVM_SINGLE_CHANNEL:
        return feature_dir(config, dataset, model)
    if experiment.endswith("_foundation") or experiment in STRICT_MLP_FOUNDATION:
        return feature_dir(config, dataset, model)
    return trial_dir(config, dataset)


def load_experiment_subjects(
    config: dict, experiment: str, dataset: str, model: str
):
    flatten = (
        experiment.endswith("_foundation")
        or experiment in STRICT_MLP_FOUNDATION
        or experiment in STRICT_SVM_ABLATION
    )
    path = input_dir(config, experiment, dataset, model)
    return path, load_subjects(path, flatten=flatten)


def mlp_config(config: dict, experiment: str) -> dict:
    value = dict(config["mlp"])
    value["dropout"] = MLP_DROPOUT
    return value


def evaluation_subjects(subjects: dict, max_subjects: int | None) -> list[str]:
    values = sorted(subjects)
    return values[:max_subjects] if max_subjects else values


def start_run(
    output_dir: Path,
    restart: bool,
    metadata: dict | None = None,
) -> list[dict]:
    if metadata is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "run_metadata.json", metadata)
    return read_csv(
        output_dir / "per_subject_metrics.csv",
        restart,
    )


def metadata_payload(identity: dict, payload: dict) -> dict:
    excluded = {"data_dir", "output_dir", "manifest"}
    return {
        key: value
        for key, value in {**identity, **payload}.items()
        if key not in excluded
    }


def channel_names_from_features(data_dir: Path) -> list[str]:
    """Return the stable channel order stored in SingLEM feature metadata.

    Single-channel SVM evaluation keeps the feature tensor unflattened with
    shape ``[trials, channels, ...]``. This validation prevents channel-order
    mismatches from silently corrupting per-electrode metrics.
    """
    reference: list[str] | None = None
    for path in sorted(data_dir.glob("*.pkl")):
        with path.open("rb") as handle:
            obj = pickle.load(handle)
        data = np.asarray(obj["data"])
        if data.ndim < 3:
            raise ValueError(
                f"single-channel SVM expects unflattened features with "
                f"shape [trials, channels, ...]: {path} has {data.shape}"
            )
        metadata = obj.get("metadata", {})
        channels = metadata.get("selected_channels")
        if channels is None:
            channels = metadata.get("channel_policy", {}).get(
                "selected_channels"
            )
        if not channels:
            raise ValueError(f"missing selected channel metadata: {path}")
        channels = [str(value) for value in channels]
        if len(channels) != data.shape[1]:
            raise ValueError(
                f"channel metadata length {len(channels)} does not match "
                f"feature channel dimension {data.shape[1]}: {path}"
            )
        if reference is None:
            reference = channels
        elif channels != reference:
            raise ValueError(
                f"inconsistent channel order in {path}; expected "
                f"{reference}, found {channels}"
            )
    if reference is None:
        raise ValueError(f"no feature files found in {data_dir}")
    return reference


def flatten_channel(x: np.ndarray, channel_index: int) -> np.ndarray:
    """Select one channel and flatten all remaining feature dimensions."""
    if x.ndim < 3:
        raise ValueError(
            f"single-channel SVM expects shape [trials, channels, ...], got {x.shape}"
        )
    selected = x[:, channel_index]
    return np.ascontiguousarray(selected.reshape(selected.shape[0], -1))


def write_channel_summary(output_dir: Path, rows: list[dict]) -> None:
    """Aggregate single-channel subject rows into one row per channel.

    Accuracy, macro-F1, precision, and recall are stored in percentages. Kappa
    remains on its native scale. Standard deviations use the sample definition
    to match the revised manuscript summaries.
    """
    metrics = [
        "accuracy",
        "f1_macro",
        "precision_macro",
        "recall_macro",
        "kappa",
    ]
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if str(row.get("valid", "True")).lower() in {"false", "0"}:
            continue
        grouped.setdefault(row["channel"], []).append(row)
    summary = []
    for channel, channel_rows in grouped.items():
        item = {"channel": channel, "n_subjects": len(channel_rows)}
        for metric in metrics:
            values = np.asarray(
                [float(row[metric]) for row in channel_rows],
                dtype=np.float64,
            )
            scale = 1.0 if metric == "kappa" else 100.0
            item[f"{metric}_mean"] = float(values.mean() * scale)
            item[f"{metric}_std"] = float(values.std(ddof=1) * scale)
        summary.append(item)
    write_csv(output_dir / "channel_summary.csv", summary)


def prepare_neural_inputs(
    model: str, x: np.ndarray, sfreq: float
) -> np.ndarray:
    from scipy import signal

    if model == "ifnetv2":
        bands = []
        for low, high in [(4, 16), (16, 40)]:
            sos = signal.butter(
                4,
                [low, high],
                btype="bandpass",
                fs=sfreq,
                output="sos",
            )
            bands.append(
                signal.sosfiltfilt(sos, x, axis=-1).astype(np.float32)
            )
        values = np.stack(bands, axis=1)
        return values.reshape(values.shape[0], -1, values.shape[-1])
    if model == "eegconformer":
        return x[:, None]
    return x


def selection_classical_features(
    method: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    sfreq: float,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    train, val = classical_raw_features(
        method,
        x_train,
        y_train,
        x_val,
        sfreq,
        config["classical"],
    )
    if not config["classical"].get("standard_scaler", True):
        return train.astype(np.float32), val.astype(np.float32)
    scaler = StandardScaler().fit(train)
    return (
        scaler.transform(train).astype(np.float32),
        scaler.transform(val).astype(np.float32),
    )


def final_classical_features(
    method: str,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_apply: np.ndarray,
    sfreq: float,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    if config["classical"].get("standard_scaler", True):
        return classical_fit_transform(
            method, x_fit, y_fit, x_apply, sfreq, config["classical"]
        )
    return classical_raw_features(
        method, x_fit, y_fit, x_apply, sfreq, config["classical"]
    )


def run_strict_svm(args, config: dict) -> None:
    data_dir, subjects = load_experiment_subjects(
        config, args.experiment, args.dataset, args.model
    )
    evaluation = evaluation_subjects(subjects, args.max_subjects)
    entry = config["datasets"][args.dataset]
    classical = args.experiment == "strict_svm_classical"
    variant = (
        args.model if args.experiment in STRICT_SVM_ABLATION else None
    )
    identity = {
        "experiment": args.experiment,
        "dataset": args.dataset,
        "model": "singlem" if variant else args.model,
        "evaluation_subjects": evaluation,
        "seed": config["seed"],
        "svm": config["svm"],
    }
    if variant:
        identity["singlem_variant"] = variant
    payload = {
        "protocol": "original_stateful_shuffle_strict_loso",
        "classifier": "cuml_rbf_svm",
        "normalization": (
            "train_only_standard_scaler"
            if classical
            else "none"
        ),
        "data_dir": str(data_dir),
        "output_dir": str(args.output_dir),
    }
    if args.dry_run:
        print(json.dumps({**identity, **payload}, indent=2))
        return
    rows = start_run(
        args.output_dir,
        args.restart,
        metadata_payload(identity, payload),
    )
    completed = {row["subject"] for row in rows}
    all_subjects = sorted(subjects)
    rng = np.random.RandomState(config["seed"])
    for test_subject in evaluation:
        source_ids = [
            value for value in all_subjects if value != test_subject
        ]
        x_source, y_source = stack_subjects(
            subjects, source_ids, entry["labels"]
        )
        x_source, y_source, train_idx, val_idx = original_source_split(
            x_source, y_source, rng, config["seed"]
        )
        if test_subject in completed:
            continue
        target = filtered_subject(
            subjects[test_subject], entry["labels"]
        )
        if classical:
            x_train, x_val = selection_classical_features(
                args.model,
                x_source[train_idx],
                y_source[train_idx],
                x_source[val_idx],
                entry["sfreq"],
                config,
            )
            x_final, x_test = final_classical_features(
                args.model,
                x_source,
                y_source,
                target.data,
                entry["sfreq"],
                config,
            )
        else:
            x_train, x_val = x_source[train_idx], x_source[val_idx]
            x_final, x_test = x_source, target.data
        params, validation_f1 = tune_svm(
            x_train,
            y_source[train_idx],
            x_val,
            y_source[val_idx],
            config["svm"],
            config["seed"],
        )
        prediction = fit_predict(
            x_final, y_source, x_test, params, config["seed"]
        )
        row = {
            "subject": test_subject,
            "best_params": json.dumps(params, sort_keys=True),
            "validation_f1_macro": validation_f1,
            **classification_metrics(target.labels, prediction),
        }
        rows.append(row)
        completed.add(test_subject)
        write_csv(args.output_dir / "per_subject_metrics.csv", rows)
        print(
            f"{args.experiment}/{args.model}/{args.dataset}/"
            f"{test_subject}: accuracy={row['accuracy']:.4f}"
        )


def run_strict_svm_single_channel(args, config: dict) -> None:
    """Run strict LOSO cuML RBF-SVM independently for each SingLEM channel.

    The source-subject split is generated once per held-out subject and reused
    for every channel. This keeps channel comparisons paired while preventing
    target-subject leakage into tuning or final training.
    """
    if args.model != "singlem":
        raise ValueError("strict_svm_single_channel supports only singlem")
    data_dir, subjects = load_experiment_subjects(
        config, args.experiment, args.dataset, args.model
    )
    channels = channel_names_from_features(data_dir)
    evaluation = evaluation_subjects(subjects, args.max_subjects)
    entry = config["datasets"][args.dataset]
    identity = {
        "experiment": args.experiment,
        "dataset": args.dataset,
        "model": args.model,
        "singlem_variant": config.get(
            "singlem_default_variant", "downstream_excluded"
        ),
        "evaluation_subjects": evaluation,
        "channels": channels,
        "seed": config["seed"],
        "svm": config["svm"],
    }
    payload = {
        "protocol": "original_stateful_shuffle_strict_loso_single_channel",
        "classifier": "cuml_rbf_svm",
        "normalization": "none",
        "data_dir": str(data_dir),
        "output_dir": str(args.output_dir),
    }
    if args.dry_run:
        print(json.dumps({**identity, **payload}, indent=2))
        return
    write_json(
        args.output_dir / "run_metadata.json",
        metadata_payload(identity, payload),
    )
    rows = read_csv(
        args.output_dir / "per_channel_subject_metrics.csv",
        args.restart,
    )
    completed = {(row["channel"], row["subject"]) for row in rows}
    all_subjects = sorted(subjects)
    rng = np.random.RandomState(config["seed"])
    for test_subject in evaluation:
        source_ids = [
            value for value in all_subjects if value != test_subject
        ]
        x_source, y_source = stack_subjects(
            subjects, source_ids, entry["labels"]
        )
        x_source, y_source, train_idx, val_idx = original_source_split(
            x_source, y_source, rng, config["seed"]
        )
        target = filtered_subject(
            subjects[test_subject], entry["labels"]
        )
        for channel_index, channel in enumerate(channels):
            key = (channel, test_subject)
            if key in completed:
                continue
            x_train = flatten_channel(x_source[train_idx], channel_index)
            x_val = flatten_channel(x_source[val_idx], channel_index)
            x_final = flatten_channel(x_source, channel_index)
            x_test = flatten_channel(target.data, channel_index)
            params, validation_f1 = tune_svm(
                x_train,
                y_source[train_idx],
                x_val,
                y_source[val_idx],
                config["svm"],
                config["seed"],
            )
            prediction = fit_predict(
                x_final,
                y_source,
                x_test,
                params,
                config["seed"],
            )
            row = {
                "channel": channel,
                "subject": test_subject,
                "best_params": json.dumps(params, sort_keys=True),
                "validation_f1_macro": validation_f1,
                **classification_metrics(target.labels, prediction),
            }
            rows.append(row)
            completed.add(key)
            write_csv(
                args.output_dir / "per_channel_subject_metrics.csv", rows
            )
            print(
                f"{args.experiment}/{args.dataset}/{channel}/"
                f"{test_subject}: accuracy={row['accuracy']:.4f}"
            )
        write_channel_summary(args.output_dir, rows)
    write_channel_summary(args.output_dir, rows)


def run_strict_mlp(args, config: dict) -> None:
    import torch

    from shared.training import (
        build_mlp,
        predict,
        set_seed,
        train_best_epoch,
        train_fixed_epochs,
    )

    data_dir, subjects = load_experiment_subjects(
        config, args.experiment, args.dataset, args.model
    )
    evaluation = evaluation_subjects(subjects, args.max_subjects)
    entry = config["datasets"][args.dataset]
    classical = args.experiment == "strict_mlp_classical"
    mlp = mlp_config(config, args.experiment)
    identity = {
        "experiment": args.experiment,
        "dataset": args.dataset,
        "model": args.model,
        "evaluation_subjects": evaluation,
        "seed": config["seed"],
        "training": config["training"],
        "mlp": mlp,
    }
    payload = {
        "protocol": "original_stateful_shuffle_selection_full_source_refit",
        "normalization": (
            (
                "selection_train_then_full_source_classical_standard_scaler"
                if config["classical"].get("standard_scaler", True)
                else "classical_feature_no_scaler"
            )
            if classical
            else (
                "selection_train_then_full_source_standard_scaler"
                if mlp.get("standard_scaler", False)
                else "none"
            )
        ),
        "data_dir": str(data_dir),
        "output_dir": str(args.output_dir),
    }
    if args.dry_run:
        print(json.dumps({**identity, **payload}, indent=2))
        return
    rows = start_run(
        args.output_dir,
        args.restart,
        metadata_payload(identity, payload),
    )
    completed = {row["subject"] for row in rows}
    all_subjects = sorted(subjects)
    rng = np.random.RandomState(config["seed"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for test_subject in evaluation:
        source_ids = [
            value for value in all_subjects if value != test_subject
        ]
        x_source, y_source = stack_subjects(
            subjects, source_ids, entry["labels"]
        )
        x_source, y_source, train_idx, val_idx = original_source_split(
            x_source, y_source, rng, config["seed"]
        )
        if test_subject in completed:
            continue
        target = filtered_subject(
            subjects[test_subject], entry["labels"]
        )
        if classical:
            x_train, x_val = selection_classical_features(
                args.model,
                x_source[train_idx],
                y_source[train_idx],
                x_source[val_idx],
                entry["sfreq"],
                config,
            )
            x_refit, x_test = final_classical_features(
                args.model,
                x_source,
                y_source,
                target.data,
                entry["sfreq"],
                config,
            )
        elif mlp.get("standard_scaler", False):
            selection_scaler = StandardScaler().fit(x_source[train_idx])
            final_scaler = StandardScaler().fit(x_source)
            x_train = selection_scaler.transform(
                x_source[train_idx]
            ).astype(np.float32)
            x_val = selection_scaler.transform(
                x_source[val_idx]
            ).astype(np.float32)
            x_refit = final_scaler.transform(x_source).astype(np.float32)
            x_test = final_scaler.transform(target.data).astype(np.float32)
        else:
            x_train = x_source[train_idx].astype(np.float32)
            x_val = x_source[val_idx].astype(np.float32)
            x_refit = x_source.astype(np.float32)
            x_test = target.data.astype(np.float32)
        set_seed(config["seed"])
        selection_model = build_mlp(
            x_train.shape[1],
            len(entry["labels"]),
            mlp["hidden_width"],
            mlp["dropout"],
        ).to(device)
        selection = train_best_epoch(
            selection_model,
            x_train,
            y_source[train_idx],
            x_val,
            y_source[val_idx],
            config["training"],
            device,
        )
        del selection_model
        set_seed(config["seed"])
        final_model = build_mlp(
            x_refit.shape[1],
            len(entry["labels"]),
            mlp["hidden_width"],
            mlp["dropout"],
        ).to(device)
        train_fixed_epochs(
            final_model,
            x_refit,
            y_source,
            selection["best_epoch"],
            config["training"],
            device,
        )
        prediction = predict(final_model, x_test, device)
        row = {
            "subject": test_subject,
            "best_val_f1": selection["best_val_f1"],
            "best_epoch": selection["best_epoch"],
            "selection_epochs_run": selection["epochs_run"],
            "refit_epochs": selection["best_epoch"],
            **classification_metrics(target.labels, prediction),
        }
        rows.append(row)
        completed.add(test_subject)
        write_csv(args.output_dir / "per_subject_metrics.csv", rows)
        del final_model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def run_strict_neural(args, config: dict) -> None:
    import torch

    from models.neural import build_model
    from shared.training import (
        predict,
        set_seed,
        train_best_epoch,
        train_fixed_epochs,
    )

    data_dir, subjects = load_experiment_subjects(
        config, args.experiment, args.dataset, args.model
    )
    evaluation = evaluation_subjects(subjects, args.max_subjects)
    entry = config["datasets"][args.dataset]
    identity = {
        "experiment": args.experiment,
        "dataset": args.dataset,
        "model": args.model,
        "evaluation_subjects": evaluation,
        "seed": config["seed"],
        "training": config["training"],
    }
    payload = {
        "protocol": "original_stateful_shuffle_selection_full_source_refit",
        "normalization": "selection_train_then_full_source_global_zscore",
        "data_dir": str(data_dir),
        "output_dir": str(args.output_dir),
    }
    if args.dry_run:
        print(json.dumps({**identity, **payload}, indent=2))
        return
    rows = start_run(
        args.output_dir,
        args.restart,
        metadata_payload(identity, payload),
    )
    completed = {row["subject"] for row in rows}
    all_subjects = sorted(subjects)
    rng = np.random.RandomState(config["seed"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for test_subject in evaluation:
        source_ids = [
            value for value in all_subjects if value != test_subject
        ]
        x_source, y_source = stack_subjects(
            subjects, source_ids, entry["labels"]
        )
        x_source, y_source, train_idx, val_idx = original_source_split(
            x_source, y_source, rng, config["seed"]
        )
        if test_subject in completed:
            continue
        target = filtered_subject(
            subjects[test_subject], entry["labels"]
        )
        x_source = prepare_neural_inputs(
            args.model, x_source, entry["sfreq"]
        )
        x_target = prepare_neural_inputs(
            args.model, target.data, entry["sfreq"]
        )
        selection_mean = x_source[train_idx].mean()
        selection_std = x_source[train_idx].std() + 1e-6
        final_mean = x_source.mean()
        final_std = x_source.std() + 1e-6
        x_train = (x_source[train_idx] - selection_mean) / selection_std
        x_val = (x_source[val_idx] - selection_mean) / selection_std
        x_refit = (x_source - final_mean) / final_std
        x_test = (x_target - final_mean) / final_std
        channels, samples = x_source.shape[-2:]
        set_seed(config["seed"])
        selection_model = build_model(
            args.model,
            len(entry["labels"]),
            channels,
            samples,
        ).to(device)
        selection = train_best_epoch(
            selection_model,
            x_train,
            y_source[train_idx],
            x_val,
            y_source[val_idx],
            config["training"],
            device,
        )
        del selection_model
        set_seed(config["seed"])
        final_model = build_model(
            args.model,
            len(entry["labels"]),
            channels,
            samples,
        ).to(device)
        train_fixed_epochs(
            final_model,
            x_refit,
            y_source,
            selection["best_epoch"],
            config["training"],
            device,
        )
        prediction = predict(final_model, x_test, device)
        row = {
            "subject": test_subject,
            "best_val_f1": selection["best_val_f1"],
            "best_epoch": selection["best_epoch"],
            "selection_epochs_run": selection["epochs_run"],
            "refit_epochs": selection["best_epoch"],
            **classification_metrics(target.labels, prediction),
        }
        rows.append(row)
        completed.add(test_subject)
        write_csv(args.output_dir / "per_subject_metrics.csv", rows)
        del final_model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def run_adapted_mlp(args, config: dict) -> None:
    import torch

    from shared.training import (
        build_mlp,
        predict,
        set_seed,
        train_best_epoch,
        train_fixed_epochs,
    )

    data_dir, subjects = load_experiment_subjects(
        config, args.experiment, args.dataset, args.model
    )
    evaluation = evaluation_subjects(subjects, args.max_subjects)
    entry = config["datasets"][args.dataset]
    classical = args.experiment == "adapted_mlp_classical"
    mlp = mlp_config(config, args.experiment)
    adaptation = config.get("mlp_adaptation", {})
    fraction = adaptation.get("fraction", config["adaptation"]["fraction"])
    seeds = adaptation.get("seeds", [config["seed"]])
    methods = adaptation.get("methods", ["pooled_refit"])
    identity = {
        "experiment": args.experiment,
        "dataset": args.dataset,
        "model": args.model,
        "evaluation_subjects": evaluation,
        "seed": config["seed"],
        "training": config["training"],
        "mlp": mlp,
        "fraction": fraction,
        "seeds": seeds,
        "methods": methods,
        "manifest": str(Path(args.manifest).resolve()),
    }
    payload = {
        "protocol": "source_plus_target_calibration_pooled_refit",
        "normalization": (
            (
                "selection_train_then_pooled_classical_standard_scaler"
                if config["classical"].get("standard_scaler", True)
                else "classical_feature_no_scaler"
            )
            if classical
            else "none"
        ),
        "data_dir": str(data_dir),
        "output_dir": str(args.output_dir),
    }
    if args.dry_run:
        print(json.dumps({**identity, **payload}, indent=2))
        return
    manifest = read_json(args.manifest)["datasets"][args.dataset][
        "subjects"
    ]
    rows = start_run(
        args.output_dir,
        args.restart,
        metadata_payload(identity, payload),
    )
    completed = {adaptation_key(row) for row in rows}
    all_subjects = sorted(subjects)
    rng = np.random.RandomState(config["seed"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    metric_names = [
        "accuracy",
        "f1_macro",
        "precision_macro",
        "recall_macro",
        "kappa",
    ]
    for test_subject in evaluation:
        source_ids = [
            value for value in all_subjects if value != test_subject
        ]
        x_source, y_source = stack_subjects(
            subjects, source_ids, entry["labels"]
        )
        x_source, y_source, train_idx, val_idx = original_source_split(
            x_source, y_source, rng, config["seed"]
        )
        expected = {
            (
                test_subject,
                fraction_key(fraction),
                seed,
                method,
            )
            for seed in seeds
            for method in methods
        }
        if expected <= completed:
            continue
        target = filtered_subject(
            subjects[test_subject], entry["labels"]
        )
        if classical:
            x_train, x_val = selection_classical_features(
                args.model,
                x_source[train_idx],
                y_source[train_idx],
                x_source[val_idx],
                entry["sfreq"],
                config,
            )
            feature_dim = x_train.shape[1]
        else:
            x_train = x_source[train_idx].astype(np.float32)
            x_val = x_source[val_idx].astype(np.float32)
            feature_dim = x_source.shape[1]
        set_seed(config["seed"])
        selection_model = build_mlp(
            feature_dim,
            len(entry["labels"]),
            mlp["hidden_width"],
            mlp["dropout"],
        ).to(device)
        selection = train_best_epoch(
            selection_model,
            x_train,
            y_source[train_idx],
            x_val,
            y_source[val_idx],
            config["training"],
            device,
        )
        del selection_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for seed in seeds:
            split = manifest[test_subject]["splits"][
                fraction_key(fraction)
            ][str(seed)]
            calibration_idx, test_idx = split_indices(target, split)
            details = split_metadata(
                target, split, fraction, calibration_idx, test_idx
            )
            for method in methods:
                key = (
                    test_subject,
                    fraction_key(fraction),
                    seed,
                    method,
                )
                if key in completed:
                    continue
                row = {
                    "subject": test_subject,
                    "seed": seed,
                    "method": method,
                    **details,
                    "source_best_val_f1": selection["best_val_f1"],
                    "source_best_epoch": selection["best_epoch"],
                    "source_epochs_run": selection["epochs_run"],
                }
                if not details["valid"]:
                    row.update({name: "" for name in metric_names})
                elif method == "pooled_refit":
                    pooled_x = np.concatenate(
                        [x_source, target.data[calibration_idx]]
                    )
                    pooled_y = np.concatenate(
                        [y_source, target.labels[calibration_idx]]
                    )
                    if classical:
                        pooled_x, test_x = final_classical_features(
                            args.model,
                            pooled_x,
                            pooled_y,
                            target.data[test_idx],
                            entry["sfreq"],
                            config,
                        )
                    else:
                        pooled_x = pooled_x.astype(np.float32)
                        test_x = target.data[test_idx].astype(np.float32)
                    set_seed(config["seed"])
                    model = build_mlp(
                        pooled_x.shape[1],
                        len(entry["labels"]),
                        mlp["hidden_width"],
                        mlp["dropout"],
                    ).to(device)
                    train_fixed_epochs(
                        model,
                        pooled_x,
                        pooled_y,
                        selection["best_epoch"],
                        config["training"],
                        device,
                    )
                    prediction = predict(
                        model,
                        test_x,
                        device,
                    )
                    row.update(
                        classification_metrics(
                            target.labels[test_idx], prediction
                        )
                    )
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    raise ValueError(f"unknown MLP adaptation: {method}")
                rows.append(row)
                completed.add(key)
                write_csv(
                    args.output_dir / "per_subject_metrics.csv", rows
                )


def run_adapted_neural(args, config: dict) -> None:
    import torch

    from models.neural import build_model
    from shared.training import (
        adapt_head,
        neural_head,
        predict,
        set_seed,
        train_best_epoch,
        train_fixed_epochs,
    )

    data_dir, subjects = load_experiment_subjects(
        config, args.experiment, args.dataset, args.model
    )
    evaluation = evaluation_subjects(subjects, args.max_subjects)
    entry = config["datasets"][args.dataset]
    adaptation = config["adaptation"]
    fraction = adaptation["fraction"]
    seeds = adaptation["seeds"]
    methods = ["source_only", "existing_head_adaptation"]
    identity = {
        "experiment": args.experiment,
        "dataset": args.dataset,
        "model": args.model,
        "evaluation_subjects": evaluation,
        "seed": config["seed"],
        "training": config["training"],
        "adaptation": adaptation,
        "methods": methods,
        "manifest": str(Path(args.manifest).resolve()),
    }
    payload = {
        "protocol": "source_training_then_existing_head_adaptation",
        "normalization": "source_global_zscore",
        "data_dir": str(data_dir),
        "output_dir": str(args.output_dir),
    }
    if args.dry_run:
        print(json.dumps({**identity, **payload}, indent=2))
        return
    manifest = read_json(args.manifest)["datasets"][args.dataset][
        "subjects"
    ]
    rows = start_run(
        args.output_dir,
        args.restart,
        metadata_payload(identity, payload),
    )
    completed = {adaptation_key(row) for row in rows}
    all_subjects = sorted(subjects)
    rng = np.random.RandomState(config["seed"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for test_subject in evaluation:
        source_ids = [
            value for value in all_subjects if value != test_subject
        ]
        x_source, y_source = stack_subjects(
            subjects, source_ids, entry["labels"]
        )
        x_source, y_source, train_idx, val_idx = original_source_split(
            x_source, y_source, rng, config["seed"]
        )
        expected = {
            (
                test_subject,
                fraction_key(fraction),
                seed,
                method,
            )
            for seed in seeds
            for method in methods
        }
        if expected <= completed:
            continue
        target = filtered_subject(
            subjects[test_subject], entry["labels"]
        )
        x_source = prepare_neural_inputs(
            args.model, x_source, entry["sfreq"]
        )
        x_target = prepare_neural_inputs(
            args.model, target.data, entry["sfreq"]
        )
        selection_mean = x_source[train_idx].mean()
        selection_std = x_source[train_idx].std() + 1e-6
        final_mean = x_source.mean()
        final_std = x_source.std() + 1e-6
        x_train = (x_source[train_idx] - selection_mean) / selection_std
        x_val = (x_source[val_idx] - selection_mean) / selection_std
        x_refit = (x_source - final_mean) / final_std
        x_target = (x_target - final_mean) / final_std
        channels, samples = x_source.shape[-2:]
        set_seed(config["seed"])
        selection_model = build_model(
            args.model,
            len(entry["labels"]),
            channels,
            samples,
        ).to(device)
        selection = train_best_epoch(
            selection_model,
            x_train,
            y_source[train_idx],
            x_val,
            y_source[val_idx],
            config["training"],
            device,
        )
        del selection_model
        set_seed(config["seed"])
        model = build_model(
            args.model,
            len(entry["labels"]),
            channels,
            samples,
        ).to(device)
        train_fixed_epochs(
            model,
            x_refit,
            y_source,
            selection["best_epoch"],
            config["training"],
            device,
        )
        source_state = copy.deepcopy(model.state_dict())
        source_prediction = predict(model, x_target, device)
        for seed in seeds:
            split = manifest[test_subject]["splits"][
                fraction_key(fraction)
            ][str(seed)]
            calibration_idx, test_idx = split_indices(target, split)
            details = split_metadata(
                target, split, fraction, calibration_idx, test_idx
            )
            for method in methods:
                key = (
                    test_subject,
                    fraction_key(fraction),
                    seed,
                    method,
                )
                if key in completed:
                    continue
                row = {
                    "subject": test_subject,
                    "seed": seed,
                    "method": method,
                    **details,
                    "source_best_val_f1": selection["best_val_f1"],
                    "source_best_epoch": selection["best_epoch"],
                }
                if method == "source_only":
                    prediction = source_prediction[test_idx]
                else:
                    model.load_state_dict(source_state)
                    set_seed(config["seed"])
                    adapt_head(
                        model,
                        x_target[calibration_idx],
                        target.labels[calibration_idx],
                        neural_head(args.model, model),
                        adaptation,
                        device,
                    )
                    prediction = predict(
                        model, x_target[test_idx], device
                    )
                row.update(
                    classification_metrics(
                        target.labels[test_idx], prediction
                    )
                )
                rows.append(row)
                completed.add(key)
                write_csv(
                    args.output_dir / "per_subject_metrics.csv", rows
                )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def dispatch(args, config: dict) -> None:
    if args.experiment in STRICT_SVM or args.experiment in STRICT_SVM_ABLATION:
        run_strict_svm(args, config)
    elif args.experiment in STRICT_SVM_SINGLE_CHANNEL:
        run_strict_svm_single_channel(args, config)
    elif args.experiment in STRICT_MLP:
        run_strict_mlp(args, config)
    elif args.experiment == "strict_neural":
        run_strict_neural(args, config)
    elif args.experiment in ADAPTED_MLP:
        run_adapted_mlp(args, config)
    elif args.experiment == "adapted_neural":
        run_adapted_neural(args, config)
    else:
        raise ValueError(f"unknown experiment: {args.experiment}")
