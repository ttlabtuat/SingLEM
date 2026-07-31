from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


def configure_rapids_cuda() -> None:
    toolkit = (
        Path(sys.executable).resolve().parents[1]
        / "targets"
        / "x86_64-linux"
    )
    configured = Path(os.environ.get("CUDA_PATH", ""))
    if not (configured / "include" / "cuda_runtime.h").exists():
        os.environ["CUDA_PATH"] = str(toolkit)


def tune_svm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
    seed: int,
) -> tuple[dict, float]:
    configure_rapids_cuda()
    import cupy as cp
    import optuna
    from cuml.svm import SVC

    x_train_gpu = cp.asarray(
        np.ascontiguousarray(x_train), dtype=cp.float32
    )
    y_train_gpu = cp.asarray(y_train, dtype=cp.int32)
    x_val_gpu = cp.asarray(np.ascontiguousarray(x_val), dtype=cp.float32)
    c_values = np.logspace(
        config["c_min_exp"],
        config["c_max_exp"],
        config["c_values"],
    ).tolist()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        model = SVC(
            C=trial.suggest_categorical("C", c_values),
            kernel="rbf",
            gamma=trial.suggest_categorical("gamma", config["gamma"]),
            random_state=seed,
        )
        model.fit(x_train_gpu, y_train_gpu)
        prediction = cp.asnumpy(model.predict(x_val_gpu))
        return f1_score(
            y_val, prediction, average="macro", zero_division=0
        )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(
        objective,
        n_trials=config["n_trials"],
        n_jobs=config["n_jobs"],
    )
    result = study.best_params, float(study.best_value)
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return result


def fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    params: dict,
    seed: int,
) -> np.ndarray:
    configure_rapids_cuda()
    import cupy as cp
    from cuml.svm import SVC

    model = SVC(
        C=params["C"],
        kernel="rbf",
        gamma=params["gamma"],
        random_state=seed,
    )
    model.fit(
        cp.asarray(np.ascontiguousarray(x_train), dtype=cp.float32),
        cp.asarray(y_train, dtype=cp.int32),
    )
    prediction = cp.asnumpy(
        model.predict(
            cp.asarray(np.ascontiguousarray(x_test), dtype=cp.float32)
        )
    )
    del model
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return prediction
