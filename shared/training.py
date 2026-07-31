from __future__ import annotations

import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_mlp(
    in_dim: int,
    n_classes: int,
    hidden_width: int = 64,
    dropout: float = 0.1,
) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_width),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_width, n_classes),
    )


def optimizer(parameters, config: dict):
    return torch.optim.AdamW(
        parameters,
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )


def scheduler(value, config: dict):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        value, T_max=config["epochs"]
    )


def loader(x, y, batch_size: int) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=True,
    )


def predict(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(
                x[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            output.append(model(batch).argmax(1).cpu())
    return torch.cat(output).numpy()


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )


def train_best_epoch(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
    device: torch.device,
) -> dict:
    opt = optimizer(model.parameters(), config)
    schedule = scheduler(opt, config)
    batches = loader(x_train, y_train, config["batch_size"])
    loss_fn = nn.CrossEntropyLoss()
    best_f1, best_state, best_epoch, stale = -1.0, None, 0, 0
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        for xb, yb in batches:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss_fn(model(xb), yb).backward()
            opt.step()
        schedule.step()
        score = macro_f1(y_val, predict(model, x_val, device))
        if score > best_f1:
            best_f1 = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= config["patience"]:
                break
    model.load_state_dict(best_state)
    return {
        "best_val_f1": best_f1,
        "best_epoch": best_epoch,
        "epochs_run": epoch,
    }


def train_fixed_epochs(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    config: dict,
    device: torch.device,
) -> None:
    opt = optimizer(model.parameters(), config)
    schedule = scheduler(opt, config)
    batches = loader(x, y, config["batch_size"])
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in batches:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss_fn(model(xb), yb).backward()
            opt.step()
        schedule.step()


def adapt_head(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    head: nn.Module,
    config: dict,
    device: torch.device,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in head.parameters():
        parameter.requires_grad = True
    opt = torch.optim.AdamW(
        head.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    xb = torch.as_tensor(x, dtype=torch.float32, device=device)
    yb = torch.as_tensor(y, dtype=torch.long, device=device)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(config["neural_epochs"]):
        model.eval()
        opt.zero_grad(set_to_none=True)
        loss_fn(model(xb), yb).backward()
        opt.step()


def adapt_parameters(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    parameters,
    config: dict,
    device: torch.device,
    train_mode: bool = True,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in parameters:
        parameter.requires_grad = True
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    opt = torch.optim.AdamW(
        trainable,
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    xb = torch.as_tensor(x, dtype=torch.float32, device=device)
    yb = torch.as_tensor(y, dtype=torch.long, device=device)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(config["neural_epochs"]):
        model.train(train_mode)
        opt.zero_grad(set_to_none=True)
        loss_fn(model(xb), yb).backward()
        opt.step()


def neural_head(name: str, model: nn.Module) -> nn.Module:
    if name == "eegnet":
        return model.classifier
    if name == "eegconformer":
        return model.head
    if name == "ifnetv2":
        return model.fc
    raise ValueError(f"unsupported neural model: {name}")
