from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Iterable


METRICS = ["accuracy", "f1_macro", "precision_macro", "recall_macro", "kappa"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def truthy(value: object) -> bool:
    return str(value).strip().lower() not in {"", "0", "false", "none", "nan"}


def valid_rows(rows: Iterable[dict], method: str | None = None) -> list[dict]:
    out = []
    for row in rows:
        if method is not None and row.get("method") != method:
            continue
        if truthy(row.get("valid", "True")):
            out.append(row)
    return out


def aggregate_rows(rows: list[dict], metrics: list[str] | None = None) -> dict:
    """Aggregate per-subject metrics for manuscript-facing result summaries.

    Accuracy, macro-F1, precision, and recall are stored per subject on the
    ``0..1`` scale and converted to percentages here. Cohen's kappa remains on
    its native scale. Standard deviations use the sample definition to match the
    revised manuscript aggregation.
    """
    metrics = metrics or METRICS
    summary: dict[str, float | int] = {"n_subjects": len(rows)}
    for metric in metrics:
        values = [float(row[metric]) for row in rows if row.get(metric, "") != ""]
        if not values:
            continue
        scale = 1.0 if metric == "kappa" else 100.0
        summary[f"{metric}_mean"] = statistics.fmean(values) * scale
        summary[f"{metric}_std"] = (
            statistics.stdev(values) * scale if len(values) > 1 else 0.0
        )
    return summary


def write_summary(path: Path, rows: list[dict], method: str | None = None) -> dict:
    selected = valid_rows(rows, method)
    summary = aggregate_rows(selected)
    write_csv(path, [summary])
    return summary


def format_float(value: object) -> str:
    if value is None:
        return ""
    if hasattr(value, "item"):
        value = value.item()
    return str(value)
