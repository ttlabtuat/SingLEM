#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path(__file__).resolve().parent
MANIFEST = MODEL_ROOT / "manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Install and inspect external foundation-model source/checkpoint "
            "files used by the feature extractors."
        )
    )
    parser.add_argument("--models", nargs="+")
    parser.add_argument(
        "--install",
        action="store_true",
        help=(
            "Copy required files from full upstream repositories or download "
            "folders into the canonical placeholder paths."
        ),
    )
    parser.add_argument(
        "--source_root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Root of a downloaded upstream source repository. Can be supplied "
            "multiple times. If omitted, the model directory and its immediate "
            "subdirectories plus repository-root foundation_models/ are searched."
        ),
    )
    parser.add_argument(
        "--checkpoint_root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Folder containing downloaded checkpoint files. Can be supplied "
            "multiple times."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing non-empty canonical files during --install.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def project_path(value: str) -> Path:
    return PROJECT_ROOT / value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ready(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def unique_paths(paths: list[Path]) -> list[Path]:
    seen = set()
    unique = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def checkpoint_entries(config: dict) -> list[tuple[Path, str]]:
    if "checkpoint_path" in config:
        return [
            (
                project_path(config["checkpoint_path"]),
                config["checkpoint_sha256"],
            )
        ]
    return [
        (project_path(path), checksum)
        for path, checksum in zip(
            config.get("checkpoint_paths", []),
            config.get("checkpoint_sha256", []),
            strict=True,
        )
    ]


def source_entries(config: dict) -> list[str]:
    return list(config.get("source_paths", [config["source_path"]]))


def checkpoint_paths(config: dict) -> list[str]:
    paths = list(config.get("checkpoint_paths", []))
    if "checkpoint_path" in config:
        paths.append(config["checkpoint_path"])
    return paths


def relative_to_model_dir(model_name: str, value: str) -> str:
    path = Path(value)
    prefix = Path("models") / "foundation" / model_name
    try:
        return str(path.relative_to(prefix))
    except ValueError:
        return path.name


def install_candidates(config: dict, model_name: str, kind: str, value: str) -> list[str]:
    key = f"{kind}_candidates"
    configured = config.get("install", {}).get(key, {}).get(value)
    if configured:
        return list(configured)
    return [relative_to_model_dir(model_name, value)]


def runtime_candidates(config: dict, model_name: str, kind: str, value: str) -> list[Path]:
    paths = [project_path(value)]
    model_dir = MODEL_ROOT / model_name
    for candidate in install_candidates(config, model_name, kind, value):
        paths.append(model_dir / candidate)
    return unique_paths(paths)


def root_candidates(model_name: str, roots: list[Path]) -> list[Path]:
    candidates = [path.expanduser() for path in roots]
    model_dir = MODEL_ROOT / model_name
    candidates.append(model_dir)
    candidates.append(PROJECT_ROOT / "foundation_models")
    for root in list(candidates):
        if root.exists() and root.is_dir():
            candidates.extend(path for path in sorted(root.iterdir()) if path.is_dir())
    return unique_paths(candidates)


def install_file(
    model_name: str,
    kind: str,
    destination_value: str,
    candidates: list[str],
    roots: list[Path],
    overwrite: bool,
    dry_run: bool,
) -> None:
    destination = project_path(destination_value)
    if ready(destination) and not overwrite:
        print(f"{model_name}: keep existing {kind} ({destination})")
        return

    searched = []
    for root in roots:
        for candidate in candidates:
            source = root / candidate
            searched.append(source)
            if not ready(source):
                continue
            if source.resolve() == destination.resolve():
                print(f"{model_name}: {kind}=ready ({destination})")
                return
            print(f"{model_name}: {'would install' if dry_run else 'install'} {kind} {destination} <- {source}")
            if not dry_run:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            return

    print(f"{model_name}: no {kind} found ({destination})")
    for path in unique_paths(searched)[:12]:
        print(f"  searched: {path}")
    if len(unique_paths(searched)) > 12:
        print("  searched: ...")


def install_model(
    name: str,
    config: dict,
    source_roots: list[Path],
    checkpoint_roots: list[Path],
    overwrite: bool,
    dry_run: bool,
) -> None:
    source_root_candidates = root_candidates(name, source_roots)
    checkpoint_root_candidates = root_candidates(
        name, checkpoint_roots + source_roots
    )
    for value in source_entries(config):
        install_file(
            name,
            "source",
            value,
            install_candidates(config, name, "source", value),
            source_root_candidates,
            overwrite,
            dry_run,
        )
    for value in checkpoint_paths(config):
        install_file(
            name,
            "checkpoint",
            value,
            install_candidates(config, name, "checkpoint", value),
            checkpoint_root_candidates,
            overwrite,
            dry_run,
        )


def report(name: str, config: dict, verify: bool) -> bool:
    source_states = []
    for value in source_entries(config):
        candidates = runtime_candidates(config, name, "source", value)
        installed = next((path for path in candidates if ready(path)), None)
        canonical = candidates[0]
        if installed is None:
            state = "missing" if not canonical.exists() else "placeholder"
            source_states.append(state)
            print(f"{name}: source={state} ({canonical})")
            continue
        state = "included" if installed == canonical else "included_alternative"
        source_states.append("included")
        print(f"{name}: source={state} ({installed})")
    source_ready = all(value == "included" for value in source_states)
    checkpoint_ready = []
    for canonical_checkpoint, expected in checkpoint_entries(config):
        relative_value = str(canonical_checkpoint.relative_to(PROJECT_ROOT))
        candidates = runtime_candidates(config, name, "checkpoint", relative_value)
        checkpoint = next((path for path in candidates if ready(path)), None)
        if checkpoint is None:
            canonical = candidates[0]
            state = "missing" if not canonical.exists() else "placeholder"
            checkpoint_ready.append(False)
            print(f"  checkpoint={state} ({canonical})")
            continue
        actual = sha256(checkpoint) if verify else "not_checked"
        valid = not verify or actual == expected
        checkpoint_ready.append(valid)
        print(
            f"  checkpoint={'ok' if valid else 'checksum_mismatch'} "
            f"({checkpoint})"
        )
    model_ready = source_ready and all(checkpoint_ready)
    print(f"  status={'ready' if model_ready else 'setup_required'}")
    if not model_ready and config.get("checkpoint_page"):
        print(f"  checkpoint source: {config['checkpoint_page']}")
    if not model_ready and name != "singlem":
        print(f"  instructions: {MODEL_ROOT / name / 'README.md'}")
    return model_ready


def main() -> None:
    args = parse_args()
    models = json.loads(MANIFEST.read_text(encoding="utf-8"))["models"]
    selected = args.models or list(models)
    unknown = sorted(set(selected) - set(models))
    if unknown:
        raise SystemExit(f"unknown models: {', '.join(unknown)}")
    if args.install:
        for name in selected:
            install_model(
                name,
                models[name],
                args.source_root,
                args.checkpoint_root,
                args.overwrite,
                args.dry_run,
            )
    checks = [report(name, models[name], args.verify) for name in selected]
    complete = all(checks)
    if not complete and not args.dry_run:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
