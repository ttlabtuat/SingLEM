from __future__ import annotations

import contextlib
import json
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


MODEL_ROOT = Path(__file__).resolve().parents[3] / "models" / "foundation"
DOWNSTREAM_ADAPTERS = Path(__file__).resolve().parent / "downstream_adapters.py"
MODEL_MANIFEST = MODEL_ROOT / "manifest.json"


def project_path(value: str) -> Path:
    return MODEL_ROOT.parents[1] / value


def _ready(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen = set()
    unique = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _relative_to_model_dir(model_name: str, value: str) -> str:
    path = Path(value)
    prefix = Path("models") / "foundation" / model_name
    try:
        return str(path.relative_to(prefix))
    except ValueError:
        return path.name


def _install_candidates(config: dict, model_name: str, kind: str, value: str) -> list[str]:
    configured = (
        config.get("install", {})
        .get(f"{kind}_candidates", {})
        .get(value)
    )
    if configured:
        return list(configured)
    return [_relative_to_model_dir(model_name, value)]


def artifact_candidates(model_name: str, config: dict, kind: str, value: str) -> list[Path]:
    candidates = [project_path(value)]
    model_root = MODEL_ROOT / model_name
    candidates.extend(
        model_root / candidate
        for candidate in _install_candidates(config, model_name, kind, value)
    )
    return _unique_paths(candidates)


def resolve_model_artifact(model_name: str, value: str, kind: str = "checkpoint") -> Path:
    config = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))["models"][model_name]
    candidates = artifact_candidates(model_name, config, kind, value)
    return next((path for path in candidates if _ready(path)), candidates[0])


def require_model_artifacts(model_name: str) -> None:
    if model_name == "singlem":
        project_root = MODEL_ROOT.parents[1]
        paths = [
            project_root / "SingLEM" / "model.py",
            project_root / "SingLEM" / "model_no_feature_embedding.py",
            project_root / "SingLEM" / "checkpoints" / "singlem_downstream_excluded.pt",
            project_root / "SingLEM" / "checkpoints" / "singlem_downstream_included.pt",
            project_root / "SingLEM" / "checkpoints" / "singlem_no_feature_embedding.pt",
        ]
        unavailable = [
            f"{'missing' if not path.exists() else 'placeholder'}: {path}"
            for path in paths
            if not path.exists() or path.stat().st_size == 0
        ]
        if unavailable:
            details = "\n  ".join(unavailable)
            raise RuntimeError(f"SingLEM artifacts are incomplete:\n  {details}")
        return
    config = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))["models"][model_name]
    required = [
        ("source", value)
        for value in config.get("source_paths", [config["source_path"]])
    ]
    required.extend(
        ("checkpoint", value) for value in config.get("checkpoint_paths", [])
    )
    if "checkpoint_path" in config:
        required.append(("checkpoint", config["checkpoint_path"]))
    unavailable = []
    for kind, value in required:
        candidates = artifact_candidates(model_name, config, kind, value)
        if any(_ready(path) for path in candidates):
            continue
        primary = candidates[0]
        state = "missing" if not primary.exists() else "placeholder"
        alternatives = ", ".join(str(path) for path in candidates[1:])
        if alternatives:
            unavailable.append(
                f"{state}: {primary} (also accepts: {alternatives})"
            )
        else:
            unavailable.append(f"{state}: {primary}")
    if unavailable:
        details = "\n  ".join(unavailable)
        raise RuntimeError(
            f"{model_name} is not installed. Replace its placeholder files as "
            f"described in {MODEL_ROOT / model_name / 'README.md'}, or run "
            f"`python models/foundation/setup_models.py --models {model_name} "
            f"--install --source_root /path/to/upstream_repo "
            f"--checkpoint_root /path/to/downloads --verify`:\n  {details}"
        )


@contextlib.contextmanager
def prepend_sys_path(path: str | Path):
    path = str(path)
    sys.path.insert(0, path)
    try:
        yield
    finally:
        try:
            sys.path.remove(path)
        except ValueError:
            pass


def purge_module_prefixes(prefixes: tuple[str, ...]) -> None:
    for module_name in list(sys.modules):
        if any(module_name == p or module_name.startswith(p + ".") for p in prefixes):
            sys.modules.pop(module_name, None)


def freeze_eval(model: nn.Module) -> nn.Module:
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def batched_array(x: np.ndarray, batch_size: int):
    for start in range(0, len(x), batch_size):
        yield x[start:start + batch_size]


def load_downstream_adapters_module():
    spec = importlib.util.spec_from_file_location("singlem_revision_downstream_adapters", DOWNSTREAM_ADAPTERS)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def install_benchmark_import_compat() -> None:
    module = load_downstream_adapters_module()
    for name in (
        "install_einops_compat",
        "install_timm_compat",
        "install_opt_einsum_compat",
        "install_transformers_compat",
        "install_wandb_compat",
        "install_mamba_ssm_compat",
        "install_rotary_embedding_compat",
        "install_torcheeg_compat",
    ):
        if hasattr(module, name):
            getattr(module, name)()


def install_bendr_import_stubs() -> None:
    if importlib.util.find_spec("parse") is None and "parse" not in sys.modules:
        sys.modules["parse"] = types.ModuleType("parse")
    if importlib.util.find_spec("dn3") is not None:
        return

    dn3 = types.ModuleType("dn3")
    trainable = types.ModuleType("dn3.trainable")
    processes = types.ModuleType("dn3.trainable.processes")
    models = types.ModuleType("dn3.trainable.models")
    layers = types.ModuleType("dn3.trainable.layers")
    utils = types.ModuleType("dn3.utils")

    class BaseProcess(nn.Module):
        pass

    class StandardClassification(BaseProcess):
        @staticmethod
        def _simple_accuracy(*_args, **_kwargs):
            return 0.0

    class Classifier(nn.Module):
        def __init__(self, *_args, **_kwargs):
            super().__init__()

    class StrideClassifier(Classifier):
        pass

    class Flatten(nn.Module):
        def forward(self, x):
            return x.reshape(x.shape[0], -1)

    class Permute(nn.Module):
        def __init__(self, dims):
            super().__init__()
            self.dims = dims

        def forward(self, x):
            return x.permute(*self.dims)

    class DN3ConfigException(Exception):
        pass

    processes.BaseProcess = BaseProcess
    processes.StandardClassification = StandardClassification
    models.Classifier = Classifier
    models.StrideClassifier = StrideClassifier
    layers.Flatten = Flatten
    layers.Permute = Permute
    utils.DN3ConfigException = DN3ConfigException
    dn3.trainable = trainable
    trainable.processes = processes
    trainable.models = models
    trainable.layers = layers
    dn3.utils = utils
    sys.modules.update({
        "dn3": dn3,
        "dn3.trainable": trainable,
        "dn3.trainable.processes": processes,
        "dn3.trainable.models": models,
        "dn3.trainable.layers": layers,
        "dn3.utils": utils,
    })


def install_linear_attention_compat() -> None:
    if importlib.util.find_spec("linear_attention_transformer") is not None or "linear_attention_transformer" in sys.modules:
        return

    mod = types.ModuleType("linear_attention_transformer")

    class _Attention(nn.Module):
        def __init__(self, dim: int, heads: int):
            super().__init__()
            self.heads = heads
            self.dim_head = dim // heads
            self.scale = self.dim_head ** -0.5
            self.to_q = nn.Linear(dim, dim, bias=False)
            self.to_k = nn.Linear(dim, dim, bias=False)
            self.to_v = nn.Linear(dim, dim, bias=False)
            self.to_out = nn.Linear(dim, dim)

        def forward(self, x):
            b, n, d = x.shape
            q = self.to_q(x).view(b, n, self.heads, self.dim_head).transpose(1, 2)
            k = self.to_k(x).view(b, n, self.heads, self.dim_head).transpose(1, 2)
            v = self.to_v(x).view(b, n, self.heads, self.dim_head).transpose(1, 2)
            attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            out = torch.matmul(attn.softmax(dim=-1), v)
            return self.to_out(out.transpose(1, 2).contiguous().view(b, n, d))

    class _FeedForwardInner(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.w1 = nn.Linear(dim, dim * 4)
            self.w2 = nn.Linear(dim * 4, dim)

        def forward(self, x):
            return self.w2(torch.nn.functional.gelu(self.w1(x)))

    class _FeedForward(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.fn = _FeedForwardInner(dim)

        def forward(self, x):
            return self.fn(x)

    class _PreNorm(nn.Module):
        def __init__(self, dim: int, fn: nn.Module):
            super().__init__()
            self.fn = fn
            self.norm = nn.LayerNorm(dim)

        def forward(self, x):
            return self.fn(self.norm(x))

    class _TransformerLayers(nn.Module):
        def __init__(self, dim: int, heads: int, depth: int):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.ModuleList([
                    _PreNorm(dim, _Attention(dim, heads)),
                    _PreNorm(dim, _FeedForward(dim)),
                ])
                for _ in range(depth)
            ])

        def forward(self, x):
            for attn, ff in self.layers:
                x = x + attn(x)
                x = x + ff(x)
            return x

    class LinearAttentionTransformer(nn.Module):
        def __init__(self, dim: int, heads: int, depth: int, **_kwargs):
            super().__init__()
            self.layers = _TransformerLayers(dim, heads, depth)

        def forward(self, x):
            return self.layers(x)

    mod.LinearAttentionTransformer = LinearAttentionTransformer
    sys.modules["linear_attention_transformer"] = mod
