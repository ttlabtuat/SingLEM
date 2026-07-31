from __future__ import annotations

import importlib.util

import numpy as np
import torch

from .common import MODEL_ROOT, batched_array, load_downstream_adapters_module, resolve_model_artifact


CSBRAIN_TOPOLOGY = [
    (0, ["AF7", "AF3", "AFZ", "AF4", "AF8", "FP1", "FPZ", "FP2", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FT8", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6"]),
    (1, ["P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8"]),
    (2, ["T7", "T8", "T9", "T10", "TP7", "TP8"]),
    (3, ["PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "O1", "OZ", "O2", "IZ"]),
    (4, ["C5", "C3", "C1", "CZ", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6"]),
]


def _clean(name: str) -> str:
    return str(name).upper().replace("EEG ", "").replace(".", "").strip()


def _region(name: str) -> int:
    key = _clean(name).split("-")[0]
    if key.startswith(("FP", "AF", "F")):
        return 0
    if key.startswith(("PO", "O", "IZ")):
        return 3
    if key.startswith(("T", "FT", "TP")):
        return 2
    if key.startswith("P"):
        return 1
    if key.startswith(("C", "FC", "CP")):
        return 4
    return 0


def _csbrain_metadata(channel_names: list[str]) -> dict:
    cleaned = [_clean(name).split("-")[0] for name in channel_names]
    present = {}
    for i, name in enumerate(cleaned):
        present.setdefault(name, []).append(i)
    used, sorted_indices, sorted_regions = set(), [], []
    for region_id, ordered_names in CSBRAIN_TOPOLOGY:
        for name in ordered_names:
            for idx in present.get(name, []):
                if idx not in used:
                    sorted_indices.append(idx)
                    sorted_regions.append(region_id)
                    used.add(idx)
    for idx in sorted([i for i in range(len(channel_names)) if i not in used], key=lambda i: (_region(channel_names[i]), cleaned[i], i)):
        sorted_indices.append(idx)
        sorted_regions.append(_region(channel_names[idx]))
    return {"brain_regions": sorted_regions, "sorted_indices": sorted_indices, "sorted_channel_names": [channel_names[i] for i in sorted_indices]}


def _mirepnet_channels() -> list[str]:
    channel_file = MODEL_ROOT / "mirepnet" / "utils" / "channel_list.py"
    spec = importlib.util.spec_from_file_location("singlem_revision_mirepnet_channel_list", channel_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.use_channels_names)


def _model_cfg(name: str) -> dict:
    root = MODEL_ROOT
    def checkpoint(model_name: str, relative_path: str) -> str:
        return str(resolve_model_artifact(model_name, relative_path))

    cfgs = {
        "csbrain": {
            "name": "csbrain", "adapter_type": "csbrain", "profile_name": "CSBrain", "input_scale": 1.0,
            "fm_root": str(root / "csbrain"), "checkpoint": checkpoint("csbrain", "models/foundation/csbrain/CSBrain.pth"),
            "strict_load": False, "embed_dim": 200,
            "model_kwargs": {"in_dim": 200, "out_dim": 200, "d_model": 200, "dim_feedforward": 800, "seq_len": 30, "n_layer": 12, "nhead": 8},
        },
        "codebrain": {
            "name": "codebrain", "adapter_type": "codebrain", "profile_name": "CodeBrain", "input_scale": 1.0,
            "fm_root": str(root / "codebrain"), "checkpoint": checkpoint("codebrain", "models/foundation/codebrain/Checkpoints/CodeBrain.pth"),
            "strict_load": True, "strip_prefixes": ["module.", "backbone."], "embed_dim": 200,
            "model_kwargs": {"in_channels": 200, "res_channels": 200, "skip_channels": 200, "out_channels": 200, "num_res_layers": 8, "diffusion_step_embed_dim_in": 200, "diffusion_step_embed_dim_mid": 200, "diffusion_step_embed_dim_out": 200, "s4_lmax": 570, "s4_d_state": 64, "s4_dropout": 0.2, "s4_bidirectional": True, "s4_layernorm": True, "codebook_size_t": 4096, "codebook_size_f": 4096, "if_codebook": False},
        },
        "luna_large": {
            "name": "luna_large", "adapter_type": "luna", "profile_name": "LUNA", "input_scale": 1.0,
            "fm_root": str(root / "luna_large"), "checkpoint": checkpoint("luna_large", "models/foundation/luna_large/weights/LUNA_large.safetensors"),
            "strict_load": False, "strip_prefixes": ["module.", "model."], "embed_dim": 576, "input_dim": 0,
            "normalize_input": False, "use_native_classifier": False, "length_adjustment": "pad", "feature_pooling": "mean", "patch_size": 40,
            "model_kwargs": {"patch_size": 40, "embed_dim": 96, "num_heads": 2, "depth": 10, "num_queries": 6, "drop_path": 0.1, "num_classes": 3},
        },
        "mirepnet": {
            "name": "mirepnet", "adapter_type": "mirepnet", "profile_name": "MIRepNet", "input_scale": 1.0,
            "fm_root": str(root / "mirepnet"), "checkpoint": checkpoint("mirepnet", "models/foundation/mirepnet/weight/MIRepNet.pth"),
            "strict_load": False, "strip_prefixes": ["module.", "model."], "embed_dim": 256, "input_dim": 256,
            "mi_only": True, "apply_euclidean_alignment": False,
            "model_kwargs": {"emb_size": 256, "depth": 6, "n_classes": 3, "mask_ratio": 0.5, "pretrain": None, "pretrainmode": False},
        },
    }
    return cfgs[name]


class BenchmarkAdapter:
    def __init__(self, model_name: str, context: dict, device: torch.device):
        self.model_name = model_name
        self.context = context
        self.device = device
        self.model_cfg = _model_cfg(model_name)
        if self.model_cfg.get("mi_only") and context["task_type"] != "mi":
            raise ValueError(f"{model_name} is MI-only; skip {context['dataset']}")
        self.metadata = self._metadata()
        module = load_downstream_adapters_module()
        self.model, self.load_info = module.build_benchmark_model(self.model_cfg, {"head_type": "v14_mlp"}, self.metadata)
        self.model.to(device).eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def _metadata(self) -> dict:
        channels = _mirepnet_channels() if self.model_cfg["adapter_type"] == "mirepnet" else self.context["channel_names"]
        metadata = {
            "dataset": {
                "id": self.context["dataset"],
                "task_id": self.context["task_type"],
                "num_classes": int(self.context["num_classes"]),
                "label_values": list(range(int(self.context["num_classes"]))),
                "n_channels": int(self.context["sample_shape"][1]),
                "n_samples_per_trial": int(self.context["sample_shape"][2]),
                "sampling_rate_hz": int(round(float(self.context["sfreq"]))),
            },
            "channels": {"channel_names": channels, "channel_positions": None},
            "signal": {"unit_out": "model_ready", "normalization": {"type": "precomputed"}},
        }
        if self.model_cfg["adapter_type"] == "csbrain":
            metadata["csbrain"] = _csbrain_metadata(channels)
        return metadata

    def extract(self, x: np.ndarray, batch_size: int) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        patch = max(1, int(round(float(self.context["sfreq"]))))
        feats = []
        with torch.no_grad():
            for batch in batched_array(x, batch_size):
                if batch.ndim == 3 and batch.shape[-1] % patch == 0:
                    batch = batch.reshape(batch.shape[0], batch.shape[1], batch.shape[-1] // patch, patch)
                xb = torch.from_numpy(batch).to(self.device)
                feats.append(self.model.forward_features(xb).cpu().numpy())
        return np.concatenate(feats, axis=0).astype(np.float32, copy=False)

    def info(self) -> dict:
        return {"model": self.model_name, "model_root": self.model_cfg["fm_root"], "checkpoint": self.model_cfg.get("checkpoint"), "load_info": self.load_info}
