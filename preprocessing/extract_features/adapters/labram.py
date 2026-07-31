from __future__ import annotations

import ast

import numpy as np
import torch

from .common import MODEL_ROOT, batched_array, freeze_eval, install_benchmark_import_compat, prepend_sys_path, purge_module_prefixes, resolve_model_artifact


class LaBraMAdapter:
    def __init__(self, context: dict, device: torch.device):
        self.context = context
        self.device = device
        self.model_root = MODEL_ROOT / "labram"
        self.checkpoint = resolve_model_artifact(
            "labram",
            "models/foundation/labram/checkpoints/labram-base.pth",
        )
        self.model, self.input_chans = self._load_model()

    def _load_model(self):
        install_benchmark_import_compat()
        purge_module_prefixes(("modeling_finetune", "utils"))
        with prepend_sys_path(self.model_root):
            from modeling_finetune import NeuralTransformer

            model = NeuralTransformer(
                patch_size=200, embed_dim=200, depth=12, num_heads=10,
                mlp_ratio=4, qkv_bias=True, norm_layer=torch.nn.LayerNorm,
                init_values=0.1, EEG_size=200, use_mean_pooling=True, use_abs_pos_emb=True,
            ).to(self.device)
            checkpoint = torch.load(self.checkpoint, map_location="cpu")
            state = {k[len("student."):]: v for k, v in checkpoint["model"].items() if k.startswith("student.") and "lm_head" not in k}
            model.load_state_dict(state, strict=False)
        return freeze_eval(model), torch.tensor(self._input_chans(self.context["channel_names"]), dtype=torch.long, device=self.device)

    def _input_chans(self, names: list[str]) -> list[int]:
        tree = ast.parse((self.model_root / "utils.py").read_text())
        standard = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "standard_1020" for t in node.targets):
                standard = [str(v).upper() for v in ast.literal_eval(node.value)]
                break
        lookup = {name: i + 1 for i, name in enumerate(standard)}
        return [0] + [lookup[name.upper()] for name in names if name.upper() in lookup]

    def extract(self, x: np.ndarray, batch_size: int) -> np.ndarray:
        usable = (x.shape[-1] // 200) * 200
        x = x[:, :, :usable].astype(np.float32, copy=False)
        feats = []
        with torch.no_grad():
            for batch in batched_array(x, batch_size):
                xb = torch.from_numpy(batch).to(self.device).view(batch.shape[0], batch.shape[1], usable // 200, 200)
                out = self.model.forward_features(xb, input_chans=self.input_chans)
                feats.append(out.reshape(out.shape[0], -1).cpu().numpy())
        return np.concatenate(feats, axis=0).astype(np.float32, copy=False)

    def info(self) -> dict:
        return {"model_root": str(self.model_root), "checkpoint": str(self.checkpoint), "feature_dim": 200}
