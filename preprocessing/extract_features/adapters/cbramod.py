from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .common import MODEL_ROOT, batched_array, freeze_eval, prepend_sys_path, purge_module_prefixes, resolve_model_artifact


class CBraModAdapter:
    def __init__(self, context: dict, device: torch.device):
        self.device = device
        self.model_root = MODEL_ROOT / "cbramod"
        self.checkpoint = resolve_model_artifact(
            "cbramod",
            "models/foundation/cbramod/pretrained_weights/pretrained_weights.pth",
        )
        self.model = self._load_model()

    def _load_model(self):
        purge_module_prefixes(("models",))
        with prepend_sys_path(self.model_root):
            from models.cbramod import CBraMod

            model = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)
        model.proj_out = nn.Identity()
        state = torch.load(self.checkpoint, map_location=self.device)
        model_state = model.state_dict()
        model_state.update({k: v for k, v in state.items() if k in model_state and tuple(v.shape) == tuple(model_state[k].shape)})
        model.load_state_dict(model_state, strict=True)
        return freeze_eval(model.to(self.device))

    def extract(self, x: np.ndarray, batch_size: int) -> np.ndarray:
        usable = (x.shape[-1] // 200) * 200
        x = x[:, :, :usable].astype(np.float32, copy=False)
        feats = []
        with torch.no_grad():
            for batch in batched_array(x, batch_size):
                xb = torch.from_numpy(batch).to(self.device).view(batch.shape[0], batch.shape[1], usable // 200, 200)
                feats.append(self.model(xb).reshape(batch.shape[0], -1).cpu().numpy())
        return np.concatenate(feats, axis=0).astype(np.float32, copy=False)

    def info(self) -> dict:
        return {"model_root": str(self.model_root), "checkpoint": str(self.checkpoint)}
