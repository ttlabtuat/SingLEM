from __future__ import annotations

import numpy as np
import torch

from .common import MODEL_ROOT, batched_array, freeze_eval, install_linear_attention_compat, prepend_sys_path, purge_module_prefixes, resolve_model_artifact


class BIOTAdapter:
    def __init__(self, context: dict, device: torch.device):
        self.device = device
        self.model_root = MODEL_ROOT / "biot"
        self.checkpoint = resolve_model_artifact(
            "biot",
            "models/foundation/biot/EEG-six-datasets-18-channels.ckpt",
        )
        self.model = self._load_model()

    def _load_model(self):
        install_linear_attention_compat()
        purge_module_prefixes(("biot",))
        source_root = (
            self.model_root / "models"
            if (self.model_root / "models" / "biot.py").exists()
            else self.model_root / "model"
            if (self.model_root / "model" / "biot.py").exists()
            else self.model_root
        )
        with prepend_sys_path(source_root):
            from biot import BIOTEncoder

            model = BIOTEncoder(emb_size=256, heads=8, depth=4, n_channels=18, n_fft=200, hop_length=100)
        model.load_state_dict(torch.load(self.checkpoint, map_location=self.device), strict=True)
        return freeze_eval(model.to(self.device))

    def extract(self, x: np.ndarray, batch_size: int) -> np.ndarray:
        feats = []
        with torch.no_grad():
            for batch in batched_array(x.astype(np.float32, copy=False), batch_size):
                xb = torch.from_numpy(batch).to(self.device)
                feats.append(self.model(xb).reshape(batch.shape[0], -1).cpu().numpy())
        return np.concatenate(feats, axis=0).astype(np.float32, copy=False)

    def info(self) -> dict:
        return {"model_root": str(self.model_root), "checkpoint": str(self.checkpoint), "expected_channels": 18}
