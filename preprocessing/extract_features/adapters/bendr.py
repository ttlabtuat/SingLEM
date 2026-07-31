from __future__ import annotations

import numpy as np
import torch

from .common import MODEL_ROOT, batched_array, freeze_eval, install_bendr_import_stubs, prepend_sys_path, purge_module_prefixes, resolve_model_artifact


class BENDRAdapter:
    def __init__(self, context: dict, device: torch.device):
        self.device = device
        self.model_root = MODEL_ROOT / "bendr"
        self.encoder_path = resolve_model_artifact(
            "bendr", "models/foundation/bendr/encoder.pt"
        )
        self.contextualizer_path = resolve_model_artifact(
            "bendr", "models/foundation/bendr/contextualizer.pt"
        )
        self.encoder, self.contextualizer = self._load_model()

    def _load_model(self):
        install_bendr_import_stubs()
        purge_module_prefixes(("dn3_ext",))
        with prepend_sys_path(self.model_root):
            from dn3_ext import BENDRContextualizer, ConvEncoderBENDR

            encoder = ConvEncoderBENDR(in_features=20, encoder_h=512).to(self.device)
            contextualizer = BENDRContextualizer(in_features=512, layers=8, heads=8).to(self.device)
        encoder.load_state_dict(torch.load(self.encoder_path, map_location=self.device), strict=True)
        contextualizer.load_state_dict(torch.load(self.contextualizer_path, map_location=self.device), strict=True)
        return freeze_eval(encoder), freeze_eval(contextualizer)

    def extract(self, x: np.ndarray, batch_size: int) -> np.ndarray:
        feats = []
        with torch.no_grad():
            for batch in batched_array(x.astype(np.float32, copy=False), batch_size):
                xb = torch.from_numpy(batch).to(self.device)
                feats.append(self.contextualizer(self.encoder(xb))[:, :, -1].cpu().numpy())
        return np.concatenate(feats, axis=0).astype(np.float32, copy=False)

    def info(self) -> dict:
        return {"model_root": str(self.model_root), "encoder": str(self.encoder_path), "contextualizer": str(self.contextualizer_path), "expected_channels": 20, "feature_dim": 512}
