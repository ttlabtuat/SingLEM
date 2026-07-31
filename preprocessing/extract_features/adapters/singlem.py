from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from .common import MODEL_ROOT, batched_array, freeze_eval, prepend_sys_path, purge_module_prefixes


class SingLEMAdapter:
    """Load a frozen SingLEM encoder and extract per-channel token features."""

    def __init__(self, context: dict, device: torch.device):
        self.device = device
        project_root = MODEL_ROOT.parents[1]
        self.model_root = project_root / "SingLEM"
        self.variant = context.get("singlem_variant") or "downstream_excluded"
        if self.variant == "default":
            self.variant = "downstream_excluded"
        checkpoints = {
            "downstream_excluded": "singlem_downstream_excluded.pt",
            "downstream_included": "singlem_downstream_included.pt",
            "no_feature_embedding": "singlem_no_feature_embedding.pt",
        }
        if self.variant not in checkpoints:
            raise ValueError(f"unknown SingLEM variant: {self.variant}")
        self.checkpoint = project_root / "SingLEM" / "checkpoints" / checkpoints[self.variant]
        self.checkpoint_sha256 = self._sha256(self.checkpoint)
        self.model = self._load_model()
        sfreq = context.get("sfreq")
        if sfreq and not np.isclose(float(sfreq), self.sample_rate_hz):
            raise ValueError(
                f"SingLEM requires {self.sample_rate_hz:g} Hz input, got {sfreq} Hz"
            )

    @staticmethod
    def _sha256(path: str | Path) -> str:
        """Return the SHA-256 digest of a checkpoint file."""
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _load_model(self):
        """Load a SingLEM checkpoint in all supported public formats.

        The final revised release uses a compact encoder payload with
        ``encoder_state_dict`` plus input metadata. Older raw state dictionaries
        are still accepted for local compatibility, but all input assumptions
        are validated before feature extraction.
        """
        purge_module_prefixes(("model", "model_no_feature_embedding"))
        with prepend_sys_path(self.model_root):
            if self.variant == "no_feature_embedding":
                from model_no_feature_embedding import Config, EEGEncoder
            else:
                from model import Config, EEGEncoder

            config = Config()
            payload = torch.load(
                self.checkpoint, map_location="cpu", weights_only=True
            )
            if isinstance(payload, dict) and "encoder_state_dict" in payload:
                input_keys = {
                    "sample_rate_hz",
                    "input_unit",
                    "input_scale",
                    "token_samples",
                    "maximum_sequence_tokens",
                }
                if not input_keys.issubset(payload):
                    raise ValueError("unsupported SingLEM checkpoint format")
                model_variant = payload.get("model_variant")
                if self.variant == "no_feature_embedding":
                    if model_variant != "no_feature_embedding":
                        raise ValueError(
                            "no_feature_embedding checkpoint is missing its model_variant tag"
                        )
                elif model_variant not in (None, self.variant):
                    raise ValueError(
                        "SingLEM checkpoint model_variant does not match the requested variant"
                    )
                model_config = payload.get("model_config", {})
                config_names = {
                    name
                    for name in vars(Config)
                    if not name.startswith("_") and name != "config"
                }
                if set(model_config) != config_names:
                    raise ValueError(
                        "SingLEM checkpoint model configuration is incomplete"
                    )
                for name, value in model_config.items():
                    setattr(config, name, value)
                state_dict = payload["encoder_state_dict"]
                input_spec = {
                    "sample_rate_hz": payload["sample_rate_hz"],
                    "unit_before_scale": payload["input_unit"],
                    "input_scale": payload["input_scale"],
                    "token_samples": payload["token_samples"],
                    "maximum_sequence_tokens": payload[
                        "maximum_sequence_tokens"
                    ],
                }
                self.checkpoint_format = "public_encoder_compact_v1"
            else:
                state_dict = payload
                input_spec = {
                    "sample_rate_hz": 128.0,
                    "unit_before_scale": "microvolt",
                    "input_scale": 0.01,
                    "token_samples": config.token_len,
                    "maximum_sequence_tokens": config.max_seq_len,
                }
                self.checkpoint_format = "legacy_state_dict"

            self.sample_rate_hz = float(input_spec.get("sample_rate_hz", 0))
            self.input_scale = float(input_spec.get("input_scale", 0))
            self.token_len = int(input_spec.get("token_samples", 0))
            self.max_seq_len = int(
                input_spec.get("maximum_sequence_tokens", 0)
            )
            if (
                input_spec.get("unit_before_scale") != "microvolt"
                or self.sample_rate_hz != 128.0
                or self.input_scale != 0.01
                or self.token_len != config.token_len
                or self.max_seq_len != config.max_seq_len
            ):
                raise ValueError("unsupported SingLEM input specification")

            config.mask_prob = 0.0
            model = EEGEncoder(config)
        model.load_state_dict(state_dict, strict=True)
        return freeze_eval(model.to(self.device))

    def _tokenize(self, x: np.ndarray) -> np.ndarray:
        """Convert ``[trials, channels, samples]`` into overlapping tokens."""
        if x.ndim != 3:
            raise ValueError(
                f"SingLEM input must have shape [trials, channels, samples], got {x.shape}"
            )
        if x.shape[-1] < self.token_len:
            raise ValueError(
                f"SingLEM input has {x.shape[-1]} samples; at least {self.token_len} are required"
            )
        starts = range(0, x.shape[-1] - self.token_len + 1, 96)
        tokens = np.stack(
            [x[:, :, start:start + self.token_len] for start in starts],
            axis=2,
        )
        if tokens.shape[2] > self.max_seq_len:
            raise ValueError(
                f"SingLEM input produces {tokens.shape[2]} tokens; maximum is {self.max_seq_len}"
            )
        return tokens

    def extract(self, x: np.ndarray, batch_size: int) -> np.ndarray:
        """Return frozen features shaped ``[trials, channels, tokens, 16]``."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not np.isfinite(x).all():
            raise ValueError("SingLEM input contains non-finite values")
        tokens = self._tokenize(x.astype(np.float32, copy=False))
        channel_features = []
        with torch.no_grad():
            for ch in range(tokens.shape[1]):
                parts = []
                ch_tokens = tokens[:, ch, :, :]
                for batch in batched_array(ch_tokens, batch_size):
                    xb = torch.from_numpy(np.ascontiguousarray(batch)).to(self.device)
                    feats, _, _ = self.model(xb)
                    parts.append(feats.cpu().numpy())
                channel_features.append(np.concatenate(parts, axis=0))
        return np.transpose(np.stack(channel_features, axis=0), (1, 0, 2, 3)).astype(np.float32, copy=False)

    def info(self) -> dict:
        """Return portable metadata recorded with each extracted feature file."""
        package_root = MODEL_ROOT.parents[1]
        return {
            "model_root": str(self.model_root.relative_to(package_root)),
            "checkpoint": str(self.checkpoint.relative_to(package_root)),
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_format": self.checkpoint_format,
            "singlem_variant": self.variant,
            "feature_dim": 16,
            "sample_rate_hz": self.sample_rate_hz,
            "unit_before_scale": "microvolt",
            "input_scale": self.input_scale,
            "token_len": self.token_len,
            "stride": 96,
            "max_seq_len": self.max_seq_len,
        }
