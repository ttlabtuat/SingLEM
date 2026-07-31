"""Foundation-model adapters for WBCIC 3C fine-tuning."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import math
import pickle
import struct
import sys
import types
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint


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


def make_activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "gelu":
        return nn.GELU()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


def purge_module_prefixes(prefixes: Iterable[str]) -> None:
    for module_name in list(sys.modules):
        for prefix in prefixes:
            if module_name == prefix or module_name.startswith(prefix + "."):
                sys.modules.pop(module_name, None)
                break


def install_einops_compat() -> None:
    if importlib.util.find_spec("einops") is not None:
        return

    einops_mod = types.ModuleType("einops")

    def _normalize(pattern: str) -> str:
        return " ".join(str(pattern).strip().split())

    def rearrange(tensor, pattern=None, **axes):
        if isinstance(tensor, str):
            tensor, pattern = pattern, tensor
        pattern = _normalize(pattern)
        if pattern == "b c l d -> b (c l) d":
            b, c, l, d = tensor.shape
            return tensor.contiguous().view(b, c * l, d)
        if pattern == "b (c l) d -> b c l d":
            b, cl, d = tensor.shape
            l = int(axes["l"])
            return tensor.contiguous().view(b, cl // l, l, d)
        if pattern == "b c l d -> b d c l":
            return tensor.permute(0, 3, 1, 2).contiguous()
        if pattern == "b d c l -> b (c l) d":
            b, d, c, l = tensor.shape
            return tensor.permute(0, 2, 3, 1).contiguous().view(b, c * l, d)
        if pattern == "b d c l -> b c l d":
            return tensor.permute(0, 2, 3, 1).contiguous()
        if pattern == "b n (h d) -> b h n d":
            b, n, hd = tensor.shape
            h = int(axes["h"])
            return tensor.contiguous().view(b, n, h, hd // h).permute(0, 2, 1, 3).contiguous()
        if pattern == "b h n d -> b n (h d)":
            b, h, n, d = tensor.shape
            return tensor.permute(0, 2, 1, 3).contiguous().view(b, n, h * d)
        if pattern == "b c h e -> b (c h) e":
            b, c, h, e = tensor.shape
            return tensor.contiguous().view(b, c * h, e)
        if pattern == "b (c h) e -> b c h e":
            b, ch, e = tensor.shape
            c = int(axes["c"])
            h = int(axes["h"])
            return tensor.contiguous().view(b, c, h, e)
        if pattern == "b c t e -> b (c t) e":
            b, c, t, e = tensor.shape
            return tensor.contiguous().view(b, c * t, e)
        if pattern == "b e (h) (w) -> b (h w) e":
            b, e, h, w = tensor.shape
            return tensor.permute(0, 2, 3, 1).contiguous().view(b, h * w, e)
        if pattern == "B C (S P) -> B (C S) P":
            b, c, t = tensor.shape
            p = int(axes["P"])
            s = t // p
            return tensor.contiguous().view(b, c, s, p).view(b, c * s, p)
        if pattern == "B E CS D -> B CS (D E)":
            b, e, cs, d = tensor.shape
            return tensor.permute(0, 2, 3, 1).contiguous().view(b, cs, d * e)
        if pattern == "B C t D -> B (C t) D":
            b, c, t, d = tensor.shape
            return tensor.contiguous().view(b, c * t, d)
        if pattern == "B H N D -> B N (H D)":
            b, h, n, d = tensor.shape
            return tensor.permute(0, 2, 1, 3).contiguous().view(b, n, h * d)
        if pattern == "B (C t) D -> (B t) C D":
            b, ct, d = tensor.shape
            c = int(axes["C"])
            t = ct // c
            return tensor.contiguous().view(b, c, t, d).permute(0, 2, 1, 3).contiguous().view(b * t, c, d)
        if pattern == "(B t) Q D -> B t (Q D)":
            bt, q, d = tensor.shape
            b = int(axes["B"])
            return tensor.contiguous().view(b, bt // b, q * d)
        if pattern == "B N A T -> B (N A) T":
            b, n, a, t = tensor.shape
            return tensor.contiguous().view(b, n * a, t)
        if pattern == "B C NA T -> B NA (T C)":
            b, c, na, t = tensor.shape
            return tensor.permute(0, 2, 3, 1).contiguous().view(b, na, t * c)
        if pattern == "b c s p -> b p (c s)":
            b, c, s, p = tensor.shape
            return tensor.permute(0, 3, 1, 2).contiguous().view(b, p, c * s)
        if pattern == "b p (c s) -> b c s p":
            b, p, _ = tensor.shape
            c = int(axes["c"])
            s = int(axes["s"])
            return tensor.contiguous().view(b, p, c, s).permute(0, 2, 3, 1).contiguous()
        if pattern in {"b c l -> b l c", "b l c -> b c l"}:
            return tensor.transpose(1, 2).contiguous()
        if pattern == "b d ... -> b ... d":
            return tensor.movedim(1, -1).contiguous()
        if pattern == "b ... d -> b d ...":
            return tensor.movedim(-1, 1).contiguous()
        if pattern == "b d ... -> b d (...)":
            return tensor.contiguous().view(tensor.shape[0], tensor.shape[1], -1)
        if pattern in {"b ... d -> b (...)d", "b ... d -> b (...) d"}:
            return tensor.contiguous().view(tensor.shape[0], -1, tensor.shape[-1])
        if pattern == "(s c) h l -> s c h l":
            s = int(axes["s"])
            sc, h, l = tensor.shape
            return tensor.contiguous().view(s, sc // s, h, l)
        if pattern == "... c h l -> ... (c h) l":
            *leading, c, h, l = tensor.shape
            return tensor.contiguous().view(*leading, c * h, l)
        if pattern == "... h n -> ... (h n)":
            *leading, h, n = tensor.shape
            return tensor.contiguous().view(*leading, h * n)
        raise NotImplementedError(f"Local einops compatibility shim does not support pattern: {pattern}")

    def repeat(tensor, pattern=None, **axes):
        if isinstance(tensor, str):
            tensor, pattern = pattern, tensor
        pattern = _normalize(pattern)
        if pattern == "n -> n d":
            return tensor.unsqueeze(-1).expand(-1, int(axes["d"]))
        raise NotImplementedError(f"Local einops compatibility shim does not support repeat pattern: {pattern}")

    def reduce(tensor, pattern=None, reduction=None, **axes):
        raise NotImplementedError("The local einops reduce shim is only for unused imports.")

    einops_mod.rearrange = rearrange
    einops_mod.repeat = repeat
    einops_mod.reduce = reduce
    layers_mod = types.ModuleType("einops.layers")
    torch_layers_mod = types.ModuleType("einops.layers.torch")

    class _RearrangeLayer(nn.Module):
        def __init__(self, pattern: str, **axes) -> None:
            super().__init__()
            self.pattern = pattern
            self.axes = axes

        def forward(self, x):
            return rearrange(x, self.pattern, **self.axes)

    class _ReduceLayer(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

        def forward(self, x):
            raise RuntimeError("The local einops Reduce shim is only for unused imports.")

    torch_layers_mod.Rearrange = _RearrangeLayer
    torch_layers_mod.Reduce = _ReduceLayer
    sys.modules["einops"] = einops_mod
    sys.modules["einops.layers"] = layers_mod
    sys.modules["einops.layers.torch"] = torch_layers_mod


def install_opt_einsum_compat() -> None:
    if importlib.util.find_spec("opt_einsum") is not None:
        return
    opt_einsum_mod = types.ModuleType("opt_einsum")
    opt_einsum_mod.contract = torch.einsum
    sys.modules["opt_einsum"] = opt_einsum_mod


def _has_module(module_name: str) -> bool:
    if module_name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(module_name) is not None
    except ValueError:
        return False


def install_transformers_compat() -> None:
    if _has_module("transformers"):
        return

    transformers_mod = types.ModuleType("transformers")

    class PretrainedConfig:
        model_type = ""

        def __init__(self, **kwargs) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    class PreTrainedModel(nn.Module):
        config_class = PretrainedConfig

        def __init__(self, config: PretrainedConfig | None = None, *args, **kwargs) -> None:
            super().__init__()
            self.config = config

        def post_init(self) -> None:
            return None

    class AutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise RuntimeError("transformers is not installed; local benchmark adapters do not support AutoModel.from_pretrained")

    transformers_mod.AutoModel = AutoModel
    transformers_mod.PretrainedConfig = PretrainedConfig
    transformers_mod.PreTrainedModel = PreTrainedModel
    sys.modules["transformers"] = transformers_mod


def install_wandb_compat() -> None:
    if _has_module("wandb"):
        return
    wandb_mod = types.ModuleType("wandb")
    wandb_mod.init = lambda *args, **kwargs: None
    wandb_mod.log = lambda *args, **kwargs: None
    wandb_mod.finish = lambda *args, **kwargs: None
    sys.modules["wandb"] = wandb_mod


def install_torcheeg_compat() -> None:
    if _has_module("torcheeg"):
        return
    torcheeg_mod = types.ModuleType("torcheeg")
    datasets_mod = types.ModuleType("torcheeg.datasets")
    constants_mod = types.ModuleType("torcheeg.datasets.constants")
    constants_mod.SEED_CHANNEL_LIST = [
        "FP1", "FPZ", "FP2", "AF7", "AF3", "AFZ", "AF4", "AF8",
        "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
        "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
        "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
        "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
        "P9", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "P10",
        "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "O1", "OZ", "O2",
    ]
    datasets_mod.constants = constants_mod
    torcheeg_mod.datasets = datasets_mod
    sys.modules["torcheeg"] = torcheeg_mod
    sys.modules["torcheeg.datasets"] = datasets_mod
    sys.modules["torcheeg.datasets.constants"] = constants_mod


def install_rotary_embedding_compat() -> None:
    if _has_module("rotary_embedding_torch"):
        return
    rotary_mod = types.ModuleType("rotary_embedding_torch")

    class RotaryEmbedding(nn.Module):
        def __init__(self, dim: int, learned_freq: bool = False, **kwargs) -> None:
            super().__init__()
            self.dim = int(dim)
            freqs = torch.arange(0, self.dim, 2, dtype=torch.float32) / max(self.dim, 1)
            freqs = 1.0 / (10000 ** freqs)
            self.register_buffer("freqs", freqs, persistent=True)

        def rotate_queries_or_keys(self, x: torch.Tensor) -> torch.Tensor:
            seq_len = x.shape[-2]
            half = x.shape[-1] // 2
            if half == 0:
                return x
            freqs = self.freqs[:half].to(device=x.device, dtype=x.dtype)
            positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
            angles = positions[:, None] * freqs[None, :]
            cos = angles.cos().view(1, 1, seq_len, half)
            sin = angles.sin().view(1, 1, seq_len, half)
            x1 = x[..., :half]
            x2 = x[..., half:half * 2]
            rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
            if x.shape[-1] > half * 2:
                rotated = torch.cat([rotated, x[..., half * 2:]], dim=-1)
            return rotated

    rotary_mod.RotaryEmbedding = RotaryEmbedding
    sys.modules["rotary_embedding_torch"] = rotary_mod


def install_mamba_ssm_compat() -> None:
    if "mamba_ssm.modules.mamba2" in sys.modules:
        return

    from dataclasses import dataclass, field
    import torch.nn.functional as F

    mamba_mod = types.ModuleType("mamba_ssm")
    models_mod = types.ModuleType("mamba_ssm.models")
    modules_mod = types.ModuleType("mamba_ssm.modules")
    ops_mod = types.ModuleType("mamba_ssm.ops")
    triton_mod = types.ModuleType("mamba_ssm.ops.triton")
    utils_mod = types.ModuleType("mamba_ssm.utils")

    config_mod = types.ModuleType("mamba_ssm.models.config_mamba")

    @dataclass
    class MambaConfig:
        d_model: int = 200
        d_intermediate: int = 0
        n_layer: int = 12
        ssm_cfg: dict = field(default_factory=dict)
        attn_layer_idx: list = field(default_factory=list)
        attn_cfg: dict = field(default_factory=dict)
        rms_norm: bool = True
        residual_in_fp32: bool = True
        fused_add_norm: bool = True

    config_mod.MambaConfig = MambaConfig

    layer_norm_mod = types.ModuleType("mamba_ssm.ops.triton.layer_norm")

    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-5, **kwargs) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(int(dim), **{k: v for k, v in kwargs.items() if k in {"device", "dtype"}}))
            self.register_parameter("bias", None)
            self.eps = float(eps)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return (y.to(dtype=x.dtype) * self.weight.to(dtype=x.dtype))

    def rms_norm_fn(x, weight, bias=None, residual=None, prenorm=False, residual_in_fp32=False, eps=1e-5, **kwargs):
        if residual is not None:
            x = x + residual
        normed = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
        out = normed.to(dtype=x.dtype) * weight.to(dtype=x.dtype)
        if bias is not None:
            out = out + bias.to(dtype=x.dtype)
        return (out, x.float() if residual_in_fp32 else x) if prenorm else out

    def layer_norm_fn(x, weight, bias=None, residual=None, prenorm=False, residual_in_fp32=False, eps=1e-5, is_rms_norm=False, **kwargs):
        if is_rms_norm:
            return rms_norm_fn(x, weight, bias, residual=residual, prenorm=prenorm, residual_in_fp32=residual_in_fp32, eps=eps)
        if residual is not None:
            x = x + residual
        out = F.layer_norm(x.to(dtype=weight.dtype), x.shape[-1:], weight=weight, bias=bias, eps=eps).to(dtype=x.dtype)
        return (out, x.float() if residual_in_fp32 else x) if prenorm else out

    layer_norm_mod.RMSNorm = RMSNorm
    layer_norm_mod.layer_norm_fn = layer_norm_fn
    layer_norm_mod.rms_norm_fn = rms_norm_fn

    mamba2_mod = types.ModuleType("mamba_ssm.modules.mamba2")

    class Mamba2(nn.Module):
        def __init__(
            self,
            d_model: int,
            d_state: int = 64,
            d_conv: int = 4,
            expand: int = 2,
            headdim: int = 50,
            ngroups: int = 1,
            layer_idx: int | None = None,
            bias: bool = False,
            conv_bias: bool = True,
            device=None,
            dtype=None,
            **kwargs,
        ) -> None:
            super().__init__()
            factory_kwargs = {"device": device, "dtype": dtype}
            self.d_model = int(d_model)
            self.d_state = int(d_state)
            self.d_conv = int(d_conv)
            self.expand = int(expand)
            self.d_inner = self.expand * self.d_model
            self.headdim = int(headdim)
            self.nheads = self.d_inner // self.headdim
            self.ngroups = int(ngroups)
            self.layer_idx = layer_idx
            d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
            conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
            self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
            self.conv1d = nn.Conv1d(conv_dim, conv_dim, kernel_size=self.d_conv, groups=conv_dim, padding=self.d_conv - 1, bias=conv_bias, **factory_kwargs)
            self.norm = RMSNorm(self.d_inner, eps=1e-5, **factory_kwargs)
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
            self.dt_bias = nn.Parameter(torch.zeros(self.nheads, **factory_kwargs))
            self.A_log = nn.Parameter(torch.zeros(self.nheads, **factory_kwargs))
            self.D = nn.Parameter(torch.ones(self.nheads, **factory_kwargs))

        def forward(self, u: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            batch, length, _ = u.shape
            zxbcdt = self.in_proj(u)
            z = zxbcdt[..., :self.d_inner]
            xbc = zxbcdt[..., self.d_inner:self.d_inner + self.d_inner + 2 * self.ngroups * self.d_state]
            xbc = self.conv1d(xbc.transpose(1, 2))[..., :length].transpose(1, 2)
            x = F.silu(xbc[..., :self.d_inner])
            x = self.norm(x)
            x = x * F.silu(z)
            return self.out_proj(x)

    mamba2_mod.Mamba2 = Mamba2

    mamba_simple_mod = types.ModuleType("mamba_ssm.modules.mamba_simple")
    mamba_simple_mod.Mamba = Mamba2

    mha_mod = types.ModuleType("mamba_ssm.modules.mha")

    class MHA(nn.Module):
        def __init__(self, d_model: int, layer_idx: int | None = None, **kwargs) -> None:
            super().__init__()
            self.proj = nn.Linear(int(d_model), int(d_model), bias=False)

        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            return self.proj(x)

    mha_mod.MHA = MHA

    mlp_mod = types.ModuleType("mamba_ssm.modules.mlp")

    class GatedMLP(nn.Module):
        def __init__(self, d_model: int, hidden_features: int, out_features: int, **kwargs) -> None:
            super().__init__()
            self.fc1 = nn.Linear(int(d_model), int(hidden_features), bias=False)
            self.fc2 = nn.Linear(int(hidden_features), int(out_features), bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(F.silu(self.fc1(x)))

    mlp_mod.GatedMLP = GatedMLP

    block_mod = types.ModuleType("mamba_ssm.modules.block")

    class Block(nn.Module):
        def __init__(self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False) -> None:
            super().__init__()
            self.residual_in_fp32 = bool(residual_in_fp32)
            self.fused_add_norm = bool(fused_add_norm)
            self.norm = norm_cls(dim)
            self.mixer = mixer_cls(dim)
            self.mlp = None if mlp_cls is nn.Identity else mlp_cls(dim)
            if self.mlp is not None:
                self.norm2 = norm_cls(dim)

        def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor | None = None, inference_params=None, **mixer_kwargs):
            residual = hidden_states if residual is None else hidden_states + residual
            hidden_states = self.norm(residual)
            hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)
            if self.mlp is not None:
                residual = hidden_states + residual
                hidden_states = self.mlp(self.norm2(residual))
            return hidden_states, residual.float() if self.residual_in_fp32 else residual

        def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
            return {}

    block_mod.Block = Block

    generation_mod = types.ModuleType("mamba_ssm.utils.generation")
    generation_mod.GenerationMixin = object
    hf_mod = types.ModuleType("mamba_ssm.utils.hf")
    hf_mod.load_config_hf = lambda *args, **kwargs: {}
    hf_mod.load_state_dict_hf = lambda *args, **kwargs: {}

    sys.modules.update({
        "mamba_ssm": mamba_mod,
        "mamba_ssm.models": models_mod,
        "mamba_ssm.models.config_mamba": config_mod,
        "mamba_ssm.modules": modules_mod,
        "mamba_ssm.modules.mamba_simple": mamba_simple_mod,
        "mamba_ssm.modules.mamba2": mamba2_mod,
        "mamba_ssm.modules.mha": mha_mod,
        "mamba_ssm.modules.mlp": mlp_mod,
        "mamba_ssm.modules.block": block_mod,
        "mamba_ssm.ops": ops_mod,
        "mamba_ssm.ops.triton": triton_mod,
        "mamba_ssm.ops.triton.layer_norm": layer_norm_mod,
        "mamba_ssm.utils": utils_mod,
        "mamba_ssm.utils.generation": generation_mod,
        "mamba_ssm.utils.hf": hf_mod,
    })


def install_timm_compat() -> None:
    if _has_module("timm"):
        return

    timm_mod = types.ModuleType("timm")
    models_mod = types.ModuleType("timm.models")
    layers_mod = types.ModuleType("timm.models.layers")
    registry_mod = types.ModuleType("timm.models.registry")
    vision_transformer_mod = types.ModuleType("timm.models.vision_transformer")

    def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
        if drop_prob == 0.0 or not training:
            return x
        keep_prob = 1.0 - float(drop_prob)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def to_2tuple(x):
        return x if isinstance(x, tuple) else (x, x)

    def trunc_normal_(tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0):
        return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

    def register_model(fn):
        return fn

    class DropPath(nn.Module):
        def __init__(self, drop_prob: float = 0.0) -> None:
            super().__init__()
            self.drop_prob = float(drop_prob)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return drop_path(x, self.drop_prob, self.training)

    class Mlp(nn.Module):
        def __init__(
            self,
            in_features: int,
            hidden_features: int | None = None,
            out_features: int | None = None,
            act_layer=nn.GELU,
            drop: float = 0.0,
            norm_layer=None,
            **kwargs,
        ) -> None:
            super().__init__()
            hidden_features = int(hidden_features or in_features)
            out_features = int(out_features or in_features)
            self.fc1 = nn.Linear(int(in_features), hidden_features)
            self.act = act_layer()
            self.drop1 = nn.Dropout(float(drop))
            self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.drop2 = nn.Dropout(float(drop))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop1(x)
            x = self.norm(x)
            x = self.fc2(x)
            return self.drop2(x)

    class _CompatAttention(nn.Module):
        def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True) -> None:
            super().__init__()
            self.fused_attn = False
            self.attn_scores = None
            self.num_heads = int(num_heads)
            self.head_dim = int(dim) // self.num_heads
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)

        def enable_gradient_hooks(self, enabled: bool = True) -> None:
            return None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, n, d = x.shape
            qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            out = F.scaled_dot_product_attention(q, k, v)
            out = out.transpose(1, 2).reshape(b, n, d)
            self.attn_scores = None
            return self.proj(out)

    class _CompatMlp(nn.Module):
        def __init__(self, dim: int, hidden_features: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(dim, hidden_features)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(hidden_features, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(self.act(self.fc1(x)))

    class _CompatBlock(nn.Module):
        def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            norm_layer=nn.LayerNorm,
        ) -> None:
            super().__init__()
            self.norm1 = norm_layer(dim)
            self.attn = _CompatAttention(dim, num_heads, qkv_bias=qkv_bias)
            self.norm2 = norm_layer(dim)
            hidden = int(dim * float(mlp_ratio))
            self.mlp = _CompatMlp(dim, hidden)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x

    class VisionTransformer(nn.Module):
        def __init__(
            self,
            *,
            patch_size: int = 16,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            norm_layer=nn.LayerNorm,
            num_classes: int = 1000,
            drop_rate: float = 0.0,
            **kwargs,
        ) -> None:
            super().__init__()
            self.patch_size = int(patch_size)
            self.embed_dim = int(embed_dim)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.pos_drop = nn.Dropout(float(drop_rate))
            self.blocks = nn.ModuleList(
                _CompatBlock(self.embed_dim, int(num_heads), mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer)
                for _ in range(int(depth))
            )
            self.norm = norm_layer(self.embed_dim)
            self.head_drop = nn.Dropout(float(drop_rate))
            self.head = nn.Identity() if int(num_classes) <= 0 else nn.Linear(self.embed_dim, int(num_classes))
            trunc_normal_(self.cls_token, std=0.02)

        def forward_features(self, x: torch.Tensor) -> torch.Tensor:
            for block in self.blocks:
                x = block(x)
            x = self.norm(x)
            return x[:, 0]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.head_drop(self.forward_features(x)))

    layers_mod.drop_path = drop_path
    layers_mod.DropPath = DropPath
    layers_mod.Mlp = Mlp
    layers_mod.to_2tuple = to_2tuple
    layers_mod.trunc_normal_ = trunc_normal_
    registry_mod.register_model = register_model
    vision_transformer_mod.VisionTransformer = VisionTransformer
    models_mod.layers = layers_mod
    models_mod.registry = registry_mod
    models_mod.vision_transformer = vision_transformer_mod
    timm_mod.models = models_mod
    sys.modules.update({
        "timm": timm_mod,
        "timm.layers": layers_mod,
        "timm.models": models_mod,
        "timm.models.layers": layers_mod,
        "timm.models.registry": registry_mod,
        "timm.models.vision_transformer": vision_transformer_mod,
    })


class V14MLPHead(nn.Module):
    """Two-layer downstream MLP matching the v14 WBCIC fair config."""

    def __init__(self, input_dim: int | None, num_classes: int, *, hidden_dim: int = 256, dropout: float = 0.2, activation: str = "elu") -> None:
        super().__init__()
        first = nn.LazyLinear(int(hidden_dim)) if input_dim is None or int(input_dim) <= 0 else nn.Linear(int(input_dim), int(hidden_dim))
        self.net = nn.Sequential(
            first,
            make_activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)
        return self.net(x)


def _state_dict_from_checkpoint(obj: Any, *, filter_name: str = "") -> dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        if filter_name == "labram_student":
            source = obj.get("model", obj)
            return OrderedDict((k[8:], v) for k, v in source.items() if isinstance(k, str) and k.startswith("student."))
        for key in ("model", "module", "state_dict", "teacher", "student"):
            value = obj.get(key)
            if isinstance(value, dict):
                return value
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(obj)!r}")
    return obj


def _load_safetensors_state_dict(path: Path) -> OrderedDict[str, torch.Tensor]:
    dtype_map = {
        "F64": torch.float64,
        "F32": torch.float32,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "I64": torch.int64,
        "I32": torch.int32,
        "I16": torch.int16,
        "I8": torch.int8,
        "U8": torch.uint8,
        "BOOL": torch.bool,
    }
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        data = bytearray(f.read())
    state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        dtype_name = str(meta["dtype"])
        if dtype_name not in dtype_map:
            raise ValueError(f"Unsupported safetensors dtype {dtype_name!r} in {path}")
        shape = tuple(int(x) for x in meta["shape"])
        start, end = (int(x) for x in meta["data_offsets"])
        count = math.prod(shape) if shape else 1
        tensor = torch.frombuffer(data, dtype=dtype_map[dtype_name], count=count, offset=start).reshape(shape)
        expected_bytes = tensor.element_size() * count
        if end - start != expected_bytes:
            raise ValueError(f"Invalid safetensors byte span for {key}: got {end - start}, expected {expected_bytes}")
        state[key] = tensor
    return state


def _strip_prefixes(state: dict[str, torch.Tensor], prefixes: Iterable[str]) -> OrderedDict[str, torch.Tensor]:
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state.items():
        new_key = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        out[new_key] = value
    return out


def load_checkpoint_state_dict(checkpoint_path: str | Path, *, filter_name: str = "") -> dict[str, torch.Tensor]:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix == ".safetensors":
        return _load_safetensors_state_dict(checkpoint_path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return _state_dict_from_checkpoint(checkpoint, filter_name=filter_name)


def load_checkpoint_into(
    module: nn.Module,
    checkpoint_path: str | Path,
    *,
    filter_name: str = "",
    strip_prefixes: Iterable[str] = ("module.", "backbone."),
    strict: bool = False,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    raw_state = load_checkpoint_state_dict(checkpoint_path, filter_name=filter_name)
    state = _strip_prefixes(raw_state, strip_prefixes)
    model_state = module.state_dict()
    matching = OrderedDict((k, v) for k, v in state.items() if k in model_state and tuple(v.shape) == tuple(model_state[k].shape))
    missing_or_mismatch = sorted(k for k in state.keys() if k not in matching)
    if strict:
        module.load_state_dict(state, strict=True)
    else:
        model_state.update(matching)
        module.load_state_dict(model_state, strict=True)
    return {
        "checkpoint": str(checkpoint_path),
        "loaded_keys": len(matching),
        "skipped_keys": len(missing_or_mismatch),
    }


def _load_reve_cls_query_token(model_cfg: dict[str, Any], embed_dim: int) -> tuple[torch.Tensor, bool]:
    checkpoint = model_cfg.get("checkpoint")
    init = torch.randn(1, 1, embed_dim)
    if not checkpoint:
        return init, False
    raw_state = load_checkpoint_state_dict(checkpoint)
    state = _strip_prefixes(raw_state, tuple(model_cfg.get("strip_prefixes", ["module.", "model."])))
    token = state.get("cls_query_token")
    if token is None or tuple(token.shape) != tuple(init.shape):
        return init, False
    return token.detach().float().clone(), True


def _metadata_channel_names(metadata: dict[str, Any]) -> list[str]:
    names = metadata.get("channels", {}).get("channel_names")
    if not names:
        raise ValueError("metadata.json must include channels.channel_names for this adapter")
    return [str(name) for name in names]


def _clean_channel_name(name: str) -> str:
    return str(name).upper().replace("EEG ", "").strip()


def _channel_candidates(name: str, aliases: dict[str, Any] | None = None) -> list[str]:
    cleaned = _clean_channel_name(name)
    candidates = [cleaned]
    alias_lookup = {_clean_channel_name(k): v for k, v in (aliases or {}).items()}
    for key in [cleaned, cleaned.split("-")[0] if "-" in cleaned else None]:
        if key is None:
            continue
        if key != cleaned:
            candidates.append(key)
        alias_value = alias_lookup.get(key)
        if alias_value is not None:
            values = alias_value if isinstance(alias_value, list) else [alias_value]
            candidates.extend(_clean_channel_name(value) for value in values)
    return list(dict.fromkeys(candidates))


def _metadata_channel_positions(metadata: dict[str, Any]) -> torch.Tensor | None:
    positions = metadata.get("channels", {}).get("channel_positions")
    if positions is None:
        return None
    arr = torch.tensor(positions, dtype=torch.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return None
    return arr.unsqueeze(0)


def _load_reve_positions(metadata: dict[str, Any], positions_path: str | Path, channel_aliases: dict[str, Any] | None = None) -> torch.Tensor:
    with Path(positions_path).open("r", encoding="utf-8") as f:
        bank = json.load(f)
    lookup = {str(name).upper(): value for name, value in bank.items()}
    coords = []
    missing = []
    for name in _metadata_channel_names(metadata):
        value = None
        for candidate in _channel_candidates(name, channel_aliases):
            value = lookup.get(candidate)
            if value is not None:
                break
        if value is None:
            missing.append(name)
        else:
            coords.append(value)
    if missing:
        raise ValueError(f"REVE positions are missing channels: {missing}")
    return torch.tensor(coords, dtype=torch.float32).unsqueeze(0)


def _load_steegformer_channel_indices(
    metadata: dict[str, Any],
    sensor_index_path: str | Path,
    channel_aliases: dict[str, Any] | None = None,
) -> torch.Tensor:
    with Path(sensor_index_path).open("rb") as f:
        obj = pickle.load(f)
    mapping = obj.get("channels_mapping", obj) if isinstance(obj, dict) else obj
    if not isinstance(mapping, dict):
        raise ValueError(f"Unsupported STEEGFormer sensor index file: {sensor_index_path}")
    lookup = {str(name).upper(): int(value) for name, value in mapping.items()}
    indices = []
    missing = []
    for name in _metadata_channel_names(metadata):
        value = None
        for candidate in _channel_candidates(name, channel_aliases):
            value = lookup.get(candidate)
            if value is not None:
                break
        if value is None:
            missing.append(name)
        else:
            indices.append(value)
    if missing:
        raise ValueError(f"STEEGFormer channel index mapping is missing channels: {missing}")
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0)


def _build_mirepnet_channel_weights(metadata: dict[str, Any], fm_root: str | Path) -> torch.Tensor:
    with prepend_sys_path(fm_root):
        from utils.channel_list import channel_positions, use_channels_names

    actual_channels = [name.upper() for name in _metadata_channel_names(metadata)]
    target_channels = [name.upper() for name in use_channels_names]
    actual_pos = torch.tensor([channel_positions[name] for name in actual_channels], dtype=torch.float32)
    rows = []
    for target in target_channels:
        row = torch.zeros(len(actual_channels), dtype=torch.float32)
        if target in actual_channels:
            row[actual_channels.index(target)] = 1.0
        else:
            target_pos = torch.tensor(channel_positions[target], dtype=torch.float32)
            dist = torch.linalg.norm(actual_pos - target_pos, dim=1).clamp_min(1e-6)
            row = (1.0 / dist)
            row = row / row.sum()
        rows.append(row)
    return torch.stack(rows, dim=0)


def _load_mne_channel_locations(metadata: dict[str, Any]) -> torch.Tensor:
    direct = _metadata_channel_positions(metadata)
    if direct is not None:
        return direct
    import mne

    names = _metadata_channel_names(metadata)
    info = mne.create_info(ch_names=names, sfreq=float(metadata["dataset"].get("sampling_rate_hz", 200)), ch_types=["eeg"] * len(names))
    montage = mne.channels.make_standard_montage("standard_1005")
    info = info.set_montage(montage, match_case=False, match_alias={"cb1": "POO7", "cb2": "POO8"})
    positions = info.get_montage().get_positions()["ch_pos"]
    missing = [name for name in names if name not in positions]
    if missing:
        raise ValueError(f"MNE standard_1005 montage is missing WBCIC channels: {missing}")
    return torch.from_numpy(np.stack([positions[name] for name in names], axis=0)).to(torch.float32).unsqueeze(0)


def _patch_codebrain_device_bug(model: nn.Module) -> None:
    def residual_forward(self, input_data):
        x, original = input_data
        h = x
        batch_size, channels, length = x.shape
        x = self.sn(x)
        if channels != self.res_channels:
            raise ValueError(f"Expected {self.res_channels} channels, got {channels}")
        h = h + original.view(batch_size, self.res_channels, length)
        h = self.conv_layer(h)
        h = self.gelu(h)
        h_t, _ = self.S41(h)
        h_s = h_t.transpose(1, 2)
        mask = self.generate_local_window_mask(length, 1).to(device=h_s.device, dtype=h_s.dtype)
        h_s, _ = self.attention(h_s, h_s, h_s, attn_mask=mask)
        h_s = h_s.transpose(1, 2)
        h = h_t + h_s
        out = torch.tanh(h[:, :self.res_channels, :]) * torch.sigmoid(h[:, self.res_channels:, :])
        res = self.res_conv(out)
        skip = self.skip_conv(out)
        return (x + res) * math.sqrt(0.5), skip

    residual_layer = getattr(model, "residual_layer", None)
    blocks = getattr(residual_layer, "residual_blocks", [])
    for block in blocks:
        block.forward = types.MethodType(residual_forward, block)


class BenchmarkFineTuneModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        input_dim: int,
        num_classes: int,
        head_cfg: dict[str, Any],
        adapter_type: str,
        metadata: dict[str, Any],
        model_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.adapter_type = adapter_type
        self.metadata = metadata
        self.model_cfg = model_cfg or {}
        self.input_scale = float(self.model_cfg.get("input_scale", 1.0))
        self.adapter_normalization = str(self.model_cfg.get("adapter_normalization", "none")).lower()
        self.use_native_classifier = self.adapter_type == "luna" and bool(self.model_cfg.get("use_native_classifier", False))
        head_type = str(head_cfg.get("head_type", "v14_mlp")).lower()
        if self.use_native_classifier:
            self.head = nn.Identity()
        else:
            if head_type not in {"v14", "v14_mlp", "v14a_mlp"}:
                raise ValueError(f"Only v14_mlp is implemented for the fair benchmark runner, got {head_type}")
            self.head = V14MLPHead(
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_dim=int(head_cfg.get("hidden_dim", 256)),
                dropout=float(head_cfg.get("dropout", 0.2)),
                activation=str(head_cfg.get("activation", "elu")),
            )
        if self.adapter_type == "labram":
            chans = metadata["labram"]["standard_1020_input_chans_with_cls"]
            self.register_buffer("labram_input_chans", torch.tensor(chans, dtype=torch.long), persistent=False)
        elif self.adapter_type == "reve":
            positions_path = self.model_cfg.get("positions_path")
            if not positions_path:
                raise ValueError("REVE adapter requires positions_path")
            self.register_buffer(
                "reve_positions",
                _load_reve_positions(metadata, positions_path, self.model_cfg.get("channel_aliases")),
                persistent=False,
            )
            raw_pooling = self.model_cfg.get("pooling", self.model_cfg.get("feature_pooling", "flatten"))
            self.reve_pooling = "no" if raw_pooling is False else str(raw_pooling).lower()
            self.reve_normalize_features = bool(self.model_cfg.get("normalize_features", False))
            if self.reve_pooling in {"last", "attention", "no", "official_no", "flatten_with_context"}:
                embed_dim = int(self.model_cfg.get("embed_dim", input_dim))
                cls_query_token, self.reve_cls_query_token_loaded = _load_reve_cls_query_token(self.model_cfg, embed_dim)
                self.reve_cls_query_token = nn.Parameter(cls_query_token)
        elif self.adapter_type == "steegformer":
            sensor_index_path = self.model_cfg.get("sensor_index_path")
            if not sensor_index_path:
                raise ValueError("STEEGFormer adapter requires sensor_index_path")
            self.register_buffer(
                "steegformer_chan_idx",
                _load_steegformer_channel_indices(metadata, sensor_index_path, self.model_cfg.get("channel_aliases")),
                persistent=False,
            )
        elif self.adapter_type == "mirepnet":
            self.register_buffer("mirepnet_channel_weights", _build_mirepnet_channel_weights(metadata, self.model_cfg["fm_root"]), persistent=False)
            self.register_buffer("mirepnet_ea_matrix", torch.empty(0), persistent=False)
        elif self.adapter_type == "luna":
            self.register_buffer("luna_channel_locations", _load_mne_channel_locations(metadata), persistent=False)

    def input_contract(self) -> dict[str, Any]:
        return {
            "adapter_type": self.adapter_type,
            "stored_unit": self.metadata.get("signal", {}).get("unit_out"),
            "sampling_rate_hz": self.metadata.get("dataset", {}).get("sampling_rate_hz"),
            "input_scale": self.input_scale,
            "adapter_normalization": self.adapter_normalization,
            "target_sampling_rate_hz": self.model_cfg.get("target_sampling_rate"),
            "n_channels": self.metadata.get("dataset", {}).get("n_channels"),
            "n_samples_per_trial": self.metadata.get("dataset", {}).get("n_samples_per_trial"),
            "feature_pooling": self.model_cfg.get("feature_pooling"),
        }

    def prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        out = dict(batch)
        out["x"] = self.prepare_input(out["x"])
        return out

    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if self.input_scale != 1.0:
            x = x * self.input_scale
        x = self._maybe_resample_time(x)
        if self.adapter_normalization in {"zscore_per_trial_channel", "zscore"}:
            flat = x.reshape(x.shape[0], x.shape[1], -1)
            mean = flat.mean(dim=2, keepdim=True)
            std = flat.std(dim=2, keepdim=True).clamp_min(1e-6)
            flat = (flat - mean) / std
            x = flat.reshape_as(x) if x.dim() == 3 else flat.reshape(x.shape)
        elif self.adapter_normalization not in {"none", "identity", ""}:
            raise ValueError(f"Unsupported adapter_normalization: {self.adapter_normalization}")
        return x

    def _maybe_resample_time(self, x: torch.Tensor) -> torch.Tensor:
        target = self.model_cfg.get("target_sampling_rate")
        if target in {None, "", 0}:
            return x
        source = float(self.metadata.get("dataset", {}).get("sampling_rate_hz", 0.0))
        target_f = float(target)
        if source <= 0.0 or target_f <= 0.0 or math.isclose(source, target_f, rel_tol=0.0, abs_tol=1.0e-6):
            return x
        if x.dim() < 3:
            raise ValueError(f"Cannot resample input with shape {tuple(x.shape)}")
        original_shape = x.shape
        flat = x.reshape(x.shape[0], x.shape[1], -1)
        new_len = max(1, int(round(flat.shape[-1] * target_f / source)))
        flat = F.interpolate(flat, size=new_len, mode="linear", align_corners=False)
        if len(original_shape) == 3:
            return flat
        return flat.reshape(flat.shape[0], flat.shape[1], -1)

    def diagnostic_input(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prepare_input(x)
        if self.adapter_type == "luna":
            return self._luna_inputs(x)[0]
        return x.reshape(x.shape[0], x.shape[1], -1)

    def fit_input_transform(self, dataset: Any) -> dict[str, Any]:
        if self.adapter_type != "mirepnet" or not bool(self.model_cfg.get("apply_euclidean_alignment", False)):
            return {"applied": False}
        max_trials = int(self.model_cfg.get("ea_max_trials", 0))
        length = len(dataset)
        if length <= 0:
            raise ValueError("Cannot fit MIRepNet Euclidean alignment on an empty dataset")
        if max_trials > 0 and length > max_trials:
            rng = np.random.default_rng(int(self.model_cfg.get("ea_seed", 2023)))
            selected = sorted(int(v) for v in rng.choice(length, size=max_trials, replace=False))
        else:
            selected = list(range(length))

        cov_sum = None
        n_used = 0
        for idx in selected:
            item = dataset[idx]
            x = item["x"] if isinstance(item, dict) else item[0]
            x_tensor = x if torch.is_tensor(x) else torch.as_tensor(x)
            x_tensor = self.prepare_input(x_tensor.unsqueeze(0)).squeeze(0)
            arr = x_tensor.detach().cpu().numpy().astype(np.float64, copy=False).reshape(x_tensor.shape[0], -1)
            if not np.isfinite(arr).all():
                continue
            cov = np.cov(arr)
            if cov.shape[0] != arr.shape[0]:
                continue
            cov_sum = cov if cov_sum is None else cov_sum + cov
            n_used += 1
        if cov_sum is None or n_used == 0:
            raise ValueError("Could not compute MIRepNet Euclidean alignment covariance from finite trials")
        ref = cov_sum / float(n_used)
        eps = float(self.model_cfg.get("ea_eps", 1.0e-5))
        ref = ref + np.eye(ref.shape[0], dtype=np.float64) * eps
        eigvals, eigvecs = np.linalg.eigh(ref)
        eigvals = np.maximum(eigvals, eps)
        matrix = (eigvecs @ np.diag(eigvals ** -0.5) @ eigvecs.T).astype(np.float32)
        self.mirepnet_ea_matrix = torch.from_numpy(matrix)
        return {"applied": True, "n_trials": int(n_used), "n_channels": int(matrix.shape[0]), "eps": eps}

    def _maybe_checkpoint_block(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if not self.training or not bool(self.model_cfg.get("activation_checkpoint_blocks", False)):
            return block(x)
        try:
            return activation_checkpoint(block, x, use_reentrant=False)
        except TypeError:
            return activation_checkpoint(block, x)

    def _steegformer_forward_features(self, eeg: torch.Tensor, chan_idx: torch.Tensor) -> torch.Tensor:
        batch = eeg.shape[0]
        x = self.backbone.patch_embed(eeg)
        batch, seq, ch_all, dim = x.shape
        seq_total = seq * ch_all
        x = x.view(batch, seq_total, dim)

        eeg_chan_indices = chan_idx.unsqueeze(1).repeat(1, seq, 1).view(batch, seq_total)
        seq_tensor = torch.arange(1, seq + 1, device=eeg.device)
        eeg_seq_indices = seq_tensor.unsqueeze(0).unsqueeze(-1).repeat(batch, 1, ch_all).view(batch, seq_total)
        x = x + self.backbone.enc_temporal_emd(eeg_seq_indices) + self.backbone.enc_channel_emd(eeg_chan_indices)

        cls_token = self.backbone.cls_token + self.backbone.enc_temporal_emd.get_cls_token()
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.backbone.pos_drop(x)
        for block in self.backbone.blocks:
            x = self._maybe_checkpoint_block(block, x)

        if getattr(self.backbone, "global_pool", False):
            return x[:, 1:, :].mean(dim=1)
        x = self.backbone.norm(x)
        return x[:, 0]

    def _reve_attention_context(self, tokens: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "reve_cls_query_token"):
            raise ValueError(f"REVE pooling={self.reve_pooling!r} requires reve_cls_query_token")
        query = self.reve_cls_query_token.to(device=tokens.device, dtype=tokens.dtype).expand(tokens.shape[0], -1, -1)
        scores = torch.matmul(query, tokens.transpose(-1, -2)) / math.sqrt(tokens.shape[-1])
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, tokens)

    def _luna_inputs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_seq = x.reshape(x.shape[0], x.shape[1], -1)
        patch_size = int(self.model_cfg.get("patch_size", self.model_cfg.get("model_kwargs", {}).get("patch_size", 40)))
        remainder = x_seq.shape[-1] % max(1, patch_size)
        if remainder:
            mode = str(self.model_cfg.get("length_adjustment", "pad")).lower()
            if mode == "pad":
                pad = patch_size - remainder
                x_seq = F.pad(x_seq, (0, pad))
            elif mode == "crop":
                x_seq = x_seq[..., : x_seq.shape[-1] - remainder]
            elif mode in {"none", "error"}:
                raise ValueError(f"LUNA input length {x_seq.shape[-1]} is not divisible by patch_size={patch_size}")
            else:
                raise ValueError(f"Unsupported LUNA length_adjustment: {mode}")
        if bool(self.model_cfg.get("normalize_input", True)):
            mean = x_seq.mean(dim=2, keepdim=True)
            std = x_seq.std(dim=2, keepdim=True).clamp_min(1e-6)
            x_seq = (x_seq - mean) / std
        mask = torch.zeros_like(x_seq, dtype=torch.bool)
        channel_locations = self.luna_channel_locations.to(device=x_seq.device, dtype=x_seq.dtype).expand(x_seq.shape[0], -1, -1)
        return x_seq, mask, channel_locations

    def _backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.adapter_type == "labram":
            return self.backbone.forward_features(x, input_chans=self.labram_input_chans, return_patch_tokens=True)
        if self.adapter_type == "reve":
            x_seq = x.reshape(x.shape[0], x.shape[1], -1)
            pos = self.reve_positions.to(device=x_seq.device, dtype=x_seq.dtype).expand(x_seq.shape[0], -1, -1).clone()
            tokens = self.backbone(x_seq, pos=pos, return_output=False)
            if self.reve_pooling in {"mean", "last_avg"}:
                return tokens.mean(dim=1)
            if self.reve_pooling in {"last", "attention"}:
                return self._reve_attention_context(tokens).squeeze(1)
            if self.reve_pooling in {"no", "official_no", "flatten_with_context"}:
                return torch.cat([self._reve_attention_context(tokens), tokens], dim=1)
            if self.reve_pooling in {"flatten", "tokens", "token_flatten"}:
                return tokens
            raise ValueError(f"Unsupported REVE pooling mode: {self.reve_pooling}")
        if self.adapter_type == "steegformer":
            x_seq = x.reshape(x.shape[0], x.shape[1], -1)
            chan_idx = self.steegformer_chan_idx.to(device=x_seq.device).expand(x_seq.shape[0], -1)
            if bool(self.model_cfg.get("activation_checkpoint_blocks", False)):
                return self._steegformer_forward_features(x_seq, chan_idx)
            return self.backbone.forward_features(x_seq, chan_idx)
        if self.adapter_type == "mirepnet":
            x_seq = x.reshape(x.shape[0], x.shape[1], -1)
            if self.mirepnet_ea_matrix.numel() > 0:
                ea = self.mirepnet_ea_matrix.to(device=x_seq.device, dtype=x_seq.dtype)
                x_seq = torch.einsum("oc,bct->bot", ea, x_seq)
            weights = self.mirepnet_channel_weights.to(device=x_seq.device, dtype=x_seq.dtype)
            x_mirepnet = torch.einsum("oc,bct->bot", weights, x_seq)
            original_x = self.backbone.embedding(x_mirepnet)
            transformed = self.backbone.transformer(original_x)
            return transformed.mean(dim=1)
        if self.adapter_type == "luna":
            x_seq, mask, channel_locations = self._luna_inputs(x)
            tokens, _ = self.backbone.prepare_tokens(x_seq, channel_locations, mask=mask)
            tokens, _ = self.backbone.cross_attn(tokens)
            batch = x_seq.shape[0]
            tokens = tokens.reshape(batch, tokens.shape[0] // batch, -1)
            for block in self.backbone.blocks:
                tokens = block(tokens)
            tokens = self.backbone.norm(tokens)
            pooling = str(self.model_cfg.get("feature_pooling", "flatten")).lower()
            if pooling in {"mean", "mean_tokens", "token_mean"}:
                return tokens.mean(dim=1)
            if pooling in {"flatten", "tokens", "token_flatten"}:
                return tokens
            raise ValueError(f"Unsupported LUNA feature_pooling: {pooling}")
        return self.backbone(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prepare_input(x)
        feats = self._backbone_features(x)
        if feats.dim() == 3 and self.adapter_type == "codebrain":
            feats = feats.unsqueeze(0)
        if feats.dim() < 2:
            raise ValueError(f"Backbone returned invalid feature shape {tuple(feats.shape)}")
        feats = feats.reshape(feats.shape[0], -1)
        if self.adapter_type == "reve" and self.reve_normalize_features:
            feats = feats * torch.rsqrt(feats.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        return feats

    def forward_logits(self, batch_or_x: dict[str, Any] | torch.Tensor) -> torch.Tensor:
        x = batch_or_x["x"] if isinstance(batch_or_x, dict) else batch_or_x
        return self.forward(x)

    def build_classifier(self, num_classes: int) -> nn.Module:
        head_cfg = dict(self.model_cfg.get("head", {}))
        input_dim = int(self.model_cfg.get("input_dim", 0))
        return V14MLPHead(
            input_dim=input_dim,
            num_classes=int(num_classes),
            hidden_dim=int(head_cfg.get("hidden_dim", 256)),
            dropout=float(head_cfg.get("dropout", 0.2)),
            activation=str(head_cfg.get("activation", "elu")),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prepare_input(x)
        if self.adapter_type == "luna" and self.use_native_classifier:
            x_seq, mask, channel_locations = self._luna_inputs(x)
            logits, _ = self.backbone(x_seq, mask, channel_locations)
            return logits
        feats = self._backbone_features(x)
        if feats.dim() == 3 and self.adapter_type == "codebrain":
            feats = feats.unsqueeze(0)
        feats = feats.reshape(feats.shape[0], -1)
        if self.adapter_type == "reve" and self.reve_normalize_features:
            feats = feats * torch.rsqrt(feats.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        return self.head(feats)


def _set_projection_identity(backbone: nn.Module) -> None:
    if hasattr(backbone, "proj_out"):
        backbone.proj_out = nn.Identity()


def build_benchmark_model(model_cfg: dict[str, Any], run_model_cfg: dict[str, Any], metadata: dict[str, Any]) -> tuple[BenchmarkFineTuneModel, dict[str, Any]]:
    adapter_type = str(model_cfg["adapter_type"]).lower()
    fm_root = Path(model_cfg["fm_root"])
    checkpoint = model_cfg.get("checkpoint")
    num_classes = int(metadata["dataset"]["num_classes"])
    n_channels = int(metadata["dataset"]["n_channels"])
    n_windows = max(1, int(metadata["dataset"]["n_samples_per_trial"]) // max(1, int(round(float(metadata["dataset"]["sampling_rate_hz"])))))
    embed_dim = int(model_cfg.get("embed_dim", 200))
    input_dim = int(model_cfg.get("input_dim", n_channels * n_windows * embed_dim))
    model_kwargs = dict(model_cfg.get("model_kwargs", {}))
    if bool(model_cfg.get("auto_num_classes", True)):
        for class_key in ("num_classes", "n_classes"):
            if class_key in model_kwargs:
                model_kwargs[class_key] = num_classes
    load_info: dict[str, Any] = {"checkpoint": checkpoint, "loaded_keys": 0, "skipped_keys": 0}

    if adapter_type == "cbramod":
        purge_module_prefixes(("models",))
        with prepend_sys_path(fm_root):
            from models.cbramod import CBraMod

            backbone = CBraMod(**model_kwargs)
        if checkpoint:
            load_info = load_checkpoint_into(backbone, checkpoint, strict=bool(model_cfg.get("strict_load", True)))
        _set_projection_identity(backbone)
    elif adapter_type == "eegmamba":
        purge_module_prefixes(("models", "modules", "mamba_ssm", "einops"))
        install_einops_compat()
        mamba_root = Path(model_cfg.get("mamba_root", fm_root / "mamba-main"))
        try:
            with prepend_sys_path(mamba_root):
                with prepend_sys_path(fm_root):
                    from models.eegmamba import EEGMamba
        except Exception:
            purge_module_prefixes(("models", "modules", "mamba_ssm"))
            install_mamba_ssm_compat()
            with prepend_sys_path(fm_root):
                from models.eegmamba import EEGMamba
        backbone = EEGMamba(**model_kwargs)
        if checkpoint:
            load_info = load_checkpoint_into(backbone, checkpoint, strict=bool(model_cfg.get("strict_load", True)))
        _set_projection_identity(backbone)
    elif adapter_type == "csbrain":
        purge_module_prefixes(("models", "einops"))
        install_einops_compat()
        with prepend_sys_path(fm_root):
            from models.CSBrain import CSBrain

            backbone = CSBrain(
                **model_kwargs,
                brain_regions=metadata["csbrain"]["brain_regions"],
                sorted_indices=metadata["csbrain"]["sorted_indices"],
            )
        if checkpoint:
            load_info = load_checkpoint_into(backbone, checkpoint, strict=bool(model_cfg.get("strict_load", False)))
        _set_projection_identity(backbone)
    elif adapter_type == "codebrain":
        purge_module_prefixes(("Models", "Utils", "einops", "opt_einsum"))
        install_einops_compat()
        install_opt_einsum_compat()
        with prepend_sys_path(fm_root):
            from Models.SSSM import SSSM

            backbone = SSSM(**model_kwargs)
        _patch_codebrain_device_bug(backbone)
        if checkpoint:
            load_info = load_checkpoint_into(
                backbone,
                checkpoint,
                strip_prefixes=tuple(model_cfg.get("strip_prefixes", ["module.", "backbone."])),
                strict=bool(model_cfg.get("strict_load", True)),
            )
    elif adapter_type == "labram":
        purge_module_prefixes(("modeling_finetune", "utils", "einops", "timm"))
        install_einops_compat()
        install_timm_compat()
        with prepend_sys_path(fm_root):
            import modeling_finetune

            factory = getattr(modeling_finetune, str(model_cfg.get("model_name", "labram_base_patch200_200")))
            backbone = factory(pretrained=False, **model_kwargs)
        if checkpoint:
            load_info = load_checkpoint_into(
                backbone,
                checkpoint,
                filter_name=str(model_cfg.get("checkpoint_filter", "labram_student")),
                strip_prefixes=tuple(model_cfg.get("strip_prefixes", ["module.", "model."])),
                strict=bool(model_cfg.get("strict_load", False)),
            )
    elif adapter_type in {"reve", "reve-base"}:
        adapter_type = "reve"
        purge_module_prefixes(("models", "einops", "transformers"))
        install_einops_compat()
        install_transformers_compat()
        config_path = Path(model_cfg.get("config_path", fm_root / "hf" / "reve-base" / "config.json"))
        with config_path.open("r", encoding="utf-8") as f:
            reve_cfg = json.load(f)
        args_backbone = types.SimpleNamespace(
            embed_dim=int(reve_cfg["embed_dim"]),
            depth=int(reve_cfg["depth"]),
            heads=int(reve_cfg["heads"]),
            head_dim=int(reve_cfg["head_dim"]),
            mlp_dim_ratio=float(reve_cfg["mlp_dim_ratio"]),
            use_geglu=bool(reve_cfg["use_geglu"]),
        )
        source_root = Path(model_cfg.get("source_root", fm_root / "src"))
        with prepend_sys_path(source_root):
            from models.encoder import REVE

            backbone = REVE(
                args_backbone=args_backbone,
                freqs=int(reve_cfg["freqs"]),
                patch_size=int(reve_cfg["patch_size"]),
                overlap_size=int(reve_cfg["patch_overlap"]),
                noise_ratio=float(reve_cfg["noise_ratio"]),
            )
        if checkpoint:
            load_info = load_checkpoint_into(
                backbone,
                checkpoint,
                strip_prefixes=tuple(model_cfg.get("strip_prefixes", ["module.", "model."])),
                strict=bool(model_cfg.get("strict_load", False)),
            )
    elif adapter_type == "steegformer":
        purge_module_prefixes(("models_vit_eeg", "timm"))
        install_timm_compat()
        source_root = Path(model_cfg.get("source_root", fm_root / "easy_start"))
        with prepend_sys_path(source_root):
            import models_vit_eeg

            factory = getattr(models_vit_eeg, str(model_cfg.get("model_name", "vit_small_patch16")))
            backbone = factory(**model_kwargs)
        if checkpoint:
            load_info = load_checkpoint_into(
                backbone,
                checkpoint,
                strip_prefixes=tuple(model_cfg.get("strip_prefixes", ["module.", "model."])),
                strict=bool(model_cfg.get("strict_load", False)),
            )
    elif adapter_type == "mirepnet":
        purge_module_prefixes(("model", "utils", "einops", "wandb"))
        install_einops_compat()
        install_wandb_compat()
        with prepend_sys_path(fm_root):
            from model.mlm import mlm_mask

            backbone = mlm_mask(**model_kwargs)
        if checkpoint:
            load_info = load_checkpoint_into(
                backbone,
                checkpoint,
                strip_prefixes=tuple(model_cfg.get("strip_prefixes", ["module.", "model."])),
                strict=bool(model_cfg.get("strict_load", False)),
            )
    elif adapter_type in {"luna", "luna-large", "luna_large"}:
        adapter_type = "luna"
        purge_module_prefixes(("models", "timm", "einops", "rotary_embedding_torch", "torcheeg"))
        install_einops_compat()
        install_timm_compat()
        install_rotary_embedding_compat()
        install_torcheeg_compat()
        with prepend_sys_path(fm_root):
            from models.LUNA import LUNA

            backbone = LUNA(**model_kwargs)
        if checkpoint:
            load_info = load_checkpoint_into(
                backbone,
                checkpoint,
                strip_prefixes=tuple(model_cfg.get("strip_prefixes", ["module.", "model."])),
                strict=bool(model_cfg.get("strict_load", False)),
            )
        if hasattr(backbone, "classifier") and not bool(model_cfg.get("use_native_classifier", False)):
            backbone.classifier = nn.Identity()
    else:
        raise ValueError(f"Unsupported adapter_type: {adapter_type}")

    model = BenchmarkFineTuneModel(
        backbone,
        input_dim=input_dim,
        num_classes=num_classes,
        head_cfg=run_model_cfg,
        adapter_type=adapter_type,
        metadata=metadata,
        model_cfg=model_cfg,
    )
    if adapter_type == "reve" and getattr(model, "reve_cls_query_token_loaded", False):
        load_info["loaded_aux_keys"] = 1
        load_info["aux_keys"] = ["cls_query_token"]
    return model, load_info


def load_model(config: dict[str, Any], metadata: dict[str, Any]) -> tuple[BenchmarkFineTuneModel, dict[str, Any]]:
    return build_benchmark_model(config["foundation_model"], config.get("model", {}), metadata)
