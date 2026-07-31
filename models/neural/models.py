from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1.0, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super().forward(x)


class EEGNet(nn.Module):
    def __init__(
        self,
        n_classes: int,
        chans: int,
        samples: int,
        f1: int = 8,
        d: int = 2,
        kernel: int = 64,
    ):
        super().__init__()
        dropout = 0.2
        f2 = f1 * d
        self.features = nn.Sequential(
            nn.Conv2d(1, f1, (1, kernel), padding=(0, kernel // 2), bias=False),
            nn.BatchNorm2d(f1),
            Conv2dWithConstraint(
                f1, f2, (chans, 1), groups=f1, bias=False, max_norm=1
            ),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=4),
            nn.Dropout(dropout),
            nn.Conv2d(
                f2, f2, (1, 22), padding=(0, 11), groups=f2, bias=False
            ),
            nn.Conv2d(f2, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(dropout),
            nn.Flatten(),
        )
        with torch.no_grad():
            dim = self.features(torch.zeros(1, 1, chans, samples)).shape[1]
        self.classifier = nn.Linear(dim, n_classes)

    def forward(self, x):
        return self.classifier(self.features(x.unsqueeze(1)))


class PatchEmbedding(nn.Module):
    def __init__(self, chans: int, emb_size: int = 40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25)),
            nn.Conv2d(40, 40, (chans, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(0.5),
            nn.Conv2d(40, emb_size, (1, 1)),
        )

    def forward(self, x):
        return self.net(x).squeeze(2).transpose(1, 2)


class TransformerBlock(nn.Module):
    def __init__(
        self, emb_size: int = 40, heads: int = 10, dropout: float = 0.5
    ):
        super().__init__()
        self.attn_norm = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(
            emb_size, heads, dropout=dropout, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(emb_size)
        self.ff = nn.Sequential(
            nn.Linear(emb_size, emb_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size * 4, emb_size),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.attn_norm(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.drop(y)
        return x + self.drop(self.ff(self.ff_norm(x)))


class EEGConformer(nn.Module):
    def __init__(
        self,
        n_classes: int,
        chans: int,
        samples: int,
        emb_size: int = 40,
        depth: int = 6,
    ):
        super().__init__()
        self.patch = PatchEmbedding(chans, emb_size)
        self.encoder = nn.Sequential(
            *[TransformerBlock(emb_size) for _ in range(depth)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(emb_size), nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        return self.head(self.encoder(self.patch(x)).mean(dim=1))


class Conv1dWithConstraint(nn.Conv1d):
    def __init__(self, *args, max_norm=0.5, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super().forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=0.5, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super().forward(x)


class IFNetStem(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int = 64,
        kernel_size: int = 63,
        patch_size: int = 125,
        radix: int = 2,
    ):
        super().__init__()
        self.out_planes = out_planes
        self.patch_size = patch_size
        self.sconv = nn.Sequential(
            Conv1dWithConstraint(
                in_planes,
                out_planes * radix,
                1,
                bias=False,
                groups=radix,
            ),
            nn.BatchNorm1d(out_planes * radix),
        )
        self.tconv = nn.ModuleList()
        for _ in range(radix):
            self.tconv.append(
                nn.Sequential(
                    Conv1dWithConstraint(
                        out_planes,
                        out_planes,
                        kernel_size,
                        groups=out_planes,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_planes),
                )
            )
            kernel_size //= 2
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        n, _, t = x.shape
        parts = torch.split(self.sconv(x), self.out_planes, dim=1)
        out = F.gelu(sum(conv(part) for conv, part in zip(self.tconv, parts)))
        out = out.reshape(
            n, self.out_planes, t // self.patch_size, self.patch_size
        )
        out = torch.log(
            torch.clamp(torch.mean(out**2, dim=3), 1e-4, 1e4)
        )
        return self.drop(out)


class IFNetV2(nn.Module):
    def __init__(
        self,
        n_classes: int,
        chans: int,
        samples: int,
        patch_size: int = 125,
    ):
        super().__init__()
        usable = max(patch_size, samples // patch_size * patch_size)
        self.patch_size = patch_size
        self.stem = IFNetStem(
            chans, patch_size=patch_size, radix=2
        )
        self.fc = LinearWithConstraint(
            64 * (usable // patch_size), n_classes
        )
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(
            module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)
        ):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        usable = x.shape[-1] // self.patch_size * self.patch_size
        return self.fc(self.stem(x[..., :usable]).flatten(1))


MODEL_NAMES = ["eegnet", "eegconformer", "ifnetv2"]


def build_model(
    name: str,
    n_classes: int,
    chans: int,
    samples: int,
):
    if name == "eegnet":
        return EEGNet(n_classes, chans, samples)
    if name == "eegconformer":
        return EEGConformer(n_classes, chans, samples)
    if name == "ifnetv2":
        return IFNetV2(n_classes, chans, samples)
    raise ValueError(f"unknown neural model: {name}")
