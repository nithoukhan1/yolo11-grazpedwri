# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""NCHW multi-scale dilated local attention inspired by DilateFormer.

Reference: Jiao et al., TMM 2023, https://arxiv.org/abs/2302.01791.
This is an independently implemented YOLO adapter, not recovered original code.
Head count/latent width are rounded up to support YOLO widths not divisible by
the number of dilation scales. See MSDA_RESEARCH.md for the exact adaptation.
"""

import math

import torch
from torch import nn


class MSDA(nn.Module):
    """Residual local attention and pointwise feedforward block, preserving B,C,H,W."""

    def __init__(self, dim, num_heads=8, dilations=(1, 2, 4), kernel_size=3, mlp_ratio=2):
        super().__init__()
        if dim < 1 or num_heads < 1 or kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("MSDA requires positive dimensions/heads and an odd positive kernel")
        self.dilations = tuple(dilations)
        if not self.dilations or any(not isinstance(d, int) or d < 1 for d in self.dilations):
            raise ValueError("MSDA dilations must be nonempty positive integers")
        self.num_scales = len(self.dilations)
        self.num_heads = math.ceil(num_heads / self.num_scales) * self.num_scales
        self.head_dim = math.ceil(dim / self.num_heads)
        self.heads_per_scale = self.num_heads // self.num_scales
        self.inner_dim = self.head_dim * self.num_heads
        self.kernel_area = kernel_size**2
        self.scale = self.head_dim**-0.5
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.qkv = nn.Conv2d(dim, 3 * self.inner_dim, 1, bias=False)
        self.unfolds = nn.ModuleList(
            nn.Unfold(kernel_size, dilation=d, padding=d * (kernel_size // 2)) for d in self.dilations
        )
        self.proj = nn.Conv2d(self.inner_dim, dim, 1)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, int(dim * mlp_ratio), 1), nn.GELU(), nn.Conv2d(int(dim * mlp_ratio), dim, 1)
        )

    @staticmethod
    def _norm(x, norm):
        return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def forward(self, x):
        batch, _, height, width = x.shape
        qkv = self.qkv(self._norm(x, self.norm1)).reshape(batch, 3, self.num_scales, -1, height, width)
        outputs = []
        for i, unfold in enumerate(self.unfolds):
            q = qkv[:, 0, i].reshape(batch, self.heads_per_scale, self.head_dim, height * width)
            k = unfold(qkv[:, 1, i]).reshape(
                batch, self.heads_per_scale, self.head_dim, self.kernel_area, height * width
            )
            v = unfold(qkv[:, 2, i]).reshape_as(k)
            weights = (torch.einsum("bhdn,bhdkn->bhnk", q, k) * self.scale).softmax(dim=-1)
            attended = torch.einsum("bhnk,bhdkn->bhdn", weights, v)
            outputs.append(attended.reshape(batch, -1, height, width))
        x = x + self.proj(torch.cat(outputs, dim=1))
        return x + self.ffn(self._norm(x, self.norm2))
