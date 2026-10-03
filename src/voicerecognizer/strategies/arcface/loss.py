"""Additive Angular Margin (ArcFace) loss module."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class ArcMarginProduct(nn.Module):
    r"""Additive Angular Margin (ArcFace) Linear Layer (Deng et al., CVPR 2019).

    Computes:
    $$
    \cos(\theta_j) = \frac{W_j^T x}{\|W_j\| \|x\|}
    $$
    For the ground truth class $y_i$:
    $$
    \cos(\theta_{y_i} + m) = \cos \theta_{y_i} \cos m - \sin \theta_{y_i} \sin m
    $$
    Scaled by $s$:
    $$
    \text{output}_j = s \cdot (\text{one\_hot}_{y_i} \cdot \cos(\theta_{y_i} + m) + (1 - \text{one\_hot}_{y_i}) \cdot \cos \theta_j)
    $$
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 24.0,
        m: float = 0.30,
        easy_margin: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(
        self, input_features: torch.Tensor, labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            input_features: (batch_size, in_features)
            labels: (batch_size,) or None for unpenalized inference

        Returns:
            tuple (penalized_scaled_logits, raw_unpenalized_logits)
        """
        cosine = F.linear(
            F.normalize(input_features, p=2, dim=-1),
            F.normalize(self.weight, p=2, dim=-1),
        )
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

        raw_logits = cosine * self.s

        if labels is None:
            return raw_logits, raw_logits

        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=1e-7, max=1.0))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        penalized_logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        penalized_logits *= self.s

        return penalized_logits, raw_logits
