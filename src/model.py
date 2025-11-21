"""多模态协同预测模型定义。"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class PairSynergyNet(nn.Module):
    """融合两个药物指纹与细胞上下文的 MLP 分类器，支持亚型门控。"""

    def __init__(
        self,
        drug_dim: int,
        context_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        subtype_dim: int = 0,
    ):
        super().__init__()
        input_dim = drug_dim * 4 + context_dim
        dims = [input_dim] + list(hidden_dims)
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], 1)
        self.subtype_dim = subtype_dim
        if subtype_dim > 0:
            gate_hidden = max(64, subtype_dim * 2)
            self.subtype_gate = nn.Sequential(
                nn.LayerNorm(subtype_dim),
                nn.Linear(subtype_dim, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, dims[-1]),
                nn.Sigmoid(),
            )
        else:
            self.subtype_gate = None


    @staticmethod
    def _pair_features(drug_a: torch.Tensor, drug_b: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(drug_a - drug_b)
        prod = drug_a * drug_b
        return torch.cat([drug_a, drug_b, diff, prod], dim=1)

    def forward(
        self,
        drug_a: torch.Tensor,
        drug_b: torch.Tensor,
        context: torch.Tensor,
        subtype: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pair = self._pair_features(drug_a, drug_b)
        feats = torch.cat([pair, context], dim=1)
        hidden = self.mlp(feats)
        if self.subtype_gate is not None and subtype is not None:
            gate = self.subtype_gate(subtype)
            hidden = hidden * gate
        return self.head(hidden).squeeze(1)


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    pos_weight: float,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal Loss，用于缓解正负样本极度不平衡。"""

    weight = torch.tensor(pos_weight, device=logits.device)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=weight, reduction="none")
    probs = torch.sigmoid(logits)
    pt = torch.where(targets == 1, probs, 1 - probs)
    loss = ((1 - pt) ** gamma) * bce
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()
