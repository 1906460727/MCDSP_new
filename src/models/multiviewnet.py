"""更精简的 MultiViewNet 结构，专注于双药协同预测，不再包含单药微调逻辑。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device() -> torch.device:
    """优先使用 GPU。"""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Highway(nn.Module):
    """Highway 层用于在不同模态之间混合特征。"""

    def __init__(self, num_layers: int, input_size: int, dropout: float):
        super().__init__()
        self.num_layers = num_layers
        self.non_linear = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            non_linear = F.gelu(self.non_linear[layer](x))
            linear = self.linear[layer](x)
            x = gate * non_linear + (1 - gate) * linear
            x = self.dropout(x)
        return x


class DualInteract(nn.Module):
    """将两个模态（例如 drug A / drug B）通过 Highway + Self-Attention 进行融合。"""

    def __init__(self, field_dim: int, embed_size: int, dropout: float = 0.3, layers: int = 1):
        super().__init__()
        self.bit_wise_net = Highway(num_layers=layers, input_size=field_dim * embed_size, dropout=dropout)
        hidden_dim = max(256, embed_size * field_dim // 2)
        self.trans_bit_nn = nn.Sequential(
            nn.LayerNorm(field_dim * embed_size),
            nn.Linear(field_dim * embed_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, field_dim * embed_size),
            nn.Dropout(dropout),
        )
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=4, dropout=dropout, batch_first=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, field_dim, embed_size = x.shape
        x_flat = x.reshape(batch_size, field_dim * embed_size)

        bit_wise_x = self.bit_wise_net(x_flat)

        # Self-attention across the fields.
        x_reshaped = x.permute(1, 0, 2)  # (field_dim, batch_size, embed_size)
        attn_output, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        attn_output = attn_output.permute(1, 0, 2).reshape(batch_size, field_dim * embed_size)

        fused = self.trans_bit_nn(bit_wise_x)
        return fused + x_flat + attn_output


class MultiViewNet(nn.Module):
    """
    纯双药协同预测网络：输入两个药物指纹 + 细胞系表型，输出协同概率。

    - 不再包含单药微调路径，训练逻辑完全由外部 loss 控制。
    - 若提供 encoder，则先将细胞系表达送入共享编码器再投影。
    """

    def __init__(
        self,
        encoder: nn.Module | None = None,
        drug_dim: int = 1024,
        cline_dim: int = 128,
        proj_dim: int = 256,
        dropout: float = 0.3,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.device = device or get_device()
        self.drug_projector = nn.Sequential(
            nn.Linear(drug_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cline_projector = nn.Sequential(
            nn.Linear(cline_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.drug_interact = DualInteract(field_dim=2, embed_size=proj_dim, dropout=dropout, layers=2)
        fusion_in_dim = proj_dim * 2 + proj_dim * 2  # drug pair (flatten) + drug-pair与cell concat
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(fusion_in_dim),
            nn.Linear(fusion_in_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def _encode_cline(self, cline_features: torch.Tensor) -> torch.Tensor:
        if self.encoder is not None:
            cline_features = self.encoder(cline_features)
        return cline_features

    def forward(
        self,
        drug_a: torch.Tensor,
        drug_b: torch.Tensor,
        cline_features: torch.Tensor,
    ) -> torch.Tensor:
        cline_features = self._encode_cline(cline_features)

        proj_drug_a = self.drug_projector(drug_a)
        proj_drug_b = self.drug_projector(drug_b)
        proj_cline = self.cline_projector(cline_features)

        drug_pair = torch.stack([proj_drug_a, proj_drug_b], dim=1)
        pair_feat = self.drug_interact(drug_pair)  # (batch, proj_dim * 2)

        fusion = torch.cat([pair_feat, torch.cat([proj_drug_a, proj_cline], dim=1)], dim=1)
        logits = self.fusion_head(fusion).squeeze(1)
        return logits

    def infer(self, drug_a: torch.Tensor, drug_b: torch.Tensor, cline_features: torch.Tensor) -> torch.Tensor:
        """兼容旧接口的推理函数。"""

        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward(drug_a, drug_b, cline_features))
