"""封装 CODE-AE (basis autoencoder) 的训练与编码逻辑。"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.code_adv import train_code_adv, get_device


class ExpressionDataset(Dataset):
    """简单的表达矩阵 Dataset，每条样本只包含一个张量。"""

    def __init__(self, data: np.ndarray):
        self.tensor = torch.from_numpy(data.astype(np.float32))

    def __len__(self) -> int:  # noqa: D401
        return self.tensor.shape[0]

    def __getitem__(self, idx: int):
        return (self.tensor[idx],)


class CodeAEManager:
    """负责训练并缓存 CODE-AE 共享编码器。"""

    def __init__(self, cfg, model_dir: Path):
        self.cfg = cfg
        self.model_dir = model_dir / cfg.model_name
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _split(self, array: np.ndarray) -> Tuple[Dataset, Dataset]:
        idx = np.arange(len(array))
        if len(idx) < 2:
            return ExpressionDataset(array), ExpressionDataset(array)
        train_idx = idx[: int(len(idx) * 0.8)]
        val_idx = idx[int(len(idx) * 0.8) :]
        return ExpressionDataset(array[train_idx]), ExpressionDataset(array[val_idx])

    def train_or_load(self, patient_expr: pd.DataFrame, cell_expr: pd.DataFrame) -> torch.nn.Module:
        device = get_device()
        pat = patient_expr.to_numpy(dtype=np.float32)
        cell = cell_expr.to_numpy(dtype=np.float32)
        s_train, s_val = self._split(cell)
        t_train, t_val = self._split(pat)

        train_batch = max(1, min(self.cfg.batch_size, len(s_train), len(t_train)))
        val_batch = max(1, min(self.cfg.batch_size, len(s_val), len(t_val)))

        def make_loader(ds: Dataset, batch_size: int, shuffle: bool, drop_last: bool) -> DataLoader:
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

        encoder, _ = train_code_adv(
            s_dataloaders=(
                make_loader(s_train, train_batch, True, len(s_train) > train_batch),
                make_loader(s_val, val_batch, False, False),
            ),
            t_dataloaders=(
                make_loader(t_train, train_batch, True, len(t_train) > train_batch),
                make_loader(t_val, val_batch, False, False),
            ),
            input_dim=patient_expr.shape[1],
            latent_dim=self.cfg.latent_dim,
            encoder_hidden_dims=list(self.cfg.encoder_hidden_dims),
            classifier_hidden_dims=list(self.cfg.classifier_hidden_dims),
            dop=self.cfg.dop,
            num_geo_layer=self.cfg.num_geo_layer,
            norm_flag=self.cfg.norm_flag,
            lr=self.cfg.lr,
            pretrain_num_epochs=self.cfg.pretrain_epochs,
            train_num_epochs=self.cfg.adv_epochs,
            retrain_flag=self.cfg.retrain,
            es_flag=False,
            model_save_folder=str(self.model_dir),
            device=device,
        )
        return encoder.to(device)

    @staticmethod
    def encode(encoder: torch.nn.Module, expr: pd.DataFrame, batch_size: int = 512) -> pd.DataFrame:
        """批量将表达矩阵映射到共享潜在空间。"""

        encoder.eval()
        device = next(encoder.parameters()).device
        array = expr.to_numpy(dtype=np.float32)
        outputs = []
        with torch.no_grad():
            for idx in range(0, len(array), batch_size):
                batch = torch.from_numpy(array[idx : idx + batch_size]).to(device)
                embeds = encoder(batch)
                outputs.append(embeds.cpu().numpy())
        matrix = np.vstack(outputs)
        return pd.DataFrame(matrix, index=expr.index)
