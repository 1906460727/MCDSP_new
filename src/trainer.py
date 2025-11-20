"""训练循环与评估指标。"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.config import TrainingConfig
from src.model import PairSynergyNet, focal_loss


def build_sampler(records: List, cfg: TrainingConfig) -> WeightedRandomSampler:
    """为 cell_line / patient 样本设置不同采样权重，提升域混合能力。"""

    weights = torch.zeros(len(records))
    cell_idx = [i for i, rec in enumerate(records) if rec.domain == "cell_line"]
    pat_idx = [i for i, rec in enumerate(records) if rec.domain == "patient"]
    if cell_idx:
        weights[cell_idx] = cfg.domain_mix_ratio / max(len(cell_idx), 1)
    if pat_idx:
        weights[pat_idx] = (1 - cfg.domain_mix_ratio) / max(len(pat_idx), 1)
    if not cell_idx or not pat_idx:
        weights[:] = 1 / len(records)
    return WeightedRandomSampler(weights, num_samples=len(records), replacement=True)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """输出 AUROC / AUPRC / F1 等指标。"""

    from sklearn import metrics

    y_score = np.nan_to_num(y_score, nan=0.5, posinf=1.0, neginf=0.0)
    y_pred = (y_score >= 0.5).astype(int)
    metrics_dict = {
        "auroc": metrics.roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan"),
        "auprc": metrics.average_precision_score(y_true, y_score),
        "f1": metrics.f1_score(y_true, y_pred),
    }
    return metrics_dict


class Trainer:
    """封装训练与验证流程。"""

    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _run_epoch(
        self,
        model: PairSynergyNet,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        pos_weight: float,
        gamma: float,
    ) -> float:
        model.train()
        losses: List[float] = []
        for drug_a, drug_b, context, _, labels, _ in loader:
            drug_a = drug_a.to(self.device)
            drug_b = drug_b.to(self.device)
            context = context.to(self.device)
            labels = labels.to(self.device)
            logits = model(drug_a, drug_b, context)
            if self.cfg.loss_strategy == "focal":
                loss = focal_loss(logits, labels, gamma=gamma, pos_weight=pos_weight)
            else:
                criterion = torch.nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor(pos_weight, device=self.device)
                )
                loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return float(np.mean(losses)) if losses else 0.0

    def _predict(self, model: PairSynergyNet, loader: DataLoader) -> Dict[str, np.ndarray]:
        model.eval()
        preds: List[np.ndarray] = []
        labels_all: List[np.ndarray] = []
        with torch.no_grad():
            for drug_a, drug_b, context, _, labels, _ in loader:
                logits = model(drug_a.to(self.device), drug_b.to(self.device), context.to(self.device))
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.append(np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0))
                labels_all.append(labels.numpy())
        return {
            "y_true": np.concatenate(labels_all) if labels_all else np.zeros(0),
            "y_score": np.concatenate(preds) if preds else np.zeros(0),
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model: PairSynergyNet,
        pos_weight: float,
        gamma: float,
    ) -> PairSynergyNet:
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        best_score = -np.inf
        best_state = None
        patience_left = self.cfg.patience
        for epoch in range(self.cfg.max_epochs):
            loss = self._run_epoch(model, train_loader, optimizer, pos_weight=pos_weight, gamma=gamma)
            pred = self._predict(model, val_loader)
            metrics = compute_metrics(pred["y_true"], pred["y_score"]) if len(pred["y_true"]) else {}
            auprc = metrics.get("auprc", float("nan"))
            if auprc > best_score:
                best_score = auprc
                best_state = model.state_dict()
                patience_left = self.cfg.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model
