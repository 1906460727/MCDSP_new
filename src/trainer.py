"""训练循环与评估指标。"""

from __future__ import annotations

from typing import Dict, List

import math
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
    """输出 AUROC / AUPRC / 动态阈值下的 F1/ACC 等指标。"""

    from sklearn import metrics

    y_score = np.nan_to_num(y_score, nan=0.5, posinf=1.0, neginf=0.0)
    try:
        auroc = metrics.roc_auc_score(y_true, y_score)
    except ValueError:
        auroc = float("nan")
    auprc = metrics.average_precision_score(y_true, y_score)

    thresholds = np.unique(np.concatenate([y_score, np.array([0.5])]))
    best_f1 = 0.0
    best_acc = 0.0
    best_threshold = 0.5
    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        f1_val = metrics.f1_score(y_true, y_pred, zero_division=0)
        if f1_val >= best_f1:
            best_f1 = f1_val
            best_acc = metrics.accuracy_score(y_true, y_pred)
            best_threshold = thr

    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": best_f1,
        "acc": best_acc,
        "best_threshold": best_threshold,
    }


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
        for drug_a, drug_b, context, subtype, labels, _ in loader:
            drug_a = drug_a.to(self.device)
            drug_b = drug_b.to(self.device)
            context = context.to(self.device)
            subtype = subtype.to(self.device)
            labels = labels.to(self.device)
            logits = model(drug_a, drug_b, context, subtype=subtype)
            if self.cfg.loss_strategy == "focal":
                per_sample = focal_loss(
                    logits,
                    labels,
                    gamma=gamma,
                    pos_weight=pos_weight,
                    reduction="none",
                )
            else:
                criterion = torch.nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor(pos_weight, device=self.device),
                    reduction="none",
                )
                per_sample = criterion(logits, labels)
            weights = self._compute_subtype_weight(subtype)
            loss = torch.mean(per_sample * weights)
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
            for drug_a, drug_b, context, subtype, labels, _ in loader:
                logits = model(
                    drug_a.to(self.device),
                    drug_b.to(self.device),
                    context.to(self.device),
                    subtype=subtype.to(self.device),
                )
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
            auroc = metrics.get("auroc", float("nan"))
            print(
                f"[Epoch {epoch+1}/{self.cfg.max_epochs}] "
                f"train_loss={loss:.4f} | val_acc={metrics.get('acc', float('nan')):.4f} | "
                f"val_auroc={auroc:.4f} | best_thr={metrics.get('best_threshold', float('nan')):.3f} | "
                f"val_metrics={metrics}"
            )
            score = auroc if not np.isnan(auroc) else -np.inf
            if score > best_score:
                best_score = score
                best_state = model.state_dict()
                patience_left = self.cfg.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    def _compute_subtype_weight(self, subtype: torch.Tensor) -> torch.Tensor:
        """根据亚型概率向量计算样本权重，越“典型”的样本权重越高。"""

        if subtype.numel() == 0 or subtype.shape[1] == 0:
            return torch.ones(subtype.shape[0], device=self.device)
        mask = torch.sum(subtype, dim=1) == 0
        probs = subtype.clone()
        probs[mask] = 1.0 / max(1, subtype.shape[1])
        entropy = -(probs * probs.clamp(min=1e-6).log()).sum(dim=1)
        max_entropy = math.log(probs.shape[1])
        typicality = 1.0 - entropy / max_entropy
        weights = typicality.clamp(0.3, 1.0)
        weights[mask] = 1.0
        return weights
