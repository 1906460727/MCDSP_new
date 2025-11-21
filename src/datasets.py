"""组合样本构建与 Dataset 定义。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

from src.config import TrainingConfig, AblationConfig
from src.drug_features import canonicalise_drug_name


@dataclass
class CombinationRecord:
    """统一描述任意一个药物组合样本。"""

    drug_a: str
    drug_b: str
    context_id: str
    domain: str  # cell_line 或 patient
    label: int
    subtype_probs: Optional[np.ndarray] = None


class CombinationBuilder:
    """根据配置从 CSV 中读取组合样本。"""

    def __init__(self, cfg: TrainingConfig, ablation: AblationConfig):
        self.cfg = cfg
        self.ablation = ablation

    @staticmethod
    def _parse_pair(pair: str) -> Tuple[str, str]:
        tokens = [tok.strip() for tok in pair.replace("+", "|").split("|") if tok.strip()]
        if len(tokens) != 2:
            raise ValueError(f"非法的药物组合格式: {pair}")
        return tokens[0], tokens[1]

    @staticmethod
    def _map_patient_sample(sample_id: str, expr_index: Iterable[str]) -> Optional[str]:
        if sample_id in expr_index:
            return sample_id
        if sample_id.endswith("D"):
            candidate = f"{sample_id[:-1]}R"
            if candidate in expr_index:
                return candidate
        return None

    def load_cell_line_records(self, df: pd.DataFrame) -> List[CombinationRecord]:
        metric = self.cfg.cell_synergy_column
        if metric not in df.columns:
            raise ValueError(f"细胞系文件缺少列: {metric}")
        records: List[CombinationRecord] = []
        for _, row in df.iterrows():
            score = row[metric]
            if pd.isna(score):
                continue
            if self.cfg.cell_synergy_column == "synergy_bliss":
                label = int(float(score) > self.cfg.cell_synergy_threshold)
            else:
                label = int(float(score) >= self.cfg.cell_synergy_threshold)
            records.append(
                CombinationRecord(
                    drug_a=str(row["drug_row"]),
                    drug_b=str(row["drug_col"]),
                    context_id=str(row["cell_line_name"]),
                    domain="cell_line",
                    label=label,
                )
            )
        return records

    def load_patient_records(
        self,
        df: pd.DataFrame,
        expr_index: Iterable[str],
        subtype_probs: pd.DataFrame,
    ) -> List[CombinationRecord]:
        label_col = self.cfg.patient_label_column
        if label_col not in df.columns:
            raise ValueError(f"患者文件缺少列: {label_col}")
        records: List[CombinationRecord] = []
        for _, row in df.iterrows():
            drug_a, drug_b = self._parse_pair(str(row["doublet_pair"]))
            sample_id = str(row["dbgap_dnaseq_sample"])
            mapped = self._map_patient_sample(sample_id, expr_index)
            if mapped is None:
                continue
            probs = subtype_probs.loc[mapped].to_numpy(dtype=np.float32)
            records.append(
                CombinationRecord(
                    drug_a=drug_a,
                    drug_b=drug_b,
                    context_id=mapped,
                    domain="patient",
                    label=int(row[label_col]),
                    subtype_probs=probs,
                )
            )
        return records


def build_context_feature_matrix(
    expr: pd.DataFrame,
    embeddings: Optional[pd.DataFrame],
    pathway_features: Optional[pd.DataFrame],
    use_embeddings: bool,
    use_pathway: bool,
) -> pd.DataFrame:
    """将表达嵌入与通路特征拼接成最终上下文特征。"""

    pieces: List[pd.DataFrame] = []
    if use_embeddings and embeddings is not None:
        pieces.append(embeddings.loc[expr.index])
    elif use_embeddings:
        raise ValueError("需要 CODE-AE 嵌入，但未提供编码器输出。")
    if use_pathway and pathway_features is not None:
        pieces.append(pathway_features.loc[expr.index])
    if not pieces:
        scaled = StandardScaler().fit_transform(expr)
        pieces.append(pd.DataFrame(scaled, index=expr.index))
    matrix = pd.concat(pieces, axis=1).fillna(0.0)
    matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return matrix.astype(np.float32)


def pca_project(matrix: pd.DataFrame, target_dim: int = 256) -> pd.DataFrame:
    """在没有 CODE-AE 的情况下，使用 PCA 压缩表达维度。"""

    n_components = max(1, min(target_dim, matrix.shape[0], matrix.shape[1]))
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(matrix)
    return pd.DataFrame(reduced, index=matrix.index)


class CombinationDataset(Dataset):
    """供 DataLoader 使用的组合样本 Dataset。"""

    def __init__(
        self,
        records: List[CombinationRecord],
        context_matrix: pd.DataFrame,
        drug_dict: Dict[str, np.ndarray],
        ablation: AblationConfig,
        subtype_dim: int,
    ):
        self.records = records
        context_array = context_matrix.to_numpy(dtype=np.float32, copy=True)
        self.context_lookup: Dict[str, torch.Tensor] = {
            ctx_id: torch.from_numpy(context_array[i])
            for i, ctx_id in enumerate(context_matrix.index)
        }
        self.drug_dict = drug_dict
        self.ablation = ablation
        self.subtype_dim = subtype_dim

    def __len__(self) -> int:  # noqa: D401
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        drug_a = self.drug_dict[canonicalise_drug_name(record.drug_a)]
        drug_b = self.drug_dict[canonicalise_drug_name(record.drug_b)]
        context_vec = self.context_lookup[record.context_id]
        context_vec = torch.nan_to_num(context_vec, nan=0.0, posinf=0.0, neginf=0.0)
        if record.subtype_probs is None or self.ablation.skip_subtype_weighting:
            subtype = np.zeros(self.subtype_dim, dtype=np.float32)
        else:
            subtype = np.nan_to_num(record.subtype_probs.astype(np.float32), nan=0.0)
        return (
            torch.from_numpy(drug_a),
            torch.from_numpy(drug_b),
            context_vec.clone(),
            torch.from_numpy(subtype),
            torch.tensor(record.label, dtype=torch.float32),
            record.domain,
        )
