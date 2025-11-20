"""通用数据工具函数，用于读写数据与划分集合。"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.pathway_activity import build_gene_symbol_map, prepare_expression_matrix


def set_seed(seed: int) -> None:
    """统一设置 Python/NumPy 的随机种子，保证实验可复现。"""

    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> Path:
    """确保目录存在，若不存在则递归创建。"""

    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(payload: dict, path: Path) -> None:
    """将结果对象保存为 JSON 文件。"""

    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_processed_expression(expr_path: Path, gene_col: str, annotation_path: Path) -> pd.DataFrame:
    """
    载入已经对齐过 Ensembl ID 的表达矩阵。
    通过注释文件保证患者与细胞系使用一致的基因符号。
    """

    symbol_map = build_gene_symbol_map(annotation_path)
    return prepare_expression_matrix(expr_path, gene_col, symbol_map)


def align_expression_domains(
    patient_expr: pd.DataFrame,
    cell_expr: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """只保留两个表达矩阵的交集基因。"""

    common_genes = sorted(set(patient_expr.columns).intersection(cell_expr.columns))
    if not common_genes:
        raise ValueError("患者与细胞系表达矩阵没有共享基因，无法对齐。")
    return patient_expr[common_genes].copy(), cell_expr[common_genes].copy()


def balanced_test_split(
    df: pd.DataFrame,
    label_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    ensure_balanced_test: bool,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按照 7:2:1 的比例划分数据，并可在测试集保持 1:1 正负样本。"""

    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-3):
        raise ValueError("train/val/test 比例之和必须等于 1。")

    labels = df[label_col]
    if ensure_balanced_test:
        pos = df[labels == 1]
        neg = df[labels == 0]
        per_class = int(math.floor(len(df) * test_ratio / 2))
        per_class = min(per_class, len(pos), len(neg))
        test = pd.concat(
            [
                pos.sample(per_class, random_state=seed),
                neg.sample(per_class, random_state=seed),
            ]
        )
        remaining = df.drop(test.index)
    else:
        remaining, test = train_test_split(df, test_size=test_ratio, stratify=labels, random_state=seed)

    rel_labels = remaining[label_col]
    train_frac = train_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(remaining, train_size=train_frac, stratify=rel_labels, random_state=seed)
    return train, val, test
