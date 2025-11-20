"""通路活性推断与患者亚型聚类相关模块。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from src.utils.pathway_activity import (
    StabilityResult,
    coral_align,
    compute_hallmark_scores,
    compute_progeny_scores,
    compute_viper_scores,
    kmeans_with_stability,
    load_dorothea_resource,
    load_hallmark_sets,
    load_progeny_resource,
    prefix_columns,
    soft_assignments,
    zscore_domain,
)


@dataclass
class PathwayFeatureResult:
    """封装患者与细胞系的通路/TF 活性矩阵。"""

    patient_features: pd.DataFrame
    cell_features: pd.DataFrame


class PathwayFeatureBuilder:
    """负责载入 signature 并计算 PROGENy、DoRothEA、Hallmark 活性。"""

    def __init__(self, cfg, paths):
        self.cfg = cfg
        self.paths = paths

    def _compute_single_domain(self, expr: pd.DataFrame, tag: str) -> pd.DataFrame:
        features: List[pd.DataFrame] = []
        net = load_progeny_resource(self.paths.progeny, top=self.cfg.progeny_top)
        if self.cfg.use_progeny and net is not None:
            prog = compute_progeny_scores(expr, net, tmin=self.cfg.min_targets)
            features.append(prefix_columns(prog, f"{tag}_progeny"))

        regulon = load_dorothea_resource(self.paths.dorothea, levels=self.cfg.dorothea_levels)
        if self.cfg.use_dorothea and regulon is not None:
            tf_scores = compute_viper_scores(expr, regulon, tmin=self.cfg.min_targets)
            features.append(prefix_columns(tf_scores, f"{tag}_tf"))

        hallmark_sets = load_hallmark_sets(self.paths.hallmark)
        if self.cfg.use_hallmark and hallmark_sets:
            hallmark_scores = compute_hallmark_scores(expr, hallmark_sets, threads=self.cfg.hallmark_threads)
            features.append(prefix_columns(hallmark_scores, f"{tag}_hallmark"))

        if not features:
            raise RuntimeError("无法生成任何通路或 TF 特征，请检查配置。")
        merged = pd.concat(features, axis=1)
        merged.index = expr.index
        return merged

    def compute(self, patient_expr: pd.DataFrame, cell_expr: pd.DataFrame) -> PathwayFeatureResult:
        pat_features = self._compute_single_domain(patient_expr, "pat")
        cell_features = self._compute_single_domain(cell_expr, "cell")
        pat_z, _ = zscore_domain(pat_features)
        cell_z, _ = zscore_domain(cell_features)
        aligned_cell = coral_align(cell_z, pat_z, reg=self.cfg.coral_reg)
        return PathwayFeatureResult(patient_features=pat_z, cell_features=aligned_cell)


@dataclass
class SubtypeResult:
    """保存聚类模型与软分配结果。"""

    assignments: pd.DataFrame
    prototypes: pd.DataFrame
    stability: StabilityResult


class SubtypeClusterer:
    """使用 K-means + bootstrap 稳定性选择最佳 K。"""

    def __init__(self, cfg):
        self.cfg = cfg

    def cluster(self, features: pd.DataFrame) -> SubtypeResult:
        stability = kmeans_with_stability(
            features,
            k_values=self.cfg.k_values,
            n_boot=self.cfg.bootstrap,
            sample_frac=self.cfg.sample_frac,
            random_state=self.cfg.random_state,
        )
        model = stability.best_model
        if model is None:
            raise RuntimeError("亚型聚类失败：未找到有效的 KMeans 模型。")
        assignments = soft_assignments(model, features)
        prototypes = pd.DataFrame(model.cluster_centers_, columns=features.columns)
        return SubtypeResult(assignments=assignments, prototypes=prototypes, stability=stability)
