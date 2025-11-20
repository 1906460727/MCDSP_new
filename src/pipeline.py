"""主流程封装，负责串联特征工程、模型训练与消融实验。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.config import PipelineConfig, AblationConfig
from src.data_utils import (
    align_expression_domains,
    balanced_test_split,
    ensure_dir,
    load_processed_expression,
    save_json,
    set_seed,
)
from src.pathway_features import PathwayFeatureBuilder, SubtypeClusterer
from src.codeae_manager import CodeAEManager
from src.drug_features import DrugFeaturizer, canonicalise_drug_name
from src.datasets import (
    CombinationBuilder,
    CombinationDataset,
    CombinationRecord,
    build_context_feature_matrix,
    pca_project,
)
from src.model import PairSynergyNet
from src.trainer import Trainer, build_sampler, compute_metrics


class ResearchPipeline:
    """封装完整实验流程，提供 `run` 方法外部调用。"""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.paths = cfg.data
        ensure_dir(self.paths.output_dir)
        ensure_dir(self.paths.model_dir)
        set_seed(cfg.random_seed)

    def _load_expression(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        patient_expr = load_processed_expression(self.paths.aml_expr, "stable_id", self.paths.annotation)
        cell_expr = load_processed_expression(self.paths.ccle_expr, "gene_id", self.paths.annotation)
        return align_expression_domains(patient_expr, cell_expr)

    def _build_context_features(
        self,
        ablation: AblationConfig,
        pathway_res,
        encoder,
        patient_expr: pd.DataFrame,
        cell_expr: pd.DataFrame,
        use_codeae: bool,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        embeddings_pat = None
        embeddings_cell = None
        if use_codeae and encoder is not None:
            embeddings_pat = CodeAEManager.encode(encoder, patient_expr)
            embeddings_cell = CodeAEManager.encode(encoder, cell_expr)
            if not np.isfinite(embeddings_pat.to_numpy()).all() or not np.isfinite(embeddings_cell.to_numpy()).all():
                use_codeae = False
                embeddings_pat = None
                embeddings_cell = None
        if not use_codeae:
            embeddings_pat = pca_project(patient_expr)
            embeddings_cell = pca_project(cell_expr)

        pat_matrix = build_context_feature_matrix(
            patient_expr,
            embeddings_pat,
            None if ablation.skip_pathway else pathway_res.patient_features,
            use_embeddings=use_codeae,
            use_pathway=not ablation.skip_pathway,
        )
        cell_matrix = build_context_feature_matrix(
            cell_expr,
            embeddings_cell,
            None if ablation.skip_pathway else pathway_res.cell_features,
            use_embeddings=use_codeae,
            use_pathway=not ablation.skip_pathway,
        )
        return pat_matrix, cell_matrix

    def _prepare_patient_splits(
        self,
        pat_records: List[CombinationRecord],
        subtype_probs: pd.DataFrame,
    ) -> Dict[str, List[CombinationRecord]]:
        df = pd.DataFrame(
            [
                {
                    "drug_a": rec.drug_a,
                    "drug_b": rec.drug_b,
                    "context_id": rec.context_id,
                    "label": rec.label,
                }
                for rec in pat_records
            ]
        )
        train_df, val_df, test_df = balanced_test_split(
            df,
            label_col="label",
            train_ratio=self.cfg.training.train_ratio,
            val_ratio=self.cfg.training.val_ratio,
            test_ratio=self.cfg.training.test_ratio,
            ensure_balanced_test=self.cfg.training.ensure_balanced_test,
            seed=self.cfg.random_seed,
        )

        def to_records(frame: pd.DataFrame) -> List[CombinationRecord]:
            records = []
            for _, row in frame.iterrows():
                probs = subtype_probs.loc[row["context_id"]].to_numpy(dtype=np.float32)
                records.append(
                    CombinationRecord(
                        drug_a=row["drug_a"],
                        drug_b=row["drug_b"],
                        context_id=row["context_id"],
                        domain="patient",
                        label=int(row["label"]),
                        subtype_probs=probs,
                    )
                )
            return records

        return {"train": to_records(train_df), "val": to_records(val_df), "test": to_records(test_df)}

    def _build_drug_dict(
        self, featurizer: DrugFeaturizer, combos: Iterable[CombinationRecord]
    ) -> Tuple[Dict[str, np.ndarray], set[str]]:
        unique = {canonicalise_drug_name(rec.drug_a) for rec in combos}
        unique.update(canonicalise_drug_name(rec.drug_b) for rec in combos)
        features = {}
        missing = set()
        for drug in unique:
            try:
                features[drug] = featurizer.featurize(drug)
            except KeyError:
                missing.add(drug)
        return features, missing

    def _build_loaders(
        self,
        splits: Dict[str, List[CombinationRecord]],
        context_matrix: pd.DataFrame,
        drug_dict: Dict[str, np.ndarray],
        subtype_dim: int,
        ablation: AblationConfig,
    ) -> Dict[str, DataLoader]:
        datasets = {
            split: CombinationDataset(records, context_matrix, drug_dict, ablation, subtype_dim)
            for split, records in splits.items()
        }
        sampler = build_sampler(splits["train"], self.cfg.training)
        return {
            "train": DataLoader(datasets["train"], batch_size=self.cfg.training.batch_size, sampler=sampler),
            "val": DataLoader(datasets["val"], batch_size=self.cfg.training.batch_size, shuffle=False),
            "test": DataLoader(datasets["test"], batch_size=self.cfg.training.batch_size, shuffle=False),
        }

    def _run_single(self, ablation: AblationConfig) -> Dict[str, Dict[str, float]]:
        patient_expr, cell_expr = self._load_expression()
        pathway_builder = PathwayFeatureBuilder(self.cfg.pathway, self.paths)
        pathway_res = pathway_builder.compute(patient_expr, cell_expr)
        subtype_clusterer = SubtypeClusterer(self.cfg.clustering)
        subtype_res = subtype_clusterer.cluster(pathway_res.patient_features)
        subtype_probs = subtype_res.assignments

        encoder = None
        use_codeae = not ablation.skip_codeae
        if use_codeae:
            try:
                encoder = CodeAEManager(self.cfg.codeae, self.paths.model_dir).train_or_load(patient_expr, cell_expr)
            except Exception as exc:
                use_codeae = False
                print(f"CODE-AE 训练失败，回退到 PCA：{exc}")
        pat_matrix, cell_matrix = self._build_context_features(
            ablation, pathway_res, encoder, patient_expr, cell_expr, use_codeae
        )

        builder = CombinationBuilder(self.cfg.training, ablation)
        pdsp_df = pd.read_csv(self.paths.pdsp_combo)
        aml_df = pd.read_csv(self.paths.aml_combo)
        cell_records = builder.load_cell_line_records(pdsp_df)
        pat_records = builder.load_patient_records(aml_df, pat_matrix.index, subtype_probs)

        featurizer = DrugFeaturizer(
            [
                self.paths.pdsp_drug_smiles,
                self.paths.patient_drug_smiles,
            ]
        )
        drug_dict, missing_drugs = self._build_drug_dict(featurizer, cell_records + pat_records)
        if missing_drugs:
            logging.warning("以下药物缺少 SMILES，已从训练集中剔除：%s", ", ".join(sorted(list(missing_drugs))[:10]))

            def filter_records(records: List[CombinationRecord]) -> List[CombinationRecord]:
                return [
                    rec
                    for rec in records
                    if canonicalise_drug_name(rec.drug_a) not in missing_drugs
                    and canonicalise_drug_name(rec.drug_b) not in missing_drugs
                ]

            cell_records = filter_records(cell_records)
            pat_records = filter_records(pat_records)

        splits_pat = self._prepare_patient_splits(pat_records, subtype_probs)
        splits = {
            "train": cell_records + splits_pat["train"],
            "val": splits_pat["val"],
            "test": splits_pat["test"],
        }
        context_matrix = pd.concat([cell_matrix, pat_matrix], axis=0)
        loaders = self._build_loaders(splits, context_matrix, drug_dict, subtype_probs.shape[1], ablation)

        sample_drug = next(iter(drug_dict.values()))
        model = PairSynergyNet(
            drug_dim=len(sample_drug),
            context_dim=context_matrix.shape[1],
            hidden_dims=self.cfg.training.hidden_dims,
            dropout=self.cfg.training.dropout,
        ).to(Trainer(self.cfg.training).device)
        trainer = Trainer(self.cfg.training)
        model = trainer.train(loaders["train"], loaders["val"], model, pos_weight=3.0, gamma=self.cfg.training.focal_gamma)
        preds = trainer._predict(model, loaders["test"])
        metrics_test = compute_metrics(preds["y_true"], preds["y_score"])
        return {"test": metrics_test, "ablation": ablation.__dict__}

    def run(self, run_ablations: bool = False) -> None:
        """执行主实验及可选消融实验，并将结果写入 outputs。"""

        base = self._run_single(self.cfg.ablation)
        save_json(base, self.paths.output_dir / f"results_{self.cfg.ablation.description}.json")
        if not run_ablations:
            return
        ablations = [
            AblationConfig(skip_pathway=True, description="no_pathway"),
            AblationConfig(skip_codeae=True, description="no_codeae"),
            AblationConfig(skip_subtype_finetune=True, description="no_subtype_finetune"),
            AblationConfig(skip_subtype_weighting=True, description="no_subtype_weight"),
            AblationConfig(description="focal_loss"),
        ]
        for abl in ablations:
            result = self._run_single(abl)
            save_json(result, self.paths.output_dir / f"results_{abl.description}.json")
