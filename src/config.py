"""
集中维护训练流程所需的所有配置数据类。
通过在一个文件内定义默认路径和参数，可以方便地在 main.py 中载入。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class DataPaths:
    """模型依赖的所有数据文件路径。"""

    processed_dir: Path = Path("data")
    aml_expr: Path = processed_dir / "Processed_AML2_Expression_matrix.csv"
    ccle_expr: Path = processed_dir / "Processed_CCLE_Expression_filtered.csv"
    aml_combo: Path = processed_dir / "AML2_drugcomb.csv"
    pdsp_combo: Path = processed_dir / "PDSP_drugcomb_filtered.csv"
    pdsp_drug_smiles: Path = processed_dir / "PDSP_drug_smiles.csv"
    patient_drug_smiles: Path = processed_dir / "Drug_Smile.csv"
    annotation: Path = Path("pre") / "raw_data" / "data" / "AML2_Expression.txt"
    progeny: Path = Path("pre") / "raw_data" / "data" / "signatures" / "progeny_human_full.csv"
    dorothea: Path = Path("pre") / "raw_data" / "data" / "signatures" / "dorothea_hs_full.csv"
    hallmark: Path = Path("pre") / "raw_data" / "data" / "signatures" / "hallmark_human.gmt"
    output_dir: Path = Path("outputs")
    model_dir: Path = Path("models")


@dataclass
class PathwayConfig:
    """通路/TF 活性推断相关配置。"""

    use_progeny: bool = True
    use_dorothea: bool = True
    use_hallmark: bool = True
    progeny_top: int = 500
    dorothea_levels: Tuple[str, ...] = ("A", "B")
    hallmark_threads: int = 4
    coral_reg: float = 1e-6
    min_targets: int = 5


@dataclass
class ClusteringConfig:
    """患者亚型聚类配置。"""

    k_values: Tuple[int, ...] = (2, 3, 4, 5)
    bootstrap: int = 50
    sample_frac: float = 0.8
    random_state: int = 42


@dataclass
class CodeAEConfig:
    """CODE-AE 共享编码器训练配置。"""

    latent_dim: int = 128
    encoder_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    classifier_hidden_dims: Tuple[int, ...] = (256, 128)
    dop: float = 0.1
    num_geo_layer: int = 1
    norm_flag: bool = True
    lr: float = 1e-3
    pretrain_epochs: int = 50
    adv_epochs: int = 200
    batch_size: int = 256
    retrain: bool = True
    model_name: str = "code_ae"


@dataclass
class TrainingConfig:
    """下游协同预测模型的训练参数。"""

    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    ensure_balanced_test: bool = True
    cell_synergy_column: str = "synergy_zip"
    cell_synergy_threshold: float = 10.0
    patient_label_column: str = "synergy_label"
    batch_size: int = 512
    lr: float = 2e-4
    weight_decay: float = 1e-4
    dropout: float = 0.3
    hidden_dims: Tuple[int, ...] = (1024, 512, 256, 128)
    max_epochs: int = 50
    patience: int = 8
    domain_mix_ratio: float = 0.7
    focal_gamma: float = 1.5
    loss_strategy: str = "class_weight"


@dataclass
class AblationConfig:
    """不同消融实验开关。"""

    skip_pathway: bool = False
    skip_codeae: bool = False
    skip_subtype_finetune: bool = False
    skip_subtype_weighting: bool = False
    description: str = "full"


@dataclass
class PipelineConfig:
    """完整流程的聚合配置。"""

    data: DataPaths = field(default_factory=DataPaths)
    pathway: PathwayConfig = field(default_factory=PathwayConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    codeae: CodeAEConfig = field(default_factory=CodeAEConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    random_seed: int = 42
