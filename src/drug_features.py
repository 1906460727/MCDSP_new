"""药物指纹特征模块，统一处理 SMILES 合并与 Morgan 指纹生成。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd


def canonicalise_drug_name(name: str) -> str:
    """将药物名称转为统一的小写格式，便于匹配。"""

    return name.strip().lower()


class DrugFeaturizer:
    """封装 RDKit 的 Morgan 指纹（新版 MorganGenerator API）。"""

    def __init__(self, smiles_paths: Sequence[Path], radius: int = 2, n_bits: int = 1024):
        try:
            from rdkit import Chem
            from rdkit import DataStructs
            from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
        except ImportError as exc:  # pragma: no cover
            raise ImportError("缺少 RDKit 依赖，无法生成药物指纹。") from exc

        self.Chem = Chem
        self.DataStructs = DataStructs
        self.generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
        self.n_bits = n_bits

        self.smiles: Dict[str, str] = {}
        for path in smiles_paths:
            if not path or not path.exists():
                continue
            df = pd.read_csv(path)
            if not {"drug_name", "smile"}.issubset(df.columns):
                continue
            for name, smile in zip(df["drug_name"], df["smile"]):
                key = canonicalise_drug_name(str(name))
                if key and key not in self.smiles and isinstance(smile, str):
                    self.smiles[key] = smile
        if not self.smiles:
            raise ValueError("未能从任何 SMILES 文件中读取到药物定义。")
        self.cache: Dict[str, np.ndarray] = {}

    def _featurize_smile(self, smile: str) -> np.ndarray:
        mol = self.Chem.MolFromSmiles(smile)
        if mol is None:
            raise ValueError(f"无效的 SMILES: {smile}")
        fp = self.generator.GetFingerprint(mol)
        arr = np.zeros((self.n_bits,), dtype=np.float32)
        self.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def featurize(self, drug_name: str) -> np.ndarray:
        key = canonicalise_drug_name(drug_name)
        if key in self.cache:
            return self.cache[key]
        smile = self.smiles.get(key)
        if smile is None:
            raise KeyError(f"未找到药物 {drug_name} 的 SMILES。")
        fp = self._featurize_smile(smile)
        self.cache[key] = fp
        return fp
