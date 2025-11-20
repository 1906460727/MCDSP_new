"""项目入口，解析命令行参数并启动完整流程。"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from src.config import PipelineConfig
from src.pipeline import ResearchPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-drug synergy pipeline")
    parser.add_argument("--config", type=Path, help="可选 JSON 配置文件，覆盖默认参数。")
    parser.add_argument("--run-ablations", action="store_true", help="是否运行全部消融实验。")
    parser.add_argument("--log-level", default="INFO", help="Python logging 等级。")
    return parser.parse_args()


def load_config(path: Path | None) -> PipelineConfig:
    cfg = PipelineConfig()
    if not path:
        return cfg
    payload = json.loads(path.read_text(encoding="utf-8"))

    def update(obj: Any, data: dict):
        for key, value in data.items():
            attr = getattr(obj, key)
            if hasattr(attr, "__dict__"):
                update(attr, value)
            else:
                setattr(obj, key, value)

    update(cfg, payload)
    return cfg


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    cfg = load_config(args.config)
    pipeline = ResearchPipeline(cfg)
    pipeline.run(run_ablations=args.run_ablations)


if __name__ == "__main__":
    main()
