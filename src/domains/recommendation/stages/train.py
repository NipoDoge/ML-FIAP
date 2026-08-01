"""Entrypoint DVC — train."""

from __future__ import annotations

import logging
from pathlib import Path

from domains.recommendation.pipeline_runner import RecommendationPipelineRunner, build_run_context

logging.basicConfig(level=logging.INFO)


def main() -> None:
    ctx = build_run_context()
    runner = RecommendationPipelineRunner()
    features_dir = Path("data/recommendation/features")
    bundle = {
        "train": features_dir / "train.parquet",
        "test_truth": features_dir / "test_truth.json",
        "interactions": Path("data/recommendation/processed/interactions.parquet"),
    }
    for path in bundle.values():
        if not Path(path).is_file():
            raise FileNotFoundError(f"Dependência em falta para train: {path}")
    runner.train(bundle, ctx)


if __name__ == "__main__":
    main()
