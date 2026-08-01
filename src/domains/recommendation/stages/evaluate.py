"""Entrypoint DVC — evaluate (+ MLflow)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from domains.recommendation.pipeline_runner import RecommendationPipelineRunner, build_run_context
from ml_core_ring.artifact_manifest import ArtifactManifest
from ml_core_ring.run_context import ModelCandidate

logging.basicConfig(level=logging.INFO)


def _candidate_manifest(name: str, params: dict) -> ArtifactManifest:
    artifacts = {}
    metadata = {
        "backend": name,
        "top_k": int(params.get("top_k", 10)),
    }
    engine = f"sklearn_{name}"
    if "torch" in name:
        artifacts["prefix"] = str(Path("models/recommendation/torch_embedding"))
        engine = "torch_embedding"
        metadata.update(
            {
                "embedding_dim": int(params.get("embedding_dim", 32)),
                "hidden_dim": int(params.get("hidden_dim", 64)),
                "use_item_bias": bool(params.get("use_item_bias", True)),
                "n_negatives": int(params.get("n_negatives", 4)),
                "min_positive_rating": float(params.get("min_positive_rating", 0.0)),
            }
        )
    return ArtifactManifest(
        domain="recommendation",
        problem_type="recommendation",
        engine=engine,
        artifacts=artifacts,
        metadata=metadata,
    )


def main() -> None:
    ctx = build_run_context()
    runner = RecommendationPipelineRunner()
    metrics_path = Path("models/recommendation/train_metrics.json")
    if not metrics_path.is_file():
        raise FileNotFoundError("Execute train antes de evaluate.")
    raw = json.loads(metrics_path.read_text(encoding="utf-8"))

    candidates = []
    for name, metrics in raw.items():
        manifest = _candidate_manifest(name, ctx.params)
        candidates.append(
            ModelCandidate(
                name=name,
                engine=name.split("_")[0] if "_" in name else name,
                metrics=metrics,
                manifest=manifest,
                artifact_paths=manifest.artifacts,
            )
        )
    champion = runner.evaluate(candidates, ctx)
    runner.log_mlflow(champion, ctx)


if __name__ == "__main__":
    main()
