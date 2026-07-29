"""Pipeline DVC de recomendação (Tech Challenge Fase 02)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ml_core_ring.mlflow_setup import ensure_mlflow_experiment
import mlflow
import pandas as pd

from core.configs import settings
from ml_core_ring.artifact_manifest import ArtifactManifest
from ml_core_ring.pipeline_runner import PipelineRunner, register_runner
from ml_core_ring.run_context import ModelCandidate, RunContext
from domains.recommendation.metrics.ranking import aggregate_ranking_metrics
from domains.recommendation.models.factory import create_model

logger = logging.getLogger(__name__)


def _repo_path(*parts: str) -> Path:
    return Path(settings.ml_project_root).joinpath(*parts)


def _load_params(ctx: RunContext) -> dict[str, Any]:
    defaults = {
        "random_state": settings.random_state,
        "top_k": 10,
        "min_interactions": 10000,
        "train_models": ["popularity", "nmf", "torch_embedding"],
        "nmf_components": 32,
        "embedding_dim": 32,
        "hidden_dim": 64,
        "dropout": 0.1,
        "n_epochs": 10,
        "batch_size": 512,
        "lr": 0.01,
        "n_negatives": 4,
        "validation_fraction": 0.1,
        "early_stopping_patience": 3,
        "early_stopping_min_delta": 1e-4,
        "min_positive_rating": 0.0,
        "champion_model": "torch_embedding",
        "mlflow_experiment": "tc02_recommendation",
    }
    merged = {**defaults, **ctx.params}
    return merged


def _ensure_mlflow(experiment: str) -> None:
    ensure_mlflow_experiment(experiment)


@register_runner
class RecommendationPipelineRunner(PipelineRunner):
    runner_id = "recommendation_dvc"

    def preprocess(self, ctx: RunContext) -> Path:
        params = _load_params(ctx)
        raw_dir = _repo_path("data", "recommendation", "raw")
        out_dir = _repo_path("data", "recommendation", "processed")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "interactions.parquet"

        ratings_file = raw_dir / "ratings.csv"
        if not ratings_file.is_file():
            from domains.recommendation.data.download import ensure_movielens_sample

            ensure_movielens_sample(raw_dir)

        df = pd.read_csv(ratings_file)
        df = df.rename(
            columns={
                "userId": "user_id",
                "movieId": "item_id",
                "rating": "rating",
                "timestamp": "timestamp",
            }
        )
        required = {"user_id", "item_id", "rating"}
        if not required.issubset(df.columns):
            raise ValueError(f"ratings.csv deve conter colunas {required}")
        df = df[list(required | {"timestamp"} & set(df.columns))].dropna()
        df["user_id"] = df["user_id"].astype(int)
        df["item_id"] = df["item_id"].astype(int)
        df["rating"] = df["rating"].astype(float)
        if len(df) < int(params["min_interactions"]):
            logger.warning(
                "Dataset com %s interações (< min_interactions=%s). Continuando para MVP académico.",
                len(df),
                params["min_interactions"],
            )
        df.to_parquet(out_path, index=False)
        logger.info("Preprocess concluído: %s linhas → %s", len(df), out_path)
        return out_path

    def feature_eng(self, data: Any, ctx: RunContext) -> dict[str, Path]:
        interactions_path = Path(data) if not isinstance(data, dict) else Path(data["interactions"])
        df = pd.read_parquet(interactions_path)

        df = df.sort_values(["user_id", "timestamp"] if "timestamp" in df.columns else ["user_id"])
        train_rows = []
        test_truth: dict[int, set[int]] = {}
        for user_id, group in df.groupby("user_id"):
            items = group["item_id"].astype(int).tolist()
            if len(items) < 2:
                train_rows.extend(
                    {"user_id": int(user_id), "item_id": int(it), "rating": float(r)}
                    for it, r in zip(items, group["rating"], strict=False)
                )
                continue
            test_item = items[-1]
            test_truth[int(user_id)] = {test_item}
            for it, r in zip(items[:-1], group["rating"].iloc[:-1], strict=False):
                train_rows.append({"user_id": int(user_id), "item_id": int(it), "rating": float(r)})

        train_df = pd.DataFrame(train_rows)
        features_dir = _repo_path("data", "recommendation", "features")
        features_dir.mkdir(parents=True, exist_ok=True)
        train_path = features_dir / "train.parquet"
        test_truth_path = features_dir / "test_truth.json"
        train_df.to_parquet(train_path, index=False)
        test_truth_path.write_text(json.dumps({str(k): list(v) for k, v in test_truth.items()}), encoding="utf-8")
        logger.info("Feature eng: train=%s users_test=%s", len(train_df), len(test_truth))
        return {
            "train": train_path,
            "test_truth": test_truth_path,
            "interactions": interactions_path,
        }

    def train(self, data: Any, ctx: RunContext) -> list[ModelCandidate]:
        params = _load_params(ctx)
        bundle = data if isinstance(data, dict) else {}
        train_path = Path(bundle["train"])
        test_truth_path = Path(bundle["test_truth"])
        train_df = pd.read_parquet(train_path)
        test_truth = {int(k): set(map(int, v)) for k, v in json.loads(test_truth_path.read_text()).items()}
        k = int(params["top_k"])

        candidates: list[ModelCandidate] = []
        models_dir = _repo_path("models", "recommendation")
        models_dir.mkdir(parents=True, exist_ok=True)

        for backend in params["train_models"]:
            model = create_model(
                backend,
                random_state=params["random_state"],
                nmf_components=params["nmf_components"],
                embedding_dim=params["embedding_dim"],
                hidden_dim=params["hidden_dim"],
                dropout=params["dropout"],
                n_epochs=params["n_epochs"],
                batch_size=params["batch_size"],
                lr=params["lr"],
                n_negatives=params["n_negatives"],
                validation_fraction=params["validation_fraction"],
                early_stopping_patience=params["early_stopping_patience"],
                early_stopping_min_delta=params["early_stopping_min_delta"],
                min_positive_rating=params["min_positive_rating"],
            )
            model.fit(train_df)
            recommendations = {
                user: model.recommend(user, k, exclude_items=set())
                for user in test_truth
            }
            metrics = aggregate_ranking_metrics(recommendations, test_truth, k)
            prefix = models_dir / backend
            artifact_paths: dict[str, str] = {}
            if backend == "torch_embedding" and hasattr(model, "save"):
                model.save(prefix)
                artifact_paths["prefix"] = str(prefix)
            manifest = ArtifactManifest(
                domain="recommendation",
                problem_type="recommendation",
                engine="torch_embedding" if backend == "torch_embedding" else f"sklearn_{backend}",
                artifacts=artifact_paths,
                metadata={
                    "backend": backend,
                    "top_k": k,
                    "embedding_dim": params["embedding_dim"],
                    "hidden_dim": params["hidden_dim"],
                    "n_negatives": params["n_negatives"],
                    "min_positive_rating": params["min_positive_rating"],
                },
            )
            candidates.append(
                ModelCandidate(
                    name=model.name,
                    engine=backend,
                    metrics=metrics,
                    manifest=manifest,
                    artifact_paths=artifact_paths,
                )
            )
        metrics_path = models_dir / "train_metrics.json"
        metrics_path.write_text(
            json.dumps({c.name: c.metrics for c in candidates}, indent=2),
            encoding="utf-8",
        )
        return candidates

    def evaluate(self, candidates: list[ModelCandidate], ctx: RunContext) -> ModelCandidate:
        if not candidates:
            raise RuntimeError("Nenhum candidato para avaliar.")
        params = _load_params(ctx)
        metric_winner = max(candidates, key=lambda c: c.metrics.get("ndcg_at_k", 0.0))
        champion_model = str(params.get("champion_model", "")).strip()
        champion = next(
            (c for c in candidates if c.name == champion_model or c.engine == champion_model),
            metric_winner,
        )
        reports_dir = _repo_path("reports", "recommendation")
        reports_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "champion": champion.name,
            "metric_winner": metric_winner.name,
            "selection_reason": (
                "champion_model configurado para produção"
                if champion.name != metric_winner.name
                else "melhor ndcg_at_k"
            ),
            "metrics": champion.metrics,
            "all_models": {c.name: c.metrics for c in candidates},
        }
        (reports_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        manifest_path = _repo_path("models", "recommendation", "champion_manifest.json")
        champion.manifest.to_json_file(manifest_path)
        logger.info("Campeão: %s | ndcg@k=%.4f", champion.name, champion.metrics.get("ndcg_at_k", 0))
        return champion

    def log_mlflow(self, champion: ModelCandidate, ctx: RunContext) -> str | None:
        params = _load_params(ctx)
        experiment = str(params["mlflow_experiment"])
        _ensure_mlflow(experiment)
        with mlflow.start_run(run_name=f"tc02_{champion.name}") as run:
            mlflow.log_params(
                {
                    "domain": "recommendation",
                    "champion": champion.name,
                    "top_k": params["top_k"],
                    "random_state": params["random_state"],
                    "embedding_dim": params["embedding_dim"],
                    "hidden_dim": params["hidden_dim"],
                    "n_epochs": params["n_epochs"],
                    "n_negatives": params["n_negatives"],
                    "min_positive_rating": params["min_positive_rating"],
                }
            )
            mlflow.log_metrics({k: float(v) for k, v in champion.metrics.items()})
            manifest_path = _repo_path("models", "recommendation", "champion_manifest.json")
            if manifest_path.is_file():
                mlflow.log_artifact(str(manifest_path), artifact_path="manifest")
            if champion.engine == "torch_embedding" and champion.artifact_paths.get("prefix"):
                prefix = Path(champion.artifact_paths["prefix"])
                for suffix in (".pt", ".meta.json"):
                    p = prefix.with_suffix(suffix)
                    if p.is_file():
                        mlflow.log_artifact(str(p), artifact_path="model")
                try:
                    mlflow.pytorch.log_model(
                        pytorch_model=_load_torch_module(prefix),
                        artifact_path="pytorch_model",
                    )
                    mlflow.register_model(
                        f"runs:/{run.info.run_id}/pytorch_model",
                        "tc02_recommender",
                    )
                except Exception as exc:
                    # Cliente MLflow 3.x vs servidor 2.x: registry/logged-models pode falhar;
                    # artefactos .pt já foram registados — treino não deve abortar.
                    logger.warning(
                        "mlflow.pytorch.log_model/register_model ignorado (continua run): %s",
                        exc,
                    )
            return run.info.run_id
        return None


def _load_torch_module(prefix: Path):
    from domains.recommendation.models.embedding_model import TorchEmbeddingRecommender

    rec = TorchEmbeddingRecommender.load(prefix)
    if rec._model is None:
        raise RuntimeError("Modelo torch inválido para MLflow.")
    return rec._model


def build_run_context(params: dict[str, Any] | None = None) -> RunContext:
    loaded: dict[str, Any] = {}
    params_path = _repo_path("params.yaml")
    if params_path.is_file():
        try:
            import yaml

            loaded = yaml.safe_load(params_path.read_text(encoding="utf-8")) or {}
        except ImportError as exc:
            raise ImportError(
                "PyYAML necessário para ler params.yaml (pip install pyyaml ou uv sync --extra tc02)."
            ) from exc
    merged = {**(loaded or {}), **(params or {})}
    return RunContext(
        domain="recommendation",
        params=merged,
        mlflow_experiment=str(merged.get("mlflow_experiment", "tc02_recommendation")),
    )
