"""Persistência de runs de recomendação (worker HTTP + fallback in-process)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import models._all_models  # noqa: F401
from core.database import Session, engine
from ml_core_ring.train_backend import TrainBackendResult
from models.pipeline_runs import PipelineRuns
from services.utils import utcnow

logger = logging.getLogger(__name__)


def run_async(coro):
    """Executa corrotina com dispose do engine no mesmo loop (Airflow/worker sync)."""

    async def _with_cleanup() -> Any:
        try:
            return await coro
        finally:
            await engine.dispose()

    return asyncio.run(_with_cleanup())


async def persist_recommendation_run(
    *,
    user_id: int,
    result: TrainBackendResult,
    airflow_dag_run_id: str | None = None,
) -> int:
    """Grava run de recomendação concluído (pipeline_type=recommendation)."""
    if result.run_result is None:
        raise ValueError(
            "TrainBackendResult.run_result é obrigatório para persistência de recomendação."
        )

    manifest = result.run_result.manifest
    metrics: dict[str, Any] = dict(result.metrics or {})
    metrics["champion_name"] = result.run_result.champion_name
    metrics["domain"] = manifest.domain
    metrics["problem_type"] = manifest.problem_type
    metrics["manifest_engine"] = manifest.engine
    if manifest.artifacts:
        metrics["artifact_paths"] = dict(manifest.artifacts)
    if result.mlflow_run_id:
        metrics["mlflow_run_id"] = result.mlflow_run_id
    if airflow_dag_run_id:
        metrics["airflow_dag_run_id"] = airflow_dag_run_id
    metrics["airflow_dag"] = "ml_training_dispatch"

    model_path = manifest.artifacts.get("prefix") or manifest.artifacts.get("joblib_path")
    inference_backend = "sklearn"
    if "torch" in str(manifest.engine).lower():
        inference_backend = "mlp"

    session = Session()
    try:
        run = PipelineRuns(
            user_id=int(user_id),
            pipeline_type="recommendation",
            objective=manifest.domain,
            status="completed",
            original_filename=str(metrics.get("data_source", "recommendation_pipeline")),
            model_path=str(model_path) if model_path else None,
            metrics=metrics,
            completed_at=utcnow(),
            is_airflow_run=True,
            inference_backend=inference_backend,
        )
        session.add(run)
        await session.commit()
        await session.refresh(run)
        logger.info("Recomendação guardada (pipeline_run_id=%s).", run.id)
        return run.id
    finally:
        await session.close()
