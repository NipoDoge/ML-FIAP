"""Treino sync de recomendação (debug API) — delega worker ou in-process + BD."""

from __future__ import annotations

import os
from typing import Any

import httpx

from core.configs import settings
from executors_ring.recommendation.persist_run import persist_recommendation_run, run_async
from executors_ring.recommendation.worker import run_training_job
from ml_core_ring.dispatch_params import flatten_dispatch_train_params


def _worker_url() -> str:
    return (
        settings.worker_recommendation_url or os.environ.get("WORKER_RECOMMENDATION_URL", "")
    ).strip()


def train_recommendation_sync(
    *,
    user_id: int,
    params: dict[str, Any] | None = None,
    domain: str = "recommendation",
) -> dict[str, Any]:
    """
    Executa treino reco síncrono (worker HTTP ou in-process) e persiste ``pipeline_runs``.

    Returns
    -------
    dict
        ``pipeline_run_id``, ``champion_name``, ``metrics``, ``mlflow_run_id``.
    """
    import domains  # noqa: F401
    import executors_ring  # noqa: F401

    flat = flatten_dispatch_train_params({**(params or {}), "domain": domain})
    worker = _worker_url()

    if worker:
        payload = {
            "domain": domain.strip().lower(),
            "user_id": int(user_id),
            "params": flat,
            "airflow_dag_run_id": None,
        }
        base = worker.rstrip("/")
        with httpx.Client(timeout=7200.0) as client:
            resp = client.post(f"{base}/train", json=payload)
            resp.raise_for_status()
            data = resp.json()
        return {
            "pipeline_run_id": int(data["pipeline_run_id"]),
            "champion_name": data.get("champion_name"),
            "metrics": data.get("metrics") or {},
            "mlflow_run_id": data.get("mlflow_run_id"),
            "status": "completed",
        }

    result = run_training_job({**flat, "domain": domain.strip().lower()})
    if result.status != "completed":
        raise RuntimeError(f"Treino não concluído (status={result.status!r}): {result.detail}")

    run_id = run_async(
        persist_recommendation_run(
            user_id=int(user_id),
            result=result,
            airflow_dag_run_id=None,
        )
    )
    return {
        "pipeline_run_id": run_id,
        "champion_name": result.run_result.champion_name if result.run_result else None,
        "metrics": result.metrics or {},
        "mlflow_run_id": result.mlflow_run_id,
        "status": "completed",
    }
