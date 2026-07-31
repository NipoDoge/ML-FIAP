"""Treino de recomendação via worker HTTP ou in-process (fallback local)."""

from __future__ import annotations

import logging
import os

import domains  # noqa: F401
import executors_ring  # noqa: F401

from orchestration_ring.airflow_env import DEFAULT_PIPELINE_USER_ID
from orchestration_ring.conf import merge_run_conf
from orchestration_ring.persist_run import persist_recommendation_run, run_async
from ml_core_ring.dispatch_params import flatten_dispatch_train_params
from orchestration_ring.recommendation_client import run_recommendation_via_worker

logger = logging.getLogger(__name__)


def task_run_recommendation(**context) -> None:
    """Executa treino no worker integrado (Docker) ou in-process (dev sem worker)."""
    ti = context["task_instance"]
    domain = ti.xcom_pull(key="domain", task_ids="validate_dispatch")
    user_id = int(
        ti.xcom_pull(key="user_id", task_ids="validate_dispatch") or DEFAULT_PIPELINE_USER_ID
    )

    conf = merge_run_conf(context, variable_key="ml_training_dispatch_conf")
    train_params = flatten_dispatch_train_params(conf)
    logger.info("Parâmetros de treino reco (flatten): %s", list(train_params.keys()))
    dag_run_id = context["dag_run"].run_id

    worker_url = os.environ.get("WORKER_RECOMMENDATION_URL", "").strip()

    if worker_url:
        payload = run_recommendation_via_worker(
            worker_url=worker_url,
            domain=domain,
            user_id=user_id,
            params=train_params,
            airflow_dag_run_id=dag_run_id,
        )
        run_id = payload.get("pipeline_run_id")
        metrics = payload.get("metrics") or {}
        champion = payload.get("champion_name")
        mlflow_run_id = payload.get("mlflow_run_id")
    else:
        from ml_core_ring.orchestration_hooks import run_training_for_domain

        logger.info("WORKER_RECOMMENDATION_URL vazio — treino in-process | domain=%s", domain)
        result = run_training_for_domain(domain, train_params)
        if result.status != "completed":
            raise RuntimeError(
                f"Treino recomendação não concluído (status={result.status!r}): {result.detail}"
            )
        run_id = run_async(
            persist_recommendation_run(
                user_id=user_id,
                result=result,
                airflow_dag_run_id=dag_run_id,
            )
        )
        metrics = result.metrics
        champion = result.run_result.champion_name if result.run_result else None
        mlflow_run_id = result.mlflow_run_id

    ti.xcom_push(key="recommendation_pipeline_run_id", value=run_id)
    ti.xcom_push(key="recommendation_metrics", value=metrics)
    ti.xcom_push(key="recommendation_champion", value=champion)
    ti.xcom_push(key="mlflow_run_id", value=mlflow_run_id)
    logger.info(
        "Recomendação concluída | pipeline_run_id=%s | champion=%s | worker=%s",
        run_id,
        champion,
        worker_url or "in-process",
    )


def task_notify_recommendation_complete(**context) -> None:
    ti = context["task_instance"]
    domain = ti.xcom_pull(key="domain", task_ids="validate_dispatch")
    run_id = ti.xcom_pull(key="recommendation_pipeline_run_id", task_ids="run_recommendation")
    champion = ti.xcom_pull(key="recommendation_champion", task_ids="run_recommendation")
    metrics = ti.xcom_pull(key="recommendation_metrics", task_ids="run_recommendation")

    logger.info("=" * 60)
    logger.info("PIPELINE RECOMENDAÇÃO CONCLUÍDO | domain=%s", domain)
    logger.info("pipeline_run_id : %s", run_id)
    logger.info("campeão         : %s", champion)
    logger.info("métricas        : %s", metrics)
    logger.info("=" * 60)
