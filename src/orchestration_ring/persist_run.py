"""
Persistência partilhada Airflow → ``pipeline_runs``.

Tabular (baseline/FE): delega a ``services.processor.airflow_persistence`` (transição Fase 4).
Recomendação: re-export de ``executors_ring.recommendation.persist_run``.
"""

from __future__ import annotations

from executors_ring.recommendation.persist_run import persist_recommendation_run

# Re-export tabular (legado platform — migrar na Fase 4)
from services.processor.airflow_persistence import (
    deactivate_manual_pipeline_runs_for_objective,
    persist_airflow_baseline_run,
    persist_airflow_feature_engineering_run,
    promote_airflow_fe_if_requested,
    reserve_airflow_fe_pipeline_run,
    run_async,
)

__all__ = [
    "deactivate_manual_pipeline_runs_for_objective",
    "persist_airflow_baseline_run",
    "persist_airflow_feature_engineering_run",
    "persist_recommendation_run",
    "promote_airflow_fe_if_requested",
    "reserve_airflow_fe_pipeline_run",
    "run_async",
]
