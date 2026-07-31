"""
DAG: ml_training_dispatch

Orquestrador central de treino por ``domain`` (Fase 3):

  validate_dispatch → branch
    tabular (churn, …): deactivate → baseline → FE → promote opcional → notify
    recommendation:     run_recommendation → notify

Conf efectiva = merge(Variable ``ml_training_dispatch_conf``, ``dag_run.conf``).
Bootstrap: ``airflow/bootstrap/ml_training_dispatch_conf.json``.

Exemplos de conf
----------------
Tabular (churn)::

  {"domain": "churn", "csv_path": "/opt/airflow/ml_project/uploads/churn.csv", "optimization_metric": "recall"}

Recomendação::

  {"domain": "recommendation", "top_k": 10, "user_id": 2}
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

# Bootstrap antes de importar orchestration_ring (Airflow parseia o módulo top-level).
_ML_CODE = os.environ.get("ML_CODE_ROOT", "/opt/airflow/ml_code")
_ML_ROOT = os.environ.get("ML_PROJECT_ROOT", "/opt/airflow/ml_project")
_ML_LIBS = os.environ.get("ML_AIRFLOW_SITE_PACKAGES", "/opt/airflow/ml_libs")
for _p in (_ML_LIBS, _ML_CODE, _ML_ROOT):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, os.path.abspath(_p))

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

from orchestration_ring.airflow_env import bootstrap_ml_sys_path, prepend_airflow_ml_site_packages

bootstrap_ml_sys_path()

DEFAULT_ARGS = {
    "owner": "ml-engineering",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


def _wrap_task(fn):
    def _inner(**context):
        prepend_airflow_ml_site_packages()
        return fn(**context)

    return _inner


with DAG(
    dag_id="ml_training_dispatch",
    description="Dispatch central de treino por domain (tabular | recommendation).",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["ml", "training", "dispatch"],
) as dag:
    from orchestration_ring.dispatch_tasks import task_validate_dispatch
    from orchestration_ring.recommendation_training import (
        task_notify_recommendation_complete,
        task_run_recommendation,
    )
    from orchestration_ring.tabular_training import (
        task_deactivate_manual_runs,
        task_notify_tabular_complete,
        task_promote_fe_optional,
        task_run_baseline,
        task_run_fe,
    )

    _dispatch_ctx = {
        "validate_task_id": "validate_dispatch",
        "baseline_task_id": "run_tabular_baseline",
        "fe_task_id": "run_tabular_fe",
    }

    validate_dispatch = BranchPythonOperator(
        task_id="validate_dispatch",
        python_callable=_wrap_task(task_validate_dispatch),
    )

    deactivate_manual = PythonOperator(
        task_id="deactivate_manual_runs",
        python_callable=_wrap_task(task_deactivate_manual_runs),
        op_kwargs=_dispatch_ctx,
    )
    run_tabular_baseline = PythonOperator(
        task_id="run_tabular_baseline",
        python_callable=_wrap_task(task_run_baseline),
        op_kwargs=_dispatch_ctx,
    )
    run_tabular_fe = PythonOperator(
        task_id="run_tabular_fe",
        python_callable=_wrap_task(task_run_fe),
        op_kwargs=_dispatch_ctx,
    )
    promote_tabular = PythonOperator(
        task_id="promote_tabular_optional",
        python_callable=_wrap_task(task_promote_fe_optional),
        op_kwargs=_dispatch_ctx,
    )
    notify_tabular = PythonOperator(
        task_id="notify_tabular_complete",
        python_callable=_wrap_task(task_notify_tabular_complete),
        op_kwargs=_dispatch_ctx,
    )

    run_recommendation = PythonOperator(
        task_id="run_recommendation",
        python_callable=_wrap_task(task_run_recommendation),
    )
    notify_recommendation = PythonOperator(
        task_id="notify_recommendation_complete",
        python_callable=_wrap_task(task_notify_recommendation_complete),
    )

    skip_tabular = EmptyOperator(task_id="skip_tabular_branch")
    skip_recommendation = EmptyOperator(task_id="skip_recommendation_branch")
    join = EmptyOperator(task_id="join", trigger_rule="none_failed_min_one_success")

    validate_dispatch >> [deactivate_manual, run_recommendation]
    deactivate_manual >> run_tabular_baseline >> run_tabular_fe >> promote_tabular >> notify_tabular
    notify_tabular >> skip_recommendation >> join
    run_recommendation >> notify_recommendation >> skip_tabular >> join
