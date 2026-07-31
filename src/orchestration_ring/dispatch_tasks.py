"""Tasks partilhadas da DAG dispatch (validação + branch)."""

from __future__ import annotations

import logging

from orchestration_ring.airflow_env import DEFAULT_PIPELINE_USER_ID
from orchestration_ring.conf import merge_run_conf
from orchestration_ring.dispatch import resolve_domain, validate_dispatch_conf

logger = logging.getLogger(__name__)


def task_validate_dispatch(**context) -> str:
    """
    Valida conf, grava metadados em XCom e devolve o task_id seguinte (branch).

    Returns
    -------
    str
        ``run_tabular_baseline`` ou ``run_recommendation`` (BranchPythonOperator).
    """
    conf = merge_run_conf(
        context,
        variable_key="ml_training_dispatch_conf",
        fallback_keys=(
            ("objective", "ml_training_objective"),
            ("csv_path", "ml_training_csv_path"),
        ),
    )
    route = validate_dispatch_conf(conf)
    domain = resolve_domain(conf)

    ti = context["task_instance"]
    ti.xcom_push(key="domain", value=domain)
    ti.xcom_push(key="route", value=route)
    ti.xcom_push(key="user_id", value=int(conf.get("user_id", DEFAULT_PIPELINE_USER_ID)))

    if route == "tabular":
        from orchestration_ring.tabular_training import task_validate_tabular_input

        task_validate_tabular_input(**context)
        logger.info("Dispatch → tabular (%s)", domain)
        return "deactivate_manual_runs"

    logger.info("Dispatch → recommendation (%s)", domain)
    return "run_recommendation"
