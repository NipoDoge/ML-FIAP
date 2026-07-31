"""
DAG: ml_drift_monitoring

Monitorização offline de **drift de dados** (PSI): exporta predições da BD PostgreSQL
para CSV e executa ``src/scripts/maintenance/drift_report.py`` contra um CSV de referência
(alinhado ao contrato de features em produção, tipicamente o ``baseline_sample`` activo).

**Não** faz parte do pipeline de treino — corre em paralelo temporalmente à vida do modelo
(habitualmente semanal ou sob demanda).

Configuração (merge Airflow Variable ``drift_monitoring_conf`` + ``dag_run.conf`` do trigger)::

    {
      "objective": "churn",
      "train_csv_path": "/opt/airflow/ml_project/src/data/pre_processed/baseline_sample_automatic.csv",
      "predictions_row_limit": 100000,
      "target_col": "target"
    }

- ``train_csv_path``: referência PSI (sem coluna ``target`` na comparação — o script remove-a).
  Caminho relativo ao projecto resolve-se via ``ML_PROJECT_ROOT``.
- ``predictions_row_limit``: tecto de linhas exportadas (ordenadas por ``id``).

Saída: CSV timestampado emitido por ``drift_report.py`` em ``PATH_MAINTENANCE_REPORTS``
(por omissão ``src/artifacts/reports/`` no volume do projeto).

Trigger manual (UI) ou ``airflow dags trigger ml_drift_monitoring``.
Schedule por omissão: ``None`` (activar cron na DAG se quiseres corrida semanal).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

log = logging.getLogger(__name__)

ML_PROJECT_ROOT = os.environ.get("ML_PROJECT_ROOT", "/opt/airflow/ml_project")
ML_CODE_ROOT = os.environ.get("ML_CODE_ROOT", "").strip()

VARIABLE_KEY = "drift_monitoring_conf"

DEFAULT_ARGS = {
    "owner": "ml-engineering",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


def _bootstrap_ml_sys_path() -> None:
    code_root = ML_CODE_ROOT
    if not code_root:
        _candidate = Path(__file__).resolve().parents[2] / "src"
        if _candidate.is_dir():
            code_root = str(_candidate)
    for p in (ML_PROJECT_ROOT, code_root):
        if p and p not in sys.path:
            sys.path.insert(0, p)


_bootstrap_ml_sys_path()


def _prepend_airflow_ml_site_packages() -> None:
    root = os.environ.get("ML_AIRFLOW_SITE_PACKAGES", "/opt/airflow/ml_libs").strip()
    if root and os.path.isdir(root):
        rp = os.path.abspath(root)
        if rp not in sys.path:
            sys.path.insert(0, rp)


def _repo_src_scripts_drift() -> str:
    """Caminho absoluto a ``drift_report.py`` (Docker: ML_CODE_ROOT = mount de ``src/``)."""
    if ML_CODE_ROOT:
        candidate = Path(ML_CODE_ROOT) / "scripts/maintenance/drift_report.py"
        if candidate.is_file():
            return str(candidate)
    fallback = Path(__file__).resolve().parents[2] / "src/scripts/maintenance/drift_report.py"
    return str(fallback)


def _resolve_train_csv(path: str) -> str:
    path = os.path.normpath(path.strip())
    if os.path.isfile(path):
        return os.path.abspath(path)
    under_root = os.path.join(ML_PROJECT_ROOT, path.lstrip("/"))
    if os.path.isfile(under_root):
        return os.path.abspath(under_root)
    return os.path.abspath(path)


def _merged_drift_conf(context: dict) -> dict:
    from airflow.models import Variable

    defaults: dict = {}
    try:
        raw = Variable.get(VARIABLE_KEY, default_var=None)
        if raw:
            defaults = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (json.JSONDecodeError, TypeError) as e:
        log.warning("Variable %s inválida: %s — uso só dag_run.conf.", VARIABLE_KEY, e)

    conf = context["dag_run"].conf or {}
    if isinstance(conf, str):
        try:
            conf = json.loads(conf)
        except json.JSONDecodeError:
            conf = {}
    if not isinstance(conf, dict):
        conf = {}

    merged = {**defaults, **conf}
    return merged


def task_export_predictions(**context) -> str:
    """Exporta ``predictions`` (+ join ``pipeline_runs.objective``) para CSV com ``input_data`` JSON por linha."""
    _prepend_airflow_ml_site_packages()

    import asyncpg

    merged = _merged_drift_conf(context)
    objective = str(merged.get("objective", "churn")).strip().lower()
    limit = int(merged.get("predictions_row_limit", 100_000))
    if limit <= 0:
        raise ValueError("predictions_row_limit deve ser > 0.")

    host = os.environ.get("DATABASE_SERVER", "localhost").strip()
    port = int(os.environ.get("DATABASE_PORT", "5432"))
    user = os.environ["DATABASE_USER"]
    password = os.environ["DATABASE_PASS"]
    database = os.environ["DATABASE_NAME"]

    sql = """
        SELECT p.id, p.pipeline_run_id, p.input_data, p.prediction, p.probability
        FROM predictions p
        INNER JOIN pipeline_runs pr ON pr.id = p.pipeline_run_id
        WHERE p.active = true AND lower(pr.objective) = lower($1)
        ORDER BY p.id ASC
        LIMIT $2
    """

    async def _fetch() -> list[dict]:
        conn = await asyncpg.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
        )
        try:
            rows = await conn.fetch(sql, objective, limit)
            return [dict(r) for r in rows]
        finally:
            await conn.close()

    records = asyncio.run(_fetch())
    if not records:
        raise ValueError(
            f"Nenhuma predição activa para objective={objective!r}. "
            "Gera inferências via POST /predict antes de correr esta DAG."
        )

    import pandas as pd

    def _input_as_json_str(v: object) -> str:
        if isinstance(v, dict):
            return json.dumps(v, ensure_ascii=False)
        if isinstance(v, str):
            return v
        return json.dumps(v, ensure_ascii=False)

    df = pd.DataFrame(records)
    df["input_data"] = df["input_data"].apply(_input_as_json_str)

    out_dir = os.path.join(ML_PROJECT_ROOT, "src", "artifacts", "reports")
    os.makedirs(out_dir, exist_ok=True)
    safe_run_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in context["dag_run"].run_id)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    pred_path = os.path.abspath(
        os.path.join(out_dir, f"drift_predictions_export_{stamp}_{safe_run_id}.csv")
    )
    df.to_csv(pred_path, index=False)

    log.info("Exportadas %s predições → %s", len(df), pred_path)
    return pred_path


def task_run_drift_report(**context) -> None:
    """Invoca ``drift_report.py`` com CSV de treino (Variable/conf) e CSV exportado na task anterior."""
    _prepend_airflow_ml_site_packages()

    ti = context["task_instance"]
    pred_path = ti.xcom_pull(task_ids="export_predictions")
    if not pred_path or not os.path.isfile(pred_path):
        raise FileNotFoundError(f"CSV de predições ausente ou path inválido: {pred_path!r}")

    merged = _merged_drift_conf(context)
    default_train = os.path.join(
        ML_PROJECT_ROOT,
        "src/data/pre_processed/baseline_sample_automatic.csv",
    )
    train_csv = merged.get("train_csv_path") or default_train
    train_csv = _resolve_train_csv(str(train_csv))
    if not os.path.isfile(train_csv):
        raise FileNotFoundError(
            f"CSV de referência não encontrado: {train_csv}. "
            "Ajuste ``train_csv_path`` na Variable drift_monitoring_conf ou publique o baseline."
        )

    target_col = str(merged.get("target_col", "target"))

    reports_dir = os.path.join(ML_PROJECT_ROOT, "src", "artifacts", "reports")
    os.makedirs(reports_dir, exist_ok=True)

    env = os.environ.copy()
    env["PATH_MAINTENANCE_REPORTS"] = reports_dir

    script = _repo_src_scripts_drift()
    if not os.path.isfile(script):
        raise FileNotFoundError(f"Script drift não encontrado: {script}")

    cmd = [
        sys.executable,
        script,
        "--train-csv",
        train_csv,
        "--predictions-csv",
        pred_path,
        "--target-col",
        target_col,
    ]
    log.info("Executando: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    log.info("Drift PSI concluído (saída em PATH_MAINTENANCE_REPORTS=%s).", reports_dir)


with DAG(
    dag_id="ml_drift_monitoring",
    description="Export predictions → relatório PSI (drift) vs CSV de referência.",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["ml", "maintenance", "drift"],
) as dag:
    export_predictions = PythonOperator(
        task_id="export_predictions",
        python_callable=task_export_predictions,
    )
    run_drift = PythonOperator(
        task_id="run_drift_report",
        python_callable=task_run_drift_report,
    )
    export_predictions >> run_drift
