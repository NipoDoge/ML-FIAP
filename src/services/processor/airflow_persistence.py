"""
Persistência de ``PipelineRuns`` para treino tabular via DAG ``ml_training_dispatch`` (Airflow).

Desactiva runs **manuais** antes do treino automático e grava baseline/FE com
``is_airflow_run=True``, reutilizando os mesmos comparadores de campeão da API.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

from sqlalchemy import func, update

# Import side-effect: regista Roles, Users, Predictions, DeployedModels antes de
# ``PipelineRuns`` resolver ``relationship("Users", ...)``.
import models._all_models  # noqa: F401
from core.configs import settings
from core.database import Session
from models.pipeline_runs import PipelineRuns
from services.processor.deployment_service import (
    get_active_deployment,
    promote_active_feature_engineering_for_objective,
)
from services.processor.inference_report import (
    attach_fe_model_comparison_table,
    attach_mlp_metrics_snapshot,
)
from services.processor.processor_service import (
    _baseline_recall_winner,
    _fe_recall_winner,
    fetch_active_baseline_metrics_snapshot,
)
from services.utils import utcnow

logger = logging.getLogger(__name__)

_STRATEGY_MEDIAN = "strategy_monthly_charges_median"


async def _load_json_file(path: str) -> dict:
    raw = await asyncio.to_thread(Path(path).read_text, encoding="utf-8")
    return json.loads(raw)


async def deactivate_manual_pipeline_runs_for_objective(objective: str) -> None:
    obj = objective.strip().lower()
    session = Session()
    try:
        # Bulk ORM update com func.* no WHERE não sincroniza sessão em modo default.
        await session.execute(
            update(PipelineRuns)
            .where(
                PipelineRuns.pipeline_type.in_(("baseline", "feature_engineering")),
                func.lower(PipelineRuns.objective) == obj,
                PipelineRuns.is_airflow_run.is_(False),
            )
            .values(active=False),
            execution_options={"synchronize_session": False},
        )
        await session.commit()
        logger.info("Runs manuais desactivados (objective=%r, is_airflow_run=false).", obj)
    finally:
        await session.close()


async def persist_airflow_baseline_run(
    *,
    objective: str,
    user_id: int,
    run_ts: str,
    pipeline: Any,
    original_filename: str,
    airflow_dag_run_id: str | None = None,
) -> int:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
    )

    model_path = pipeline.baseline_model_joblib_path()
    csv_path = os.path.join(pipeline.snapshot_path, pipeline.contract_sample_name)

    from services.pipelines.binary_decision_threshold import labels_from_probability_threshold

    model = pipeline.model
    thr = float(getattr(pipeline, "decision_threshold", settings.classification_decision_threshold))
    y_pred_test = labels_from_probability_threshold(model, pipeline.x_test, thr)
    y_proba_test = model.predict_proba(pipeline.x_test)[:, 1]
    _zd = {"zero_division": 0}
    yt = pipeline.y_test
    metrics: dict = {
        "classification_decision_threshold": thr,
        "test_accuracy": float(accuracy_score(yt, y_pred_test)),
        "test_f1": float(f1_score(yt, y_pred_test, **_zd)),
        "test_precision": float(precision_score(yt, y_pred_test, **_zd)),
        "test_recall": float(recall_score(yt, y_pred_test, **_zd)),
    }
    if int(yt.sum()) > 0 and int(len(yt) - yt.sum()) > 0:
        metrics["test_pr_auc"] = float(average_precision_score(yt, y_proba_test))

    _ml = getattr(pipeline, "mlflow_run_id", None)
    if _ml:
        metrics["mlflow_run_id"] = _ml
    if airflow_dag_run_id:
        metrics["airflow_dag_run_id"] = airflow_dag_run_id

    session = Session()
    try:
        run = PipelineRuns(
            user_id=user_id,
            pipeline_type="baseline",
            objective=objective,
            status="completed",
            original_filename=original_filename,
            model_path=model_path,
            csv_output_path=csv_path,
            metrics=metrics,
            completed_at=utcnow(),
            is_airflow_run=True,
        )
        session.add(run)
        await session.commit()
        await session.refresh(run)

        await _baseline_recall_winner(session, run, run_ts)
        metrics = dict(run.metrics or metrics)
        metrics["baseline_fe_contract_published"] = bool(run.active)
        run.metrics = metrics
        session.add(run)
        await session.commit()
        logger.info(
            "Baseline Airflow guardado (pipeline_run_id=%s, active=%s).", run.id, run.active
        )
        return run.id
    finally:
        await session.close()


async def reserve_airflow_fe_pipeline_run(
    *,
    objective: str,
    user_id: int,
    manifest_path: str,
    airflow_dag_run_id: str | None = None,
) -> int:
    """
    Cria um ``PipelineRuns`` FE em ``processing`` antes do treino — necessário para a pasta de
    logs ``src/data/old/<ts>_fe<id>/pipeline_<ts>.txt`` (só texto; artefactos pesados vêm ZIP MLflow).
    """
    resolved_manifest_path = os.path.abspath(manifest_path)
    baseline_manifest = await _load_json_file(resolved_manifest_path)
    csv_baseline = baseline_manifest.get("output_sample_csv_stable")
    if not csv_baseline:
        raise ValueError("Manifest inválido: campo 'output_sample_csv_stable' ausente.")
    original_filename = os.path.basename(os.path.abspath(csv_baseline))

    session = Session()
    try:
        meta: dict = {}
        if airflow_dag_run_id:
            meta["airflow_dag_run_id"] = airflow_dag_run_id
        run = PipelineRuns(
            user_id=user_id,
            pipeline_type="feature_engineering",
            objective=objective,
            status="processing",
            original_filename=original_filename,
            is_airflow_run=True,
            metrics=meta or None,
        )
        session.add(run)
        await session.commit()
        await session.refresh(run)
        logger.info("FE Airflow reservado (pipeline_run_id=%s, processing).", run.id)
        return run.id
    finally:
        await session.close()


async def persist_airflow_feature_engineering_run(
    *,
    objective: str,
    user_id: int,
    run_ts: str,
    manifest_path: str,
    pipeline: Any,
    optimization_metric: str,
    min_precision: float | None,
    min_roc_auc: float | None,
    tuning_n_iter: int | None,
    time_limit_minutes: int,
    effective_tuning_minutes: int,
    airflow_dag_run_id: str | None = None,
    existing_run_id: int | None = None,
    bundle_run_root: str | None = None,
) -> tuple[int, bool]:
    from services.pipelines.fe_model_selection import normalize_optimization_metric

    metric = normalize_optimization_metric(optimization_metric)
    resolved_manifest_path = os.path.abspath(manifest_path)
    baseline_manifest = await _load_json_file(resolved_manifest_path)

    csv_baseline = baseline_manifest.get("output_sample_csv_stable")
    if not csv_baseline:
        raise ValueError("Manifest inválido: campo 'output_sample_csv_stable' ausente.")
    csv_baseline = os.path.abspath(csv_baseline)

    baseline_input_path = baseline_manifest.get("input_csv_snapshot") or baseline_manifest.get(
        "input_csv_source"
    )
    baseline_input_path = os.path.abspath(baseline_input_path) if baseline_input_path else None
    fe_training_csv_basename = os.path.basename(csv_baseline)
    original_filename = fe_training_csv_basename
    fe_joblib = pipeline.fe_sklearn_joblib_path()

    session = Session()
    try:
        active_dep = None
        baseline_ref = None
        dep = await get_active_deployment(objective.strip().lower(), session)
        if dep is not None:
            active_dep = dep.id
        baseline_ref = await fetch_active_baseline_metrics_snapshot(session, objective)

        merged_metrics: dict = {
            "fe_training_csv": csv_baseline,
            "fe_training_csv_basename": fe_training_csv_basename,
            "baseline_upstream_input_basename": (
                os.path.basename(baseline_input_path) if baseline_input_path else None
            ),
            "baseline_upstream_input_path": baseline_input_path,
            "baseline_manifest_ref": {
                "run_timestamp": baseline_manifest.get("run_timestamp"),
                "sample_schema": baseline_manifest.get("sample_schema"),
            },
            "optimization_metric": metric,
            "min_precision": min_precision,
            "min_roc_auc": min_roc_auc,
            "tuning_n_iter": tuning_n_iter if tuning_n_iter is not None else 100,
            "manifest_path_used": resolved_manifest_path,
            "tuning_time_limit_effective_minutes": effective_tuning_minutes,
            "tuning_time_limit_requested": time_limit_minutes,
            "airflow_dag": True,
        }
        if baseline_ref:
            merged_metrics["baseline_reference_metrics"] = baseline_ref
        _ml = getattr(pipeline, "mlflow_run_id", None)
        if _ml:
            merged_metrics["mlflow_run_id"] = _ml
            merged_metrics["mlflow_fe_run_id"] = _ml
        if airflow_dag_run_id:
            merged_metrics["airflow_dag_run_id"] = airflow_dag_run_id
        _bcs = float(pipeline.best_cv_score)
        merged_metrics["best_cv_score"] = _bcs
        if math.isfinite(_bcs):
            merged_metrics[f"cv_{metric}"] = _bcs
        med = getattr(pipeline.strategy, "monthly_median", None)
        if med is not None:
            try:
                merged_metrics[_STRATEGY_MEDIAN] = float(med)
            except (TypeError, ValueError):
                pass
        merged_metrics.update(dict(pipeline.tuned_metrics))
        merged_metrics.update(dict(pipeline.guardrails_summary))
        attach_mlp_metrics_snapshot(merged_metrics, pipeline)
        attach_fe_model_comparison_table(merged_metrics, pipeline)
        if pipeline.best_model_name:
            merged_metrics["best_model_name"] = pipeline.best_model_name
        if active_dep is not None:
            merged_metrics["active_deployment_id_at_train"] = active_dep

        backend = "mlp" if settings.use_mlp_for_prediction else "sklearn"
        mlp_prefix = getattr(pipeline, "mlp_artifact_prefix", None)
        merged_metrics["inference_backend"] = backend
        merged_metrics["predict_model"] = "pytorch_mlp" if backend == "mlp" else "sklearn_pipeline"
        merged_metrics["sklearn_benchmark_classifier"] = pipeline.best_model_name
        if backend == "mlp" and mlp_prefix:
            merged_metrics["mlp_artifact_prefix"] = mlp_prefix

        if bundle_run_root and existing_run_id is not None:
            import tempfile

            from services.processor.artifact_bundle import safe_unlink
            from services.processor.fe_bundle_export import (
                finalize_fe_bundle_pipeline_outputs,
                log_fe_bundle_zip_to_mlflow_run,
                write_fe_manifest_zip_from_run_root,
            )

            zip_bn = f"fe_artifacts_{existing_run_id}_{run_ts}.zip"
            merged_metrics["fe_snapshot_dirname"] = f"{run_ts}_fe{existing_run_id}"
            merged_metrics["fe_bundle_zip_filename"] = zip_bn

            finalize_fe_bundle_pipeline_outputs(bundle_run_root, pipeline)
            zip_path = os.path.join(tempfile.gettempdir(), zip_bn)
            safe_unlink(zip_path)
            write_fe_manifest_zip_from_run_root(
                bundle_run_root,
                pipeline_run_id=existing_run_id,
                objective=objective,
                run_timestamp=run_ts,
                csv_baseline=csv_baseline,
                original_filename=original_filename,
                merged_metrics=merged_metrics,
                mlflow_fe_run_id=getattr(pipeline, "mlflow_run_id", None),
                best_model_name=pipeline.best_model_name,
                active_deployment_id=active_dep,
                zip_path=zip_path,
            )
            rel = log_fe_bundle_zip_to_mlflow_run(
                mlflow_run_id=getattr(pipeline, "mlflow_run_id", None),
                objective=objective,
                zip_path=zip_path,
            )
            if rel:
                merged_metrics["fe_bundle_zip_mlflow_relative"] = rel
            safe_unlink(zip_path)

        if existing_run_id is not None:
            run = await session.get(PipelineRuns, existing_run_id)
            if run is None:
                raise ValueError(f"PipelineRuns id={existing_run_id} não encontrado.")
            if run.pipeline_type != "feature_engineering":
                raise ValueError(f"Run id={existing_run_id} não é feature_engineering.")
            if int(run.user_id) != int(user_id):
                raise ValueError(f"Run id={existing_run_id}: user_id não coincide com o reservado.")
            if str(run.objective).strip().lower() != str(objective).strip().lower():
                raise ValueError(
                    f"Run id={existing_run_id}: objective não coincide com o reservado."
                )
            run.status = "completed"
            run.objective = objective
            run.original_filename = original_filename
            run.model_path = fe_joblib
            run.csv_output_path = None
            run.metrics = merged_metrics
            run.completed_at = utcnow()
            run.inference_backend = backend
            run.is_airflow_run = True
            session.add(run)
            await session.commit()
            await session.refresh(run)
        else:
            run = PipelineRuns(
                user_id=user_id,
                pipeline_type="feature_engineering",
                objective=objective,
                status="completed",
                original_filename=original_filename,
                model_path=fe_joblib,
                csv_output_path=None,
                metrics=merged_metrics,
                completed_at=utcnow(),
                is_airflow_run=True,
                inference_backend=backend,
            )
            session.add(run)
            await session.commit()
            await session.refresh(run)

        await _fe_recall_winner(session, run)
        m_final = dict(run.metrics or {})
        m_final["fe_recall_champion"] = bool(run.active)
        run.metrics = m_final
        session.add(run)
        await session.commit()
        champion = bool(run.active)
        logger.info(
            "FE Airflow guardado (pipeline_run_id=%s, fe_recall_champion=%s).",
            run.id,
            champion,
        )
        return run.id, champion
    finally:
        await session.close()


async def promote_airflow_fe_if_requested(
    *, objective: str, user_id: int, auto_promote: bool
) -> None:
    if not auto_promote:
        logger.info("auto_promote=false — skip promote.")
        return
    session = Session()
    try:
        await promote_active_feature_engineering_for_objective(
            objective=objective,
            promoted_by_user_id=user_id,
            db=session,
        )
        logger.info("Promote automático concluído (objective=%r).", objective)
    except ValueError as e:
        logger.warning("Promote automático não aplicado: %s", e)
    finally:
        await session.close()


def run_async(coro):
    """
    Executa uma corrotina num loop novo (``asyncio.run``).

    Em workers Airflow há frequentemente **várias** chamadas a ``run_async`` na mesma task
    (ex.: reservar FE e depois persistir). O pool do ``AsyncEngine`` associa ligações ao
    event loop; é preciso ``await engine.dispose()`` **no mesmo loop** antes de o
    ``asyncio.run`` terminar. Um segundo ``asyncio.run`` só para o dispose fecha o loop
    primeiro e deixa as conexões órfãs (erro "attached to a different loop").
    """
    from core.database import engine

    async def _with_cleanup() -> Any:
        try:
            return await coro
        finally:
            await engine.dispose()

    return asyncio.run(_with_cleanup())
