"""Tasks Airflow — treino tabular (Baseline + Feature Engineering)."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone

from orchestration_ring.airflow_env import (
    DEFAULT_PIPELINE_USER_ID,
    resolve_ml_artifact_path,
)
from orchestration_ring.persist_run import (
    deactivate_manual_pipeline_runs_for_objective,
    persist_airflow_baseline_run,
    persist_airflow_feature_engineering_run,
    promote_airflow_fe_if_requested,
    reserve_airflow_fe_pipeline_run,
    run_async,
)

logger = logging.getLogger(__name__)


def task_validate_tabular_input(**context) -> None:
    """Valida CSV e colunas mínimas do domínio tabular."""
    from orchestration_ring.conf import merge_run_conf

    conf = merge_run_conf(
        context,
        variable_key="ml_training_dispatch_conf",
        fallback_keys=(
            ("objective", "ml_training_objective"),
            ("csv_path", "ml_training_csv_path"),
        ),
    )
    from core.configs import settings as svc_settings

    objective = conf.get("objective") or conf.get("domain")
    csv_path = conf.get("csv_path")

    if not objective:
        raise ValueError("Parâmetro 'objective' ou 'domain' é obrigatório no conf.")
    if not csv_path:
        raise ValueError("Parâmetro 'csv_path' é obrigatório no conf.")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    try:
        import pandas as pd

        df = pd.read_csv(csv_path, nrows=5)
        logger.info("CSV validado — colunas=%s | arquivo=%s", len(df.columns), csv_path)
    except Exception as exc:
        raise ValueError(f"Erro ao ler o CSV: {exc}") from exc

    objective = str(objective).strip().lower()
    try:
        from executors_ring.tabular_classification.strategies import STRATEGY_REGISTRY

        if objective not in STRATEGY_REGISTRY:
            available = list(STRATEGY_REGISTRY.keys())
            raise ValueError(f"Domínio {objective!r} não registrado. Disponíveis: {available}")
        strategy = STRATEGY_REGISTRY[objective]()
        strategy.validate(df)
        logger.info("Validação de colunas OK para domínio %r.", objective)
    except ImportError:
        logger.warning("STRATEGY_REGISTRY indisponível — validação de colunas ignorada.")

    ti = context["task_instance"]
    ti.xcom_push(key="objective", value=objective)
    ti.xcom_push(key="csv_path", value=csv_path)
    ti.xcom_push(key="user_id", value=int(conf.get("user_id", DEFAULT_PIPELINE_USER_ID)))
    ti.xcom_push(key="optimization_metric", value=conf.get("optimization_metric", "accuracy"))
    ti.xcom_push(key="time_limit_minutes", value=int(conf.get("time_limit_minutes", 2)))
    ti.xcom_push(key="acc_target", value=float(conf.get("acc_target", 0.90)))

    if conf.get("min_precision") is not None:
        ti.xcom_push(key="min_precision", value=float(conf["min_precision"]))
    if conf.get("min_roc_auc") is not None:
        ti.xcom_push(key="min_roc_auc", value=float(conf["min_roc_auc"]))
    if conf.get("tuning_n_iter") is not None:
        ti.xcom_push(key="tuning_n_iter", value=int(conf["tuning_n_iter"]))

    ti.xcom_push(key="auto_promote", value=bool(conf.get("auto_promote", False)))

    dt_raw = conf.get("decision_threshold")
    ti.xcom_push(
        key="decision_threshold",
        value=float(svc_settings.classification_decision_threshold if dt_raw is None else dt_raw),
    )


def task_deactivate_manual_runs(**context) -> None:
    ti = context["task_instance"]
    validate_task = context.get("validate_task_id", "validate_input")
    objective = ti.xcom_pull(key="objective", task_ids=validate_task)
    run_async(deactivate_manual_pipeline_runs_for_objective(objective))


def task_run_baseline(**context) -> None:
    ti = context["task_instance"]
    validate_task = context.get("validate_task_id", "validate_input")
    objective = ti.xcom_pull(key="objective", task_ids=validate_task)
    csv_path = ti.xcom_pull(key="csv_path", task_ids=validate_task)
    user_id = ti.xcom_pull(key="user_id", task_ids=validate_task)
    decision_threshold = float(ti.xcom_pull(key="decision_threshold", task_ids=validate_task))

    logger.info(
        "Baseline | objective=%s | csv=%s | decision_threshold=%s",
        objective,
        csv_path,
        decision_threshold,
    )

    from core.configs import settings as ml_settings
    from core.custom_logger import setup_pipeline_run_logging
    from executors_ring.tabular_classification.baseline import Baseline
    from executors_ring.tabular_classification.strategies import get_class_labels

    now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    snapshot_path = os.path.join(ml_settings.path_data, ml_settings.path_logs, now)
    setup_pipeline_run_logging(
        snapshot_path,
        now,
        run_id=None,
        objective=objective,
        pipeline_type="baseline",
    )

    pipeline = Baseline(
        pobjective=objective,
        run_timestamp=now,
        csv_path=csv_path,
        class_labels=get_class_labels(objective),
        defer_global_preprocess_contract=True,
        decision_threshold=decision_threshold,
        artifact_name_suffix="_automatic",
    )
    pipeline.run(start_time=datetime.now(timezone.utc))
    pipeline.save_artifacts()

    if pipeline.defer_global_preprocess_contract:
        baseline_manifest_path = os.path.abspath(
            os.path.join(pipeline.snapshot_path, pipeline.contract_manifest_name)
        )
        baseline_sample_csv_path = os.path.abspath(
            os.path.join(pipeline.snapshot_path, pipeline.contract_sample_name)
        )
    else:
        manifest_rel = os.path.join(
            pipeline.path_data_preprocessed, pipeline.contract_manifest_name
        )
        sample_rel = os.path.join(pipeline.path_data_preprocessed, pipeline.contract_sample_name)
        baseline_manifest_path = resolve_ml_artifact_path(manifest_rel)
        baseline_sample_csv_path = resolve_ml_artifact_path(sample_rel)

    if not os.path.isfile(baseline_manifest_path):
        raise FileNotFoundError(f"Manifest do baseline não encontrado: {baseline_manifest_path}")
    if not os.path.isfile(baseline_sample_csv_path):
        raise FileNotFoundError(
            f"CSV estável do baseline não encontrado: {baseline_sample_csv_path}"
        )

    br_id = run_async(
        persist_airflow_baseline_run(
            objective=objective,
            user_id=int(user_id),
            run_ts=now,
            pipeline=pipeline,
            original_filename=os.path.basename(csv_path),
            airflow_dag_run_id=context["dag_run"].run_id,
        )
    )

    ti.xcom_push(key="baseline_manifest_path", value=baseline_manifest_path)
    ti.xcom_push(key="baseline_sample_csv_path", value=baseline_sample_csv_path)
    ti.xcom_push(key="baseline_run_ts", value=now)
    ti.xcom_push(key="baseline_pipeline_run_id", value=br_id)


def task_run_fe(**context) -> None:
    ti = context["task_instance"]
    validate_task = context.get("validate_task_id", "validate_input")
    baseline_task = context.get("baseline_task_id", "run_baseline")
    objective = ti.xcom_pull(key="objective", task_ids=validate_task)
    optimization_metric = ti.xcom_pull(key="optimization_metric", task_ids=validate_task)
    time_limit_minutes = int(ti.xcom_pull(key="time_limit_minutes", task_ids=validate_task))
    acc_target = ti.xcom_pull(key="acc_target", task_ids=validate_task)
    manifest_path = ti.xcom_pull(key="baseline_manifest_path", task_ids=baseline_task)
    user_id = ti.xcom_pull(key="user_id", task_ids=validate_task)
    min_precision = ti.xcom_pull(key="min_precision", task_ids=validate_task)
    min_roc_auc = ti.xcom_pull(key="min_roc_auc", task_ids=validate_task)
    tuning_n_iter = ti.xcom_pull(key="tuning_n_iter", task_ids=validate_task)
    decision_threshold = float(ti.xcom_pull(key="decision_threshold", task_ids=validate_task))

    if not manifest_path or not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Manifest do Baseline não encontrado para FE: {manifest_path}")

    from core.configs import settings as ml_settings
    from core.custom_logger import setup_pipeline_run_logging
    from executors_ring.tabular_classification.feature_engineering import FeatureEngineering
    from executors_ring.tabular_classification.strategies import STRATEGY_REGISTRY
    from services.processor.artifact_bundle import safe_rmtree
    from services.processor.fe_bundle_export import prepare_fe_bundle_baseline_tree

    fe_run_id = run_async(
        reserve_airflow_fe_pipeline_run(
            objective=objective,
            user_id=int(user_id),
            manifest_path=manifest_path,
            airflow_dag_run_id=context["dag_run"].run_id,
        )
    )
    now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    fe_snapshot = os.path.join(ml_settings.path_data, ml_settings.path_logs, f"{now}_fe{fe_run_id}")
    os.makedirs(fe_snapshot, exist_ok=True)
    setup_pipeline_run_logging(
        fe_snapshot,
        now,
        run_id=fe_run_id,
        objective=objective,
        pipeline_type="feature_engineering",
    )

    run_root: str | None = None
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            baseline_manifest_fe = json.load(handle)
        run_root = tempfile.mkdtemp(prefix=f"fe_airflow_bundle_{fe_run_id}_")
        prepare_fe_bundle_baseline_tree(
            run_root,
            resolved_manifest_path=os.path.abspath(manifest_path),
            baseline_manifest=baseline_manifest_fe,
        )
        fe_d = os.path.join(run_root, "20_feature_engineering")
        fe_plots = os.path.join(fe_d, "plots")
        os.makedirs(fe_plots, exist_ok=True)

        strategy = STRATEGY_REGISTRY[objective]()
        pipeline = FeatureEngineering(
            objective=objective,
            strategy=strategy,
            run_timestamp=now,
            csv_path=None,
            manifest_path=manifest_path,
            optimization_metric=optimization_metric,
            min_precision=min_precision,
            min_roc_auc=min_roc_auc,
            tuning_n_iter=tuning_n_iter,
            export_figures_dir=fe_plots,
            decision_threshold=decision_threshold,
            artifact_name_suffix="_automatic",
            is_airflow_run=True,
        )
        pipeline.run(time_limit_minutes=time_limit_minutes, acc_target=acc_target)

        fe_id, champion = run_async(
            persist_airflow_feature_engineering_run(
                objective=objective,
                user_id=int(user_id),
                run_ts=now,
                manifest_path=manifest_path,
                pipeline=pipeline,
                optimization_metric=optimization_metric,
                min_precision=min_precision,
                min_roc_auc=min_roc_auc,
                tuning_n_iter=tuning_n_iter,
                time_limit_minutes=time_limit_minutes,
                effective_tuning_minutes=time_limit_minutes,
                airflow_dag_run_id=context["dag_run"].run_id,
                existing_run_id=fe_run_id,
                bundle_run_root=run_root,
            )
        )
    finally:
        if run_root:
            safe_rmtree(run_root)

    ti.xcom_push(key="fe_run_ts", value=now)
    ti.xcom_push(key="fe_best_model", value=pipeline.best_model_name)
    ti.xcom_push(key="fe_metrics", value=str(pipeline.tuned_metrics))
    ti.xcom_push(key="fe_pipeline_run_id", value=fe_id)
    ti.xcom_push(key="fe_recall_champion", value=champion)


def task_promote_fe_optional(**context) -> None:
    ti = context["task_instance"]
    validate_task = context.get("validate_task_id", "validate_input")
    fe_task = context.get("fe_task_id", "run_fe")
    if not ti.xcom_pull(key="auto_promote", task_ids=validate_task):
        logger.info("auto_promote=false — promote automático ignorado.")
        return
    if not ti.xcom_pull(key="fe_recall_champion", task_ids=fe_task):
        logger.info("FE não venceu comparador — promote automático ignorado.")
        return
    objective = ti.xcom_pull(key="objective", task_ids=validate_task)
    user_id = int(ti.xcom_pull(key="user_id", task_ids=validate_task))
    run_async(
        promote_airflow_fe_if_requested(
            objective=objective,
            user_id=user_id,
            auto_promote=True,
        )
    )


def task_notify_tabular_complete(**context) -> None:
    ti = context["task_instance"]
    validate_task = context.get("validate_task_id", "validate_input")
    baseline_task = context.get("baseline_task_id", "run_baseline")
    fe_task = context.get("fe_task_id", "run_fe")
    objective = ti.xcom_pull(key="objective", task_ids=validate_task)
    fe_best_model = ti.xcom_pull(key="fe_best_model", task_ids=fe_task)
    fe_metrics = ti.xcom_pull(key="fe_metrics", task_ids=fe_task)
    bl_id = ti.xcom_pull(key="baseline_pipeline_run_id", task_ids=baseline_task)
    fe_id = ti.xcom_pull(key="fe_pipeline_run_id", task_ids=fe_task)
    champion = ti.xcom_pull(key="fe_recall_champion", task_ids=fe_task)

    logger.info("=" * 60)
    logger.info("PIPELINE TABULAR CONCLUÍDO | domain=%s", objective)
    logger.info("Baseline pipeline_run : %s", bl_id)
    logger.info("FE pipeline_run       : %s", fe_id)
    logger.info("FE campeão            : %s", champion)
    logger.info("Melhor modelo FE      : %s", fe_best_model)
    logger.info("Métricas FE           : %s", fe_metrics)
    logger.info("=" * 60)
