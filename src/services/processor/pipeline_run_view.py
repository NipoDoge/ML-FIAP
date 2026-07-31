"""Reorganização do blob plano ``metrics`` de um ``PipelineRuns`` em secções
nomeadas para a resposta de ``GET /admin/runs``.

Mantém ``_legacy_flat_metrics`` com cópia integral do ``metrics`` original para
compatibilidade com clientes antigos.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any

from models.pipeline_runs import PipelineRuns

_BASELINE_MODEL_DISPLAY = "Regressão Logística"

_LEGACY_NOTE = (
    "Opcional: cópia idêntica ao objeto `metrics` atual para APIs/clientes "
    "antigos — pode ser omitido na UI."
)

_PRIMARY_METRIC_SCOPE_FE = (
    "Hold-out de teste após FE + seleção/tuning (threshold de decisão aplicado "
    "conforme classification_decision_threshold)"
)

_PRIMARY_METRIC_SCOPE_BASELINE = (
    "Hold-out de teste do run de baseline (sem tuning, sem comparativo entre algoritmos)."
)

_PYTORCH_MLP_ROLE = (
    "Rede neural tabular (MVP): treinamento + métricas em validação/hold-out; "
    "promoção a backend em /predict depende do modo MLP na execução do FE."
)

_COMPARISON_DESCRIPTION = (
    "Tabela resumo (sklearn pré/pós tuning + MLP). Ordem sugerida na UI: "
    "ler primeiro `inference`, depois esta tabela."
)


def _isoformat(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def _build_run_section(run: PipelineRuns) -> dict[str, Any]:
    return {
        "id": run.id,
        "user_id": run.user_id,
        "pipeline_type": run.pipeline_type,
        "objective": run.objective,
        "status": run.status,
        "is_airflow_run": bool(run.is_airflow_run),
        "timestamps": {
            "created_at": _isoformat(run.created_at),
            "completed_at": _isoformat(run.completed_at),
        },
        "artifacts": {
            "original_filename": run.original_filename,
            "model_path": run.model_path,
            "csv_output_path": run.csv_output_path,
            "active": True if run.active is None else bool(run.active),
            "error_message": run.error_message,
        },
    }


def _build_training_context(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "optimization_metric": metrics.get("optimization_metric"),
        "classification_decision_threshold": metrics.get("classification_decision_threshold"),
        "tuning_time_limit_requested_minutes": metrics.get("tuning_time_limit_requested"),
        "tuning_time_limit_effective_minutes": metrics.get("tuning_time_limit_effective_minutes"),
        "tuning_n_iter_configured": metrics.get("tuning_n_iter"),
        "guardrails": {
            "min_precision": metrics.get("min_precision"),
            "min_roc_auc": metrics.get("min_roc_auc"),
            "tuning_guardrails_passed": metrics.get("tuning_guardrails_passed"),
            "selection_guardrails_passed": metrics.get("selection_guardrails_passed"),
        },
        "data": {
            "fe_training_csv_basename": metrics.get("fe_training_csv_basename"),
            "fe_training_csv": metrics.get("fe_training_csv"),
            "manifest_path_used": metrics.get("manifest_path_used"),
            "baseline_upstream_input_basename": metrics.get("baseline_upstream_input_basename"),
            "baseline_upstream_input_path": metrics.get("baseline_upstream_input_path"),
        },
        "baseline_manifest_ref": metrics.get("baseline_manifest_ref") or {},
        "champion_flags": {
            "fe_recall_champion": metrics.get("fe_recall_champion"),
        },
        "strategy_notes": {
            "strategy_monthly_charges_median": metrics.get("strategy_monthly_charges_median"),
            "sklearn_benchmark_classifier": metrics.get("sklearn_benchmark_classifier"),
        },
    }


def _build_inference_fe(run: PipelineRuns, metrics: dict[str, Any]) -> dict[str, Any]:
    backend = (run.inference_backend or metrics.get("inference_backend") or "sklearn").lower()
    best_name = metrics.get("best_model_name")
    predict_model_key = metrics.get("predict_model")
    opt = (metrics.get("optimization_metric") or "").strip().lower()
    cv_metric_key = f"cv_{opt}" if opt else "best_cv_score"
    cv_value = metrics.get(cv_metric_key)
    if cv_value is None:
        cv_value = metrics.get("best_cv_score")

    if backend == "mlp":
        promoted_name = "PyTorch MLP"
        explain = (
            "Modelo servido / escolhido para inferência: rede neural PyTorch (MLP). "
            "O sklearn permanece como benchmark no comparativo desta run."
        )
    else:
        promoted_name = (
            f"{best_name} (tuned)" if best_name else (predict_model_key or "modelo desconhecido")
        )
        explain = (
            f"Modelo servido / escolhido para inferência: pipeline sklearn "
            f"({best_name} após tuning)."
            if best_name
            else "Modelo servido / escolhido para inferência: pipeline sklearn (após tuning)."
        )
        if metrics.get("mlp_training"):
            explain += (
                " PyTorch MLP foi treinado como benchmark experimental; a chave de uso do "
                "MLP em inferência estava desligada — por isso `inference_backend` permanece `sklearn`."
            )

    primary: dict[str, Any] = {
        "metric_suite_scope": _PRIMARY_METRIC_SCOPE_FE,
        "Acurácia": metrics.get("Acurácia"),
        "Precisão": metrics.get("Precisão"),
        "Recall": metrics.get("Recall"),
        "F1": metrics.get("F1"),
        "ROC AUC": metrics.get("ROC AUC"),
    }
    if opt:
        primary[f"cv_{opt}"] = metrics.get(f"cv_{opt}")

    return {
        "backend": backend,
        "explain_backend_choice": explain,
        "promoted_model": {
            "name": promoted_name,
            "family": predict_model_key,
            "predict_model_key": predict_model_key,
            "best_model_name_raw": best_name,
            "best_cv_score_metric": cv_metric_key,
            "best_cv_score_value": cv_value,
        },
        "test_metrics_primary": primary,
    }


_RUN_TS_FROM_MODEL_PATH = re.compile(r"_(\d{8}_\d{6})\.joblib$")


def _infer_baseline_paths(run: PipelineRuns) -> dict[str, str | None]:
    """Deriva caminhos auxiliares (snapshot, manifest, log) a partir do que está
    persistido em ``model_path`` / ``csv_output_path`` da run de baseline."""

    snapshot_dir: str | None = None
    if run.csv_output_path:
        snapshot_dir = os.path.dirname(run.csv_output_path) or None

    run_ts: str | None = None
    if run.model_path:
        m = _RUN_TS_FROM_MODEL_PATH.search(run.model_path)
        if m:
            run_ts = m.group(1)

    manifest_path: str | None = None
    log_path: str | None = None
    if snapshot_dir:
        candidate_manifest = os.path.join(snapshot_dir, "manifest.json")
        manifest_path = candidate_manifest
        if run_ts:
            log_path = os.path.join(snapshot_dir, f"pipeline_{run_ts}.txt")

    return {
        "snapshot_dir": snapshot_dir,
        "manifest_path": manifest_path,
        "log_path": log_path,
        "run_timestamp": run_ts,
    }


def _build_baseline_view(run: PipelineRuns, metrics: dict[str, Any]) -> dict[str, Any]:
    """Vista compacta para runs ``baseline``: modelo, métricas, ficheiros e logs.

    Sem ``training_context`` extenso, sem ``comparison``/``experiments`` e sem
    ``_legacy_flat_metrics``: o baseline é simples por desenho e a UI deve mostrar
    apenas o essencial.
    """
    paths = _infer_baseline_paths(run)
    return {
        "run": {
            "id": run.id,
            "user_id": run.user_id,
            "pipeline_type": run.pipeline_type,
            "objective": run.objective,
            "status": run.status,
            "is_airflow_run": bool(run.is_airflow_run),
            "active": True if run.active is None else bool(run.active),
            "error_message": run.error_message,
            "timestamps": {
                "created_at": _isoformat(run.created_at),
                "completed_at": _isoformat(run.completed_at),
                "run_timestamp": paths["run_timestamp"],
            },
        },
        "model": {
            "name": _BASELINE_MODEL_DISPLAY,
            "sklearn_estimator": "LogisticRegression",
            "role": (
                "Referência interpretável antes do feature engineering. "
                "Sem comparativo entre algoritmos: o baseline tem um único modelo por desenho."
            ),
            "classification_decision_threshold": metrics.get("classification_decision_threshold"),
        },
        "metrics": {
            "scope": _PRIMARY_METRIC_SCOPE_BASELINE,
            "accuracy": metrics.get("test_accuracy"),
            "precision": metrics.get("test_precision"),
            "recall": metrics.get("test_recall"),
            "f1": metrics.get("test_f1"),
            "pr_auc": metrics.get("test_pr_auc"),
        },
        "files": {
            "input_csv_original_filename": run.original_filename,
            "model_path": run.model_path,
            "sample_csv": run.csv_output_path,
            "manifest_path": paths["manifest_path"],
            "log_path": paths["log_path"],
            "snapshot_dir": paths["snapshot_dir"],
        },
        "flags": {
            "baseline_fe_contract_published": metrics.get("baseline_fe_contract_published"),
        },
    }


def _build_experiments(metrics: dict[str, Any]) -> dict[str, Any]:
    mlp_training = metrics.get("mlp_training") or {}
    return {
        "pytorch_mlp": {
            "ran": bool(mlp_training),
            "role": _PYTORCH_MLP_ROLE,
            "training_hyperparameters": mlp_training,
            "metrics_validation": metrics.get("mlp_metrics_val") or {},
            "metrics_test": metrics.get("mlp_metrics_test") or {},
        },
    }


def _build_comparison(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "description": _COMPARISON_DESCRIPTION,
        "fe_model_comparison_table": metrics.get("fe_model_comparison_table") or [],
    }


def _build_baseline_reference_snapshot(metrics: dict[str, Any]) -> dict[str, Any]:
    brm = metrics.get("baseline_reference_metrics") or {}
    if not brm:
        return {}
    return {
        "baseline_before_fe": {
            "description": brm.get("description"),
            "baseline_pipeline_run_id": brm.get("baseline_pipeline_run_id"),
            "model_selection": _BASELINE_MODEL_DISPLAY,
            "role": brm.get("role"),
            "classification_decision_threshold": brm.get("classification_decision_threshold"),
            "test_metrics": {
                "accuracy": brm.get("test_accuracy"),
                "precision": brm.get("test_precision"),
                "recall": brm.get("test_recall"),
                "f1": brm.get("test_f1"),
                "pr_auc": brm.get("test_pr_auc"),
                "roc_auc": brm.get("test_roc_auc"),
            },
        },
    }


def _build_fe_artifacts_bundle(metrics: dict[str, Any]) -> dict[str, Any]:
    """Metadados para localizar log em ``data/old`` e ZIP FE no MLflow (runs Airflow/API)."""
    return {
        "log_snapshot_dirname": metrics.get("fe_snapshot_dirname"),
        "bundle_zip_filename": metrics.get("fe_bundle_zip_filename"),
        "mlflow_fe_run_id": metrics.get("mlflow_fe_run_id") or metrics.get("mlflow_run_id"),
        "bundle_zip_mlflow_relative_path": metrics.get("fe_bundle_zip_mlflow_relative"),
    }


def _build_legacy(metrics: dict[str, Any]) -> dict[str, Any]:
    legacy: dict[str, Any] = {"note": _LEGACY_NOTE}
    legacy.update(metrics)
    return legacy


def build_pipeline_run_view(run: PipelineRuns) -> dict[str, Any]:
    """Devolve um dict com a vista estruturada de um ``PipelineRuns``.

    - ``feature_engineering``: vista completa (``run``, ``training_context``,
      ``inference``, ``experiments``, ``comparison``, ``baseline_reference_snapshot``
      e ``_legacy_flat_metrics``) a partir do blob ``metrics``.
    - ``baseline``: vista compacta (``run``, ``model``, ``metrics``, ``files``,
      ``flags``) — sem comparativo, sem experimentos e sem cópia legacy, porque
      o baseline é simples por desenho.
    - Outros: apenas ``run`` + ``_legacy_flat_metrics`` como fallback.
    """
    metrics = dict(run.metrics or {})
    pipeline_type = (run.pipeline_type or "").strip().lower()

    if pipeline_type == "feature_engineering":
        return {
            "run": _build_run_section(run),
            "training_context": _build_training_context(metrics),
            "inference": _build_inference_fe(run, metrics),
            "experiments": _build_experiments(metrics),
            "comparison": _build_comparison(metrics),
            "baseline_reference_snapshot": _build_baseline_reference_snapshot(metrics),
            "artifacts_bundle": _build_fe_artifacts_bundle(metrics),
            "_legacy_flat_metrics": _build_legacy(metrics),
        }

    if pipeline_type == "baseline":
        return _build_baseline_view(run, metrics)

    return {
        "run": _build_run_section(run),
        "_legacy_flat_metrics": _build_legacy(metrics),
    }
