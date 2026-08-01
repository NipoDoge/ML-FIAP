"""Execução programática Baseline + FE (sem persistência BD — responsabilidade da orquestração)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from core.configs import settings
from executors_ring.tabular_classification.baseline import Baseline
from executors_ring.tabular_classification.feature_engineering import FeatureEngineering
from executors_ring.tabular_classification.strategies import STRATEGY_REGISTRY, get_class_labels


@dataclass
class BaselineJobResult:
    run_timestamp: str
    manifest_path: str
    sample_csv_path: str
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class FeatureEngineeringJobResult:
    run_timestamp: str
    best_model_name: str | None
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class TabularPipelineResult:
    baseline: BaselineJobResult | None = None
    feature_engineering: FeatureEngineeringJobResult | None = None


def _resolve_artifact_path(path: str) -> str:
    path = os.path.normpath(path.strip())
    if os.path.isfile(path):
        return os.path.abspath(path)
    under_data = os.path.join(settings.path_data, path)
    if os.path.isfile(under_data):
        return os.path.abspath(under_data)
    return os.path.abspath(path)


def run_baseline_job(
    *,
    objective: str,
    csv_path: str,
    decision_threshold: float | None = None,
    artifact_name_suffix: str = "",
    defer_global_preprocess_contract: bool = True,
    run_timestamp: str | None = None,
) -> BaselineJobResult:
    """Executa Baseline e devolve caminhos do contrato manifest + sample."""
    if objective not in STRATEGY_REGISTRY:
        known = ", ".join(sorted(STRATEGY_REGISTRY))
        raise ValueError(f"Domínio {objective!r} não registrado. Disponíveis: {known}")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    now = run_timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    threshold = (
        float(settings.classification_decision_threshold)
        if decision_threshold is None
        else float(decision_threshold)
    )

    pipeline = Baseline(
        pobjective=objective,
        run_timestamp=now,
        csv_path=csv_path,
        class_labels=get_class_labels(objective),
        defer_global_preprocess_contract=defer_global_preprocess_contract,
        decision_threshold=threshold,
        artifact_name_suffix=artifact_name_suffix,
    )
    pipeline.run(start_time=datetime.now(timezone.utc))
    pipeline.save_artifacts()

    if defer_global_preprocess_contract:
        manifest_path = os.path.abspath(
            os.path.join(pipeline.snapshot_path, pipeline.contract_manifest_name)
        )
        sample_path = os.path.abspath(
            os.path.join(pipeline.snapshot_path, pipeline.contract_sample_name)
        )
    else:
        manifest_path = _resolve_artifact_path(
            os.path.join(pipeline.path_data_preprocessed, pipeline.contract_manifest_name)
        )
        sample_path = _resolve_artifact_path(
            os.path.join(pipeline.path_data_preprocessed, pipeline.contract_sample_name)
        )

    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Manifest do baseline não encontrado: {manifest_path}")
    if not os.path.isfile(sample_path):
        raise FileNotFoundError(f"Sample CSV do baseline não encontrado: {sample_path}")

    return BaselineJobResult(
        run_timestamp=now,
        manifest_path=manifest_path,
        sample_csv_path=sample_path,
        metrics={},
    )


def run_feature_engineering_job(
    *,
    objective: str,
    manifest_path: str,
    optimization_metric: str = "recall",
    time_limit_minutes: int = 30,
    acc_target: float = 0.90,
    min_precision: float | None = None,
    min_roc_auc: float | None = None,
    tuning_n_iter: int | None = None,
    decision_threshold: float | None = None,
    artifact_name_suffix: str = "",
    run_timestamp: str | None = None,
) -> FeatureEngineeringJobResult:
    """Executa FE a partir do manifest produzido pelo Baseline."""
    manifest_path = _resolve_artifact_path(manifest_path)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Manifest do baseline não encontrado: {manifest_path}")

    with open(manifest_path, encoding="utf-8") as handle:
        baseline_manifest = json.load(handle)
    manifest_objective = str(baseline_manifest.get("objective", "")).strip().lower()
    if manifest_objective and manifest_objective != objective.strip().lower():
        raise ValueError(
            f"Manifest objective={manifest_objective!r} difere do pedido {objective!r}."
        )

    if objective not in STRATEGY_REGISTRY:
        known = ", ".join(sorted(STRATEGY_REGISTRY))
        raise ValueError(f"Domínio {objective!r} não registrado. Disponíveis: {known}")

    now = run_timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    threshold = (
        float(settings.classification_decision_threshold)
        if decision_threshold is None
        else float(decision_threshold)
    )

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
        decision_threshold=threshold,
        artifact_name_suffix=artifact_name_suffix,
        is_airflow_run=False,
    )
    pipeline.run(time_limit_minutes=time_limit_minutes, acc_target=acc_target)

    tuned = pipeline.tuned_metrics or {}
    metrics = {str(k): float(v) for k, v in tuned.items() if isinstance(v, (int, float))}
    return FeatureEngineeringJobResult(
        run_timestamp=now,
        best_model_name=getattr(pipeline, "best_model_name", None),
        metrics=metrics,
    )


def run_tabular_pipeline(params: dict[str, Any]) -> TabularPipelineResult:
    """
    Executa Baseline e/ou FE conforme ``stage`` em params.

    stage: ``baseline`` | ``fe`` | ``full`` (default ``full`` quando csv_path presente).
    """
    objective = str(params["objective"]).strip().lower()
    stage = str(params.get("stage", "full")).strip().lower()
    csv_path = params.get("csv_path")
    manifest_path = params.get("manifest_path") or params.get("baseline_manifest_path")

    baseline_result: BaselineJobResult | None = None
    fe_result: FeatureEngineeringJobResult | None = None

    if stage in ("baseline", "full"):
        if not csv_path:
            raise ValueError("csv_path é obrigatório para stage baseline/full.")
        baseline_result = run_baseline_job(
            objective=objective,
            csv_path=str(csv_path),
            decision_threshold=params.get("decision_threshold"),
            artifact_name_suffix=str(params.get("artifact_name_suffix", "")),
            defer_global_preprocess_contract=bool(
                params.get("defer_global_preprocess_contract", True)
            ),
            run_timestamp=params.get("baseline_run_timestamp"),
        )
        manifest_path = baseline_result.manifest_path

    if stage in ("fe", "full"):
        if not manifest_path:
            raise ValueError("manifest_path é obrigatório para stage fe/full.")
        fe_result = run_feature_engineering_job(
            objective=objective,
            manifest_path=str(manifest_path),
            optimization_metric=str(params.get("optimization_metric", "recall")),
            time_limit_minutes=int(params.get("time_limit_minutes", 30)),
            acc_target=float(params.get("acc_target", 0.90)),
            min_precision=params.get("min_precision"),
            min_roc_auc=params.get("min_roc_auc"),
            tuning_n_iter=params.get("tuning_n_iter"),
            decision_threshold=params.get("decision_threshold"),
            artifact_name_suffix=str(params.get("artifact_name_suffix", "")),
            run_timestamp=params.get("fe_run_timestamp"),
        )

    return TabularPipelineResult(baseline=baseline_result, feature_engineering=fe_result)
