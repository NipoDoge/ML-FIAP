"""Backend de treino tabular (Baseline + Feature Engineering)."""

from __future__ import annotations

from ml_core_ring.train_backend import (
    TrainBackendResult,
    TrainRequestContext,
    register_train_backend,
)

from executors_ring.tabular_classification.runner import run_tabular_pipeline


class LegacyFeTrainBackend:
    backend_id = "legacy_fe"

    def run_train(self, ctx: TrainRequestContext) -> TrainBackendResult:
        params = dict(ctx.params)
        execute_locally = bool(params.get("execute_locally")) or bool(params.get("csv_path"))

        if not execute_locally:
            return TrainBackendResult(
                status="orchestrated",
                detail=(
                    "Treino tabular via Airflow DAG ml_training_dispatch "
                    "(Baseline + Feature Engineering). Passe execute_locally=true e csv_path "
                    "para execução programática no executor."
                ),
            )

        params.setdefault("objective", ctx.domain)
        result = run_tabular_pipeline(params)

        metrics: dict[str, float] = {}
        if result.baseline and result.baseline.metrics:
            metrics.update(result.baseline.metrics)
        if result.feature_engineering and result.feature_engineering.metrics:
            metrics.update(result.feature_engineering.metrics)

        champion = None
        if result.feature_engineering:
            champion = result.feature_engineering.best_model_name

        detail_parts = []
        if result.baseline:
            detail_parts.append(f"baseline_ts={result.baseline.run_timestamp}")
        if result.feature_engineering:
            detail_parts.append(f"fe_ts={result.feature_engineering.run_timestamp}")
        if champion:
            detail_parts.append(f"champion={champion}")

        return TrainBackendResult(
            status="completed",
            detail=" | ".join(detail_parts) or "tabular pipeline concluído",
            metrics=metrics,
        )


register_train_backend(LegacyFeTrainBackend())
