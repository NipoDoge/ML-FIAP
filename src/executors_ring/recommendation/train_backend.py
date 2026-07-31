"""Backend de treino recomendação — pipeline DVC / embedding PyTorch."""

from __future__ import annotations

from ml_core_ring.train_backend import (
    TrainBackendResult,
    TrainRequestContext,
    register_train_backend,
)

from executors_ring.recommendation.pipeline_runner import (
    RecommendationPipelineRunner,
    build_run_context,
)


class RecommendationDvcTrainBackend:
    backend_id = "recommendation_dvc"

    def run_train(self, ctx: TrainRequestContext) -> TrainBackendResult:
        run_ctx = build_run_context({**ctx.params, "domain": ctx.domain})
        result = RecommendationPipelineRunner().run(run_ctx)
        return TrainBackendResult(
            status="completed",
            detail=f"Champion: {result.champion_name}",
            metrics=result.metrics,
            mlflow_run_id=result.mlflow_run_id,
            run_result=result,
        )


register_train_backend(RecommendationDvcTrainBackend())
