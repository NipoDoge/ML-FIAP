"""Pipeline de recomendação (DVC, embedding, baselines)."""

from executors_ring.recommendation import inference_engine as recommendation_inference  # noqa: F401
from executors_ring.recommendation import train_backend as recommendation_train_backend  # noqa: F401
from executors_ring.recommendation.pipeline_runner import (
    RecommendationPipelineRunner,
    build_run_context,
)
from executors_ring.recommendation.worker import result_to_dict, run_training_job

__all__ = [
    "RecommendationPipelineRunner",
    "build_run_context",
    "recommendation_train_backend",
    "run_training_job",
    "result_to_dict",
]
