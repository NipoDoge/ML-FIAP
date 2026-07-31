"""Testes do executors_ring — tabular e recomendação."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import executors_ring  # noqa: F401
from executors_ring.recommendation.worker import result_to_dict, run_training_job
from executors_ring.tabular_classification.train_backend import LegacyFeTrainBackend
from ml_core_ring.inference_engine import ENGINE_REGISTRY
from ml_core_ring.train_backend import TrainRequestContext


def test_legacy_fe_backend_orchestrated_by_default():
    backend = LegacyFeTrainBackend()
    result = backend.run_train(TrainRequestContext(domain="churn", params={}))
    assert result.status == "orchestrated"


@patch("executors_ring.tabular_classification.train_backend.run_tabular_pipeline")
def test_legacy_fe_backend_executes_when_csv_path(mock_run):
    mock_run.return_value = MagicMock(
        baseline=MagicMock(run_timestamp="ts1", metrics={}),
        feature_engineering=MagicMock(
            run_timestamp="ts2",
            best_model_name="random_forest",
            metrics={"cv_recall": 0.8},
        ),
    )
    backend = LegacyFeTrainBackend()
    result = backend.run_train(
        TrainRequestContext(
            domain="churn", params={"csv_path": "/tmp/data.csv", "objective": "churn"}
        )
    )
    mock_run.assert_called_once()
    assert result.status == "completed"
    assert "champion=random_forest" in (result.detail or "")


def test_torch_bundle_engine_registered_via_executors():
    assert "torch_bundle" in ENGINE_REGISTRY


@patch("executors_ring.recommendation.train_backend.RecommendationPipelineRunner")
@patch("executors_ring.recommendation.train_backend.build_run_context")
def test_recommendation_worker_job(mock_build_ctx, mock_runner_cls):
    from ml_core_ring.artifact_manifest import ArtifactManifest
    from ml_core_ring.run_context import RunResult

    mock_build_ctx.return_value = MagicMock()
    mock_runner_cls.return_value.run.return_value = RunResult(
        manifest=ArtifactManifest(
            domain="recommendation",
            problem_type="recommendation",
            engine="sklearn_joblib",
            artifacts={},
        ),
        metrics={"hit_rate": 0.5},
        champion_name="popularity",
    )
    payload = result_to_dict(run_training_job({"top_k": 3}))
    assert payload["status"] == "completed"
    assert payload["champion_name"] == "popularity"
