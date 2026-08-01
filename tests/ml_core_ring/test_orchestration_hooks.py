"""Testes de orchestration_hooks — resolução domínio → backend."""

from __future__ import annotations

from unittest.mock import patch

import pytest

import domains  # noqa: F401
import executors_ring  # noqa: F401
from ml_core_ring.artifact_manifest import ArtifactManifest
from ml_core_ring.orchestration_hooks import run_training_for_domain
from ml_core_ring.run_context import RunContext, RunResult


def test_run_training_churn_returns_orchestrated():
    result = run_training_for_domain("churn", {"objective": "churn"})
    assert result.status == "orchestrated"
    assert "ml_training_dispatch" in (result.detail or "")


def test_run_training_unknown_domain_raises():
    with pytest.raises(KeyError, match="não registrado"):
        run_training_for_domain("unknown_domain_xyz")


@patch("executors_ring.recommendation.train_backend.RecommendationPipelineRunner")
@patch("executors_ring.recommendation.train_backend.build_run_context")
def test_run_training_recommendation_delegates_to_runner(mock_build_ctx, mock_runner_cls):
    mock_build_ctx.return_value = RunContext(domain="recommendation", params={})
    mock_runner_cls.return_value.run.return_value = RunResult(
        manifest=ArtifactManifest(
            domain="recommendation",
            problem_type="recommendation",
            engine="torch_bundle",
            artifacts={"prefix": "/tmp/model"},
        ),
        metrics={"hit_rate": 0.42},
        mlflow_run_id="run-123",
        champion_name="torch_embedding",
    )

    result = run_training_for_domain("recommendation", {"top_k": 5})

    mock_build_ctx.assert_called_once()
    mock_runner_cls.return_value.run.assert_called_once()
    assert result.status == "completed"
    assert result.metrics["hit_rate"] == 0.42
    assert result.mlflow_run_id == "run-123"
