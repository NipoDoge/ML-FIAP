"""Testes do core ML — manifest e registry de inferência."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import executors_ring  # noqa: F401 — registra torch_bundle
from core.ml import engines  # noqa: F401
from core.ml.artifact_manifest import ArtifactManifest
from core.ml.inference_engine import ENGINE_REGISTRY, get_engine


@dataclass
class FakeRun:
    objective: str
    inference_backend: str | None
    model_path: str | None
    metrics: dict | None


def test_artifact_manifest_from_sklearn_run():
    run = FakeRun(
        objective="churn",
        inference_backend="sklearn",
        model_path="/tmp/model.joblib",
        metrics={"classification_decision_threshold": 0.3},
    )
    manifest = ArtifactManifest.from_pipeline_run(run)
    assert manifest.engine == "sklearn_joblib"
    assert manifest.problem_type == "binary_classification"
    assert manifest.artifacts["joblib_path"] == "/tmp/model.joblib"
    assert manifest.metadata["decision_threshold"] == 0.3


def test_artifact_manifest_from_mlp_run():
    run = FakeRun(
        objective="churn",
        inference_backend="mlp",
        model_path=None,
        metrics={
            "mlp_artifact_prefix": "/opt/airflow/ml_project/src/artifacts/models/mlp_prefix",
            "classification_decision_threshold": 0.5,
        },
    )
    manifest = ArtifactManifest.from_pipeline_run(run)
    assert manifest.engine == "torch_bundle"
    assert manifest.artifacts["prefix"].endswith("mlp_prefix")


def test_engine_registry_contains_adapters():
    assert "sklearn_joblib" in ENGINE_REGISTRY
    assert "torch_bundle" in ENGINE_REGISTRY


def test_get_engine_unknown_raises():
    manifest = ArtifactManifest(
        domain="x",
        problem_type="binary_classification",
        engine="unknown_engine",
        artifacts={},
    )
    with pytest.raises(ValueError, match="não registrado"):
        get_engine(manifest)
