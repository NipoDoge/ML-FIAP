"""Testes de mlflow_setup e paths MLflow."""

from __future__ import annotations

from unittest.mock import patch

from ml_core_ring.mlflow_setup import (
    MLFLOW_ARTIFACT_CONTAINER_ROOT,
    ensure_mlflow_experiment,
    resolved_mlflow_artifact_dir,
    uses_remote_mlflow_tracking,
)
from ml_core_ring.paths import resolve_shared_artifact_path


def test_uses_remote_mlflow_tracking():
    assert uses_remote_mlflow_tracking("http://mlflow_server:5000") is True
    assert uses_remote_mlflow_tracking("sqlite:///tmp/mlflow.db") is False


def test_resolved_mlflow_artifact_dir_container():
    from core.configs import settings

    settings.mlflow_artifact_root = MLFLOW_ARTIFACT_CONTAINER_ROOT
    assert resolved_mlflow_artifact_dir() == "/mlflow/artifacts"


def test_resolved_mlflow_artifact_dir_host_relative(monkeypatch):
    monkeypatch.setenv("ML_PROJECT_ROOT", "/repo")
    from core.configs import settings

    settings.ml_project_root = "/repo"
    settings.mlflow_artifact_root = "src/artifacts/mlruns"
    assert resolved_mlflow_artifact_dir() == "/repo/src/artifacts/mlruns"


def test_resolve_mlflow_artifact_prefix(monkeypatch):
    monkeypatch.setenv("ML_PROJECT_ROOT", "/repo")
    from core.configs import settings

    settings.ml_project_root = "/repo"
    resolved = resolve_shared_artifact_path("/mlflow/artifacts/2/abc/artifacts/model")
    assert resolved == "/repo/src/artifacts/mlruns/2/abc/artifacts/model"


@patch("mlflow.set_experiment")
@patch("mlflow.create_experiment")
@patch("mlflow.get_experiment_by_name", return_value=None)
@patch("ml_core_ring.mlflow_setup.configure_mlflow_tracking")
def test_ensure_mlflow_experiment_remote_no_custom_location(
    _mock_configure,
    _mock_get,
    mock_create,
    _mock_set,
):
    from core.configs import settings

    settings.mlflow_tracking_uri = "http://mlflow_server:5000"
    settings.mlflow_artifact_root = "/mlflow/artifacts"

    ensure_mlflow_experiment("tc02_recommendation")

    mock_create.assert_called_once_with("tc02_recommendation")
