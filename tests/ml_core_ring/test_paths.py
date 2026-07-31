"""Testes de resolução de paths partilhados."""

from __future__ import annotations

from ml_core_ring.paths import airflow_upload_path, resolve_shared_artifact_path


def test_airflow_upload_path():
    assert airflow_upload_path("x.csv") == "/opt/airflow/ml_project/uploads/x.csv"


def test_resolve_shared_to_repo_root(monkeypatch):
    monkeypatch.setenv("ML_PROJECT_ROOT", "/repo")
    from core.configs import settings

    settings.ml_project_root = "/repo"
    resolved = resolve_shared_artifact_path(
        "/var/www/ml_shared/models/recommendation/torch_embedding"
    )
    assert resolved == "/repo/models/recommendation/torch_embedding"


def test_resolve_mlflow_artifact_path(monkeypatch):
    monkeypatch.setenv("ML_PROJECT_ROOT", "/repo")
    from core.configs import settings

    settings.ml_project_root = "/repo"
    assert resolve_shared_artifact_path("/mlflow/artifacts/0") == "/repo/src/artifacts/mlruns/0"
