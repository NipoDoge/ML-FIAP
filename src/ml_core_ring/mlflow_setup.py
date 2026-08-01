"""MLflow — tracking unificado (Postgres ``mlflow`` + artefactos partilhados)."""

from __future__ import annotations

import os
from pathlib import Path

# Alinhado a ``mlflow_server --default-artifact-root`` e bind mount compose.
MLFLOW_ARTIFACT_CONTAINER_ROOT = "/mlflow/artifacts"

# Subpath no repo host (bind mount ./src/artifacts/mlruns → /mlflow/artifacts).
MLFLOW_ARTIFACT_HOST_SUBPATH = "src/artifacts/mlruns"


def uses_remote_mlflow_tracking(tracking_uri: str | None = None) -> bool:
    from core.configs import settings

    uri = (tracking_uri or settings.mlflow_tracking_uri or "").strip()
    return uri.startswith(("http://", "https://"))


def resolved_mlflow_artifact_dir() -> str:
    """Directório local para ``mkdir`` / escrita directa (host ou contentor)."""
    from core.configs import settings

    raw = (settings.mlflow_artifact_root or MLFLOW_ARTIFACT_HOST_SUBPATH).strip()
    if raw in {MLFLOW_ARTIFACT_CONTAINER_ROOT, MLFLOW_ARTIFACT_CONTAINER_ROOT.rstrip("/")}:
        return MLFLOW_ARTIFACT_CONTAINER_ROOT
    if raw.startswith("/mlflow/"):
        return raw
    if os.path.isabs(raw):
        return raw
    from ml_core_ring.paths import ml_project_root

    return str(Path(ml_project_root()) / raw)


def configure_mlflow_tracking() -> None:
    import mlflow

    from core.configs import settings

    os.makedirs(resolved_mlflow_artifact_dir(), exist_ok=True)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)


def ensure_mlflow_experiment(experiment_name: str) -> None:
    """Cria experimento sem ``artifact_location`` errado quando tracking é HTTP."""
    import mlflow

    configure_mlflow_tracking()
    if mlflow.get_experiment_by_name(experiment_name):
        mlflow.set_experiment(experiment_name)
        return
    if uses_remote_mlflow_tracking():
        # Servidor central: Postgres + --default-artifact-root /mlflow/artifacts
        mlflow.create_experiment(experiment_name)
    else:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=resolved_mlflow_artifact_dir(),
        )
    mlflow.set_experiment(experiment_name)
