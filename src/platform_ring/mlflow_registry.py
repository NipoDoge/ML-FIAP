"""Side-effect MLflow Model Registry no promote (Fase 6).

A fonte de verdade para ``/predict`` continua ``deployed_models`` (Postgres).
Este módulo espelha o promote no MLflow Registry (Staging → Production), best-effort.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from core.configs import settings

_logger = logging.getLogger(__name__)

# Nome registado no MLflow Model Registry por domínio.
REGISTRY_MODEL_BY_DOMAIN: dict[str, str] = {
    "recommendation": "tc02_recommender",
    "churn": "churn_fe_model",
}

# Subpath do artefacto dentro do MLflow run (``runs:/<id>/<subpath>``).
REGISTRY_ARTIFACT_SUBPATH: dict[str, str] = {
    "recommendation": "pytorch_model",
    "churn": "sklearn_model",
}


@dataclass(frozen=True)
class RegistryPromoteResult:
    model_name: str
    version: str
    stage: str = "Production"
    mlflow_run_id: str | None = None
    warning: str | None = None
    skipped: bool = False
    skip_reason: str | None = None


def _tracking_uri() -> str:
    return (settings.mlflow_tracking_uri or "").strip() or "http://localhost:5000"


def _metrics_dict(metrics: Any) -> dict[str, Any]:
    return dict(metrics) if isinstance(metrics, dict) else {}


def resolve_registry_model_name(domain: str) -> str | None:
    return REGISTRY_MODEL_BY_DOMAIN.get(domain.strip().lower())


def _promote_registry_version(client: Any, *, model_name: str, version: str) -> None:
    """Coloca versão em Production — aliases (MLflow ≥2.9 / 3.x) ou stages legados."""

    if hasattr(client, "set_registered_model_alias"):
        client.set_registered_model_alias(model_name, "Production", version)
        return

    if hasattr(client, "transition_model_version"):
        client.transition_model_version(
            name=model_name,
            version=version,
            stage="Staging",
            archive_existing_versions=False,
        )
        client.transition_model_version(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )
        return

    raise AttributeError(
        "MlflowClient sem API Registry (set_registered_model_alias ou transition_model_version)"
    )


def sync_mlflow_registry_on_promote(
    *,
    domain: str,
    metrics: dict[str, Any] | None,
    pipeline_type: str,
) -> RegistryPromoteResult | None:
    """Espelha promote no MLflow Registry. Nunca levanta — falhas viram ``warning``."""

    d = domain.strip().lower()
    model_name = resolve_registry_model_name(d)
    if not model_name:
        return RegistryPromoteResult(
            model_name="",
            version="",
            skipped=True,
            skip_reason=f"domínio {d!r} sem modelo Registry configurado",
        )

    m = _metrics_dict(metrics)
    run_id = m.get("mlflow_run_id") or m.get("mlflow_fe_run_id")
    if isinstance(run_id, str):
        run_id = run_id.strip() or None

    try:
        from mlflow import MlflowClient
        from mlflow.tracking import set_tracking_uri
    except ImportError as exc:
        _logger.warning("MLflow indisponível no promote Registry: %s", exc)
        return RegistryPromoteResult(
            model_name=model_name,
            version="",
            mlflow_run_id=run_id,
            warning=f"mlflow não instalado: {exc}",
        )

    set_tracking_uri(_tracking_uri())
    client = MlflowClient()

    try:
        version = _resolve_model_version(
            client,
            model_name=model_name,
            mlflow_run_id=run_id,
            pipeline_type=pipeline_type.strip().lower(),
            domain=d,
        )
        if not version:
            return RegistryPromoteResult(
                model_name=model_name,
                version="",
                mlflow_run_id=run_id,
                warning=(
                    "Nenhuma versão Registry encontrada ou criada. "
                    "Confirme mlflow_run_id em pipeline_runs.metrics e treino com log_model."
                ),
            )

        _promote_registry_version(client, model_name=model_name, version=version)
        _logger.info(
            "MLflow Registry: %s v%s → Production (run_id=%s)",
            model_name,
            version,
            run_id,
        )
        return RegistryPromoteResult(
            model_name=model_name,
            version=str(version),
            stage="Production",
            mlflow_run_id=run_id,
        )
    except Exception as exc:
        _logger.warning(
            "MLflow Registry side-effect falhou (promote BD OK): domain=%s model=%s: %s",
            d,
            model_name,
            exc,
            exc_info=True,
        )
        return RegistryPromoteResult(
            model_name=model_name,
            version="",
            mlflow_run_id=run_id,
            warning=str(exc),
        )


def _resolve_model_version(
    client: Any,
    *,
    model_name: str,
    mlflow_run_id: str | None,
    pipeline_type: str,
    domain: str,
) -> str | None:
    """Encontra ou cria versão Registry; devolve número de versão como string."""

    if mlflow_run_id:
        matched = _version_for_run_id(client, model_name, mlflow_run_id)
        if matched:
            return matched
        created = _create_version_from_run(
            client,
            model_name=model_name,
            mlflow_run_id=mlflow_run_id,
            pipeline_type=pipeline_type,
            domain=domain,
        )
        if created:
            return created

    return _latest_version(client, model_name)


def _version_for_run_id(client: Any, model_name: str, run_id: str) -> str | None:
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception as exc:  # noqa: BLE001
        _logger.debug("search_model_versions falhou para %s/run_id=%s: %s", model_name, run_id, exc)
        return None
    for mv in versions:
        if getattr(mv, "run_id", None) == run_id:
            return str(mv.version)
    return None


def _create_version_from_run(
    client: Any,
    *,
    model_name: str,
    mlflow_run_id: str,
    pipeline_type: str,
    domain: str,
) -> str | None:
    subpath = REGISTRY_ARTIFACT_SUBPATH.get(domain)
    if pipeline_type == "feature_engineering" and domain == "churn":
        subpath = subpath or "sklearn_model"
    elif pipeline_type == "recommendation":
        subpath = subpath or "pytorch_model"
    if not subpath:
        return None

    source = f"runs:/{mlflow_run_id}/{subpath}"
    try:
        try:
            client.create_registered_model(model_name)
        except Exception as exc:  # noqa: BLE001
            _logger.debug("create_registered_model ignorado para %s: %s", model_name, exc)
        mv = client.create_model_version(
            name=model_name,
            source=source,
            run_id=mlflow_run_id,
        )
        return str(mv.version)
    except Exception as exc:  # noqa: BLE001
        _logger.debug("create_model_version falhou (%s): %s", source, exc)
        # Fallback: artefacto alternativo no run churn
        if domain == "churn" and subpath == "sklearn_model":
            alt = f"runs:/{mlflow_run_id}/model"
            try:
                mv = client.create_model_version(
                    name=model_name,
                    source=alt,
                    run_id=mlflow_run_id,
                )
                return str(mv.version)
            except Exception as exc:  # noqa: BLE001
                _logger.debug("create_model_version fallback falhou (%s): %s", alt, exc)
        return None


def _latest_version(client: Any, model_name: str) -> str | None:
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception as exc:  # noqa: BLE001
        _logger.debug("search_model_versions falhou para latest %s: %s", model_name, exc)
        return None
    if not versions:
        return None
    latest = max(versions, key=lambda v: int(v.version))
    return str(latest.version)
