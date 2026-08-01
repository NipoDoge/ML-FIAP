"""Helpers partilhados pelas rotas ``/v1/domains/{domain}/…``."""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any, Literal

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from models.users import Users as users_models
from platform_ring.promote_response import build_deployed_model_response
from platform_ring.promote_service import promote_for_domain
from platform_ring.runs_service import list_runs_for_domain
from platform_ring.schemas import contracts as platform_schemas
from services.processor import processor_service
from services.processor.deployment_service import (
    NoActiveDeploymentError,
    RollbackError,
    get_deployment_history,
    rollback_deployment,
)
from services.processor.pipeline_run_view import build_pipeline_run_view

_logger = logging.getLogger(__name__)

PipelineTypeFilter = Literal["baseline", "feature_engineering", "recommendation"]
StatusFilter = Literal["processing", "completed", "failed"]


def metrics_json_for_response_header(metrics: dict | None) -> str:
    """Serializa métricas para cabeçalhos HTTP (ASCII-only)."""

    def norm(o: Any) -> Any:
        if o is None:
            return None
        if isinstance(o, dict):
            return {str(k): norm(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [norm(v) for v in o]
        if isinstance(o, bool):
            return o
        if isinstance(o, float):
            return None if not math.isfinite(o) else o
        if isinstance(o, int):
            return o
        if isinstance(o, str):
            return o
        if hasattr(o, "tolist") and callable(o.tolist):
            try:
                return norm(o.tolist())
            except (AttributeError, TypeError, ValueError) as exc:
                _logger.debug("Métricas header: tolist ignorado para %r: %s", type(o), exc)
        if hasattr(o, "item") and callable(o.item):
            try:
                return norm(o.item())
            except (AttributeError, TypeError, ValueError) as exc:
                _logger.debug("Métricas header: item ignorado para %r: %s", type(o), exc)
        return str(o)

    if not metrics:
        return ""
    try:
        payload = norm(metrics)
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        _logger.warning("Métricas header: fallback após falha de serialização: %s", exc)
        return "{}"


async def list_domain_runs(
    db: AsyncSession,
    *,
    domain: str,
    pipeline_type: PipelineTypeFilter | None = None,
    run_status: StatusFilter | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    runs = await list_runs_for_domain(
        db,
        domain=domain.strip().lower(),
        pipeline_type=pipeline_type,
        status=run_status,
        limit=limit,
    )
    return [build_pipeline_run_view(r) for r in runs]


async def promote_domain(
    db: AsyncSession,
    *,
    domain: str,
    admin: users_models,
    pipeline_run_id: int | None = None,
) -> platform_schemas.DeployedModelResponse:
    try:
        result = await promote_for_domain(
            domain=domain.strip().lower(),
            promoted_by_user_id=admin.id,
            db=db,
            pipeline_run_id=pipeline_run_id,
        )
        return build_deployed_model_response(result.deployment, result.mlflow_registry)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


async def deployment_history(
    db: AsyncSession,
    *,
    domain: str,
) -> list[platform_schemas.DeployedModelResponse]:
    records = await get_deployment_history(domain=domain.strip().lower(), db=db)
    if not records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Nenhum deployment encontrado para o domínio '{domain}'.",
        )
    return records


async def rollback_domain(
    db: AsyncSession,
    *,
    domain: str,
) -> platform_schemas.DeployedModelResponse:
    try:
        return await rollback_deployment(domain=domain.strip().lower(), db=db)
    except RollbackError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Rollback falhou: {exc}",
        ) from exc


async def predict_for_domain_route(
    db: AsyncSession,
    *,
    domain: str,
    features: dict[str, Any],
    user: users_models,
) -> platform_schemas.PredictResponse:
    try:
        pred, inference_report, reco_extra = await processor_service.predict_for_domain(
            domain=domain.strip().lower(),
            features=features,
            user_id=user.id,
            db=db,
        )
    except NoActiveDeploymentError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao realizar predição: {exc}",
        ) from exc

    prob_pct = None
    if pred.probability is not None:
        prob_pct = round(float(pred.probability) * 100, 2)
    prob_display = f"{prob_pct}%" if prob_pct is not None else None
    recommended = reco_extra.get("recommended_items") if reco_extra else None
    return platform_schemas.PredictResponse(
        id=pred.id,
        domain=domain.strip().lower(),
        pipeline_run_id=pred.pipeline_run_id,
        prediction=pred.prediction,
        probability=prob_pct,
        probability_display=prob_display,
        recommended_items=recommended,
        input_data=pred.input_data if isinstance(pred.input_data, dict) else dict(pred.input_data),
        inference_report=inference_report,
    )


def file_response_for_run(run, pipeline_type: str):
    """Devolve CSV em disco + metadados em cabeçalhos (treino sync tabular)."""
    from starlette.responses import FileResponse

    if run.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"run_id": run.id, "error": run.error_message},
        )
    if not run.csv_output_path or not os.path.isfile(run.csv_output_path):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Arquivo CSV de saída não encontrado após o pipeline.",
        )
    return FileResponse(
        path=run.csv_output_path,
        filename=os.path.basename(run.csv_output_path),
        media_type="text/csv",
        status_code=status.HTTP_201_CREATED,
        headers={
            "X-Pipeline-Run-Id": str(run.id),
            "X-Pipeline-Type": pipeline_type,
            "X-Pipeline-Objective": run.objective,
            "X-Pipeline-Metrics": metrics_json_for_response_header(run.metrics),
        },
    )
