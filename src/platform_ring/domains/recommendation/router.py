"""Rotas HTTP — domínio recommendation (TC02)."""

from __future__ import annotations

import json
from typing import Any, Literal

import httpx
from fastapi import APIRouter, Depends, Form, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from core.deps import (
    get_current_user,
    get_session,
    require_admin,
    require_airflow_api_trigger_enabled,
    require_sync_training_routes_enabled,
)
from ml_core_ring.dispatch_params import flatten_dispatch_train_params
from models.users import Users as users_models
from platform_ring.domains.common import (
    deployment_history,
    list_domain_runs,
    predict_for_domain_route,
    promote_domain,
    rollback_domain,
)
from platform_ring.schemas import contracts as platform_schemas
from platform_ring.schemas.recommendation_features import RecommendationFeaturesInput
from platform_ring.training_trigger import trigger_training_dag

DOMAIN = "recommendation"

router = APIRouter(prefix=f"/domains/{DOMAIN}", tags=[f"domain-{DOMAIN}"])


class RecommendationTrainSyncRequest(BaseModel):
    """Corpo opcional para treino sync (debug) — parâmetros flatten como no dispatch."""

    model_config = ConfigDict(extra="allow")

    train_models: list[str] | None = Field(default=None, example=["torch_embedding"])
    n_epochs: int | None = Field(default=None, ge=1)
    top_k: int | None = Field(default=None, ge=1, le=100)
    mlflow_experiment: str | None = None


class RecommendationTrainSyncResponse(BaseModel):
    pipeline_run_id: int
    champion_name: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    mlflow_run_id: str | None = None
    status: str = "completed"


@router.post(
    "/predict", status_code=status.HTTP_200_OK, response_model=platform_schemas.PredictResponse
)
async def recommendation_predict(
    payload: RecommendationFeaturesInput,
    db: AsyncSession = Depends(get_session),
    user_logged: users_models = Depends(get_current_user),
):
    """Predição reco — corpo = ``user_id`` + ``top_k`` (sem campo ``domain``)."""
    features = payload.model_dump(mode="json")
    return await predict_for_domain_route(db, domain=DOMAIN, features=features, user=user_logged)


@router.post(
    "/admin/promote",
    status_code=status.HTTP_201_CREATED,
    response_model=platform_schemas.DeployedModelResponse,
)
async def recommendation_promote(
    db: AsyncSession = Depends(get_session),
    admin: users_models = Depends(require_admin),
):
    return await promote_domain(db, domain=DOMAIN, admin=admin)


@router.get("/admin/runs", status_code=status.HTTP_200_OK)
async def recommendation_list_runs(
    run_status: Literal["processing", "completed", "failed"] | None = Query(
        None, alias="status", description="Estado da execução."
    ),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_session),
    admin: users_models = Depends(require_admin),
):
    return await list_domain_runs(
        db,
        domain=DOMAIN,
        pipeline_type="recommendation",
        run_status=run_status,
        limit=limit,
    )


@router.get(
    "/admin/deployments/history",
    status_code=status.HTTP_200_OK,
    response_model=list[platform_schemas.DeployedModelResponse],
)
async def recommendation_deployment_history(
    db: AsyncSession = Depends(get_session),
    admin: users_models = Depends(require_admin),
):
    return await deployment_history(db, domain=DOMAIN)


@router.post(
    "/admin/rollback",
    status_code=status.HTTP_200_OK,
    response_model=platform_schemas.DeployedModelResponse,
)
async def recommendation_rollback(
    db: AsyncSession = Depends(get_session),
    admin: users_models = Depends(require_admin),
):
    return await rollback_domain(db, domain=DOMAIN)


@router.post(
    "/admin/train/trigger",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=platform_schemas.TriggerDagResponse,
)
async def recommendation_train_trigger(
    top_k: int = Form(10, ge=1, le=100),
    train_models: str = Form(
        '["torch_embedding"]',
        description='JSON array, ex.: ["torch_embedding"]',
    ),
    n_epochs: int = Form(1, ge=1),
    mlflow_experiment: str = Form("tc02_recommendation"),
    admin: users_models = Depends(require_airflow_api_trigger_enabled),
):
    """Treino automático reco via Airflow → worker HTTP."""
    try:
        models_list = json.loads(train_models)
        if not isinstance(models_list, list):
            raise TypeError("train_models deve ser JSON array.")
    except (json.JSONDecodeError, TypeError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    extra = flatten_dispatch_train_params(
        {
            "top_k": top_k,
            "train_models": models_list,
            "n_epochs": n_epochs,
            "mlflow_experiment": mlflow_experiment,
        }
    )

    try:
        result = await trigger_training_dag(
            domain=DOMAIN,
            user_id=admin.id,
            extra=extra,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Airflow recusou o trigger: {exc.response.text}",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Airflow indisponível: {exc}",
        ) from exc

    return platform_schemas.TriggerDagResponse(
        dag_run_id=result.dag_run_id,
        dag_id=result.dag_id,
        domain=result.domain,
        objective=result.domain,
        csv_path=result.csv_path,
        message=result.message,
    )


@router.post(
    "/admin/train/sync",
    status_code=status.HTTP_201_CREATED,
    response_model=RecommendationTrainSyncResponse,
    summary="Treino reco sync via worker (debug)",
)
async def recommendation_train_sync(
    body: RecommendationTrainSyncRequest | None = None,
    admin: users_models = Depends(require_sync_training_routes_enabled),
):
    """Treino síncrono — chama worker (``WORKER_RECOMMENDATION_URL``) ou in-process se vazio."""
    from services.processor.recommendation_sync_service import train_recommendation_sync

    params = body.model_dump(exclude_none=True) if body else {}
    try:
        data = train_recommendation_sync(user_id=admin.id, params=params, domain=DOMAIN)
        return RecommendationTrainSyncResponse(**data)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Worker reco indisponível: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Treino sync falhou: {exc}",
        ) from exc
