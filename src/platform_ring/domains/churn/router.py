"""Rotas HTTP — domínio churn (TC01 tabular)."""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import Literal

import httpx
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import FileResponse

from core.deps import (
    get_current_user,
    get_session,
    require_admin,
    require_airflow_api_trigger_enabled,
    require_sync_training_routes_enabled,
)
from ml_core_ring.paths import airflow_upload_path, resolved_ml_shared_uploads_dir
from models.users import Users as users_models
from platform_ring.domains.common import (
    deployment_history,
    file_response_for_run,
    list_domain_runs,
    metrics_json_for_response_header,
    predict_for_domain_route,
    promote_domain,
    rollback_domain,
)
from platform_ring.schemas import contracts as platform_schemas
from platform_ring.schemas.churn_features import ChurnFeaturesInput
from platform_ring.training_trigger import trigger_training_dag
from services.processor import processor_service

DOMAIN = "churn"

router = APIRouter(prefix=f"/domains/{DOMAIN}", tags=[f"domain-{DOMAIN}"])


def _schedule_remove(path: str) -> None:
    try:
        if path and os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


@router.post(
    "/predict", status_code=status.HTTP_200_OK, response_model=platform_schemas.PredictResponse
)
async def churn_predict(
    payload: ChurnFeaturesInput,
    db: AsyncSession = Depends(get_session),
    user_logged: users_models = Depends(get_current_user),
):
    """Predição churn — corpo = features Telco (sem campo ``domain``)."""
    features = payload.model_dump(mode="json", by_alias=True)
    return await predict_for_domain_route(db, domain=DOMAIN, features=features, user=user_logged)


@router.post(
    "/admin/promote",
    status_code=status.HTTP_201_CREATED,
    response_model=platform_schemas.DeployedModelResponse,
)
async def churn_promote(
    db: AsyncSession = Depends(get_session),
    admin: users_models = Depends(require_admin),
):
    """Promove run FE activo de churn para servir em ``/domains/churn/predict``."""
    return await promote_domain(db, domain=DOMAIN, admin=admin)


@router.get("/admin/runs", status_code=status.HTTP_200_OK)
async def churn_list_runs(
    pipeline_type: Literal["baseline", "feature_engineering"] | None = Query(
        None, description="Filtrar baseline ou feature_engineering."
    ),
    run_status: Literal["processing", "completed", "failed"] | None = Query(
        None, alias="status", description="Estado da execução."
    ),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_session),
    admin: users_models = Depends(require_admin),
):
    return await list_domain_runs(
        db, domain=DOMAIN, pipeline_type=pipeline_type, run_status=run_status, limit=limit
    )


@router.get(
    "/admin/deployments/history",
    status_code=status.HTTP_200_OK,
    response_model=list[platform_schemas.DeployedModelResponse],
)
async def churn_deployment_history(
    db: AsyncSession = Depends(get_session),
    admin: users_models = Depends(require_admin),
):
    return await deployment_history(db, domain=DOMAIN)


@router.post(
    "/admin/rollback",
    status_code=status.HTTP_200_OK,
    response_model=platform_schemas.DeployedModelResponse,
)
async def churn_rollback(
    db: AsyncSession = Depends(get_session),
    admin: users_models = Depends(require_admin),
):
    return await rollback_domain(db, domain=DOMAIN)


@router.post(
    "/admin/train/trigger",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=platform_schemas.TriggerDagResponse,
)
async def churn_train_trigger(
    file: UploadFile = File(..., description="CSV Telco (obrigatório)."),
    optimization_metric: Literal["accuracy", "precision", "recall", "f1", "roc_auc"] = Form(
        "recall"
    ),
    min_precision: float | None = Form(None),
    min_roc_auc: float | None = Form(None),
    tuning_n_iter: int | None = Form(None),
    time_limit_minutes: int = Form(30),
    acc_target: float | None = Form(None),
    auto_promote: bool = Form(False),
    admin: users_models = Depends(require_airflow_api_trigger_enabled),
):
    """Treino automático churn via Airflow ``ml_training_dispatch`` (branch tabular)."""
    upload_dir = resolved_ml_shared_uploads_dir()
    os.makedirs(upload_dir, exist_ok=True)
    filename = f"{DOMAIN}_{uuid.uuid4().hex[:8]}_{file.filename}"
    host_path = os.path.join(upload_dir, filename)
    content = await file.read()
    await asyncio.to_thread(Path(host_path).write_bytes, content)

    csv_path_airflow = airflow_upload_path(filename)

    try:
        result = await trigger_training_dag(
            domain=DOMAIN,
            user_id=admin.id,
            csv_path=csv_path_airflow,
            optimization_metric=optimization_metric,
            min_precision=min_precision,
            min_roc_auc=min_roc_auc,
            tuning_n_iter=tuning_n_iter,
            time_limit_minutes=time_limit_minutes,
            acc_target=acc_target,
            auto_promote=auto_promote,
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
    "/admin/train/baseline",
    status_code=status.HTTP_201_CREATED,
    response_class=FileResponse,
    summary="Treino baseline sync (debug)",
)
async def churn_train_baseline_sync(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_session),
    admin: users_models = Depends(require_sync_training_routes_enabled),
):
    """Treino baseline síncrono — só dev + admin (``ENVIRONMENT≠prd``)."""
    try:
        run = await processor_service.run_baseline(
            file=file, objective=DOMAIN, user_id=admin.id, db=db
        )
        return file_response_for_run(run, "baseline")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Treino baseline falhou: {exc}",
        ) from exc


@router.post(
    "/admin/train/feature-engineering",
    status_code=status.HTTP_201_CREATED,
    response_class=FileResponse,
    summary="Treino FE sync (debug)",
)
async def churn_train_fe_sync(
    background_tasks: BackgroundTasks,
    optimization_metric: Literal["accuracy", "precision", "recall", "f1", "roc_auc"] = Form(
        "recall"
    ),
    min_precision: float | None = Form(None),
    min_roc_auc: float | None = Form(None),
    tuning_n_iter: int | None = Form(None),
    time_limit_minutes: int = Form(2),
    acc_target: float | None = Form(None),
    decision_threshold: float = Form(0.3),
    db: AsyncSession = Depends(get_session),
    admin: users_models = Depends(require_sync_training_routes_enabled),
):
    """Treino FE síncrono — só dev + admin."""
    try:
        run, zip_path = await processor_service.run_feature_engineering(
            objective=DOMAIN,
            user_id=admin.id,
            db=db,
            optimization_metric=optimization_metric,
            min_precision=min_precision,
            min_roc_auc=min_roc_auc,
            tuning_n_iter=tuning_n_iter,
            time_limit_minutes=time_limit_minutes,
            acc_target=acc_target,
            decision_threshold=decision_threshold,
        )
        if run.status != "completed":
            if zip_path:
                _schedule_remove(zip_path)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"run_id": run.id, "error": run.error_message or "Pipeline não concluiu."},
            )
        if not zip_path or not os.path.isfile(zip_path):
            if zip_path:
                _schedule_remove(zip_path)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="ZIP de artefactos não gerado.",
            )
        background_tasks.add_task(_schedule_remove, zip_path)
        return FileResponse(
            path=zip_path,
            filename=f"fe_artifacts_run_{run.id}.zip",
            media_type="application/zip",
            status_code=status.HTTP_201_CREATED,
            headers={
                "X-Pipeline-Run-Id": str(run.id),
                "X-Pipeline-Type": "feature_engineering",
                "X-Pipeline-Objective": run.objective,
                "X-Pipeline-Metrics": metrics_json_for_response_header(run.metrics),
            },
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Treino FE falhou: {exc}",
        ) from exc
