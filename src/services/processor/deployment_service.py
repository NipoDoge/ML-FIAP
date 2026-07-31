"""Gestão de modelos promovidos (produção) por domínio."""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import and_, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from core.configs import settings
from ml_core_ring.paths import resolve_shared_artifact_path
from models.deployed_models import DeployedModels
from models.pipeline_runs import PipelineRuns
from services.utils import utcnow


class RollbackError(Exception):
    """Não existe deployment archived para restaurar no domínio."""


class NoActiveDeploymentError(Exception):
    """Não existe modelo ativo para o domínio solicitado."""


def _normalize_domain(domain: str) -> str:
    return domain.strip().lower()


async def get_active_deployment(domain: str, db: AsyncSession) -> DeployedModels | None:
    d = _normalize_domain(domain)
    stmt = (
        select(DeployedModels)
        .where(
            DeployedModels.domain == d,
            DeployedModels.status == "active",
            DeployedModels.active.is_(True),
        )
        .options(selectinload(DeployedModels.pipeline_run))
    )
    result = await db.execute(stmt)
    return result.scalars().one_or_none()


async def promote_active_feature_engineering_for_objective(
    objective: str,
    promoted_by_user_id: int,
    db: AsyncSession,
) -> DeployedModels:
    """
    Promove o único ``PipelineRuns`` de feature engineering **activo**, concluído e com o ``objective``
    dado (tipicamente ``settings.objective``). Exige exactamente um candidato.
    """
    obj = _normalize_domain(objective)
    backend = "mlp" if settings.use_mlp_for_prediction else "sklearn"
    predicates = [
        PipelineRuns.pipeline_type == "feature_engineering",
        PipelineRuns.status == "completed",
        PipelineRuns.active.is_(True),
        func.lower(PipelineRuns.objective) == obj,
        PipelineRuns.inference_backend == backend,
    ]
    if settings.is_production:
        predicates.append(PipelineRuns.is_airflow_run.is_(True))
    stmt = select(PipelineRuns).where(and_(*predicates))
    res = await db.execute(stmt)
    runs = list(res.scalars().all())
    if len(runs) == 0:
        raise ValueError(
            f"Nenhum pipeline feature_engineering activo e concluído para objective={objective!r} "
            f"e inference_backend={backend!r}. Treine FE com USE_MLP_FOR_PREDICTION coerente "
            "ou reveja o estado em pipeline_runs."
        )
    if len(runs) > 1:
        raise ValueError(
            f"Ambiguidade: {len(runs)} runs FE activos para objective={objective!r} "
            f"e inference_backend={backend!r}. Desactive os obsoletos antes de promover."
        )
    run_id = runs[0].id
    return await promote_pipeline_run(
        domain=objective,
        pipeline_run_id=run_id,
        promoted_by_user_id=promoted_by_user_id,
        pipeline_type="feature_engineering",
        db=db,
    )


async def promote_pipeline_run(
    domain: str,
    pipeline_run_id: int,
    promoted_by_user_id: int,
    pipeline_type: str,
    db: AsyncSession,
) -> DeployedModels:
    """
    Arquiva o deployment ativo anterior do domínio e cria um novo como active.
    ``pipeline_type`` deve ser ``feature_engineering`` e o run na BD tem de coincidir.
    """
    d = _normalize_domain(domain)
    pt = pipeline_type.strip().lower()
    allowed = frozenset({"feature_engineering", "recommendation"})
    if pt not in allowed:
        raise ValueError(
            f"pipeline_type inválido para promote: {pipeline_type!r}. "
            f"Permitidos: {sorted(allowed)}."
        )

    stmt_run = select(PipelineRuns).where(
        PipelineRuns.id == pipeline_run_id,
        PipelineRuns.active.is_(True),
    )
    res = await db.execute(stmt_run)
    run = res.scalars().one_or_none()
    if not run:
        raise ValueError(f"PipelineRun {pipeline_run_id} não encontrado.")
    if run.status != "completed":
        raise ValueError(f"PipelineRun {pipeline_run_id} não está concluído (status={run.status}).")
    if run.pipeline_type != pt:
        raise ValueError(
            f"PipelineRun {pipeline_run_id} é do tipo '{run.pipeline_type}', "
            f"mas foi pedida promoção como '{pt}'."
        )
    if _normalize_domain(run.objective) != d:
        raise ValueError(
            f"Domínio '{domain}' não coincide com o objective do run ('{run.objective}')."
        )
    if pt == "feature_engineering":
        local_model_path = resolve_shared_artifact_path(run.model_path)
        if not local_model_path or not os.path.exists(local_model_path):
            raise ValueError(
                "Artefato do modelo não encontrado. "
                f"caminho_na_BD={run.model_path!r} "
                f"caminho_resolvido_na_API={local_model_path!r} "
                "(esperado existir se o treino foi no Airflow e o volume ml_shared está montado em "
                "/var/www/ml_shared no contentor api_processing). "
                "Se acabou de alterar o código Python, faça rebuild: "
                "`docker compose build api_processing && docker compose up -d api_processing`."
            )
    elif pt == "recommendation":
        prefix = run.model_path or (run.metrics or {}).get("artifact_paths", {}).get("prefix")
        if not prefix:
            raise ValueError(
                "Run de recomendação sem prefix de artefacto (model_path ou metrics.artifact_paths)."
            )
        resolved = resolve_shared_artifact_path(str(prefix))
        if not resolved:
            raise ValueError(f"Prefix de recomendação inválido: {prefix!r}")
        base = Path(resolved)
        pt_file = base.with_suffix(".pt")
        meta_file = base.with_suffix(".meta.json")
        if not pt_file.is_file() and not base.is_file():
            raise ValueError(f"Modelo torch de recomendação não encontrado: {pt_file}")
        if not meta_file.is_file():
            raise ValueError(f"Meta do modelo de recomendação em falta: {meta_file}")

    await db.execute(
        update(PipelineRuns)
        .where(
            PipelineRuns.id != pipeline_run_id,
            PipelineRuns.pipeline_type == pt,
            func.lower(PipelineRuns.objective) == d,
        )
        .values(active=False),
        execution_options={"synchronize_session": False},
    )

    await db.execute(
        update(DeployedModels)
        .where(
            DeployedModels.domain == d,
            DeployedModels.status == "active",
        )
        .values(status="archived")
    )

    row = DeployedModels(
        domain=d,
        pipeline_run_id=pipeline_run_id,
        status="active",
        promoted_at=utcnow(),
        promoted_by_user_id=promoted_by_user_id,
        metrics_snapshot=run.metrics,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    await db.refresh(row, attribute_names=["pipeline_run"])
    return row


async def promote_recommendation_for_domain(
    domain: str,
    promoted_by_user_id: int,
    db: AsyncSession,
    *,
    pipeline_run_id: int | None = None,
) -> DeployedModels:
    """Promove run de recomendação activo (ou id explícito) para ``/predict``."""
    d = _normalize_domain(domain)
    if pipeline_run_id is not None:
        return await promote_pipeline_run(
            domain=domain,
            pipeline_run_id=pipeline_run_id,
            promoted_by_user_id=promoted_by_user_id,
            pipeline_type="recommendation",
            db=db,
        )

    predicates = [
        PipelineRuns.pipeline_type == "recommendation",
        PipelineRuns.status == "completed",
        PipelineRuns.active.is_(True),
        func.lower(PipelineRuns.objective) == d,
    ]
    if settings.is_production:
        predicates.append(PipelineRuns.is_airflow_run.is_(True))

    res = await db.execute(select(PipelineRuns).where(and_(*predicates)))
    runs = list(res.scalars().all())
    if len(runs) == 0:
        raise ValueError(
            f"Nenhum run recommendation activo e concluído para domain={domain!r}. "
            "Treine via ml_training_dispatch e confirme pipeline_runs."
        )
    if len(runs) > 1:
        raise ValueError(
            f"Ambiguidade: {len(runs)} runs recommendation activos para domain={domain!r}."
        )
    return await promote_pipeline_run(
        domain=domain,
        pipeline_run_id=runs[0].id,
        promoted_by_user_id=promoted_by_user_id,
        pipeline_type="recommendation",
        db=db,
    )


async def get_deployment_history(
    domain: str, db: AsyncSession, limit: int = 10
) -> list[DeployedModels]:
    """Retorna os últimos `limit` deployments (active + archived) para o domínio, do mais recente ao mais antigo."""
    d = _normalize_domain(domain)
    stmt = (
        select(DeployedModels)
        .where(DeployedModels.domain == d)
        .order_by(DeployedModels.promoted_at.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def rollback_deployment(domain: str, db: AsyncSession) -> DeployedModels:
    """
    Reverte para o deployment archived mais recente do domínio.
    Arquiva o active atual e reativa o último archived.
    """
    d = _normalize_domain(domain)

    stmt_archived = (
        select(DeployedModels)
        .where(DeployedModels.domain == d, DeployedModels.status == "archived")
        .order_by(DeployedModels.promoted_at.desc())
        .limit(1)
    )
    res = await db.execute(stmt_archived)
    previous = res.scalars().one_or_none()
    if not previous:
        raise RollbackError(f"Nenhum deployment archived encontrado para o domínio '{domain}'.")

    await db.execute(
        update(DeployedModels)
        .where(DeployedModels.domain == d, DeployedModels.status == "active")
        .values(status="archived")
    )

    previous.status = "active"
    db.add(previous)
    await db.commit()
    await db.refresh(previous)
    return previous
