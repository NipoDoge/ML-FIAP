"""Consultas sobre execuções de pipeline (lista para promoção / auditoria)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.pipeline_runs import PipelineRuns

PipelineTypeLiteral = Literal["baseline", "feature_engineering", "recommendation"]
StatusLiteral = Literal["processing", "completed", "failed"]


def _normalize_domain(domain: str) -> str:
    return domain.strip().lower()


async def list_pipeline_runs(
    db: AsyncSession,
    *,
    objective: str | None = None,
    pipeline_type: PipelineTypeLiteral | None = None,
    status: StatusLiteral | None = None,
    limit: int = 50,
) -> Sequence[PipelineRuns]:
    """
    Lista ``PipelineRuns`` por ordem descendente de ``created_at`` (mais recentes primeiro).
    Filtra opcionalmente por domínio, tipo e status (todos combináveis).

    Parameters
    ----------
    objective:
        Igual ao ``objective`` / ``domain`` gravado nos runs (normalizado case-insensitive).
    pipeline_type:
        ``baseline``, ``feature_engineering`` ou ``recommendation``.
    status:
        ``processing``, ``completed`` ou ``failed``.
    limit:
        Máximo de linhas devolvidas (cap a 200 no endpoint).
    """
    stmt = select(PipelineRuns)
    predicates = []

    if objective is not None and objective.strip():
        d = _normalize_domain(objective)
        predicates.append(func.lower(PipelineRuns.objective) == d)
    if pipeline_type is not None:
        predicates.append(PipelineRuns.pipeline_type == pipeline_type)
    if status is not None:
        predicates.append(PipelineRuns.status == status)

    if predicates:
        stmt = stmt.where(and_(*predicates))

    stmt = stmt.order_by(PipelineRuns.created_at.desc()).limit(limit)

    result = await db.execute(stmt)
    return result.scalars().all()
