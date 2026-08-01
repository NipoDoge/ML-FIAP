"""Facade de consulta de runs — extensão multi-domínio."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession

from models.pipeline_runs import PipelineRuns
from services.processor.pipeline_runs_service import list_pipeline_runs as _list_legacy

PipelineTypeFilter = Literal["baseline", "feature_engineering", "recommendation"]
StatusFilter = Literal["processing", "completed", "failed"]


async def list_runs_for_domain(
    db: AsyncSession,
    *,
    domain: str,
    pipeline_type: PipelineTypeFilter | None = None,
    status: StatusFilter | None = None,
    limit: int = 50,
) -> Sequence[PipelineRuns]:
    """Lista runs filtrados por domínio (normalizado)."""
    return await _list_legacy(
        db,
        objective=domain,
        pipeline_type=pipeline_type,  # type: ignore[arg-type]
        status=status,
        limit=limit,
    )
