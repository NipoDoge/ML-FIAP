from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from core.configs import settings
from core.deps import get_session

router = APIRouter()


def _meta() -> dict[str, str]:
    return {
        "service": settings.project_name,
        "version": settings.project_version.strip() or settings.project_version,
        "environment": settings.environment,
    }


@router.get(
    "",
    summary="Health",
    description=(
        "Sem autenticação. **200** só após `SELECT 1` no Postgres — processo vivo **e** BD acessível; "
        "**503** se a BD falhar. Rota única propositada para o âmbito académico (sem separar liveness/readiness)."
    ),
    status_code=status.HTTP_200_OK,
)
async def health(db: AsyncSession = Depends(get_session)) -> dict:
    try:
        await db.execute(text("SELECT 1"))
    except SQLAlchemyError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Base de dados indisponível.",
        )
    return {"alive": True, "database": "up", **_meta()}
