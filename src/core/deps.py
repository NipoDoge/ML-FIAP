from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import Session
from core.auth import oauth2_scheme
from models.users import Users as users_models
from models.roles import Roles
from fastapi import HTTPException, status
from jose import JWTError, jwt
from core.configs import settings
from typing import Optional
from pydantic import BaseModel
from sqlalchemy.future import select


class TokenData(BaseModel):
    username: Optional[str] = None


async def get_session():
    session: AsyncSession = Session()
    try:
        yield session
    finally:
        await session.close()


async def get_current_user(
    db: Session = Depends(get_session), token: str = Depends(oauth2_scheme)
) -> users_models:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Não foi possível autenticar o usuario.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token, settings.jwt_secret, algorithms=settings.algorithm, options={"verify_aud": False}
        )
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception

        token_data: TokenData = TokenData(username=username)

    except JWTError:
        raise credentials_exception

    async with db as session:
        querie = select(users_models).filter(users_models.id == int(token_data.username))
        resultset = await session.execute(querie)
        user: users_models = resultset.scalars().unique().one_or_none()

        if user is None:
            raise credentials_exception

        return user


async def require_admin(user: users_models = Depends(get_current_user)) -> users_models:
    if user.role_id != Roles.ADMINISTRATOR:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Acesso restrito a administradores.",
        )
    return user


async def require_sync_training_routes_enabled(
    admin: users_models = Depends(require_admin),
) -> users_models:
    """Admin autenticado + treino síncrono só fora de produção (baseline / FE)."""
    if settings.is_production:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Treino síncrono não está disponível neste ambiente. Em produção use o Airflow: "
                "variável ml_training_dispatch_conf + CSV no volume partilhado + trigger manual do DAG ml_training_dispatch."
            ),
        )
    return admin


async def require_airflow_api_trigger_enabled(
    admin: users_models = Depends(require_admin),
) -> users_models:
    """Disparo do DAG via API só fora de produção; em prd o fluxo é UI do Airflow + Variables."""
    if settings.is_production:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Trigger do DAG pela API não está disponível em produção. Use a UI do Airflow: "
                "Admin → Variables (ml_training_dispatch_conf), coloque o dataset no volume acessível ao worker e Trigger DAG ml_training_dispatch."
            ),
        )
    return admin
