import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import _generate_access_token
from core.deps import get_current_user, get_session
from models.users import Users as users_models
from schemas import users_schemas
from services.auth import auth_service

logger = logging.getLogger(__name__)

router = APIRouter()


# POST user / signup
@router.post("/signup", response_model=users_schemas.users, status_code=status.HTTP_201_CREATED)
async def signup(user: users_schemas.users_create, db: AsyncSession = Depends(get_session)):
    try:
        new_user: users_models = await auth_service.register_user(user, db)
        return new_user
    except IntegrityError as exc:
        detail_log = str(exc.orig) if exc.orig else str(exc)
        logger.warning("IntegrityError no signup: %s", detail_log, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_406_NOT_ACCEPTABLE,
            detail=(
                "Ocorreu um erro ao processar esta solicitação. "
                "Os detalhes foram registrados no log do servidor."
            ),
        )


# POST Login
@router.post("/authenticate", status_code=status.HTTP_200_OK)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_session)
):
    user = await auth_service.authorize(form_data.username, form_data.password, db)
    logger.info(f"User: {user}")
    if not user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Dados incorretos")

    return JSONResponse(
        content={"access_token": _generate_access_token(sub=user.id), "token_type": "bearer"},
        status_code=status.HTTP_200_OK,
    )


# GET Logged
@router.get("/logged", response_model=users_schemas.users, status_code=status.HTTP_200_OK)
async def get_logged(user_logged: users_models = Depends(get_current_user)):
    return user_logged
