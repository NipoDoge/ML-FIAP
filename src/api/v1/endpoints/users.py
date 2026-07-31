from fastapi import APIRouter, HTTPException, status, Depends, Response
from models.users import Users as users_models

from schemas import users_schemas as users_schemas
from core.deps import get_session, get_current_user
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
import datetime
from services.user import users_services as users_service
from core.security import get_password_hash

router = APIRouter()


# GET users
@router.get("/", response_model=List[users_schemas.usersGetData], status_code=status.HTTP_200_OK)
async def get_users(
    db: AsyncSession = Depends(get_session), user_logged: users_models = Depends(get_current_user)
):
    try:
        users: List[users_schemas.usersGetData] = await users_service.select_all_users(db)
        return users
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


# GET user
@router.get("/{id_user}", response_model=users_schemas.usersGetData, status_code=status.HTTP_200_OK)
async def get_user(
    id_user: int,
    db: AsyncSession = Depends(get_session),
    user_logged: users_models = Depends(get_current_user),
):
    try:
        user: users_schemas.usersGetData | None = await users_service.select_user(id_user, db)
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuário não encontrado.")
    return user


# PUT user
@router.put("/{id_user}", status_code=status.HTTP_200_OK)
async def put_user(
    id_user: int,
    payload: users_schemas.users_update,
    db: AsyncSession = Depends(get_session),
    user_logged: users_models = Depends(get_current_user),
):
    try:
        user_id = id_user
        user_data = payload.model_dump(exclude_unset=True)

        updated = await users_service.update_user(user_id, user_data, db)
        if updated:
            return Response(
                status_code=status.HTTP_200_OK, content="Usuário atualizado com sucesso."
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Usuário não encontrado."
            )

    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


# DELETE user
@router.delete("/{id_user}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    id_user,
    db: AsyncSession = Depends(get_session),
    user_logged: users_models = Depends(get_current_user),
):
    try:
        user = await users_service.drop_user(id_user, db)
        if user:
            return Response(
                status_code=status.HTTP_204_NO_CONTENT,
                content=f"Usuário {user.name} deletado com sucesso.",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Usuário não encontrado."
            )

    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


# Forgot Password - Send Email
@router.post("/forgot-password/{email}", status_code=status.HTTP_200_OK)
async def forgot_password(email: str, db: AsyncSession = Depends(get_session)):
    try:
        token = await users_service.generate_reset_token(email, db)
        await users_service.send_email(email, token)
        return {"message": "Email de redefinição de senha enviado!."}
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


# Reset Password
@router.post("/reset-password", status_code=status.HTTP_200_OK)
async def reset_password(token: str, password: str, db: AsyncSession = Depends(get_session)):
    try:
        user: users_schemas.users_update = await users_service.get_user_by_reset_token(token, db)
        if not user or user.reset_password_expires < datetime.datetime.now():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Token inválido ou expirado"
            )

        user.password = get_password_hash(password)
        user.reset_password_token = None
        user.reset_password_expires = None
        db.add(user)
        await db.commit()
        return {"message": "Senha redefinida com sucesso!"}
    except HTTPException as e:
        if e.status_code != status.HTTP_401_UNAUTHORIZED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Ocorreu um erro durante a solicitação.",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Você não possui permissão para consultar esses dados.",
            )
