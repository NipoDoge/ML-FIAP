
from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from core.deps import get_current_user, get_session
from models.roles import Roles as roles_models
from models.users import Users as users_models
from schemas import roles_schemas
from services.roles import roles_services as roles_service

router = APIRouter()


# POST role
@router.post("/", status_code=status.HTTP_201_CREATED, response_model=roles_schemas.role)
async def post_role(
    role: roles_schemas.role,
    db: AsyncSession = Depends(get_session),
    user_logged: users_models = Depends(get_current_user),
):
    try:
        if user_logged.role_id == roles_models.ADMINISTRATOR:
            new_role: roles_models = await roles_service.register_role(role, db)
            return new_role
        else:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


# GET roles
@router.get("/", status_code=status.HTTP_200_OK, response_model=list[roles_schemas.role])
async def get_roles(
    db: AsyncSession = Depends(get_session), user_logged: users_models = Depends(get_current_user)
):
    try:
        if user_logged.role_id == roles_models.ADMINISTRATOR:
            roles: list[roles_schemas.role] = await roles_service.select_all_roles(db)
            return roles
        else:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


# GET role
@router.get("/{id_role}", status_code=status.HTTP_200_OK, response_model=roles_schemas.role)
async def get_role(
    id_role: int,
    db: AsyncSession = Depends(get_session),
    user_logged: users_models = Depends(get_current_user),
):
    try:
        if user_logged.role_id == roles_models.ADMINISTRATOR:
            role = await roles_service.select_role(id_role, db)
            if role:
                return role
            else:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        else:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


# PUT role
@router.put("/{id_role}", status_code=status.HTTP_202_ACCEPTED, response_model=roles_schemas.role)
async def put_role(
    id_role: int,
    role: roles_schemas.role_update,
    db: AsyncSession = Depends(get_session),
    user_logged: users_models = Depends(get_current_user),
):
    try:
        if user_logged.role_id == roles_models.ADMINISTRATOR:
            role_update: roles_schemas.role = await roles_service.update_role(id_role, role, db)
            return role_update
        else:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


# DELETE role
@router.delete("/{id_role}", status_code=status.HTTP_202_ACCEPTED)
async def delete_role(
    id_role: int,
    db: AsyncSession = Depends(get_session),
    user_logged: users_models = Depends(get_current_user),
):
    try:
        if user_logged.role_id == roles_models.ADMINISTRATOR:
            deleted = await roles_service.drop_role(id_role, db)
            if deleted:
                return Response(
                    status_code=status.HTTP_204_NO_CONTENT,
                    content=f"Papel {deleted.description} deletado com sucesso.",
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="Papel não encontrado."
                )
        else:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
