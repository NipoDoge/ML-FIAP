from schemas import roles_schemas as roles_schemas
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from typing import List

from models.roles import Roles as roles_models


async def register_role(role: roles_schemas.role, db: AsyncSession) -> roles_models:
    new_role: roles_models = roles_models(description=role.description, active=int(role.active))

    async with db as session:
        session.add(new_role)
        await session.commit()
        await session.refresh(new_role)
        return new_role


async def select_all_roles(db: AsyncSession) -> List[roles_schemas.role]:
    async with db as session:
        querie = select(roles_models).filter(roles_models.active.is_(True))
        resultset = await session.execute(querie)
        roles: List[roles_schemas.role] = resultset.scalars().unique().all()
        return roles


async def select_role(id_role: int, db: AsyncSession) -> roles_schemas.role:
    async with db as session:
        querie = select(roles_models).filter(
            roles_models.id == id_role,
            roles_models.active.is_(True),
        )
        resultset = await session.execute(querie)
        role: roles_schemas.role = resultset.scalars().unique().one_or_none()
        return role


async def update_role(id_role: int, role: roles_schemas.role_update, db: AsyncSession):
    async with db as session:
        querie = select(roles_models).filter(
            roles_models.id == id_role,
            roles_models.active.is_(True),
        )
        resultset = await session.execute(querie)
        role_update: roles_schemas.role = resultset.scalars().unique().one_or_none()

        if role.description:
            role_update.description = role.description

        await session.commit()
        return role_update


async def drop_role(id_role: int, db: AsyncSession):
    async with db as session:
        querie = select(roles_models).filter(
            roles_models.id == id_role,
            roles_models.active.is_(True),
        )
        resultset = await session.execute(querie)
        role_del: roles_schemas.role = resultset.scalars().unique().one_or_none()

        if role_del:
            role_del.active = False
            await session.commit()
            await session.refresh(role_del)
            return role_del
