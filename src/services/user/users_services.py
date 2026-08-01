import datetime
import secrets
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from models.roles import Roles as roles_models
from models.users import Users as users_models
from schemas import users_schemas


async def select_all_users(db: AsyncSession) -> list[users_schemas.usersGetData]:
    async with db as session:
        querie = (
            select(users_models)
            .order_by(users_models.id.asc())
            .filter(users_models.active.is_(True))
        )
        resultset = await session.execute(querie)
        users: list[users_schemas.usersGetData] = resultset.scalars().unique().all()

        users_list = []
        for user in users:
            # role
            role = select(roles_models).filter(
                roles_models.id == user.role_id,
                roles_models.active.is_(True),
            )
            role = await session.execute(role)
            role = role.scalars().unique().one_or_none()

            users_list.append(
                {
                    "id": user.id,
                    "name": user.name,
                    "email": user.email,
                    "active": user.active,
                    "role": role.get_role_display(),
                }
            )

        return users_list


async def select_user(id_user: int, db: AsyncSession) -> users_schemas.usersGetData | None:
    async with db as session:
        querie = select(users_models).filter(
            users_models.id == id_user,
            users_models.active.is_(True),
        )
        resultset = await session.execute(querie)
        user = resultset.scalars().unique().one_or_none()

        if user:
            role = select(roles_models).filter(
                roles_models.id == user.role_id,
                roles_models.active.is_(True),
            )
            role = await session.execute(role)
            role = role.scalars().unique().one_or_none()
            return users_schemas.usersGetData(
                id=user.id,
                name=user.name,
                email=user.email,
                active=user.active,
                role=role.get_role_display(),
            )
        return None


async def update_user(id_user: int, user: dict, db: AsyncSession) -> bool:
    async with db as session:
        querie = select(users_models).filter(
            users_models.id == id_user,
            users_models.active.is_(True),
        )
        resultset = await session.execute(querie)
        user_up: users_schemas.users | None = resultset.scalars().unique().one_or_none()

        if user_up:
            if user.get("name"):
                user_up.name = user["name"]
            if user.get("email"):
                user_up.email = user["email"]
            if user.get("active") is not None:
                user_up.active = user["active"]
            if user.get("role_id"):
                user_up.role_id = user["role_id"]
            await session.commit()
            await session.refresh(user_up)
            return True
        return False


async def drop_user(id_user: int, db: AsyncSession):
    async with db as session:
        querie = select(users_models).filter(
            users_models.id == int(id_user),
            users_models.active.is_(True),
        )
        result_set = await session.execute(querie)
        user_delete = result_set.scalars().unique().one_or_none()
        if user_delete:
            user_delete.active = False
            await session.commit()
            await session.refresh(user_delete)
            return user_delete
        return None


async def get_user_by_email(email: str, db: AsyncSession):
    async with db as session:
        querie = select(users_models).filter(
            users_models.email == email,
            users_models.active.is_(True),
        )
        resultset = await session.execute(querie)
        user_up: users_schemas.users = resultset.scalars().unique().one_or_none()
        if user_up:
            return user_up
        else:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)


async def generate_reset_token(email: str, db: AsyncSession):
    async with db as session:
        user: users_schemas.users = await get_user_by_email(email, db)
        token = secrets.token_urlsafe(16)
        user.reset_password_token = token
        user.reset_password_expires = datetime.datetime.now(
            datetime.timezone.utc
        ) + datetime.timedelta(hours=1)
        session.add(user)
        await session.commit()
        return token


async def send_email(email: str, token: str):
    # configuração do servidor de email
    sender_email = "gabrieldrumond211@gmail.com"
    receiver_email = email
    password = "jggp jojt pcop pjub"

    message = MIMEMultipart("alternative")
    message["Subject"] = "Recuperação de senha"
    message["From"] = f"Cod3Bit Dev Team <{sender_email}>"
    message["To"] = receiver_email

    reset_link = f"http://127.0.0.1:3000/resetpassword?email={email}&token={token}"
    text = f"""\
    Olá,
    Recebemos uma solicitação para redefinir sua senha. Clique no link abaixo para redefinir sua senha:
    {reset_link}
    Se você não solicitou a redefinição de senha, ignore este e-mail.
    Obrigado,
    Cod3Bit - Development Team
    """
    part = MIMEText(text, "plain")
    message.attach(part)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:  # Inicia a conexão TLS
            server.login(user=sender_email, password=password)
            server.sendmail(sender_email, receiver_email, message.as_string())
    except smtplib.SMTPException as e:
        print(f"Erro SMTP: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Erro ao enviar e-mail."
        )
    except OSError as e:
        print(f"Erro geral: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Erro ao enviar e-mail."
        )

    return True


async def get_user_by_reset_token(token: str, db: AsyncSession):
    async with db as session:
        query = select(users_models).filter(
            users_models.reset_password_token == token,
            users_models.active.is_(True),
        )
        resultset = await session.execute(query)
        user_up: users_schemas.users_update = resultset.scalars().unique().one_or_none()
        if user_up:
            return user_up
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Token inválido ou expirado"
            )
