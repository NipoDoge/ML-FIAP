from datetime import datetime, timezone

from pydantic import BaseModel as SC_BaseModel
from pydantic import EmailStr, Field


class users(SC_BaseModel):
    id: int | None = None
    name: str
    email: EmailStr
    created_at: datetime | None = Field(default_factory=lambda: datetime.now(timezone.utc))
    active: bool | None = True
    role_id: int | None = 1

    class Config:
        from_attributes = True


class users_update(users):
    name: str | None = None
    email: EmailStr | None = None
    password: str | None = None
    active: bool | None = True
    role_id: int | None = None


class users_create(users):
    password: str


class usersGetData(SC_BaseModel):
    id: int | None = None
    name: str | None = None
    email: EmailStr | None = None
    active: bool | None = True
    role: str | None = None
