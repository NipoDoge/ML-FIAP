
from pydantic import BaseModel as SC_basemodel


class role(SC_basemodel):
    id: int | None = None
    description: str
    active: bool

    class Config:
        from_attributes = True


class role_update(role):
    description: str | None = None
    active: bool | None = None
