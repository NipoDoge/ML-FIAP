from sqlalchemy import Boolean, Column, DateTime
from sqlalchemy.ext.declarative import as_declarative, declared_attr

from services.utils import utcnow


@as_declarative()
class modelsGeneric:
    created_at = Column(DateTime, default=utcnow, nullable=False)
    active = Column(Boolean, default=True)

    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()
