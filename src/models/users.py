from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from core.generic import modelsGeneric


class Users(modelsGeneric):
    __tablename__ = "users"
    id = Column(Integer, autoincrement=True, primary_key=True)
    password = Column(String(255), nullable=False)
    name = Column(String(50), nullable=False)
    email = Column(String(50), nullable=False)
    role_id = Column(Integer, ForeignKey("roles.id"), default=1)

    # relação da FK pra apontar o relacionamento de role para usuario
    role = relationship("Roles", lazy="joined", back_populates="user")
