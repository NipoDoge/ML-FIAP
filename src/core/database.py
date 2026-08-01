from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from core.configs import settings

engine: AsyncEngine = create_async_engine(
    settings.database_url, echo=False, future=True, pool_pre_ping=True
)

Session: AsyncSession = sessionmaker(
    autocommit=False, autoflush=False, expire_on_commit=False, class_=AsyncSession, bind=engine
)

db_base_model = declarative_base()
