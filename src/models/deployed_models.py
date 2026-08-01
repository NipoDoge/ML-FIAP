from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from core.generic import modelsGeneric


class DeployedModels(modelsGeneric):
    """
    Registo de modelos promovidos a produção por domínio.
    No máximo um registo com status='active' por domain (garantido no serviço).
    """

    __tablename__ = "deployed_models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    domain = Column(String(100), nullable=False, index=True)
    pipeline_run_id = Column(Integer, ForeignKey("pipeline_runs.id"), nullable=False)
    status = Column(String(20), nullable=False, default="active")
    promoted_at = Column(DateTime, nullable=True)
    promoted_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    metrics_snapshot = Column(JSON, nullable=True)

    pipeline_run = relationship("PipelineRuns", back_populates="deployments", lazy="joined")
    promoted_by = relationship("Users", foreign_keys=[promoted_by_user_id], lazy="joined")
