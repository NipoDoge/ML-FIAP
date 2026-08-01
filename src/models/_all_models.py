"""
Importa todos os modelos ORM no mesmo registry do declarative base.

Usado por ``airflow_persistence`` antes de instanciar ``PipelineRuns``: as
``relationship("Users", ...)`` etc. só resolvem depois das classes estarem
carregadas — sem isto, o worker Airflow levanta InvalidRequestError ao persistir.
"""

from models.deployed_models import DeployedModels  # noqa: F401
from models.pipeline_runs import PipelineRuns  # noqa: F401
from models.predictions import Predictions  # noqa: F401
from models.roles import Roles  # noqa: F401
from models.users import Users  # noqa: F401
