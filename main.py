"""Entrypoint da API na raiz do repo; adiciona ``src/`` ao ``sys.path``."""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Vários .joblib (pipelines sklearn do FE) deserializam com referência a ``dill``.
# Deve constar de docker/requirements-api.txt; sem isto, /predict falha com "No module named 'dill'".
import dill  # noqa: F401
from fastapi import FastAPI

import domains  # noqa: F401 — registra DomainPlugin
import executors_ring  # noqa: F401 — registra TrainBackend
from src.api.v1 import api
from src.core.configs import settings
from src.core.logging_api_request import setup_api_request_logging
from src.core.logging_setup import setup_root_logging
from src.core.middleware.request_record import request_record

setup_root_logging()
setup_api_request_logging()

app = FastAPI(title=settings.project_name, version=settings.project_version)
app.middleware("http")(request_record)

app.include_router(api.router, prefix=settings.project_version)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
