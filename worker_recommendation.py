"""Entrypoint do worker de recomendação (uvicorn)."""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from executors_ring.recommendation.worker_app import app  # noqa: F401
