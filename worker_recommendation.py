"""Entrypoint do worker de recomendação (uvicorn)."""

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from executors_ring.recommendation.worker_app import app  # noqa: F401
