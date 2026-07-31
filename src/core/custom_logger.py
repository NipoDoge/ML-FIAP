"""
Logging de pipelines ML.

- `setup_log`: uso em scripts `if __name__ == "__main__"` — reconfigura o logger **raiz**
  (console + arquivo). Evite chamar dentro do processo da API FastAPI.

- `setup_pipeline_run_logging`: uso no processo da API — anexa arquivo só ao logger
  `ml.pipeline`, sem limpar handlers do root (compatível com Uvicorn).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from core.configs import settings


class _PipelineContextFormatter(logging.Formatter):
    """Prefixa cada linha com run_id, objective e pipeline_type."""

    def __init__(self, run_id: Any, objective: str | None, pipeline_type: str | None) -> None:
        super().__init__(
            "%(asctime)s | %(levelname)s | run_id=%(run_id)s | objective=%(objective)s | "
            "pipeline_type=%(pipeline_type)s | %(message)s",
        )
        self._run_id = str(run_id) if run_id is not None else "-"
        self._objective = objective if objective is not None else "-"
        self._pipeline_type = pipeline_type if pipeline_type is not None else "-"

    def format(self, record: logging.LogRecord) -> str:
        record.run_id = getattr(record, "run_id", self._run_id)
        record.objective = getattr(record, "objective", self._objective)
        record.pipeline_type = getattr(record, "pipeline_type", self._pipeline_type)
        return super().format(record)


def setup_log(snapshot_path: str, now: str) -> logging.Logger:
    """
    Configura o logging para escrever no console e em um arquivo txt.
    Reconfigura o logger **raiz** (limpa handlers). Uso típico: execução standalone do pipeline.
    """
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    log_filename = os.path.join(snapshot_path, f"pipeline_{now}.txt")

    log_format = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    root = logging.getLogger()
    root.setLevel(settings.get_log_level())

    if root.handlers:
        root.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    root.addHandler(console_handler)

    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setFormatter(log_format)
    root.addHandler(file_handler)

    logging.info("Log persistente configurado em: %s", log_filename)
    return root


def setup_pipeline_run_logging(
    snapshot_path: str,
    now: str,
    *,
    run_id: int | None,
    objective: str | None,
    pipeline_type: str | None,
) -> str:
    """
    Anexa um FileHandler ao logger `ml.pipeline` com formato contextual (run_id, objective, pipeline_type).
    Não altera o logger raiz — seguro para uso dentro do processo FastAPI.

    Retorna o caminho absoluto do arquivo de log criado.
    """
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    log_filename = os.path.abspath(os.path.join(snapshot_path, f"pipeline_{now}.txt"))

    formatter = _PipelineContextFormatter(run_id, objective, pipeline_type)
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(settings.get_log_level())

    lg = logging.getLogger("ml.pipeline")
    lg.setLevel(settings.get_log_level())
    for h in list(lg.handlers):
        if isinstance(h, logging.FileHandler):
            try:
                h.close()
            except OSError:
                pass
            lg.removeHandler(h)
    lg.addHandler(file_handler)
    lg.propagate = True

    lg.info(
        "Pipeline run log file attached | path=%s",
        log_filename,
        extra={
            "run_id": str(run_id) if run_id is not None else "-",
            "objective": objective or "-",
            "pipeline_type": pipeline_type or "-",
        },
    )
    return log_filename
