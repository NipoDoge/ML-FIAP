"""
Configuração central de logging para o processo da API.

Garante que TODOS os loggers (api.request, ml.pipeline, uvicorn, etc.)
emitem para stdout — necessário para `docker logs` e Dozzle funcionarem.

Regras:
- Logger raiz: StreamHandler para stdout com formato legível.
- Logger `api.request`: propagate=True (já definido em logging_api_request.py),
  então as mensagens chegam ao root e aparecem no console.
- Logger `ml.pipeline`: idem — propagate=True garante stdout + arquivo por run.
- Uvicorn já escreve em stdout nativamente; não reconfigurar seus loggers.
"""

from __future__ import annotations

import logging
import sys

from core.configs import settings

_ROOT_CONFIGURED = False


def setup_root_logging() -> None:
    """Configura o logger raiz com StreamHandler para stdout (idempotente)."""
    global _ROOT_CONFIGURED
    if _ROOT_CONFIGURED:
        return

    root = logging.getLogger()

    already_has_stream = any(
        isinstance(h, logging.StreamHandler)
        and getattr(h, "stream", None) in (sys.stdout, sys.stderr)
        for h in root.handlers
    )

    if not already_has_stream:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        handler.setLevel(settings.get_log_level())
        root.addHandler(handler)

    root.setLevel(settings.get_log_level())
    _ROOT_CONFIGURED = True
