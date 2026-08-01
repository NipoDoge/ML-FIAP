"""Merge de conf Airflow (Variable + dag_run.conf)."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def merge_run_conf(
    context: dict, *, variable_key: str, fallback_keys: tuple[tuple[str, str], ...] = ()
) -> dict[str, Any]:
    """
    Defaults de Airflow Variable + overrides em ``dag_run.conf``.

    ``fallback_keys`` permite preencher lacunas com Variables string (ex. objective/csv_path).
    """
    from airflow.models import Variable

    defaults: dict[str, Any] = {}
    try:
        raw = Variable.get(variable_key, default_var=None)
        if raw:
            defaults = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Airflow Variable {variable_key!r} deve ser JSON válido: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read %s: %s", variable_key, exc)

    for key, var_key in fallback_keys:
        if defaults.get(key):
            continue
        try:
            value = Variable.get(var_key, default_var=None)
            if value:
                defaults[key] = value
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not read fallback Airflow Variable %s: %s", var_key, exc)

    conf_run = context["dag_run"].conf or {}
    merged = {**defaults, **conf_run}
    logger.info(
        "Conf efetiva (%s): chaves=%s (Variable + conf do run; conf tem prioridade)",
        variable_key,
        list(merged.keys()),
    )
    return merged
