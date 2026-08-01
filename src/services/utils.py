import logging
import os
from datetime import datetime
from typing import Any

import pytz

logger = logging.getLogger(__name__)


def filename_with_suffix(filename: str, suffix: str | None) -> str:
    """
    Insere ``suffix`` imediatamente antes da extensão.

    Ex.: ``manifest.json`` + ``_automatic`` → ``manifest_automatic.json``.
    Sem extensão (nada após o último ``.`` no nome base), concatena no fim: ``foo`` → ``foo_automatic``.
    """
    suf = (suffix or "").strip()
    if not suf:
        return filename
    stem, ext = os.path.splitext(filename)
    if not ext:
        return f"{filename}{suf}"
    return f"{stem}{suf}{ext}"


def to_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None

    if dt.tzinfo is None:
        sp = pytz.timezone("America/Sao_Paulo")
        dt = sp.localize(dt)
    dt_utc = dt.astimezone(pytz.UTC).replace(tzinfo=None)
    return dt_utc


def utcnow() -> datetime:
    return datetime.now(pytz.UTC).replace(tzinfo=None)


_PARAM_STR_MAX = 500


def log_training_csv_to_active_run(
    csv_path: str | None,
    *,
    df: Any | None = None,
    dataset_name: str = "training_data",
    context: str = "training",
) -> None:
    """
    Dentro de um ``mlflow.start_run`` já aberto: SHA-256 do CSV no disco, parâmetros
    ``training_data_*`` e ``mlflow.log_input`` (falhas em log_input não abortam).
    """
    import hashlib
    import os
    from pathlib import Path

    import mlflow
    import mlflow.data
    import pandas as pd

    def _truncate(value: str, max_len: int = _PARAM_STR_MAX) -> str:
        return value if len(value) <= max_len else value[: max_len - 3] + "..."

    if not csv_path:
        logger.warning("Linhagem MLflow: csv_path vazio; skip.")
        return
    resolved = os.path.abspath(csv_path)
    if not os.path.isfile(resolved):
        logger.warning("Linhagem MLflow: ficheiro inexistente %s; skip.", resolved)
        return

    try:
        h = hashlib.sha256()
        with open(resolved, "rb") as bf:
            for chunk in iter(lambda: bf.read(65536), b""):
                h.update(chunk)
        sha = h.hexdigest()
    except OSError as exc:
        logger.warning("Linhagem MLflow: leitura de %s falhou: %s", resolved, exc)
        return

    mlflow.log_param("training_data_digest_algorithm", "sha256")
    mlflow.log_param("training_data_sha256", sha)
    mlflow.log_param("training_data_size_bytes", os.path.getsize(resolved))
    mlflow.log_param("training_data_basename", os.path.basename(resolved))
    mlflow.log_param("training_data_path", _truncate(resolved))

    frame = df
    if frame is None:
        try:
            frame = pd.read_csv(resolved)
        except (OSError, pd.errors.ParserError, ValueError) as exc:
            logger.warning("Linhagem MLflow: pd.read_csv falhou para log_input: %s", exc)
            frame = None

    if frame is not None:
        mlflow.log_param("training_data_n_rows", frame.shape[0])
        mlflow.log_param("training_data_n_columns", frame.shape[1])

    try:
        if frame is None:
            return
        dataset = mlflow.data.from_pandas(
            frame,
            source=Path(resolved).resolve().as_uri(),
            name=dataset_name,
            digest=sha,
        )
        mlflow.log_input(dataset, context=context)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "mlflow.log_input falhou (parâmetros de linhagem já registados): %s",
            exc,
        )
