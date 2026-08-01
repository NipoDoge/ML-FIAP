"""Montagem de manifest MLOps e ficheiro ZIP de artefatos (runs FE com Baseline)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from typing import Any

import sklearn

from core.configs import settings

logger = logging.getLogger(__name__)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(
    *,
    pipeline_run_id: int,
    objective: str,
    run_timestamp: str,
    status: str,
    input_filename: str,
    input_path: str | None,
    paths_in_bundle: list[str],
    metrics: dict[str, Any] | None,
    mlflow_baseline_run_id: str | None,
    mlflow_fe_run_id: str | None,
    best_model_name: str | None,
    original_filename: str,
    active_deployment_id: int | None = None,
) -> dict[str, Any]:
    m: dict[str, Any] = {
        "schema_version": 1,
        "pipeline_run_id": pipeline_run_id,
        "objective": objective,
        "pipeline_type": "feature_engineering",
        "status": status,
        "run_timestamp": run_timestamp,
        "input_original_filename": original_filename,
        "best_model_name": best_model_name,
        "mlflow": {
            "tracking_uri": settings.mlflow_tracking_uri,
            "artifact_root": settings.mlflow_artifact_root,
            "baseline_run_id": mlflow_baseline_run_id,
            "fe_run_id": mlflow_fe_run_id,
        },
        "metrics": metrics or {},
        "environment": {
            "python": sys.version.split()[0],
            "sklearn": sklearn.__version__,
        },
        "active_deployment_id": active_deployment_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": [],
    }
    if input_path and os.path.isfile(input_path):
        m["input_sha256"] = _sha256_file(input_path)
        m["input_size_bytes"] = os.path.getsize(input_path)
    for p in paths_in_bundle:
        if p and os.path.isfile(p):
            rel = p
            m["files"].append(
                {
                    "path": rel,
                    "name": os.path.basename(p),
                    "size_bytes": os.path.getsize(p),
                    "sha256": _sha256_file(p),
                }
            )
    return m


def write_manifest(manifest: dict[str, Any], dest_json: str) -> None:
    os.makedirs(os.path.dirname(dest_json) or ".", exist_ok=True)
    with open(dest_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def zip_tree(source_dir: str, zip_path: str) -> None:
    """
    Cria ``zip_path`` com o conteúdo de ``source_dir`` (estrutura relativa
    a ``source_dir`` na raiz do zip).
    """
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(source_dir):
            for name in files:
                fp = os.path.join(root, name)
                arc = os.path.relpath(fp, start=source_dir)
                zf.write(fp, arcname=arc)


def copy_if_exists(src: str, dest_dir: str) -> str | None:
    if not src or not os.path.isfile(src):
        return None
    os.makedirs(dest_dir, exist_ok=True)
    base = os.path.basename(src)
    out = os.path.join(dest_dir, base)
    shutil.copy2(src, out)
    return out


def glob_copy_pattern(graph_dir: str, pattern: str, dest: str) -> list[str]:
    import glob

    out: list[str] = []
    os.makedirs(dest, exist_ok=True)
    for p in glob.glob(os.path.join(graph_dir, pattern)):
        if os.path.isfile(p):
            t = os.path.join(dest, os.path.basename(p))
            shutil.copy2(p, t)
            out.append(t)
    return out


def safe_unlink(path: str | None) -> None:
    if not path or not os.path.isfile(path):
        return
    try:
        os.remove(path)
    except OSError as e:
        logger.warning("Não foi possível apagar %s: %s", path, e)


def safe_rmtree(path: str | None) -> None:
    if not path or not os.path.isdir(path):
        return
    try:
        shutil.rmtree(path, ignore_errors=True)
    except OSError as e:
        logger.warning("Não foi possível apagar dir %s: %s", path, e)
