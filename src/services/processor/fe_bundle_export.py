"""Montagem do bundle FE (manifest + ZIP) alinhada com ``run_feature_engineering`` (API).

Partilhado entre a rota manual (temp dir) e o DAG Airflow (ZIP único no MLflow).
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any

from core.configs import settings
from services.processor.artifact_bundle import (
    build_manifest,
    copy_if_exists,
    write_manifest,
    zip_tree,
)

logger = logging.getLogger(__name__)


def format_best_model_name_txt(sklearn_champion: str | None, *, use_mlp_for_inference: bool) -> str:
    """Conteúdo de ``best_model_name.txt`` no ZIP do FE: campeão sklearn vs modelo em ``/predict``."""
    sk = (sklearn_champion or "").strip() or "-"
    lines = [f"Campeão no comparativo SKLEARN: {sk}"]
    if use_mlp_for_inference:
        lines.append("Modelo usado na inferência: PyTorch MLP")
    else:
        lines.append(f"Modelo usado na inferência: {sk}")
    return "\n".join(lines) + "\n"


def prepare_fe_bundle_baseline_tree(
    run_root: str,
    *,
    resolved_manifest_path: str,
    baseline_manifest: dict[str, Any],
) -> None:
    """Preenche ``00_input_baseline`` e ``10_baseline`` antes de executar ``FeatureEngineering``."""
    b_in = os.path.join(run_root, "00_input_baseline")
    b_out = os.path.join(run_root, "10_baseline")
    fe_d = os.path.join(run_root, "20_feature_engineering")
    os.makedirs(b_in, exist_ok=True)
    os.makedirs(fe_d, exist_ok=True)

    baseline_input_path = baseline_manifest.get("input_csv_snapshot") or baseline_manifest.get(
        "input_csv_source"
    )
    baseline_input_path = os.path.abspath(baseline_input_path) if baseline_input_path else None

    csv_baseline = baseline_manifest.get("output_sample_csv_stable")
    if not csv_baseline:
        raise ValueError("Manifest inválido: campo 'output_sample_csv_stable' ausente.")
    csv_baseline = os.path.abspath(csv_baseline)

    if baseline_input_path and os.path.isfile(baseline_input_path):
        shutil.copy2(baseline_input_path, os.path.join(b_in, os.path.basename(baseline_input_path)))
    copy_if_exists(csv_baseline, b_out)
    copy_if_exists(resolved_manifest_path, b_out)
    model_baseline = baseline_manifest.get("model_path")
    if model_baseline:
        copy_if_exists(os.path.abspath(model_baseline), b_out)
    graphs_dir = baseline_manifest.get("graphs_dir")
    if graphs_dir and os.path.isdir(os.path.abspath(graphs_dir)):
        for name in os.listdir(os.path.abspath(graphs_dir)):
            copy_if_exists(
                os.path.join(os.path.abspath(graphs_dir), name), os.path.join(b_out, "graphs")
            )


def finalize_fe_bundle_pipeline_outputs(run_root: str, pipeline: Any) -> None:
    """Após ``pipeline.run()`` / ``save()``: joblib, ``fe_export/``, ``best_model_name.txt``."""
    fe_d = os.path.join(run_root, "20_feature_engineering")
    os.makedirs(fe_d, exist_ok=True)
    fe_joblib = pipeline.fe_sklearn_joblib_path()
    copy_if_exists(fe_joblib, fe_d)
    fe_export_src = pipeline.fe_export_bundle_dir(fe_joblib)
    if os.path.isdir(fe_export_src):
        fe_export_dst = os.path.join(fe_d, "fe_export")
        shutil.copytree(fe_export_src, fe_export_dst, dirs_exist_ok=True)
    with open(os.path.join(fe_d, "best_model_name.txt"), "w", encoding="utf-8") as tf:
        tf.write(
            format_best_model_name_txt(
                pipeline.best_model_name,
                use_mlp_for_inference=settings.use_mlp_for_prediction,
            )
        )


def write_fe_manifest_zip_from_run_root(
    run_root: str,
    *,
    pipeline_run_id: int,
    objective: str,
    run_timestamp: str,
    csv_baseline: str,
    original_filename: str,
    merged_metrics: dict[str, Any],
    mlflow_fe_run_id: str | None,
    best_model_name: str | None,
    active_deployment_id: int | None,
    zip_path: str,
) -> None:
    """Escreve ``manifest.json`` na raiz de ``run_root`` e compacta para ``zip_path``."""
    all_files: list[str] = []
    for root, _dirs, files in os.walk(run_root):
        for name in files:
            all_files.append(os.path.join(root, name))

    bundle_manifest_path = os.path.join(run_root, "manifest.json")
    man = build_manifest(
        pipeline_run_id=pipeline_run_id,
        objective=objective,
        run_timestamp=run_timestamp,
        status="completed",
        input_filename=original_filename,
        input_path=csv_baseline,
        paths_in_bundle=all_files,
        metrics=merged_metrics,
        mlflow_baseline_run_id=None,
        mlflow_fe_run_id=mlflow_fe_run_id,
        best_model_name=best_model_name,
        original_filename=original_filename,
        active_deployment_id=active_deployment_id,
    )
    write_manifest(man, bundle_manifest_path)
    man = build_manifest(
        pipeline_run_id=pipeline_run_id,
        objective=objective,
        run_timestamp=run_timestamp,
        status="completed",
        input_filename=original_filename,
        input_path=csv_baseline,
        paths_in_bundle=all_files + [bundle_manifest_path],
        metrics=merged_metrics,
        mlflow_baseline_run_id=None,
        mlflow_fe_run_id=mlflow_fe_run_id,
        best_model_name=best_model_name,
        original_filename=original_filename,
        active_deployment_id=active_deployment_id,
    )
    write_manifest(man, bundle_manifest_path)
    zip_tree(run_root, zip_path)


def log_fe_bundle_zip_to_mlflow_run(
    *,
    mlflow_run_id: str | None,
    objective: str,
    zip_path: str,
    artifact_subdir: str = "fe_bundle",
) -> str | None:
    """Anexa o ZIP à mesma run FE do MLflow (subpasta ``artifact_subdir``)."""
    if not mlflow_run_id or not os.path.isfile(zip_path):
        return None
    try:
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        experiment_name = f"{objective}_feature_engineering"
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            logger.warning(
                "Experimento MLflow '%s' inexistente — ZIP não anexado.", experiment_name
            )
            return None
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_id=mlflow_run_id):
            mlflow.log_artifact(zip_path, artifact_path=artifact_subdir)
        return f"{artifact_subdir}/{os.path.basename(zip_path)}"
    except Exception as e:
        logger.warning("Falha ao registar ZIP FE no MLflow (run_id=%s): %s", mlflow_run_id, e)
        return None
