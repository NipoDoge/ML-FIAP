#!/usr/bin/env python3
"""Promove modelo no MLflow Model Registry (CLI — delega a platform_ring Fase 6)."""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from platform_ring.mlflow_registry import sync_mlflow_registry_on_promote


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promove modelo no MLflow Registry (side-effect; BD deployed_models inalterada)."
    )
    parser.add_argument(
        "--domain",
        default="recommendation",
        help="Domínio (default: recommendation).",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Override do nome Registry (default: mapa por domínio).",
    )
    parser.add_argument(
        "--mlflow-run-id",
        default=os.getenv("MLFLOW_RUN_ID"),
        help="Run MLflow a promover (opcional; senão usa última versão).",
    )
    parser.add_argument(
        "--pipeline-type",
        default=None,
        help="pipeline_type (default: recommendation ou feature_engineering para churn).",
    )
    args = parser.parse_args()

    domain = args.domain.strip().lower()
    pipeline_type = args.pipeline_type or (
        "recommendation" if domain == "recommendation" else "feature_engineering"
    )
    metrics = {"mlflow_run_id": args.mlflow_run_id} if args.mlflow_run_id else {}

    if args.model_name:
        from platform_ring import mlflow_registry as reg_mod

        original = dict(reg_mod.REGISTRY_MODEL_BY_DOMAIN)
        reg_mod.REGISTRY_MODEL_BY_DOMAIN[domain] = args.model_name
        try:
            result = sync_mlflow_registry_on_promote(
                domain=domain,
                metrics=metrics,
                pipeline_type=pipeline_type,
            )
        finally:
            reg_mod.REGISTRY_MODEL_BY_DOMAIN.clear()
            reg_mod.REGISTRY_MODEL_BY_DOMAIN.update(original)
    else:
        result = sync_mlflow_registry_on_promote(
            domain=domain,
            metrics=metrics,
            pipeline_type=pipeline_type,
        )

    if result is None:
        raise SystemExit("Promote Registry falhou (resultado vazio).")
    if result.skipped:
        raise SystemExit(result.skip_reason or "Domínio ignorado.")
    if result.warning and not result.version:
        raise SystemExit(result.warning)
    print(
        f"Registry: {result.model_name} v{result.version} → {result.stage}"
        + (f" (run_id={result.mlflow_run_id})" if result.mlflow_run_id else "")
    )
    if result.warning:
        print(f"Aviso: {result.warning}", file=sys.stderr)


if __name__ == "__main__":
    main()
