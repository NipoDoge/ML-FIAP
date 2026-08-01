#!/usr/bin/env python3
"""Validação mínima de ambiente (Tech Challenge Fase 02)."""

from __future__ import annotations

import importlib
from pathlib import Path


def _check_import(name: str, pip_hint: str | None = None) -> None:
    try:
        importlib.import_module(name)
    except ImportError as exc:
        hint = pip_hint or name
        raise SystemExit(f"Dependência em falta: {hint} ({exc})") from exc


def _check_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    for rel in ("src/core/ml", "src/domains", "dvc.yaml", "params.yaml"):
        path = root / rel
        if not path.exists():
            raise SystemExit(f"Caminho esperado em falta: {path}")


def main() -> None:
    _check_import("torch")
    _check_import("sklearn")
    _check_import("mlflow")
    _check_import("pandas")
    _check_import("yaml", "pyyaml")
    _check_paths()
    print("Ambiente OK — core ML + TC02 presentes.")


if __name__ == "__main__":
    main()
