#!/usr/bin/env python3
"""
Verifica dependências entre anéis (_ring) — Fase 0.

Uso:
  python3 scripts/check_ring_imports.py
  python3 scripts/check_ring_imports.py --strict   # falha em avisos legados

Regras: docs/DOCUMENTACAO.md (secção Arquitectura)
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

# anel → conjunto de anéis que NÃO pode importar
FORBIDDEN: dict[str, frozenset[str]] = {
    "platform_ring": frozenset({"executors_ring", "orchestration_ring"}),
    "ml_core_ring": frozenset({"platform_ring", "executors_ring", "orchestration_ring"}),
    "domains_ring": frozenset({"platform_ring", "executors_ring", "orchestration_ring"}),
    "executors_ring": frozenset({"platform_ring", "orchestration_ring"}),
    "orchestration_ring": frozenset({"platform_ring"}),
}

# Excepções temporárias (path relativo a src/) — remover na Fase 4 (platform_ring)
TRANSITION_ALLOWLIST: frozenset[tuple[str, str]] = frozenset(
    {
        ("orchestration_ring/persist_run.py", "services.processor.airflow_persistence"),
        ("orchestration_ring/tabular_training.py", "services.processor.artifact_bundle"),
        ("orchestration_ring/tabular_training.py", "services.processor.fe_bundle_export"),
    }
)

# código legado → anel pretendido (avisos até migrar)
LEGACY_MAP: dict[str, str] = {
    "api": "platform_ring",
    "services/processor": "platform_ring",
    "services/auth": "platform_ring",
    "services/user": "platform_ring",
    "services/roles": "platform_ring",
    "schemas": "platform_ring",
    "services/pipelines": "executors_ring",
    "core/ml": "ml_core_ring",
    "domains": "domains_ring",
    "core/configs.py": "infra_ring",
    "core/database.py": "infra_ring",
    "core/deps.py": "platform_ring",
}


def _ring_from_path(path: Path) -> str | None:
    rel = path.relative_to(SRC)
    parts = rel.parts
    if not parts:
        return None
    first = parts[0]
    if first.endswith("_ring"):
        return first
    return None


def _legacy_ring(path: Path) -> str | None:
    rel = path.relative_to(SRC).as_posix()
    for prefix, ring in sorted(LEGACY_MAP.items(), key=lambda x: -len(x[0])):
        if rel.startswith(prefix):
            return ring
    return None


def _module_imports(tree: ast.AST) -> list[str]:
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.append(node.module)
    return out


def _imported_ring(module: str) -> str | None:
    if module.startswith("executors_ring") or ".executors_ring" in module:
        return "executors_ring"
    if module.startswith("orchestration_ring") or ".orchestration_ring" in module:
        return "orchestration_ring"
    if module.startswith("platform_ring") or ".platform_ring" in module:
        return "platform_ring"
    if module.startswith("ml_core_ring") or module.startswith("core.ml"):
        return "ml_core_ring"
    if module.startswith("domains_ring") or module.startswith("domains."):
        return "domains_ring"
    if module.startswith("services.pipelines"):
        return "executors_ring"
    if module.startswith("services.processor") or module.startswith("api."):
        return "platform_ring"
    return None


def _check_file(path: Path, *, strict_legacy: bool) -> list[str]:
    errors: list[str] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return [f"{path}: syntax error: {exc}"]

    ring = _ring_from_path(path)
    legacy = _legacy_ring(path) if ring is None else None
    owner = ring or legacy
    if owner is None:
        return []

    forbidden = FORBIDDEN.get(owner, frozenset())
    rel = path.relative_to(REPO_ROOT)

    for mod in _module_imports(tree):
        target = _imported_ring(mod)
        if target and target in forbidden:
            rel_src = path.relative_to(SRC).as_posix()
            if (rel_src, mod) in TRANSITION_ALLOWLIST:
                continue
            level = "ERROR" if ring else ("ERROR" if strict_legacy else "WARN")
            errors.append(f"{level} {rel}: {owner} importa {mod!r} ({target} proibido)")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verifica imports entre anéis _ring.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Trata avisos em código legado como erro.",
    )
    args = parser.parse_args()

    docs = [REPO_ROOT / "docs/DOCUMENTACAO.md"]
    missing_docs = [str(p.relative_to(REPO_ROOT)) for p in docs if not p.is_file()]
    if missing_docs:
        print("Documentação oficial em falta:", ", ".join(missing_docs))
        return 1

    py_files = [
        p for p in SRC.rglob("*.py") if "artifacts" not in p.parts and "__pycache__" not in p.parts
    ]

    errors: list[str] = []
    warns: list[str] = []
    for path in py_files:
        for msg in _check_file(path, strict_legacy=args.strict):
            if msg.startswith("ERROR"):
                errors.append(msg)
            else:
                warns.append(msg)

    if warns:
        print(f"--- Avisos legado ({len(warns)}) — migrar nas fases 1–4 ---")
        for w in warns[:30]:
            print(w)
        if len(warns) > 30:
            print(f"... +{len(warns) - 30} avisos")

    if errors:
        print(f"--- Erros ({len(errors)}) ---")
        for e in errors:
            print(e)
        return 1

    print("check_ring_imports: documentação OK.")
    if warns and not args.strict:
        print(f"{len(warns)} aviso(s) em código legado (esperado até migrar).")
    else:
        print("Nenhuma violação de import.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
