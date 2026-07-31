"""
Agrega latência a partir de access.jsonl (e rotacionados access.jsonl.*).
Grava resumo CSV em PATH_MAINTENANCE_REPORTS.

Saída: duas seções no CSV
  - Uma linha por rota monitorada (predict, train, geral)
  - Coluna `slo_status`: ok | breach — baseado no threshold p95 configurável
  - Coluna `error_rate_pct`: % de respostas 4xx + 5xx por rota

SLO padrão: p95 de /predict < 300ms (ajustável via argumento --slo-ms).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.configs import settings  # noqa: E402

# Rotas que merecem linha própria no relatório
MONITORED_ROUTES = {
    "predict": "/predict",
    "train_baseline": "/train/baseline",
    "train_fe": "/train/feature-engineering",
    "promote": "/promote",
}


def _load_jsonl_files(log_dir: Path) -> list[dict]:
    rows: list[dict] = []
    if not log_dir.is_dir():
        return rows
    for name in sorted(log_dir.glob("access.jsonl*")):
        if not name.is_file():
            continue
        with open(name, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _normalize_latency_rows(raw: list[dict]) -> pd.DataFrame:
    """Mapeia campos do middleware atual para colunas estáveis de análise."""
    out = []
    for r in raw:
        if "duration_ms" not in r:
            continue
        out.append(
            {
                "timestamp": r.get("ts") or r.get("timestamp"),
                "latency_ms": float(r["duration_ms"]),
                "status_code": r.get("status")
                if r.get("status") is not None
                else r.get("status_code"),
                "request_id": r.get("request_id"),
                "path": r.get("path", ""),
            }
        )
    return pd.DataFrame(out)


def _route_label(path: str) -> str:
    """Retorna o label da rota monitorada ou 'other'."""
    for label, fragment in MONITORED_ROUTES.items():
        if fragment in path:
            return label
    return "other"


def _build_summary(df: pd.DataFrame, label: str, slo_p95_ms: float) -> dict:
    """Gera uma linha de resumo para um subconjunto do DataFrame."""
    if df.empty:
        return {}
    q = df["latency_ms"].quantile([0.5, 0.9, 0.95, 0.99])
    p95 = float(q.loc[0.95])
    error_mask = df["status_code"].apply(lambda s: isinstance(s, (int, float)) and s >= 400)
    error_rate = round(error_mask.sum() / len(df) * 100, 2)
    return {
        "route": label,
        "count": len(df),
        "mean_ms": round(float(df["latency_ms"].mean()), 2),
        "p50_ms": round(float(q.loc[0.5]), 2),
        "p90_ms": round(float(q.loc[0.9]), 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(float(q.loc[0.99]), 2),
        "error_rate_pct": error_rate,
        "slo_threshold_ms": slo_p95_ms,
        "slo_status": "ok" if p95 <= slo_p95_ms else "breach",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Relatório de latência da API")
    parser.add_argument(
        "--slo-ms",
        type=float,
        default=300.0,
        help="Threshold p95 em ms para SLO da rota /predict (padrão: 300ms)",
    )
    args = parser.parse_args()

    log_dir = Path(settings.path_api_request_logs)
    raw = _load_jsonl_files(log_dir)
    if not raw:
        print(f"Nenhuma linha válida em {log_dir} (access.jsonl*)", file=sys.stderr)
        sys.exit(1)

    df = _normalize_latency_rows(raw)
    if df.empty:
        print("Nenhuma coluna duration_ms encontrada.", file=sys.stderr)
        sys.exit(1)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["route_label"] = df["path"].apply(_route_label)

    rows = []

    # Linha por rota monitorada
    for label in list(MONITORED_ROUTES.keys()) + ["other"]:
        subset = df[df["route_label"] == label]
        summary = _build_summary(subset, label, slo_p95_ms=args.slo_ms)
        if summary:
            rows.append(summary)

    # Linha consolidada (todas as rotas)
    overall = _build_summary(df, "ALL", slo_p95_ms=args.slo_ms)
    if overall:
        rows.append(overall)

    summary_df = pd.DataFrame(rows)

    out_dir = Path(settings.path_maintenance_reports)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"latency_summary_{stamp}.csv"
    summary_df.to_csv(out_csv, index=False)

    print(f"Resumo gravado: {out_csv}\n")
    print(summary_df.to_string(index=False))

    breaches = summary_df[summary_df["slo_status"] == "breach"]
    if not breaches.empty:
        print("\n⚠ SLO breach detectado nas rotas:", breaches["route"].tolist())
    else:
        print("\n✓ Todas as rotas dentro do SLO.")


if __name__ == "__main__":
    main()
