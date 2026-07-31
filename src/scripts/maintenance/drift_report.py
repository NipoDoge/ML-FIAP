"""
Drift de dados: compara CSV de treino (referência) com produção (predições).

Produção: CSV exportado da tabela `predictions` com coluna `input_data` (JSON)
por linha, ou CSV já com colunas de features alinhadas ao treino.
Grava PSI por feature e resumo agregado em PATH_MAINTENANCE_REPORTS.

Interpretação do PSI
--------------------
  < 0.10  → ok       — distribuição estável, nenhuma ação necessária.
  0.10–0.25 → warning — mudança moderada, aumentar frequência de monitoramento.
  > 0.25  → critical  — drift significativo, avaliar retreino com dados recentes.

Referências de threshold: literatura padrão de monitoramento de modelos
(Siddiqi 2006; amplamente adotado em credit scoring e MLOps).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.configs import settings  # noqa: E402

PSI_WARNING = 0.10
PSI_CRITICAL = 0.25


def _psi_status(psi: float) -> str:
    if psi >= PSI_CRITICAL:
        return "critical"
    if psi >= PSI_WARNING:
        return "warning"
    return "ok"


def _calculate_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        # Feature com poucos valores únicos (ex.: binária) — PSI não é confiável
        raise ValueError("bins insuficientes")
    expected = np.histogram(reference, bins=breakpoints)[0] / len(reference)
    actual = np.histogram(current, bins=breakpoints)[0] / len(current)
    expected = np.where(expected == 0, 0.0001, expected)
    actual = np.where(actual == 0, 0.0001, actual)
    psi = np.sum((actual - expected) * np.log(actual / expected))
    return float(psi)


def _load_predictions_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "input_data" in df.columns:
        parsed = df["input_data"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        return pd.json_normalize(parsed)
    drop_cols = [
        c for c in ("id", "prediction", "probability", "pipeline_run_id") if c in df.columns
    ]
    return df.drop(columns=drop_cols, errors="ignore")


def main() -> None:
    parser = argparse.ArgumentParser(description="Drift treino vs predições (CSV)")
    parser.add_argument("--train-csv", required=True, type=Path, help="CSV de treino (referência)")
    parser.add_argument(
        "--predictions-csv", required=True, type=Path, help="Export de predictions ou features"
    )
    parser.add_argument(
        "--target-col", default="target", help="Coluna alvo no treino (removida na comparação)"
    )
    args = parser.parse_args()

    train = pd.read_csv(args.train_csv)
    if "dataset" in train.columns:
        train = train.drop(columns=["dataset"])
    if args.target_col in train.columns:
        train = train.drop(columns=[args.target_col])

    prod = _load_predictions_features(args.predictions_csv)

    numeric_cols = [
        c for c in train.columns if c in prod.columns and pd.api.types.is_numeric_dtype(train[c])
    ]
    if not numeric_cols:
        print("Nenhuma coluna numérica em comum entre treino e produção.", file=sys.stderr)
        sys.exit(1)

    rows = []
    skipped = []
    for col in numeric_cols:
        ref = train[col].dropna().astype(float).values
        cur = prod[col].dropna().astype(float).values
        if len(ref) < 2 or len(cur) < 2:
            skipped.append(col)
            continue
        try:
            psi = _calculate_psi(ref, cur)
        except ValueError:
            print(
                f"  [skip] '{col}' — poucos valores únicos, PSI instável (feature provavelmente binária).",
                file=sys.stderr,
            )
            skipped.append(col)
            continue
        rows.append({"feature": col, "psi": round(psi, 6), "status": _psi_status(psi)})

    if not rows:
        print("Nenhuma feature calculável após filtros.", file=sys.stderr)
        sys.exit(1)

    psi_df = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)

    # Linha de resumo agregado no topo
    mean_psi = psi_df["psi"].mean()
    summary_row = pd.DataFrame(
        [
            {
                "feature": "** RESUMO **",
                "psi": round(mean_psi, 6),
                "status": _psi_status(mean_psi),
            }
        ]
    )
    counts = psi_df["status"].value_counts().to_dict()
    n_ok = counts.get("ok", 0)
    n_warning = counts.get("warning", 0)
    n_critical = counts.get("critical", 0)

    out_dir = Path(settings.path_maintenance_reports)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"drift_psi_{stamp}.csv"

    pd.concat([summary_row, psi_df], ignore_index=True).to_csv(out_csv, index=False)

    print(f"PSI gravado: {out_csv}\n")
    print(f"Features analisadas : {len(rows)}  |  ignoradas: {len(skipped)}")
    print(f"ok: {n_ok}  |  warning: {n_warning}  |  critical: {n_critical}")
    print(f"PSI médio: {mean_psi:.4f}  →  {_psi_status(mean_psi).upper()}\n")
    print(psi_df.to_string(index=False))

    if n_critical > 0:
        critical_feats = psi_df[psi_df["status"] == "critical"]["feature"].tolist()
        print(f"\n⚠ CRITICAL — {n_critical} feature(s) com drift significativo: {critical_feats}")
        print(
            "  Decisão sugerida: avaliar retreino com dados recentes e reavaliar drift após novo deploy."
        )
    elif n_warning > 0:
        print(
            f"\n⚡ WARNING — {n_warning} feature(s) com mudança moderada. Aumentar frequência de monitoramento."
        )
    else:
        print("\n✓ Todas as features dentro do limite aceitável.")


if __name__ == "__main__":
    main()
