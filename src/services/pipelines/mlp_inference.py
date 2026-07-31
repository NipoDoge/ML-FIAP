"""
Inferência da MLP PyTorch a partir do bundle gravado em ``_run_mlp_torch_mvp``.

Bundle (3 ficheiros, mesmo *prefixo*):
  - ``<prefix>.pt``                   → ``state_dict`` da rede (``_MLPBinary``).
  - ``<prefix>_preprocess.joblib``    → ``ColumnTransformer`` ajustado no FE.
  - ``<prefix>_meta.json``            → ``hidden_dims``, ``dropout``, ``n_features_in``,
                                         ``decision_threshold``, ``feature_columns_in_order``.

A API usa estas funções no ramo ``inference_backend == "mlp"`` do ``predict_for_domain``.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MlpBundle:
    """Bundle pronto-a-servir para uma run FE com inference_backend='mlp'."""

    model: Any  # _MLPBinary (torch.nn.Module)
    preprocess: Any  # sklearn ColumnTransformer ajustado
    feature_columns_in_order: list[str]
    decision_threshold: float
    n_features_in: int
    meta: dict


def _bundle_paths(prefix: str) -> tuple[str, str, str]:
    return f"{prefix}.pt", f"{prefix}_preprocess.joblib", f"{prefix}_meta.json"


def load_mlp_bundle(prefix: str) -> MlpBundle:
    """Carrega o bundle MLP a partir do prefixo gravado em ``mlp_artifact_prefix``."""
    import torch

    from services.pipelines.mlp_torch_tabular import _MLPBinary

    ckpt_path, pre_path, meta_path = _bundle_paths(prefix)
    for p in (ckpt_path, pre_path, meta_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Bundle MLP incompleto. Falta: {p}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    hidden_dims = tuple(int(d) for d in meta["hidden_dims"])
    dropout = float(meta.get("dropout", 0.0))
    n_features_in = int(meta["n_features_in"])

    model = _MLPBinary(n_features=n_features_in, hidden_dims=hidden_dims, dropout=dropout)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    preprocess = joblib.load(pre_path)

    cols = [str(c) for c in meta.get("feature_columns_in_order", [])]
    if not cols:
        raise ValueError(f"meta.json sem 'feature_columns_in_order': {meta_path}")

    threshold = float(meta.get("decision_threshold", 0.5))
    return MlpBundle(
        model=model,
        preprocess=preprocess,
        feature_columns_in_order=cols,
        decision_threshold=threshold,
        n_features_in=n_features_in,
        meta=meta,
    )


def _align_for_mlp(df: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    """Garante o subconjunto e a ordem de colunas que o ``ColumnTransformer`` espera."""
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Colunas em falta para predição MLP (alinhar ao treino): " + ", ".join(missing)
        )
    return df[expected_cols]


def predict_with_mlp(
    bundle: MlpBundle, df_input: pd.DataFrame, threshold: float | None = None
) -> tuple[int, float]:
    """
    Aplica o mesmo pré-processamento do treino e devolve ``(label, probability)``.

    ``threshold`` é a probabilidade mínima para classe positiva (1).
    Se omitido, usa ``bundle.decision_threshold`` (que veio do treino).
    """
    import torch

    if df_input.empty:
        raise ValueError("df_input vazio para predict_with_mlp.")

    thr = float(bundle.decision_threshold if threshold is None else threshold)
    if not 0.0 <= thr <= 1.0:
        raise ValueError(f"threshold fora de [0,1]: {thr}")

    df_aligned = _align_for_mlp(df_input, bundle.feature_columns_in_order)
    X = bundle.preprocess.transform(df_aligned)
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    with torch.no_grad():
        logits = bundle.model(torch.from_numpy(X)).cpu().numpy()
    proba = 1.0 / (1.0 + np.exp(-logits))
    proba_pos = float(proba.ravel()[0])
    label = 1 if proba_pos >= thr else 0
    return label, proba_pos
