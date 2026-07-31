"""Adapter sklearn joblib para o registry de inferência."""

from __future__ import annotations

import os

import joblib
import pandas as pd

from ml_core_ring.artifact_manifest import ArtifactManifest
from ml_core_ring.inference_engine import register_engine
from ml_core_ring.paths import resolve_shared_artifact_path
from ml_core_ring.prediction_result import PredictionResult


def _sklearn_feature_names(model) -> list[str] | None:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return None


def _align_dataframe_to_model(model, df: pd.DataFrame) -> pd.DataFrame:
    names = _sklearn_feature_names(model)
    if not names:
        return df
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise ValueError(
            "Colunas em falta para predição (alinhar ao treino): " + ", ".join(missing)
        )
    return df[names]


@register_engine
class SklearnJoblibEngine:
    backend_id = "sklearn_joblib"

    def __init__(self) -> None:
        self._model = None

    def load(self, manifest: ArtifactManifest) -> None:
        raw_path = manifest.artifacts.get("joblib_path")
        if not raw_path:
            raise ValueError("Manifest sklearn_joblib sem artifacts['joblib_path'].")
        local_path = resolve_shared_artifact_path(raw_path)
        if not local_path or not os.path.exists(local_path):
            raise ValueError(f"Modelo não encontrado em: {raw_path}")
        self._model = joblib.load(local_path)

    def predict(self, df_input: pd.DataFrame) -> PredictionResult:
        if self._model is None:
            raise RuntimeError("Engine sklearn_joblib não carregado.")
        aligned = _align_dataframe_to_model(self._model, df_input)
        label = int(self._model.predict(aligned)[0])
        probability = None
        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(aligned)[0]
            probability = float(proba[1])
        return PredictionResult(
            problem_type="binary_classification",
            label=label,
            probability=probability,
        )
