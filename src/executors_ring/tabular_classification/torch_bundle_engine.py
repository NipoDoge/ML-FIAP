"""Adapter bundle PyTorch (MLP tabular) — registado via ``executors_ring``."""

from __future__ import annotations

import pandas as pd

from executors_ring.tabular_classification.mlp_inference import load_mlp_bundle, predict_with_mlp
from ml_core_ring.artifact_manifest import ArtifactManifest
from ml_core_ring.inference_engine import register_engine
from ml_core_ring.paths import resolve_shared_artifact_path
from ml_core_ring.prediction_result import PredictionResult


@register_engine
class TorchBundleEngine:
    backend_id = "torch_bundle"

    def __init__(self) -> None:
        self._bundle = None

    def load(self, manifest: ArtifactManifest) -> None:
        prefix = manifest.artifacts.get("prefix")
        if not prefix:
            raise ValueError("Manifest torch_bundle sem artifacts['prefix'].")
        resolved = resolve_shared_artifact_path(prefix)
        if not resolved:
            raise ValueError("Prefix MLP inválido no manifest.")
        self._bundle = load_mlp_bundle(resolved)

    def predict(self, df_input: pd.DataFrame) -> PredictionResult:
        if self._bundle is None:
            raise RuntimeError("Engine torch_bundle não carregado.")
        label, prob = predict_with_mlp(self._bundle, df_input)
        return PredictionResult(
            problem_type="binary_classification",
            label=int(label),
            probability=float(prob),
        )
