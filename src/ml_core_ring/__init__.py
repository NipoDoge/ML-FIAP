"""Contratos compartilhados de ML (inferência, manifest, domínios, treino)."""

from ml_core_ring import engines
from ml_core_ring.artifact_manifest import ArtifactManifest
from ml_core_ring.domain_plugin import DOMAIN_REGISTRY, DomainPlugin, get_domain, register_domain
from ml_core_ring.inference_engine import (
    ENGINE_REGISTRY,
    InferenceEngine,
    get_engine,
    register_engine,
)
from ml_core_ring.orchestration_hooks import run_training_for_domain
from ml_core_ring.pipeline_runner import (
    RUNNER_REGISTRY,
    PipelineRunner,
    get_runner,
    register_runner,
)
from ml_core_ring.prediction_result import PredictionResult
from ml_core_ring.train_backend import (
    TRAIN_BACKEND_REGISTRY,
    TrainBackend,
    TrainBackendResult,
    TrainRequestContext,
    get_train_backend,
    register_train_backend,
)

__all__ = [
    "DOMAIN_REGISTRY",
    "ENGINE_REGISTRY",
    "RUNNER_REGISTRY",
    "TRAIN_BACKEND_REGISTRY",
    "ArtifactManifest",
    "DomainPlugin",
    "InferenceEngine",
    "PipelineRunner",
    "PredictionResult",
    "TrainBackend",
    "TrainBackendResult",
    "TrainRequestContext",
    "engines",
    "get_domain",
    "get_engine",
    "get_runner",
    "get_train_backend",
    "register_domain",
    "register_engine",
    "register_runner",
    "register_train_backend",
    "run_training_for_domain",
]
