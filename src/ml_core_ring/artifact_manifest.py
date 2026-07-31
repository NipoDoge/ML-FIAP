"""Manifest tipado de artefactos promovidos para inferência."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol


class PipelineRunLike(Protocol):
    """Contrato mínimo para converter um run legado em manifest."""

    objective: str
    inference_backend: str | None
    model_path: str | None
    metrics: dict | None
    pipeline_type: str | None


@dataclass(frozen=True)
class ArtifactManifest:
    domain: str
    problem_type: Literal["binary_classification", "recommendation"]
    engine: str
    artifacts: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json_file(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactManifest:
        return cls(
            domain=str(data["domain"]),
            problem_type=data["problem_type"],
            engine=str(data["engine"]),
            artifacts={str(k): str(v) for k, v in (data.get("artifacts") or {}).items()},
            metadata=dict(data.get("metadata") or {}),
            schema_version=int(data.get("schema_version", 1)),
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> ArtifactManifest:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    @classmethod
    def from_pipeline_run(cls, run: PipelineRunLike) -> ArtifactManifest:
        """Converte o formato legado Fase 01 (Postgres + metrics JSON) em manifest."""
        metrics = dict(run.metrics or {})
        domain = str(run.objective).strip().lower()
        pipeline_type = str(getattr(run, "pipeline_type", "") or "").strip().lower()

        if pipeline_type == "recommendation" or metrics.get("problem_type") == "recommendation":
            prefix = run.model_path or (metrics.get("artifact_paths") or {}).get("prefix", "")
            if not prefix:
                raise ValueError("Run de recomendação sem prefix de artefacto.")
            engine_name = str(metrics.get("manifest_engine", "torch_embedding"))
            engine = "recommendation_torch" if "torch" in engine_name.lower() else "sklearn_joblib"
            top_k = int(metrics.get("top_k", 10))
            return cls(
                domain=domain,
                problem_type="recommendation",
                engine=engine,
                artifacts={"prefix": str(prefix)},
                metadata={"top_k": top_k, "champion_name": metrics.get("champion_name")},
            )

        backend = (run.inference_backend or "sklearn").strip().lower()

        if backend == "mlp":
            prefix = metrics.get("mlp_artifact_prefix") or ""
            if not prefix:
                raise ValueError(
                    "Run com inference_backend='mlp' sem 'mlp_artifact_prefix' nas métricas."
                )
            threshold = metrics.get("classification_decision_threshold", 0.5)
            return cls(
                domain=domain,
                problem_type="binary_classification",
                engine="torch_bundle",
                artifacts={"prefix": str(prefix)},
                metadata={
                    "inference_backend": backend,
                    "decision_threshold": float(threshold),
                },
            )

        model_path = run.model_path or ""
        if not model_path:
            raise ValueError("Run sklearn sem model_path definido.")
        threshold = metrics.get("classification_decision_threshold", 0.5)
        return cls(
            domain=domain,
            problem_type="binary_classification",
            engine="sklearn_joblib",
            artifacts={"joblib_path": str(model_path)},
            metadata={
                "inference_backend": backend,
                "decision_threshold": float(threshold),
            },
        )
