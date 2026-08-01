"""Contratos HTTP partilhados da plataforma (predict, promote, runs, trigger)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PipelineRunResponse(BaseModel):
    id: int
    user_id: int
    pipeline_type: str
    is_airflow_run: bool = False
    objective: str
    status: str
    original_filename: str
    model_path: str | None = None
    csv_output_path: str | None = None
    metrics: dict | None = None
    error_message: str | None = None
    active: bool = Field(default=True, description="Run lógico ativo no painel interno.")
    inference_backend: str = Field(
        default="sklearn",
        description="Backend servido em /predict para este run: 'sklearn' (joblib) ou 'mlp' (PyTorch).",
    )
    created_at: datetime | None = None
    completed_at: datetime | None = None

    class Config:
        from_attributes = True


class MetricSnapshot(BaseModel):
    """Métricas de classificação binária no holdout de treino (um único split de teste)."""

    model_config = ConfigDict(extra="ignore")

    accuracy: float | None = Field(
        default=None, description="Acurácia no conjunto de teste do run."
    )
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    roc_auc: float | None = None


_INFERENCE_SCOPE_NOTE = (
    "Probabilidade e classe são deste pedido. Métricas abaixo são do hold-out do treino da run "
    "ligada a `pipeline_run_id`, não desta linha."
)


class ServedModelPredict(BaseModel):
    """Modelo efectivamente usado na chamada a ``/predict``."""

    model_config = ConfigDict(extra="ignore")

    inference_backend: str = Field(..., description="`sklearn` ou `mlp`.")
    predict_model_key: str = Field(
        ..., description="Identificador lógico (ex.: sklearn_pipeline, pytorch_mlp)."
    )
    label: str = Field(default="Modelo efectivamente usado neste `/predict`")
    name: str = Field(
        ..., description="Nome legível do modelo promovido (linha do comparativo ou inferido)."
    )
    origin: str = Field(
        ..., description="Origem no comparativo de treino (pré/pós-tuning, MLP, etc.)."
    )


class TrainingSelectionSummaryPredict(BaseModel):
    model_config = ConfigDict(extra="ignore")

    optimization_metric: str | None = None
    best_cv_score: float | None = Field(
        default=None, description="Melhor média de CV na seleção interna."
    )
    classification_decision_threshold_for_holdout_metrics: float | None = Field(
        default=None,
        description="Threshold usado nas métricas de holdout registadas para o modelo FE servido.",
    )
    sklearn_classifier_from_cv_study: str | None = Field(
        default=None,
        description="Nome do classificador sklearn destacado no estudo de CV (pode coincidir com o promovido).",
    )


class PyTorchMLPExperiment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    holdout_row: dict[str, Any] = Field(default_factory=dict)
    training_summary: dict[str, Any] | None = None


class ExperimentationPredict(BaseModel):
    model_config = ConfigDict(extra="ignore")

    pytorch_mlp: PyTorchMLPExperiment | None = None


class ComparisonPredict(BaseModel):
    model_config = ConfigDict(extra="ignore")

    intro: str = Field(
        default="Ordem sugerida: modelo do predict → candidatos pré-tuning → experimentação (MLP)."
    )
    model_used_for_this_predict: dict[str, Any] = Field(default_factory=dict)
    pre_tuning_sklearn: list[dict[str, Any]] = Field(default_factory=list)
    experimentation: ExperimentationPredict = Field(default_factory=ExperimentationPredict)


class BaselinePredictBlock(BaseModel):
    """Referência ao pipeline Baseline (antes do FE), ligado ao treino deste run."""

    model_config = ConfigDict(extra="ignore")

    baseline_pipeline_run_id: int | None = None
    model_selection: str = Field(default="Regressão Logística")
    role: str | None = None
    classification_decision_threshold: float | None = None
    description: str | None = None
    holdout_metrics: dict[str, Any] = Field(default_factory=dict)


class InferenceReport(BaseModel):
    """
    Relatório narrativo para ``/predict``: modelo servido, métricas de holdout do treino,
    comparativo organizado e baseline de referência.
    """

    model_config = ConfigDict(extra="ignore")

    scope_note: str = Field(default=_INFERENCE_SCOPE_NOTE)
    served_model: ServedModelPredict
    training_selection_summary: TrainingSelectionSummaryPredict
    holdout_metrics_served_model: MetricSnapshot | None = Field(
        default=None,
        description="Métricas de teste (holdout) do modelo efectivamente servido neste domínio.",
    )
    comparison: ComparisonPredict
    baseline: BaselinePredictBlock | None = None
    notes: list[str] = Field(default_factory=list)
    summary_lines: list[str] = Field(default_factory=list)


class PredictResponse(BaseModel):
    id: int
    domain: str
    pipeline_run_id: int
    prediction: int = Field(
        default=0,
        description="Classe prevista (tabular) ou primeiro item_id (recomendação, referência).",
    )
    probability: float | None = Field(
        default=None,
        description="Probabilidade estimada da classe positiva em percentual (0–100), quando disponível.",
    )
    probability_display: str | None = Field(
        default=None,
        description="Representação legível da probabilidade (ex.: '78.29%').",
    )
    recommended_items: list[int] | None = Field(
        default=None,
        description="Lista de item_id recomendados (domínio recommendation).",
    )
    input_data: dict
    inference_report: InferenceReport = Field(
        ...,
        description="Contexto do modelo servido e métricas de holdout do treino associadas a este run.",
    )

    class Config:
        from_attributes = True


class DeployedModelResponse(BaseModel):
    id: int
    domain: str
    pipeline_run_id: int
    status: str
    promoted_at: datetime | None = None
    promoted_by_user_id: int | None = None
    metrics_snapshot: dict | None = None
    pipeline_type: str | None = Field(
        default=None,
        description="Tipo do pipeline promovido (feature_engineering, recommendation, …).",
    )
    mlflow_registry_model: str | None = Field(
        default=None,
        description="Nome do modelo no MLflow Model Registry (side-effect Fase 6).",
    )
    mlflow_registry_version: str | None = Field(
        default=None,
        description="Versão promovida a Production no Registry.",
    )
    mlflow_registry_stage: str | None = Field(
        default=None,
        description="Stage final no Registry (tipicamente Production).",
    )
    mlflow_registry_run_id: str | None = Field(
        default=None,
        description="MLflow run_id associado à versão Registry.",
    )
    mlflow_registry_warning: str | None = Field(
        default=None,
        description="Aviso se o side-effect Registry falhou (promote na BD mantém-se).",
    )

    class Config:
        from_attributes = True


class TriggerDagResponse(BaseModel):
    dag_run_id: str
    dag_id: str
    domain: str
    objective: str = Field(description="Alias de domain (tabular).")
    csv_path: str | None = None
    message: str
