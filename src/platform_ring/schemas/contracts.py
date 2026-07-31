"""Contratos HTTP partilhados da plataforma (predict, promote, runs, trigger)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class PipelineRunResponse(BaseModel):
    id: int
    user_id: int
    pipeline_type: str
    is_airflow_run: bool = False
    objective: str
    status: str
    original_filename: str
    model_path: Optional[str] = None
    csv_output_path: Optional[str] = None
    metrics: Optional[dict] = None
    error_message: Optional[str] = None
    active: bool = Field(default=True, description="Run lógico ativo no painel interno.")
    inference_backend: str = Field(
        default="sklearn",
        description="Backend servido em /predict para este run: 'sklearn' (joblib) ou 'mlp' (PyTorch).",
    )
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class MetricSnapshot(BaseModel):
    """Métricas de classificação binária no holdout de treino (um único split de teste)."""

    model_config = ConfigDict(extra="ignore")

    accuracy: Optional[float] = Field(
        default=None, description="Acurácia no conjunto de teste do run."
    )
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    roc_auc: Optional[float] = None


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

    optimization_metric: Optional[str] = None
    best_cv_score: Optional[float] = Field(
        default=None, description="Melhor média de CV na seleção interna."
    )
    classification_decision_threshold_for_holdout_metrics: Optional[float] = Field(
        default=None,
        description="Threshold usado nas métricas de holdout registadas para o modelo FE servido.",
    )
    sklearn_classifier_from_cv_study: Optional[str] = Field(
        default=None,
        description="Nome do classificador sklearn destacado no estudo de CV (pode coincidir com o promovido).",
    )


class PyTorchMLPExperiment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    holdout_row: dict[str, Any] = Field(default_factory=dict)
    training_summary: Optional[dict[str, Any]] = None


class ExperimentationPredict(BaseModel):
    model_config = ConfigDict(extra="ignore")

    pytorch_mlp: Optional[PyTorchMLPExperiment] = None


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

    baseline_pipeline_run_id: Optional[int] = None
    model_selection: str = Field(default="Regressão Logística")
    role: Optional[str] = None
    classification_decision_threshold: Optional[float] = None
    description: Optional[str] = None
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
    holdout_metrics_served_model: Optional[MetricSnapshot] = Field(
        default=None,
        description="Métricas de teste (holdout) do modelo efectivamente servido neste domínio.",
    )
    comparison: ComparisonPredict
    baseline: Optional[BaselinePredictBlock] = None
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
    probability: Optional[float] = Field(
        default=None,
        description="Probabilidade estimada da classe positiva em percentual (0–100), quando disponível.",
    )
    probability_display: Optional[str] = Field(
        default=None,
        description="Representação legível da probabilidade (ex.: '78.29%').",
    )
    recommended_items: Optional[list[int]] = Field(
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
    promoted_at: Optional[datetime] = None
    promoted_by_user_id: Optional[int] = None
    metrics_snapshot: Optional[dict] = None
    pipeline_type: Optional[str] = Field(
        default=None,
        description="Tipo do pipeline promovido (feature_engineering, recommendation, …).",
    )
    mlflow_registry_model: Optional[str] = Field(
        default=None,
        description="Nome do modelo no MLflow Model Registry (side-effect Fase 6).",
    )
    mlflow_registry_version: Optional[str] = Field(
        default=None,
        description="Versão promovida a Production no Registry.",
    )
    mlflow_registry_stage: Optional[str] = Field(
        default=None,
        description="Stage final no Registry (tipicamente Production).",
    )
    mlflow_registry_run_id: Optional[str] = Field(
        default=None,
        description="MLflow run_id associado à versão Registry.",
    )
    mlflow_registry_warning: Optional[str] = Field(
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
    csv_path: Optional[str] = None
    message: str
