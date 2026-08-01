from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import parallel_backend
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    ParameterSampler,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from core.configs import settings
from core.custom_logger import setup_log
from ml_core_ring.mlflow_setup import configure_mlflow_tracking, ensure_mlflow_experiment
from services.pipelines.binary_decision_threshold import labels_from_probability_threshold
from services.pipelines.fe_hyperparameter_tuning import param_distributions_for
from services.pipelines.fe_model_selection import (
    normalize_optimization_metric,
    result_column_for_metric,
    sklearn_scoring_parameter,
)
from services.pipelines.feature_strategies.base import FeatureStrategy
from services.utils import filename_with_suffix, log_training_csv_to_active_run

if TYPE_CHECKING:
    from services.pipelines.mlp_torch_tabular import TorchTabularMLPResult

configure_mlflow_tracking()

logger = logging.getLogger("ml.pipeline")

plt.style.use("seaborn-v0_8-darkgrid")


class FeatureEngineering:
    """
    Pipeline de Feature Engineering, seleção de features,
    treinamento comparativo, tuning e persistência.
    """

    def __init__(
        self,
        objective: str,
        strategy: FeatureStrategy,
        run_timestamp: str | None = None,
        csv_path: str | None = None,
        manifest_path: str | None = None,
        optimization_metric: str = "accuracy",
        min_precision: float | None = None,
        min_roc_auc: float | None = None,
        tuning_n_iter: int | None = None,
        export_figures_dir: str | None = None,
        # --- MLP PyTorch (MVP, Tech Challenge): treino paralelo ao sklearn, só no FE ---
        enable_mlp_torch: bool = True,
        mlp_val_fraction: float = 0.15,
        mlp_hidden_dims: tuple[int, ...] = (64, 32),
        mlp_dropout: float = 0.0,
        mlp_batch_size: int = 64,
        mlp_lr: float = 1e-3,
        mlp_weight_decay: float = 1e-5,
        mlp_max_epochs: int = 300,
        mlp_early_stopping_patience: int = 20,
        *,
        artifact_name_suffix: str | None = None,
        is_airflow_run: bool = False,
        decision_threshold: float | None = None,
    ):
        self.objective = objective
        self.strategy = strategy
        self._explicit_csv_path = os.path.abspath(csv_path) if csv_path else None
        self._explicit_manifest_path = os.path.abspath(manifest_path) if manifest_path else None
        self.optimization_metric = normalize_optimization_metric(optimization_metric)
        self.min_precision = min_precision
        self.min_roc_auc = min_roc_auc
        self.export_figures_dir = export_figures_dir
        self.mlflow_run_id: str | None = None
        self._training_csv_path_resolved: str | None = None

        self.path_data_preprocessed = settings.path_data_preprocessed
        self.path_model = settings.path_model
        self.test_size = settings.test_size
        self.random_state = settings.random_state
        self._artifact_suffix = (artifact_name_suffix or "").strip()
        self.is_airflow_run = bool(is_airflow_run)

        if run_timestamp is not None:
            self.now = run_timestamp
        else:
            self.now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.snapshot_path = os.path.join(settings.path_data, settings.path_logs, self.now)

        self.data: pd.DataFrame | None = None
        self.x_train: pd.DataFrame | None = None
        self.x_test: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.y_test: pd.Series | None = None
        self.feature_names: list[str] = []
        self.feature_groups: dict[str, list[str]] | None = None
        self.k_features = 25

        self.trained_models: dict = {}
        self.baseline_manifest: dict | None = None
        self.results_df: pd.DataFrame | None = None
        self.guardrails_summary: dict = {}
        self.best_model_name: str | None = None
        self.best_pipeline = None
        self.best_params: dict | None = None
        self.best_cv_score: float = -np.inf
        self.best_test_score: float = -np.inf
        self.tuned_metrics: dict = {}
        self.figs_to_log: list[tuple[str, plt.Figure]] = []
        self.n_jobs = (
            1 if "debugpy" in sys.modules or not self._process_parallelism_available() else -1
        )
        self.tuning_n_iter = tuning_n_iter if tuning_n_iter is not None else 100

        self.enable_mlp_torch = enable_mlp_torch
        self.mlp_val_fraction = mlp_val_fraction
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_dropout = mlp_dropout
        self.mlp_batch_size = mlp_batch_size
        self.mlp_lr = mlp_lr
        self.mlp_weight_decay = mlp_weight_decay
        self.mlp_max_epochs = mlp_max_epochs
        self.mlp_early_stopping_patience = mlp_early_stopping_patience
        self.decision_threshold = (
            float(decision_threshold)
            if decision_threshold is not None
            else float(settings.classification_decision_threshold)
        )
        if not (0.0 <= self.decision_threshold <= 1.0):
            raise ValueError(
                f"classification_decision_threshold deve estar em [0, 1]. Recebido: {self.decision_threshold!r}"
            )
        # Preenchidos em _run_mlp_torch_mvp (para MLflow / estudo)
        self.mlp_torch_result: TorchTabularMLPResult | None = None
        self.mlp_torch_hparams: dict[str, Any] = {}
        self.mlp_torch_checkpoint_path: str | None = None
        # Prefixo (sem extensão) do bundle MLP servível: <prefix>.pt + <prefix>_preprocess.joblib + <prefix>_meta.json
        self.mlp_artifact_prefix: str | None = None

        for metric_name, metric_value in (
            ("min_precision", self.min_precision),
            ("min_roc_auc", self.min_roc_auc),
        ):
            if metric_value is not None and not (0.0 <= metric_value <= 1.0):
                raise ValueError(
                    f"{metric_name} deve estar no intervalo [0, 1]. Recebido: {metric_value}"
                )
        if self.tuning_n_iter <= 0:
            raise ValueError(f"tuning_n_iter deve ser > 0. Recebido: {self.tuning_n_iter}")
        if not (0.0 < self.mlp_val_fraction < 1.0):
            raise ValueError(
                f"mlp_val_fraction deve estar em (0, 1). Recebido: {self.mlp_val_fraction}"
            )

    def fe_sklearn_joblib_path(self) -> str:
        name = filename_with_suffix(
            f"best_{self.objective}_{self.now}.joblib", self._artifact_suffix
        )
        return os.path.join(self.path_model, name)

    def fe_export_bundle_dir(self, joblib_path: str) -> str:
        dname = f"fe_export_{self.objective}_{self.now}{self._artifact_suffix or ''}"
        return os.path.join(os.path.dirname(os.path.abspath(joblib_path)), dname)

    @staticmethod
    def _process_parallelism_available() -> bool:
        try:
            os.sysconf("SC_SEM_NSEMS_MAX")
        except (AttributeError, OSError, ValueError):
            return False
        return True

    def _passes_guardrails(self, precision_value: float, roc_auc_value: float) -> bool:
        if self.min_precision is not None and precision_value < self.min_precision:
            return False
        if self.min_roc_auc is None:
            return True
        return not (np.isnan(roc_auc_value) or roc_auc_value < self.min_roc_auc)

    def _joblib_cv_parallel_cm(self):
        """
        Airflow (e outros processos filhos) fazem joblib/Loky recuar para n_jobs=1.
        ``parallel_backend('threading')`` paraleliza folds por threads, sem novos processos.
        Controlado por ``settings.resolved_joblib_parallel_backend_for_sklearn()`` /
        ``ML_PIPELINE_JOBLIB_BACKEND``.

        Com debug manual (VS Code / ``debugpy`` carregado), não força backend — alinhado a
        ``n_jobs=1`` e breakpoints mais previsíveis.
        """
        if "debugpy" in sys.modules:
            return nullcontext()

        backend = settings.resolved_joblib_parallel_backend_for_sklearn()
        if backend is not None:
            logger.debug(
                "joblib.parallel_backend(%r) para CV/permutação (setting=%s, AIRFLOW_CTX=%s).",
                backend,
                settings.ml_pipeline_joblib_backend,
                "sim" if os.environ.get("AIRFLOW_CTX_DAG_ID") else "não",
            )
            return parallel_backend(backend)
        return nullcontext()

    @staticmethod
    def _is_binary_column(series: pd.Series) -> bool:
        non_null = series.dropna()
        if non_null.empty:
            return False
        unique_values = set(non_null.unique().tolist())
        return unique_values.issubset({0, 1, np.int8(0), np.int8(1)})

    def _classify_feature_columns(self, x_df: pd.DataFrame) -> dict[str, list[str]]:
        categorical_cols = x_df.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_cols = x_df.select_dtypes(include=[np.number, "bool"]).columns.tolist()

        binary_cols: list[str] = []
        continuous_cols: list[str] = []
        for col in numeric_cols:
            if self._is_binary_column(x_df[col]):
                binary_cols.append(col)
            else:
                continuous_cols.append(col)

        grouped = {
            "binary": binary_cols,
            "continuous": continuous_cols,
            "categorical": categorical_cols,
        }
        self.feature_groups = grouped
        return grouped

    def _build_preprocess_transformer(
        self, x_train: pd.DataFrame, groups: dict[str, list[str]] | None = None
    ) -> ColumnTransformer:
        """
        Pré-processamento no mesmo padrão do Baseline:
        - binárias: passthrough (sem transformação)
        - contínuas: StandardScaler (+ imputação mediana só se houver nulos)
        - categóricas: OneHotEncoder (+ imputação moda só se houver nulos)
        """
        groups = groups or self._classify_feature_columns(x_train)
        if groups is None or not any(groups.values()):
            groups = self._classify_feature_columns(x_train)

        binary_cols = groups["binary"]
        continuous_cols = groups["continuous"]
        categorical_cols = groups["categorical"]

        transformers: list[tuple] = []

        if continuous_cols:
            continuous_steps = []
            has_nulls_cont = bool(x_train[continuous_cols].isna().any().any())
            if has_nulls_cont:
                continuous_steps.append(("imputer", SimpleImputer(strategy="median")))
            continuous_steps.append(("scaler", StandardScaler()))
            transformers.append(("continuous", SkPipeline(continuous_steps), continuous_cols))

        if binary_cols:
            has_nulls_bin = bool(x_train[binary_cols].isna().any().any())
            if has_nulls_bin:
                raise ValueError(
                    "Colunas binárias com nulos detectadas. "
                    "Como binárias estão em passthrough por definição, "
                    "preencha nulos antes do treino."
                )
            transformers.append(("binary", "passthrough", binary_cols))

        if categorical_cols:
            categorical_steps = []
            has_nulls_cat = bool(x_train[categorical_cols].isna().any().any())
            if has_nulls_cat:
                categorical_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
            categorical_steps.append(
                ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
            )
            transformers.append(("categorical", SkPipeline(categorical_steps), categorical_cols))

        if not transformers:
            raise ValueError("Nenhuma coluna válida encontrada para pré-processamento no FE.")

        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )

    def _build_model_pipeline(self, model_name: str):
        preprocess = self._build_preprocess_transformer(self.x_train, groups=self.feature_groups)
        model_map = {
            "Decision Tree": DecisionTreeClassifier(random_state=self.random_state),
            "Random Forest": RandomForestClassifier(random_state=self.random_state),
            "SVM": SVC(kernel="rbf", probability=True, random_state=self.random_state),
            "Gradient Boosting": GradientBoostingClassifier(random_state=self.random_state),
        }
        if model_name not in model_map:
            raise ValueError(f"Modelo não suportado no FE: {model_name}")
        return SkPipeline(
            [
                ("preprocess", preprocess),
                ("remove_constant", VarianceThreshold(threshold=0.0)),
                ("selector", SelectKBest(f_classif, k=self.k_features)),
                ("model", model_map[model_name]),
            ]
        )

    # ------------------------------------------------------------------
    # Etapa 1 — Carregar dados pré-processados (saída do baseline)
    # ------------------------------------------------------------------

    def load_data(self):
        logger.info("Carregando dataset pré-processado...")
        manifest_path = self._explicit_manifest_path or os.path.join(
            self.path_data_preprocessed, "manifest.json"
        )
        if not os.path.isfile(manifest_path):
            raise ValueError(f"Manifest não encontrado: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.baseline_manifest = manifest

        if manifest.get("objective") != self.objective:
            raise ValueError(
                f"Manifest do baseline com objective='{manifest.get('objective')}', "
                f"mas FE foi chamado para '{self.objective}'."
            )
        if manifest.get("sample_schema") != "raw_clean":
            raise ValueError(
                f"sample_schema inválido no manifest: {manifest.get('sample_schema')!r}. "
                "Esperado: 'raw_clean'."
            )

        file_from_manifest = manifest.get("output_sample_csv_stable")
        if not file_from_manifest:
            raise ValueError("Manifest inválido: campo 'output_sample_csv_stable' ausente.")

        csv_path = os.path.abspath(file_from_manifest)
        if self._explicit_csv_path and os.path.abspath(self._explicit_csv_path) != csv_path:
            logger.warning(
                "csv_path explícito (%s) difere do manifest (%s). Usando manifest como fonte de verdade.",
                self._explicit_csv_path,
                csv_path,
            )
        if not os.path.isfile(csv_path):
            raise ValueError(f"CSV do baseline não encontrado (manifest): {csv_path}")

        logger.info(f"CSV selecionado via manifest: {csv_path}")
        self._training_csv_path_resolved = csv_path
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.lower()

        has_prefixes = any("__" in col for col in df.columns)
        if has_prefixes:
            df.columns = [col.split("__", 1)[-1] if "__" in col else col for col in df.columns]
            logger.warning("Prefixos de colunas detectados e removidos (CSV legado).")

        logger.info(f"Colunas: {df.columns.tolist()}")

        if "target" not in df.columns:
            logger.error(
                "Coluna 'target' não encontrada no dataset pré-processado no final do CSV."
            )
            raise ValueError(
                "Pipeline interrompido — coluna 'target' não encontrada no dataset pré-processado."
            )

        null_total = df.isnull().sum().sum()
        if null_total > 0:
            logger.error(f"Valores nulos encontrados: {null_total}")
            raise ValueError("Pipeline interrompido — valores nulos no dataset pré-processado.")

        logger.info(f"Dataset carregado: {df.shape}")
        logger.info(f"Target:\n{df['target'].value_counts().to_string()}")

        self.data = df

    # ------------------------------------------------------------------
    # Etapa 2 — Criar features (delega para a strategy)
    # ------------------------------------------------------------------

    def build_features(self):
        logger.info(f"Construindo features com strategy: {self.strategy.__class__.__name__}")

        self.strategy.validate(self.data)
        self.data = self.strategy.build(self.data)

        logger.info(f"Dataset após feature engineering: {self.data.shape}")

    # ------------------------------------------------------------------
    # Etapa 3 — Split + Seleção de features
    # ------------------------------------------------------------------

    def select_features(self, k: int = 25):
        logger.info("Iniciando split de dados para FE...")

        y = self.data["target"]
        x = self.data.drop(columns=["target"])
        self.feature_names = x.columns.tolist()
        self.k_features = min(k, max(1, x.shape[1]))

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        groups = self._classify_feature_columns(self.x_train)
        logger.info(
            "Feature groups (pré-build): binárias=%s contínuas=%s categóricas=%s",
            len(groups["binary"]),
            len(groups["continuous"]),
            len(groups["categorical"]),
        )
        logger.debug("Colunas binárias: %s", groups["binary"])
        logger.debug("Colunas contínuas: %s", groups["continuous"])
        logger.debug("Colunas categóricas: %s", groups["categorical"])
        logger.info("k de seleção configurado para SelectKBest: %s", self.k_features)

    # ------------------------------------------------------------------
    # Etapa 4 — Treinamento comparativo
    # ------------------------------------------------------------------

    def train_models(self):
        sort_col = result_column_for_metric(self.optimization_metric)
        scoring = sklearn_scoring_parameter(self.optimization_metric)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        logger.info(
            "Iniciando comparação de modelos (métrica de otimização: %s → coluna %s)",
            self.optimization_metric,
            sort_col,
        )

        model_configs = [
            "Decision Tree",
            "Random Forest",
            "SVM",
            "Gradient Boosting",
        ]

        results = []

        with self._joblib_cv_parallel_cm():
            for name in model_configs:
                pipeline = self._build_model_pipeline(name)
                cv_scores = cross_val_score(
                    pipeline,
                    self.x_train,
                    self.y_train,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=self.n_jobs,
                )
                cv_score = float(np.mean(cv_scores))

                cv_guard = cross_validate(
                    pipeline,
                    self.x_train,
                    self.y_train,
                    cv=cv,
                    scoring={"precision": "precision", "roc_auc": "roc_auc"},
                    n_jobs=self.n_jobs,
                )
                guardrail_precision_cv = float(np.mean(cv_guard["test_precision"]))
                guardrail_roc_auc_cv = float(np.mean(cv_guard["test_roc_auc"]))

                pipeline.fit(self.x_train, self.y_train)
                y_pred = labels_from_probability_threshold(
                    pipeline, self.x_test, self.decision_threshold
                )

                metrics = {
                    "Modelo": name,
                    "Acurácia": accuracy_score(self.y_test, y_pred),
                    "Precisão": precision_score(self.y_test, y_pred, zero_division=0),
                    "Recall": recall_score(self.y_test, y_pred, zero_division=0),
                    "F1": f1_score(self.y_test, y_pred, zero_division=0),
                    "ROC AUC": np.nan,
                    "CV Score": cv_score,
                }

                if hasattr(pipeline, "predict_proba"):
                    metrics["ROC AUC"] = roc_auc_score(
                        self.y_test, pipeline.predict_proba(self.x_test)[:, 1]
                    )
                elif hasattr(pipeline, "decision_function"):
                    metrics["ROC AUC"] = roc_auc_score(
                        self.y_test, pipeline.decision_function(self.x_test)
                    )

                metrics["Pass Guardrails"] = self._passes_guardrails(
                    guardrail_precision_cv,
                    guardrail_roc_auc_cv,
                )
                metrics["CV Precision"] = guardrail_precision_cv
                metrics["CV ROC AUC"] = guardrail_roc_auc_cv

                logger.info("=== %s === cv_%s: %.4f", name, self.optimization_metric, cv_score)
                logger.debug(f"\n{classification_report(self.y_test, y_pred)}")

                self.trained_models[name] = pipeline
                results.append(metrics)

        self.results_df = (
            pd.DataFrame(results)
            .sort_values("CV Score", ascending=False)
            .reset_index(drop=True)
            .round(4)
        )
        logger.info(f"Ranking de modelos:\n{self.results_df.to_string()}")

        eligible = self.results_df[self.results_df["Pass Guardrails"]]
        if not eligible.empty:
            winner_row = eligible.iloc[0]
            self.guardrails_summary["selection_guardrails_passed"] = True
        else:
            winner_row = self.results_df.iloc[0]
            self.guardrails_summary["selection_guardrails_passed"] = False
            logger.warning(
                "Nenhum modelo passou nos guardrails (min_precision=%s, min_roc_auc=%s). "
                "Selecionando melhor modelo por cv_%s.",
                self.min_precision,
                self.min_roc_auc,
                self.optimization_metric,
            )
        self.best_model_name = str(winner_row["Modelo"])
        self.best_pipeline = self.trained_models[self.best_model_name]
        self.best_cv_score = float(winner_row["CV Score"])
        logger.info(
            "Modelo selecionado para a etapa seguinte: %s (cv_%s=%.4f)",
            self.best_model_name,
            self.optimization_metric,
            self.best_cv_score,
        )

    # ------------------------------------------------------------------
    # Etapa 5 — Tuning (time-budget)
    # ------------------------------------------------------------------

    def _finalize_tuned_metrics(self, y_pred, y_proba_vec: np.ndarray | None) -> None:
        """Preenche `tuned_metrics` a partir de predições no conjunto de teste (`y_proba_vec` = P(classe positiva))."""
        self.tuned_metrics = {
            "classification_decision_threshold": float(self.decision_threshold),
            "Acurácia": accuracy_score(self.y_test, y_pred),
            "Precisão": precision_score(self.y_test, y_pred, zero_division=0),
            "Recall": recall_score(self.y_test, y_pred, zero_division=0),
            "F1": f1_score(self.y_test, y_pred, zero_division=0),
            "ROC AUC": (
                roc_auc_score(self.y_test, y_proba_vec)
                if y_proba_vec is not None and not np.all(y_proba_vec == 0)
                else np.nan
            ),
        }

    def tune(self, time_limit_minutes: int = 60, acc_target: float | None = None):
        sort_col = result_column_for_metric(self.optimization_metric)
        scoring = sklearn_scoring_parameter(self.optimization_metric)
        if not self.best_model_name:
            raise RuntimeError("tune requer train_models prévio com best_model_name definido.")

        if acc_target is not None:
            logger.info(
                "Iniciando tuning — modelo: %s | limite: %smin | métrica: %s | alvo em %s > %s",
                self.best_model_name,
                time_limit_minutes,
                self.optimization_metric,
                sort_col,
                acc_target,
            )
        else:
            logger.info(
                "Iniciando tuning — modelo: %s | limite: %smin | métrica: %s | sem alvo explícito",
                self.best_model_name,
                time_limit_minutes,
                self.optimization_metric,
            )

        start = time.time()
        deadline = start + time_limit_minutes * 60

        param_distributions = param_distributions_for(self.best_model_name)
        sampler = ParameterSampler(
            param_distributions, n_iter=self.tuning_n_iter, random_state=self.random_state
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        self.best_cv_score = -np.inf
        self.best_params = None
        tuned_pipeline: SkPipeline | None = None
        tuned_pipeline_fallback: SkPipeline | None = None
        best_cv_fallback = -np.inf
        best_params_fallback: dict | None = None

        n_evaluated = 0

        with self._joblib_cv_parallel_cm():
            for i, params in enumerate(sampler, start=1):
                now = time.time()
                if now >= deadline:
                    logger.info("Tempo limite atingido — encerrando busca.")
                    break

                pipeline = self._build_model_pipeline(self.best_model_name)
                pipeline.set_params(**params)

                cv_scores = cross_val_score(
                    pipeline,
                    self.x_train,
                    self.y_train,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=self.n_jobs,
                )
                mean_cv = float(np.mean(cv_scores))

                guardrail_precision_cv = float("nan")
                guardrail_roc_auc_cv = float("nan")
                if self.min_precision is not None or self.min_roc_auc is not None:
                    scorers = {}
                    if self.min_precision is not None:
                        scorers["precision"] = "precision"
                    if self.min_roc_auc is not None:
                        scorers["roc_auc"] = "roc_auc"
                    cv_guard = cross_validate(
                        pipeline,
                        self.x_train,
                        self.y_train,
                        cv=cv,
                        scoring=scorers,
                        n_jobs=self.n_jobs,
                    )
                    if "test_precision" in cv_guard:
                        guardrail_precision_cv = float(np.mean(cv_guard["test_precision"]))
                    if "test_roc_auc" in cv_guard:
                        guardrail_roc_auc_cv = float(np.mean(cv_guard["test_roc_auc"]))

                pass_guardrails = self._passes_guardrails(
                    guardrail_precision_cv, guardrail_roc_auc_cv
                )

                pipeline.fit(self.x_train, self.y_train)

                if mean_cv > best_cv_fallback:
                    best_cv_fallback = mean_cv
                    best_params_fallback = params
                    tuned_pipeline_fallback = pipeline

                if pass_guardrails and mean_cv > self.best_cv_score:
                    self.best_cv_score = mean_cv
                    self.best_params = params
                    tuned_pipeline = pipeline
                    elapsed = now - start
                    logger.info(
                        "[%s] Novo melhor (guardrails ok) | cv_%s=%.4f | cv_precision=%.4f | cv_roc_auc=%.4f | t=%.1fm | params=%s",
                        i,
                        self.optimization_metric,
                        mean_cv,
                        guardrail_precision_cv,
                        guardrail_roc_auc_cv,
                        elapsed / 60,
                        params,
                    )

                if i % 5 == 0:
                    elapsed = now - start
                    remaining = max(0.0, deadline - now)
                    logger.debug(
                        "Iterações: %s | melhor_cv_%s=%.4f | decorrido=%.1fm | restante=%.1fm",
                        i,
                        self.optimization_metric,
                        self.best_cv_score,
                        elapsed / 60,
                        remaining / 60,
                    )

                n_evaluated = i

        if tuned_pipeline is None:
            self.guardrails_summary["tuning_guardrails_passed"] = False
            if tuned_pipeline_fallback is not None:
                logger.warning(
                    "Nenhuma iteração do tuning passou nos guardrails (min_precision=%s, min_roc_auc=%s). "
                    "Usando melhor configuração por cv_%s sem guardrails.",
                    self.min_precision,
                    self.min_roc_auc,
                    self.optimization_metric,
                )
                tuned_pipeline = tuned_pipeline_fallback
                self.best_params = best_params_fallback
                self.best_cv_score = best_cv_fallback
            else:
                logger.warning(
                    "Nenhuma iteração melhorou o modelo — mantendo %s da etapa comparativa.",
                    self.best_model_name,
                )
                tuned_pipeline = self.best_pipeline
        else:
            self.guardrails_summary["tuning_guardrails_passed"] = True

        self.best_pipeline = tuned_pipeline

        y_pred = labels_from_probability_threshold(
            self.best_pipeline, self.x_test, self.decision_threshold
        )
        y_proba = (
            self.best_pipeline.predict_proba(self.x_test)[:, 1]
            if hasattr(self.best_pipeline, "predict_proba")
            else None
        )
        self._finalize_tuned_metrics(y_pred, y_proba)

        elapsed_total = time.time() - start
        logger.info(f"Tuning concluído — {n_evaluated} iterações em {elapsed_total / 60:.1f}min")
        logger.info("Melhor cv_%s: %.4f", self.optimization_metric, self.best_cv_score)
        if acc_target is not None:
            logger.info(
                "Alvo (%.2f) sobre %s: %s",
                acc_target,
                sort_col,
                "atingido" if self.tuned_metrics[sort_col] > acc_target else "não atingido",
            )

    # ------------------------------------------------------------------
    # Etapa 5b — MLP PyTorch (MVP, isolado do SkPipeline)
    # ------------------------------------------------------------------

    def _run_mlp_torch_mvp(self) -> None:
        """
        Treina um perceptrão multicamadas (PyTorch) **em paralelo** aos modelos sklearn.

        **Por que existe:** requisito típico de Tech Challenge (MLP + MLflow), sem substituir
        o modelo sklearn promovido em produção.

        **O que entra:** os mesmos ``x_train`` / ``x_test`` / ``y_train`` / ``y_test`` já usados
        no FE (pós-``select_features``), ou seja, **já com as colunas criadas pela strategy**.

        **Pré-processamento:** só o ``ColumnTransformer`` (imputação, scaler, OHE) — **igual ao
        passo ``preprocess`` dos pipelines sklearn**, mas **sem** ``VarianceThreshold``,
        ``SelectKBest`` nem modelo. Assim o MLP vê **todas** as colunas numéricas/categóricas
        transformadas; os Random Forest / etc. ainda reduzem dimensão com ``SelectKBest`` lá
        dentro do ``SkPipeline``. Comparar métricas no MLflow tendo isso em mente.

        **Validação:** parte do treino é separada (estratificada) para *early stopping* pela
        loss (BCE com logits).

        Se ``torch`` não estiver instalado ou ``enable_mlp_torch=False``, o passo é ignorado.
        """
        self.mlp_torch_result = None
        self.mlp_torch_hparams = {}
        self.mlp_torch_checkpoint_path = None

        if not self.enable_mlp_torch:
            logger.info("MLP PyTorch desligado (enable_mlp_torch=False).")
            return
        if (
            self.x_train is None
            or self.x_test is None
            or self.y_train is None
            or self.y_test is None
        ):
            logger.warning("MLP PyTorch: split inexistente — passo ignorado.")
            return

        try:
            import torch

            from services.pipelines.mlp_torch_tabular import train_eval_mlp_binary_tabular
        except ImportError as e:
            logger.warning("MLP PyTorch indisponível (import): %s — passo ignorado.", e)
            return

        # 1) Montar o mesmo tipo de pré-processamento que o FE usa nos pipelines sklearn.
        preprocess = self._build_preprocess_transformer(self.x_train, groups=self.feature_groups)
        X_train_t = preprocess.fit_transform(self.x_train)
        X_test_t = preprocess.transform(self.x_test)

        def _to_float32_dense(X):
            if hasattr(X, "toarray"):
                X = X.toarray()
            return np.asarray(X, dtype=np.float32)

        X_train_t = _to_float32_dense(X_train_t)
        X_test_t = _to_float32_dense(X_test_t)
        y_all = self.y_train.to_numpy(dtype=np.int64)

        # 2) Tirar um pedaço do treino para validação (early stopping), sem tocar no teste.
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_t,
            y_all,
            test_size=self.mlp_val_fraction,
            random_state=self.random_state,
            stratify=y_all,
        )
        y_te = self.y_test.to_numpy(dtype=np.int64)

        self.mlp_torch_hparams = {
            "hidden_dims": self.mlp_hidden_dims,
            "dropout": self.mlp_dropout,
            "batch_size": self.mlp_batch_size,
            "lr": self.mlp_lr,
            "weight_decay": self.mlp_weight_decay,
            "max_epochs": self.mlp_max_epochs,
            "early_stopping_patience": self.mlp_early_stopping_patience,
            "val_fraction": self.mlp_val_fraction,
            "random_state": self.random_state,
            "torch_version": torch.__version__,
        }

        logger.info(
            "MLP PyTorch (MVP): treinando | shapes tr/val/te=%s %s %s | hparams=%s",
            X_tr.shape,
            X_val.shape,
            X_test_t.shape,
            {k: v for k, v in self.mlp_torch_hparams.items() if k != "torch_version"},
        )

        try:
            self.mlp_torch_result = train_eval_mlp_binary_tabular(
                X_tr,
                X_val,
                X_test_t,
                y_tr,
                y_val,
                y_te,
                random_state=self.random_state,
                hidden_dims=self.mlp_hidden_dims,
                dropout=self.mlp_dropout,
                batch_size=self.mlp_batch_size,
                lr=self.mlp_lr,
                weight_decay=self.mlp_weight_decay,
                max_epochs=self.mlp_max_epochs,
                early_stopping_patience=self.mlp_early_stopping_patience,
            )
        except Exception:
            logger.exception("MLP PyTorch falhou (sklearn segue normal)")
            self.mlp_torch_result = None
            self.mlp_torch_checkpoint_path = None
            return

        os.makedirs(self.path_model, exist_ok=True)
        prefix = os.path.join(
            self.path_model, f"pytorch_mlp_{self.objective}_{self.now}{self._artifact_suffix or ''}"
        )
        ckpt = f"{prefix}.pt"
        preprocess_path = f"{prefix}_preprocess.joblib"
        meta_path = f"{prefix}_meta.json"
        torch.save(self.mlp_torch_result.state_dict, ckpt)
        # Bundle de inferência: o ColumnTransformer fitted é necessário para servir o MLP.
        joblib.dump(preprocess, preprocess_path)
        meta = {
            "schema_version": 1,
            "objective": self.objective,
            "run_timestamp": self.now,
            "hidden_dims": list(self.mlp_hidden_dims),
            "dropout": float(self.mlp_dropout),
            "n_features_in": int(X_train_t.shape[1]),
            "decision_threshold": float(self.decision_threshold),
            "feature_columns_in_order": list(map(str, self.x_train.columns)),
            "feature_groups": {k: list(v) for k, v in (self.feature_groups or {}).items()},
            "best_epoch": int(self.mlp_torch_result.best_epoch),
            "best_val_loss": float(self.mlp_torch_result.best_val_loss),
            "metrics_test": dict(self.mlp_torch_result.metrics_test),
            "metrics_val": dict(self.mlp_torch_result.metrics_val),
            "torch_version": str(self.mlp_torch_hparams.get("torch_version", "")),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        self.mlp_torch_checkpoint_path = ckpt
        self.mlp_artifact_prefix = prefix
        logger.info(
            "MLP PyTorch (MVP): melhor época=%s val_loss=%.4f | test metrics=%s | bundle=%s",
            self.mlp_torch_result.best_epoch,
            self.mlp_torch_result.best_val_loss,
            self.mlp_torch_result.metrics_test,
            prefix,
        )

    # ------------------------------------------------------------------
    # Etapa 6 — Importância de features
    # ------------------------------------------------------------------

    def _model_input_column_names(self) -> np.ndarray:
        """Nomes das colunas na matriz que entra no classificador (pós preprocess + VT + SelectKBest)."""
        feat_names = np.array(self.feature_names)
        steps = getattr(self.best_pipeline, "named_steps", {})
        if "preprocess" in steps:
            feat_names = np.array(steps["preprocess"].get_feature_names_out())
        if "remove_constant" in steps:
            vt = steps["remove_constant"]
            if hasattr(vt, "get_support"):
                feat_names = feat_names[vt.get_support()]
        if "selector" in steps:
            selector = steps["selector"]
            if hasattr(selector, "get_support"):
                feat_names = feat_names[selector.get_support()]
        return feat_names

    def _model_comparison_intro_bullets(self) -> list[str]:
        """
        Resumo explícito para o utilizador (CSV + MD), sem ambiguidade.

        Com ``USE_MLP_FOR_PREDICTION=true``: sklearn no FE é benchmark (RandomSearch omitido);
        inferência após promote é PyTorch MLP. Com ``false``: inferência é o pipeline sklearn (com tuning).
        """
        sk = self.best_model_name or "-"
        if settings.use_mlp_for_prediction:
            return [
                f"Campeão no comparativo SKLEARN (benchmark · pré-tuning): {sk}",
                "Modelo usado na inferência: PyTorch MLP (USE_MLP_FOR_PREDICTION=true).",
            ]
        sk_display = f"{sk} (tuned)" if self.tuned_metrics else sk
        return [
            f"Campeão no comparativo SKLEARN: {sk_display}",
            "Modelo usado na inferência: pipeline sklearn (USE_MLP_FOR_PREDICTION=false).",
        ]

    def _model_comparison_evidence_lines(self) -> list[str]:
        """Contexto operacional (depois do resumo): modo de execução, treino PyTorch, tuning sklearn."""
        lines: list[str] = []
        lines.extend(self._model_comparison_intro_bullets())

        modo = "Airflow" if self.is_airflow_run else "manual"
        lines.append(f"Execução em modo {modo}.")

        if not self.enable_mlp_torch:
            lines.append(
                "Treino PyTorch MLP neste FE: desactivado (enable_mlp_torch=False no construtor)."
            )
        elif self.mlp_torch_result is not None:
            lines.append(
                "Treino PyTorch MLP neste FE: executado (linha «PyTorch MLP» na tabela abaixo)."
            )
        else:
            lines.append(
                "Treino PyTorch MLP neste FE: previsto mas sem resultado (torch/import ausente, erro ou dados)."
            )

        if settings.use_mlp_for_prediction:
            lines.append(
                "Neste modo, RandomSearch do sklearn não corre; o joblib sklearn mantém-se no vencedor "
                "da etapa pré-tuning (apenas referência / artefacto auxiliar)."
            )
        else:
            lines.append(
                "Neste modo, RandomSearch do sklearn corre quando aplicável; a linha «(tuned)» resume o candidato promovível."
            )

        return lines

    def _pytorch_comparison_origem(self) -> str:
        """Texto da coluna Origem para a linha PyTorch — alinhado a USE_MLP_FOR_PREDICTION."""
        if settings.use_mlp_for_prediction:
            return "PyTorch MLP (treino no FE; modelo de inferência após promote)"
        return "PyTorch MLP (treino no FE; inferência via sklearn promovido)"

    def _build_model_comparison_table(self) -> pd.DataFrame | None:
        """
        Consolida métricas de teste de todos os modelos do run num DataFrame único:
        sklearn pré-tuning (``results_df``) + sklearn pós-tuning (``tuned_metrics``) + MLP.

        Útil para o Model Card / relatório (PDF Etapa 2: tabela comparativa).
        """
        rows: list[dict[str, Any]] = []

        sk_pre = (
            "sklearn (pré-tuning; benchmark)"
            if settings.use_mlp_for_prediction
            else "sklearn (pré-tuning)"
        )
        if self.results_df is not None and not self.results_df.empty:
            for _, r in self.results_df.iterrows():
                rows.append(
                    {
                        "Modelo": str(r.get("Modelo", "")),
                        "Origem": sk_pre,
                        "Accuracy": float(r.get("Acurácia", float("nan"))),
                        "Precision": float(r.get("Precisão", float("nan"))),
                        "Recall": float(r.get("Recall", float("nan"))),
                        "F1": float(r.get("F1", float("nan"))),
                        "ROC AUC": float(r.get("ROC AUC", float("nan"))),
                    }
                )

        if self.tuned_metrics:
            rows.append(
                {
                    "Modelo": f"{self.best_model_name or 'best'} (tuned)",
                    "Origem": "sklearn (pós-tuning; promovido em /predict)",
                    "Accuracy": float(self.tuned_metrics.get("Acurácia", float("nan"))),
                    "Precision": float(self.tuned_metrics.get("Precisão", float("nan"))),
                    "Recall": float(self.tuned_metrics.get("Recall", float("nan"))),
                    "F1": float(self.tuned_metrics.get("F1", float("nan"))),
                    "ROC AUC": float(self.tuned_metrics.get("ROC AUC", float("nan"))),
                }
            )

        if self.mlp_torch_result is not None:
            mt = self.mlp_torch_result.metrics_test or {}
            rows.append(
                {
                    "Modelo": "PyTorch MLP",
                    "Origem": self._pytorch_comparison_origem(),
                    "Accuracy": float(mt.get("accuracy", float("nan"))),
                    "Precision": float(mt.get("precision", float("nan"))),
                    "Recall": float(mt.get("recall", float("nan"))),
                    "F1": float(mt.get("f1", float("nan"))),
                    "ROC AUC": float(mt.get("roc_auc", float("nan"))),
                }
            )

        if not rows:
            return None
        df = pd.DataFrame(rows)[
            ["Modelo", "Origem", "Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
        ]
        return df.round(4)

    def _export_fe_bundle(self, joblib_path: str) -> str:
        """Pasta com comparação de modelos, resumo PyTorch, CSV pré-transform (pós-strategy) e pós-transform (entrada do modelo)."""
        bundle = self.fe_export_bundle_dir(joblib_path)
        os.makedirs(bundle, exist_ok=True)

        def _f(name: str) -> str:
            return filename_with_suffix(name, self._artifact_suffix)

        if self.results_df is not None:
            self.results_df.to_csv(
                os.path.join(bundle, _f("model_selection_comparison.csv")), index=False
            )

        comparison_df = self._build_model_comparison_table()
        if comparison_df is not None:
            comparison_csv = os.path.join(bundle, _f("model_comparison_full.csv"))
            comparison_md = os.path.join(bundle, _f("model_comparison_full.md"))
            evidence_lines = self._model_comparison_evidence_lines()
            # CSV só com a grelha tabular (RFC 4180 + BOM UTF-8 para Excel). Narrativa fica no .md —
            # linhas de comentário livres no topo partiam colunas no Excel quando havia vírgulas nos textos.
            comparison_df.to_csv(
                comparison_csv,
                index=False,
                encoding="utf-8-sig",
                quoting=csv.QUOTE_NONNUMERIC,
                lineterminator="\n",
            )
            try:
                md_table = comparison_df.to_markdown(index=False, floatfmt=".4f")
            except ImportError:
                # tabulate ausente: fallback simples (cabeçalho + linhas pipe-separated)
                hdr = "| " + " | ".join(comparison_df.columns) + " |"
                sep = "| " + " | ".join(["---"] * len(comparison_df.columns)) + " |"
                lines = [
                    "| "
                    + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row)
                    + " |"
                    for row in comparison_df.itertuples(index=False, name=None)
                ]
                md_table = "\n".join([hdr, sep, *lines])
            ev_md = "".join(f"- {line}\n" for line in evidence_lines)
            header = (
                f"# Comparação de modelos — {self.objective} ({self.now})\n\n"
                f"{ev_md}"
                f"- Threshold de decisão: `{float(self.decision_threshold)}`\n"
                f"- Métrica de optimização: `{self.optimization_metric}`\n\n"
            )
            with open(comparison_md, "w", encoding="utf-8") as f:
                f.write(header + md_table + "\n")

        df_pre_tr = self.x_train.copy()
        df_pre_tr["target"] = self.y_train
        df_pre_te = self.x_test.copy()
        df_pre_te["target"] = self.y_test
        df_pre_tr.to_csv(os.path.join(bundle, _f("train_features_pre_transform.csv")), index=False)
        df_pre_te.to_csv(os.path.join(bundle, _f("test_features_pre_transform.csv")), index=False)

        kv_rows: list[tuple[str, str]] = []
        kv_rows.append(("pytorch_mvp_enabled", str(self.mlp_torch_result is not None)))
        for k, v in sorted(self.mlp_torch_hparams.items()):
            kv_rows.append((f"pytorch_hparam_{k}", str(v)))
        ckpt = self.mlp_torch_checkpoint_path
        kv_rows.append(("pytorch_checkpoint_path", "" if ckpt is None else str(ckpt)))
        if self.mlp_torch_result is not None:
            kv_rows.append(("pytorch_best_epoch", str(int(self.mlp_torch_result.best_epoch))))
            kv_rows.append(
                ("pytorch_best_val_loss", str(float(self.mlp_torch_result.best_val_loss)))
            )
            for k, v in self.mlp_torch_result.metrics_val.items():
                kv_rows.append((f"pytorch_metrics_val_{k}", str(float(v))))
            for k, v in self.mlp_torch_result.metrics_test.items():
                kv_rows.append((f"pytorch_metrics_test_{k}", str(float(v))))
        pd.DataFrame(kv_rows, columns=["key", "value"]).to_csv(
            os.path.join(bundle, _f("pytorch_mvp_summary.csv")),
            index=False,
        )

        pre = self.best_pipeline[:-1]
        Xtr = pre.transform(self.x_train)
        Xte = pre.transform(self.x_test)
        cols = self._model_input_column_names()
        if len(cols) != Xtr.shape[1]:
            logger.warning(
                "fe_export: %s nomes vs %s colunas na matriz; usando f0..fN.",
                len(cols),
                Xtr.shape[1],
            )
            cols = np.array([f"f{i}" for i in range(Xtr.shape[1])])
        df_tr = pd.DataFrame(Xtr, columns=cols)
        df_tr["target"] = self.y_train.to_numpy()
        df_te = pd.DataFrame(Xte, columns=cols)
        df_te["target"] = self.y_test.to_numpy()
        df_tr.to_csv(os.path.join(bundle, _f("train_model_input.csv")), index=False)
        df_te.to_csv(os.path.join(bundle, _f("test_model_input.csv")), index=False)

        logger.info("FE exportado em: %s", bundle)
        return bundle

    def evaluate_importance(self):
        logger.info("Calculando importância de features...")

        feat_names = self._model_input_column_names()

        model_step = (
            self.best_pipeline.named_steps.get("model", self.best_pipeline)
            if hasattr(self.best_pipeline, "named_steps")
            else self.best_pipeline
        )

        tree_imp = getattr(model_step, "feature_importances_", None)
        if tree_imp is not None and len(tree_imp) == len(feat_names):
            imp_df = pd.DataFrame({"feature": feat_names, "importance": tree_imp}).sort_values(
                "importance", ascending=False
            )
            logger.info(f"Importância Gini (top 20):\n{imp_df.head(20).to_string()}")

            fig, ax = plt.subplots(figsize=(8, min(0.45 * len(imp_df.head(20)), 10)))
            sns.barplot(data=imp_df.head(20), x="importance", y="feature", color="#1f77b4", ax=ax)
            ax.set_title("Importância de Features (Gini) — Top 20")
            ax.set_xlabel("Importância")
            ax.set_ylabel("Feature")
            plt.tight_layout()
            if self.export_figures_dir:
                os.makedirs(self.export_figures_dir, exist_ok=True)
                gini_name = filename_with_suffix(
                    "feature_importance_gini_top20.png", self._artifact_suffix
                )
                gini_png = os.path.join(self.export_figures_dir, gini_name)
                fig.savefig(gini_png, dpi=200, bbox_inches="tight")
            self.figs_to_log.append(
                (
                    filename_with_suffix(
                        "feature_importance_gini_top20.png", self._artifact_suffix
                    ),
                    fig,
                )
            )
            plt.close(fig)
        else:
            logger.info("Modelo não expõe feature_importances_. Pulando gráfico Gini.")

        with self._joblib_cv_parallel_cm():
            perm = permutation_importance(
                self.best_pipeline,
                self.x_test,
                self.y_test,
                scoring=sklearn_scoring_parameter(self.optimization_metric),
                n_repeats=10,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
        perm_feat_names = np.array(self.x_test.columns.tolist())
        if perm_feat_names.shape[0] != perm.importances_mean.shape[0]:
            logger.warning(
                "Colunas de x_test (%s) ≠ n_features da permutação (%s). Usando rótulos genéricos.",
                perm_feat_names.shape[0],
                perm.importances_mean.shape[0],
            )
            perm_feat_names = np.array(
                [f"feature_{i}" for i in range(perm.importances_mean.shape[0])]
            )
        perm_df = pd.DataFrame(
            {
                "feature": perm_feat_names,
                "importance": perm.importances_mean,
                "std": perm.importances_std,
            }
        ).sort_values("importance", ascending=False)
        logger.info(f"Importância por permutação (top 20):\n{perm_df.head(20).to_string()}")

        fig2, ax2 = plt.subplots(figsize=(8, min(0.45 * len(perm_df.head(20)), 10)))
        sns.barplot(data=perm_df.head(20), x="importance", y="feature", color="#ff7f0e", ax=ax2)
        ax2.set_title(f"Importância por Permutação ({self.optimization_metric}) — Top 20")
        ax2.set_xlabel("Queda média na métrica")
        ax2.set_ylabel("Feature")
        plt.tight_layout()
        if self.export_figures_dir:
            os.makedirs(self.export_figures_dir, exist_ok=True)
            perm_name = filename_with_suffix(
                "feature_importance_permutation_top20.png", self._artifact_suffix
            )
            p_png = os.path.join(self.export_figures_dir, perm_name)
            fig2.savefig(p_png, dpi=200, bbox_inches="tight")
        self.figs_to_log.append(
            (
                filename_with_suffix(
                    "feature_importance_permutation_top20.png", self._artifact_suffix
                ),
                fig2,
            )
        )
        plt.close(fig2)

    # ------------------------------------------------------------------
    # Etapa 7 — Persistência (joblib + MLflow)
    # ------------------------------------------------------------------

    def save(self):
        logger.info("Salvando artefatos...")

        os.makedirs(self.path_model, exist_ok=True)

        joblib_path = self.fe_sklearn_joblib_path()
        joblib.dump(self.best_pipeline, joblib_path)
        logger.info(f"Modelo salvo em: {joblib_path}")

        fe_bundle_dir: str | None = None
        try:
            fe_bundle_dir = self._export_fe_bundle(joblib_path)
        except Exception:
            logger.exception("Falha ao exportar bundle FE (CSV/comparação/PyTorch)")

        try:
            experiment_name = f"{self.objective}_feature_engineering"
            ensure_mlflow_experiment(experiment_name)
            run_nm = f"fe_{self.objective}_{self.now}{self._artifact_suffix or ''}"
            with mlflow.start_run(run_name=run_nm) as _mfe:
                self.mlflow_run_id = _mfe.info.run_id
                log_training_csv_to_active_run(
                    self._training_csv_path_resolved,
                    df=None,
                    dataset_name=f"{self.objective}_fe_baseline_sample_csv",
                    context="training",
                )
                if self.best_params:
                    for k, v in self.best_params.items():
                        mlflow.log_param(k, str(v))
                mlflow.log_param("optimization_metric", self.optimization_metric)
                mlflow.log_param(
                    "classification_decision_threshold", float(self.decision_threshold)
                )
                mlflow.log_param("tuning_n_iter", self.tuning_n_iter)
                if self.min_precision is not None:
                    mlflow.log_param("min_precision", float(self.min_precision))
                if self.min_roc_auc is not None:
                    mlflow.log_param("min_roc_auc", float(self.min_roc_auc))
                for k, v in self.guardrails_summary.items():
                    mlflow.log_param(k, bool(v))
                if self.baseline_manifest is not None:
                    mlflow.log_dict(
                        self.baseline_manifest,
                        filename_with_suffix("baseline_manifest.json", self._artifact_suffix),
                    )

                if np.isfinite(self.best_cv_score):
                    mlflow.log_metric(f"cv_{self.optimization_metric}", float(self.best_cv_score))
                for k, v in self.tuned_metrics.items():
                    if v is not None and not (
                        isinstance(v, float) and (np.isnan(v) or np.isinf(v))
                    ):
                        mlflow.log_metric(k.replace(" ", "_").lower(), float(v))

                # --- MLP PyTorch (MVP): mesma run do FE para comparar curvas/métricas no MLflow ---
                if self.mlp_torch_result is not None:
                    mlflow.log_param("pytorch_mlp_enabled", True)
                    mlflow.log_param(
                        "pytorch_mlp_hidden_dims", ",".join(str(d) for d in self.mlp_hidden_dims)
                    )
                    mlflow.log_param("pytorch_mlp_dropout", float(self.mlp_dropout))
                    mlflow.log_param("pytorch_mlp_batch_size", int(self.mlp_batch_size))
                    mlflow.log_param("pytorch_mlp_lr", float(self.mlp_lr))
                    mlflow.log_param("pytorch_mlp_weight_decay", float(self.mlp_weight_decay))
                    mlflow.log_param("pytorch_mlp_max_epochs", int(self.mlp_max_epochs))
                    mlflow.log_param(
                        "pytorch_mlp_early_stopping_patience", int(self.mlp_early_stopping_patience)
                    )
                    mlflow.log_param("pytorch_mlp_val_fraction", float(self.mlp_val_fraction))
                    if self.mlp_torch_hparams.get("torch_version"):
                        mlflow.log_param(
                            "pytorch_mlp_torch_version",
                            str(self.mlp_torch_hparams["torch_version"]),
                        )
                    mlflow.log_metric(
                        "pytorch_mlp_best_epoch", float(self.mlp_torch_result.best_epoch)
                    )
                    mlflow.log_metric(
                        "pytorch_mlp_best_val_loss", float(self.mlp_torch_result.best_val_loss)
                    )
                    for split_name, metrics in (
                        ("val", self.mlp_torch_result.metrics_val),
                        ("test", self.mlp_torch_result.metrics_test),
                    ):
                        for k, v in metrics.items():
                            if v is not None and not (
                                isinstance(v, float) and (np.isnan(v) or np.isinf(v))
                            ):
                                mlflow.log_metric(f"pytorch_mlp_{split_name}_{k}", float(v))
                    if self.mlp_torch_checkpoint_path and os.path.isfile(
                        self.mlp_torch_checkpoint_path
                    ):
                        mlflow.log_artifact(
                            self.mlp_torch_checkpoint_path, artifact_path="pytorch_mlp"
                        )
                    if self.mlp_artifact_prefix:
                        for ext in ("_preprocess.joblib", "_meta.json"):
                            p = f"{self.mlp_artifact_prefix}{ext}"
                            if os.path.isfile(p):
                                mlflow.log_artifact(p, artifact_path="pytorch_mlp")
                else:
                    mlflow.log_param("pytorch_mlp_enabled", False)

                for fname, fobj in self.figs_to_log:
                    try:
                        mlflow.log_figure(fobj, fname)
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"Falha ao logar figura {fname}: {e}")

                mlflow.sklearn.log_model(self.best_pipeline, artifact_path="sklearn_model")
                mlflow.log_artifact(joblib_path, artifact_path="joblib")
                if fe_bundle_dir and os.path.isdir(fe_bundle_dir):
                    mlflow.log_artifacts(fe_bundle_dir, artifact_path="fe_export")

            logger.info(f"Resultados registrados no MLflow (experimento: {experiment_name})")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Falha ao registrar no MLflow: {e}")

    # ------------------------------------------------------------------
    # Orquestrador
    # ------------------------------------------------------------------

    def _run_data_contract_and_fe_build(self, start_time):
        """
        Responsabilidade: carregar contrato do baseline e construir features.
        """
        self.load_data()
        logger.debug(f"Dados carregados: {datetime.now(timezone.utc) - start_time}")

        self.build_features()
        logger.debug(f"Features criadas: {datetime.now(timezone.utc) - start_time}")

    def _run_modeling_prep_and_selection(self, start_time):
        """
        Responsabilidade: preparar split e selecionar modelo base.
        """
        self.select_features()
        logger.debug(f"Features selecionadas: {datetime.now(timezone.utc) - start_time}")

        self.train_models()
        logger.debug(f"Modelos treinados: {datetime.now(timezone.utc) - start_time}")

    def _run_tuning_evaluation_and_persistence(
        self, start_time, time_limit_minutes: int, acc_target: float | None
    ):
        """
        Responsabilidade: tuning, avaliação final e persistência.

        Atalho: quando ``USE_MLP_FOR_PREDICTION=true`` o tuning sklearn é **pulado** (a MLP
        é o modelo servido em /predict). O ``best_pipeline`` sklearn fica como vencedor de
        ``train_models()`` (pré-tuning) — suficiente para preservar ``model_path``,
        importância de features, exportação de bundles e tabela comparativa.
        """
        if settings.use_mlp_for_prediction:
            logger.info(
                "Tuning sklearn DESLIGADO (USE_MLP_FOR_PREDICTION=true). best_pipeline = "
                "vencedor de train_models() (sem RandomSearch). best_model_name=%s "
                "best_cv_score=%.6f. tuned_metrics fica vazio na tabela comparativa.",
                self.best_model_name,
                float(self.best_cv_score) if np.isfinite(self.best_cv_score) else float("nan"),
            )
        else:
            self.tune(time_limit_minutes=time_limit_minutes, acc_target=acc_target)
            logger.debug(f"Tuning concluído: {datetime.now(timezone.utc) - start_time}")

        # MLP PyTorch: comparável no MLflow aos sklearn; não altera best_pipeline nem artefato promovido.
        self._run_mlp_torch_mvp()
        logger.debug(f"MLP PyTorch (MVP): {datetime.now(timezone.utc) - start_time}")

        self.evaluate_importance()
        logger.debug(f"Importância avaliada: {datetime.now(timezone.utc) - start_time}")

        self.save()
        logger.debug(f"Artefatos salvos: {datetime.now(timezone.utc) - start_time}")

    def run(self, time_limit_minutes: int, acc_target: float | None):
        """
        Orquestrador principal: executa FE por responsabilidades.
        """
        start_time = datetime.now(timezone.utc)
        logger.info(f"Pipeline de Feature Engineering iniciado: {start_time}")
        logger.info(
            "FE config | USE_MLP_FOR_PREDICTION=%s | enable_mlp_torch=%s",
            settings.use_mlp_for_prediction,
            self.enable_mlp_torch,
        )
        logger.info(
            "classification_decision_threshold: %s (métricas de teste; CV usa predict interno do sklearn)",
            self.decision_threshold,
        )

        self._run_data_contract_and_fe_build(start_time)
        self._run_modeling_prep_and_selection(start_time)
        self._run_tuning_evaluation_and_persistence(start_time, time_limit_minutes, acc_target)

        elapsed = datetime.now(timezone.utc) - start_time
        logger.info(f"Pipeline de Feature Engineering concluído em: {elapsed}")


if __name__ == "__main__":
    from services.pipelines.feature_strategies import STRATEGY_REGISTRY

    objective = "heart_disease"

    if objective not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Strategy '{objective}' não registrada. Disponíveis: {list(STRATEGY_REGISTRY.keys())}"
        )

    strategy = STRATEGY_REGISTRY[objective]()

    snapshot_path = os.path.join(
        settings.path_data, settings.path_logs, datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )
    setup_log(snapshot_path, datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))

    pipeline = FeatureEngineering(objective=objective, strategy=strategy)

    try:
        pipeline.run(time_limit_minutes=2, acc_target=None)
    except Exception as e:
        logger.error(f"Erro durante a execução do pipeline: {e}")
        raise
