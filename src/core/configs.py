from pathlib import Path
import os

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, ValidationInfo, field_validator, model_validator
import logging

from typing import Literal

# Raiz do repositório (este ficheiro: repo/src/core/configs.py)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """
    Configurações centralizadas com validação de tipo automática.
    Carrega do .env e converte para tipos corretos.

    extra='ignore': o mesmo .env serve à API e ao Docker Compose; variáveis como
    AIRFLOW_UID, PGADMIN_*, TIMEZONE não pertencem ao Settings e são ignoradas.

    Caminhos por omissão assumem árvore com código e dados em ``src/`` (ver README).
    """

    model_config = SettingsConfigDict(
        env_file_encoding="utf-8",
        env_file=(
            str(_REPO_ROOT / ".env"),
            ".env",
        ),
        case_sensitive=False,
        extra="ignore",
    )

    # Caminhos (relativos ao CWD; no Docker, WORKDIR é a raiz do repo em /var/www)
    path_data: str = Field(
        default="src/data/", validation_alias="PATH_DATA", description="Path para dados brutos"
    )
    path_data_preprocessed: str = Field(
        default="src/data/pre_processed/", validation_alias="PATH_DATA_PREPROCESSED"
    )
    path_model: str = Field(
        default="src/artifacts/models/",
        validation_alias="PATH_MODEL",
        description="Joblibs sklearn (não confundir com o pacote ORM ``models``).",
    )
    path_graphs: str = Field(default="src/graphs/", validation_alias="PATH_GRAPHS")
    path_logs: str = Field(default="logs/", validation_alias="PATH_LOGS")
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        validation_alias="MLFLOW_TRACKING_URI",
        description="Tracking server HTTP (Docker: http://mlflow_server:5000 via compose).",
    )
    mlflow_artifact_root: str = Field(
        default="src/artifacts/mlruns",
        validation_alias="MLFLOW_ARTIFACT_ROOT",
        description="Artefactos no host; Docker compose usa /mlflow/artifacts (mesmo bind mount).",
    )

    debug: bool = Field(default=False, validation_alias="DEBUG", description="Ativa modo debug")
    test_size: float = Field(
        default=0.2, validation_alias="TEST_SIZE", description="Proporção de teste (0.0 a 1.0)"
    )
    random_state: int = Field(default=42, validation_alias="RANDOM_STATE")

    objective: str = Field(
        default="churn",
        validation_alias="OBJECTIVE",
        description="Domínio ML quando o endpoint não aceita objective no formulário (baseline na API).",
    )

    use_mlp_for_prediction: bool = Field(
        default=True,
        validation_alias="USE_MLP_FOR_PREDICTION",
        description="Usar MLP para predição",
    )
    project_name: str = Field(validation_alias="PROJECT_NAME", description="Nome do projeto")
    project_version: str = Field(
        validation_alias="PROJECT_VERSION", description="Versão do projeto"
    )

    database_user: str = Field(
        validation_alias="DATABASE_USER", description="Usuário do banco de dados"
    )
    database_pass: str = Field(
        validation_alias="DATABASE_PASS", description="Senha do banco de dados"
    )
    database_server: str = Field(
        validation_alias="DATABASE_SERVER", description="Servidor do banco de dados"
    )
    database_port: int = Field(
        validation_alias="DATABASE_PORT", description="Porta do banco de dados"
    )
    database_name: str = Field(
        validation_alias="DATABASE_NAME", description="Nome do banco de dados"
    )
    airflow_database_name: str = Field(
        default="airflow",
        validation_alias="AIRFLOW_DATABASE_NAME",
        description="Base Postgres dos metadados Airflow (separada de processing).",
    )
    mlflow_database_name: str = Field(
        default="mlflow",
        validation_alias="MLFLOW_DATABASE_NAME",
        description="Base Postgres do backend store MLflow (servidor :5000).",
    )

    database_url: str | None = None

    jwt_secret: str = Field(validation_alias="SECRET", description="Chave secreta JWT")
    algorithm: str = Field(validation_alias="ALGORITHM", description="Algoritmo JWT")
    access_token_expire_minutes: int = Field(
        validation_alias="ACCESS_TOKEN_EXPIRE_MINUTES",
        description="Minutos para expiração do token de acesso",
    )

    log_http_requests: bool = Field(
        default=True,
        validation_alias="LOG_HTTP_REQUESTS",
        description="Log de método, path, status e latência por requisição",
    )

    log_http_requests_file: bool = Field(
        default=True,
        validation_alias="LOG_HTTP_REQUESTS_FILE",
        description="Persistir linhas JSONL de acesso em path_api_request_logs/access.jsonl",
    )
    path_api_request_logs: str = Field(
        default="logs/api_requests",
        validation_alias="PATH_API_REQUEST_LOGS",
        description="Diretório dos arquivos access.jsonl (rotação automática)",
    )
    log_http_requests_max_bytes: int = Field(
        default=5_242_880,
        validation_alias="LOG_HTTP_REQUESTS_MAX_BYTES",
        description="Tamanho máximo de access.jsonl antes da rotação (~5 MiB)",
    )
    log_http_requests_backup_count: int = Field(
        default=5,
        validation_alias="LOG_HTTP_REQUESTS_BACKUP_COUNT",
        description="Número de arquivos access.jsonl.* retidos após rotação",
    )
    path_maintenance_reports: str = Field(
        default="src/artifacts/reports",
        validation_alias="PATH_MAINTENANCE_REPORTS",
        description="Saídas dos scripts de manutenção (latência, drift, relatórios)",
    )

    airflow_base_url: str = Field(
        default="http://airflow-webserver:8080",
        validation_alias="AIRFLOW_BASE_URL",
        description="URL base do Airflow",
    )
    airflow_user: str = Field(
        default="airflow", validation_alias="AIRFLOW_USER", description="Usuário do Airflow"
    )
    airflow_password: str = Field(
        default="airflow", validation_alias="AIRFLOW_PASSWORD", description="Senha do Airflow"
    )
    ml_shared_path: str = Field(
        default="ml_data/uploads",
        validation_alias="ML_SHARED_PATH",
        description="Pasta de uploads CSV (relativa a ML_PROJECT_ROOT ou absoluta). Alinhada ao bind mount compose.",
    )

    ml_project_root: str = Field(
        default="",
        validation_alias="ML_PROJECT_ROOT",
        description="Raiz do projecto ML no contentor (artefactos reco, params.yaml, data/recommendation).",
    )

    worker_recommendation_url: str = Field(
        default="",
        validation_alias="WORKER_RECOMMENDATION_URL",
        description="URL base do worker HTTP de recomendação (ex.: http://worker_recommendation:8010).",
    )

    environment: str = Field(
        default="development", validation_alias="ENVIRONMENT", description="Ambiente de execução"
    )

    sync_fe_tune_max_minutes: int = Field(
        default=2,
        validation_alias="SYNC_FE_TUNE_MAX_MINUTES",
        description="Teto de minutos de tuning do FE em rotas síncronas (API, não Airflow).",
    )

    classification_decision_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        validation_alias="CLASSIFICATION_DECISION_THRESHOLD",
        description=(
            "Probabilidade mínima P(classe positiva) para classificar como positivo nos relatórios "
            "(Precision/Recall/F1Accuracy em teste; não altera thresholds internos do CV). "
            "Override por conf Airflow/decision_threshold ou FE argumento dedicado."
        ),
    )

    ml_pipeline_joblib_backend: Literal["auto", "threading", "loky", "multiprocessing"] = Field(
        default="auto",
        validation_alias="ML_PIPELINE_JOBLIB_BACKEND",
        description=(
            "Backend joblib para cross_val_score/cross_validate/permutation_importance. "
            "'auto' usa threading dentro de tasks Airflow (evita Loky→n_jobs=1). "
            "Em ambiente sem Airflow mantém omissão (loky). Ver .env_example."
        ),
    )

    @field_validator("debug", mode="before")
    @classmethod
    def _normalize_debug(cls, v: object, info: ValidationInfo) -> object:
        if isinstance(v, str):
            key = v.strip().lower()
            if key in {"", "0", "false", "no", "off", "release", "prod", "production"}:
                return False
            if key in {"1", "true", "yes", "on", "debug", "dev", "development"}:
                return True
            raise ValueError(
                f"{info.field_name} deve ser booleano ou um modo conhecido. Recebido: {v!r}."
            )
        return v

    @field_validator("ml_pipeline_joblib_backend", mode="before")
    @classmethod
    def _normalize_ml_pipeline_joblib_backend(
        cls, v: object, info: ValidationInfo
    ) -> Literal["auto", "threading", "loky", "multiprocessing"]:
        if v is None or (isinstance(v, str) and not v.strip()):
            return "auto"
        if not isinstance(v, str):
            raise TypeError(f"{info.field_name} deve ser uma string.")
        key = v.strip().lower()
        allowed = frozenset({"auto", "threading", "loky", "multiprocessing"})
        if key not in allowed:
            raise ValueError(
                f"{info.field_name} deve ser um de {sorted(allowed)}. Recebido: {v!r}."
            )
        return key  # type: ignore[return-value]

    def resolved_joblib_parallel_backend_for_sklearn(self) -> str | None:
        """
        Backend a passar a ``joblib.parallel_backend`` nos passos de CV do pipeline ML.

        ``None`` = não forçar (joblib usa loky por omissão). Com ``auto``, usa
        ``threading`` dentro de tasks Airflow (``AIRFLOW_CTX_DAG_ID`` definido)
        para evitar Loky a cair para ``n_jobs=1`` em subprocessos.
        """
        b = self.ml_pipeline_joblib_backend
        if b == "auto":
            return "threading" if os.environ.get("AIRFLOW_CTX_DAG_ID") else None
        return b

    @property
    def is_production(self) -> bool:
        """Ambientes em que treino síncrono (baseline/FE via HTTP) deve ficar desligado — usar Airflow."""
        return self.environment.strip().lower() in {"prd", "prod", "production"}

    def get_log_level(self) -> int:
        """Retorna o nível de logging baseado em debug"""
        return logging.DEBUG if self.debug else logging.INFO

    @model_validator(mode="after")
    def set_database_url(self):
        self.database_url = (
            f"postgresql+asyncpg://{self.database_user}:"
            f"{self.database_pass}@{self.database_server}:"
            f"{self.database_port}/{self.database_name}"
        )
        if not (self.ml_project_root or "").strip():
            self.ml_project_root = str(_REPO_ROOT)
        return self


# Instância global (singleton pattern)
settings: Settings = Settings()
