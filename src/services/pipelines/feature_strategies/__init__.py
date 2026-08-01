from services.pipelines.feature_strategies.base import FeatureStrategy
from services.pipelines.feature_strategies.churn_features import ChurnFeatures
from services.pipelines.feature_strategies.heart_disease_features import HeartDiseaseFeatures

STRATEGY_REGISTRY: dict[str, type[FeatureStrategy]] = {
    "heart_disease": HeartDiseaseFeatures,
    "churn": ChurnFeatures,
}

CLASS_LABELS: dict[str, tuple[str, str]] = {
    "heart_disease": ("Sem Doença", "Doença Cardíaca"),
    "churn": ("Não Churn", "Churn"),
}


def get_class_labels(objective: str) -> tuple[str, str]:
    """Retorna os labels das classes para o domínio, ou o padrão genérico."""
    return CLASS_LABELS.get(objective, (f"Sem {objective}", objective))
