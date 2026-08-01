"""Strategies tabulares por domínio (churn, heart_disease, …)."""

from services.pipelines.feature_strategies import (
    STRATEGY_REGISTRY,
    get_class_labels,
)
from services.pipelines.feature_strategies.base import FeatureStrategy

__all__ = ["STRATEGY_REGISTRY", "FeatureStrategy", "get_class_labels"]
