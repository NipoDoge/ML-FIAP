"""Implementação de treino e modelos por tipo de problema."""

from executors_ring import recommendation as recommendation_executor
from executors_ring import tabular_classification as tabular_executor

__all__ = ["recommendation_executor", "tabular_executor"]
