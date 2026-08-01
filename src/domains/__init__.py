"""Registo de domínios (import side-effect)."""

from domains.churn import plugin as churn_plugin
from domains.recommendation import plugin as recommendation_plugin

__all__ = ["churn_plugin", "recommendation_plugin"]
