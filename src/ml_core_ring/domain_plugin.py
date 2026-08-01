"""Registry único de domínios de problema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel


@dataclass(frozen=True)
class DomainPlugin:
    name: str
    problem_type: Literal["binary_classification", "recommendation"]
    feature_strategy: type | None
    input_schema: type[BaseModel]
    class_labels: tuple[str, str] | None
    allowed_metrics: frozenset[str]
    pipeline_runner_id: str


DOMAIN_REGISTRY: dict[str, DomainPlugin] = {}


def register_domain(plugin: DomainPlugin) -> DomainPlugin:
    DOMAIN_REGISTRY[plugin.name.strip().lower()] = plugin
    return plugin


def get_domain(name: str) -> DomainPlugin:
    key = name.strip().lower()
    plugin = DOMAIN_REGISTRY.get(key)
    if plugin is None:
        known = ", ".join(sorted(DOMAIN_REGISTRY)) or "(vazio)"
        raise KeyError(f"Domínio {name!r} não registrado. Disponíveis: {known}")
    return plugin
