"""Testes orchestration_ring — dispatch e persistência."""

from __future__ import annotations

import domains  # noqa: F401
import pytest

from orchestration_ring.dispatch import (
    resolve_domain,
    training_route_for_domain,
    validate_dispatch_conf,
)


def test_resolve_domain_prefers_domain_key():
    assert resolve_domain({"domain": "recommendation", "objective": "churn"}) == "recommendation"


def test_resolve_domain_falls_back_to_objective():
    assert resolve_domain({"objective": "churn"}) == "churn"


def test_training_route_churn_is_tabular():
    assert training_route_for_domain("churn") == "tabular"


def test_training_route_recommendation():
    assert training_route_for_domain("recommendation") == "recommendation"


def test_validate_dispatch_tabular_requires_csv(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    route = validate_dispatch_conf({"domain": "churn", "csv_path": str(csv_path)})
    assert route == "tabular"


def test_validate_dispatch_tabular_missing_csv():
    with pytest.raises(ValueError, match="csv_path"):
        validate_dispatch_conf({"domain": "churn"})


def test_validate_dispatch_recommendation_without_csv():
    route = validate_dispatch_conf({"domain": "recommendation"})
    assert route == "recommendation"
