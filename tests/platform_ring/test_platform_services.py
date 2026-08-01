"""Testes platform_ring — trigger e promote."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import domains  # noqa: F401
from platform_ring.training_trigger import build_dispatch_conf


def test_build_dispatch_conf_tabular_requires_csv():
    with pytest.raises(ValueError, match="csv_path"):
        build_dispatch_conf(domain="churn", user_id=1, csv_path=None)


def test_build_dispatch_conf_tabular_ok():
    conf = build_dispatch_conf(domain="churn", user_id=2, csv_path="/tmp/x.csv")
    assert conf["domain"] == "churn"
    assert conf["csv_path"] == "/tmp/x.csv"
    assert conf["objective"] == "churn"


def test_build_dispatch_conf_recommendation():
    conf = build_dispatch_conf(domain="recommendation", user_id=2)
    assert conf["domain"] == "recommendation"
    assert "csv_path" not in conf


def test_trigger_training_dag_calls_dispatch_dag():
    import asyncio

    from platform_ring.training_trigger import trigger_training_dag

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("platform_ring.training_trigger.httpx.AsyncClient", return_value=mock_client):
        result = asyncio.run(trigger_training_dag(domain="recommendation", user_id=1))

    assert result.dag_id == "ml_training_dispatch"
    assert result.domain == "recommendation"
    mock_client.post.assert_awaited_once()
    call_url = mock_client.post.await_args.args[0]
    assert "ml_training_dispatch" in call_url
