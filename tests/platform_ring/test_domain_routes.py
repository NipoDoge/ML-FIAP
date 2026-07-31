"""Smoke tests — rotas /v1/domains/{domain}/ (estrutura estática)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHURN_ROUTER = (ROOT / "src/platform_ring/domains/churn/router.py").read_text(encoding="utf-8")
RECO_ROUTER = (ROOT / "src/platform_ring/domains/recommendation/router.py").read_text(
    encoding="utf-8"
)
API = (ROOT / "src/api/v1/api.py").read_text(encoding="utf-8")


def test_churn_router_endpoints():
    assert 'DOMAIN = "churn"' in CHURN_ROUTER
    assert 'f"/domains/{DOMAIN}"' in CHURN_ROUTER
    for path in (
        '"/predict"',
        '"/admin/promote"',
        '"/admin/runs"',
        '"/admin/deployments/history"',
        '"/admin/rollback"',
        '"/admin/train/trigger"',
        '"/admin/train/baseline"',
        '"/admin/train/feature-engineering"',
    ):
        assert path in CHURN_ROUTER, f"missing {path} in churn router"


def test_recommendation_router_endpoints():
    assert "recommendation" in RECO_ROUTER
    for path in (
        '"/predict"',
        '"/admin/promote"',
        '"/admin/runs"',
        '"/admin/deployments/history"',
        '"/admin/rollback"',
        '"/admin/train/trigger"',
        '"/admin/train/sync"',
    ):
        assert path in RECO_ROUTER, f"missing {path} in recommendation router"


def test_churn_predict_uses_features_schema_not_domain():
    assert "ChurnFeaturesInput" in CHURN_ROUTER
    assert "predict_for_domain_route" in CHURN_ROUTER


def test_api_mounts_domains_router():
    assert "domains_router" in API
    assert "processor" not in API
