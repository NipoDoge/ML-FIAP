import logging

import numpy as np
import pandas as pd

from services.pipelines.feature_strategies.base import FeatureStrategy

logger = logging.getLogger(__name__)

EPS = 1e-6


class ChurnFeatures(FeatureStrategy):
    """
    Feature engineering para churn em telecom (nível produção).
    """

    def __init__(self):
        self.monthly_median = None

    def fit(self, df: pd.DataFrame):
        df = df.copy()
        df.columns = df.columns.str.lower()

        self.monthly_median = df["monthlycharges"].median()
        return self

    def required_columns(self) -> list[str]:
        return [
            "tenure",
            "monthlycharges",
            "totalcharges",
            "contract",
            "paymentmethod",
            "internetservice",
        ]

    def created_features(self) -> list[str]:
        return [
            "contract_stability",
            "is_new_customer",
            "new_customer_in_mounth_contract",
            "risk_payment_monthly",
            "new_customer_risk_payment_monthly",
            "fiber_high_cost",
            "fiber_premium_monthly",
            "fiber_premium_monthly_new_customer",
            "avg_ticket",
            "charge_ratio",
            "num_services",
            "low_engagement",
            "high_cost_low_engagement",
            "is_auto_payment",
            "tenure_log",
        ]

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.columns = out.columns.str.lower()

        missing = set(self.required_columns()) - set(out.columns)
        if missing:
            raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

        if self.monthly_median is None:
            self.fit(out)

        out["is_new_customer"] = (out["tenure"] <= 12).astype(int)
        out["tenure_log"] = np.log1p(out["tenure"])

        contract_mapping = {"Month-to-month": 0, "One year": 1, "Two year": 2}

        out["contract_stability"] = out["contract"].map(contract_mapping)

        out["new_customer_in_mounth_contract"] = (
            (out["contract"] == "Month-to-month") & (out["is_new_customer"])
        ).astype(int)
        out["risk_payment_monthly"] = (
            (out["paymentmethod"] == "Electronic check") & (out["contract"] == "Month-to-month")
        ).astype(int)
        out["new_customer_risk_payment_monthly"] = (
            out["risk_payment_monthly"] & out["is_new_customer"]
        ).astype(int)

        out["fiber_high_cost"] = (
            (out["internetservice"] == "Fiber optic")
            & (out["monthlycharges"] > self.monthly_median)
        ).astype(int)
        out["fiber_premium_monthly"] = (
            (out["internetservice"] == "Fiber optic") & (out["contract"] == "Month-to-month")
        ).astype(int)
        out["fiber_premium_monthly_new_customer"] = (
            out["fiber_premium_monthly"] & out["is_new_customer"]
        ).astype(int)

        out["avg_ticket"] = out["totalcharges"] / (out["tenure"] + 1)
        out["charge_ratio"] = out["monthlycharges"] / (out["avg_ticket"] + EPS)

        service_cols = [
            "onlinesecurity",
            "onlinebackup",
            "deviceprotection",
            "techsupport",
            "streamingtv",
            "streamingmovies",
        ]
        missing_services = set(service_cols) - set(out.columns)
        if missing_services:
            logger.warning(f"Colunas de serviço ausentes: {missing_services}")

        valid_service_cols = [c for c in service_cols if c in out.columns]
        out["num_services"] = out[valid_service_cols].sum(axis=1)
        out["low_engagement"] = (out["num_services"] <= 2).astype(int)

        out["high_cost_low_engagement"] = (
            (out["monthlycharges"] > self.monthly_median) & (out["num_services"] <= 2)
        ).astype(int)
        out["is_auto_payment"] = (
            out["paymentmethod"].astype(str).str.lower().str.contains("automatic", regex=False)
        ).astype(int)

        return out
