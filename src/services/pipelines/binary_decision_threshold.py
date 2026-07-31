"""Conversão de probabilidades em classe com limiar configurável (vs predict() ≈ limiar interno ~0.5)."""

from __future__ import annotations

from typing import Any

import numpy as np


def labels_from_probability_threshold(estimator: Any, X: Any, threshold: float) -> np.ndarray:
    """
    Para classificação binária com ``predict_proba``, usa P(classe ``classes_[1]``) >= ``threshold``.
    Caso não haja probabilidades bem definidas ou o estimador não seja binário típico, recua para
    ``predict()``.
    """
    t = float(threshold)
    if not (0.0 <= t <= 1.0):
        raise ValueError(
            f"classification_decision_threshold deve estar em [0, 1]. Recebido: {threshold!r}"
        )

    pred_default = estimator.predict(X)
    if not hasattr(estimator, "predict_proba"):
        return np.asarray(pred_default)

    proba = estimator.predict_proba(X)
    classes = getattr(estimator, "classes_", None)
    if proba.shape[1] < 2 or classes is None or len(classes) != 2:
        return np.asarray(pred_default)

    y = np.where(proba[:, 1] >= t, classes[1], classes[0])
    return np.asarray(y)
