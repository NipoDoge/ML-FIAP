"""Testes do recomendador PyTorch TC02."""

from __future__ import annotations

import json

import pandas as pd
import pytest

pytest.importorskip("torch")

from domains.recommendation.models.embedding_model import TorchEmbeddingRecommender


def test_torch_embedding_persists_seen_items_after_load(tmp_path):
    train_df = pd.DataFrame(
        [
            {"user_id": 1, "item_id": 10, "rating": 5.0},
            {"user_id": 1, "item_id": 20, "rating": 4.0},
            {"user_id": 2, "item_id": 20, "rating": 5.0},
            {"user_id": 2, "item_id": 30, "rating": 4.0},
            {"user_id": 3, "item_id": 10, "rating": 4.0},
            {"user_id": 3, "item_id": 30, "rating": 5.0},
        ]
    )
    model = TorchEmbeddingRecommender(
        embedding_dim=8,
        hidden_dim=16,
        n_epochs=2,
        batch_size=4,
        n_negatives=1,
        early_stopping_patience=1,
        random_state=7,
    )

    model.fit(train_df)
    prefix = tmp_path / "torch_embedding"
    model.save(prefix)

    meta = json.loads(prefix.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["model_type"] == "embedding_mlp_bce_item_bias"
    assert meta["use_item_bias"] is True
    assert meta["user_items"]["1"] == [10, 20]

    loaded = TorchEmbeddingRecommender.load(prefix)
    recommendations = loaded.recommend(user_id=1, n_items=3)

    assert recommendations
    assert not ({10, 20} & set(recommendations))


def test_torch_embedding_uses_low_ratings_as_explicit_negatives():
    train_df = pd.DataFrame(
        [
            {"user_id": 1, "item_id": 10, "rating": 5.0},
            {"user_id": 1, "item_id": 20, "rating": 2.0},
            {"user_id": 2, "item_id": 20, "rating": 4.0},
            {"user_id": 2, "item_id": 30, "rating": 1.0},
            {"user_id": 3, "item_id": 10, "rating": 4.5},
            {"user_id": 3, "item_id": 30, "rating": 3.0},
        ]
    )
    model = TorchEmbeddingRecommender(
        embedding_dim=8,
        hidden_dim=16,
        n_epochs=1,
        batch_size=4,
        n_negatives=1,
        validation_fraction=0.0,
        min_positive_rating=4.0,
        random_state=7,
    )

    model.fit(train_df)

    assert model.training_summary["n_positive_interactions"] == 3
    assert model.training_summary["n_explicit_negative_interactions"] == 3
