"""Factory de modelos de recomendação (Strategy + Factory — TC02)."""

from __future__ import annotations

from typing import Literal, Protocol

import numpy as np
import pandas as pd


class RecommenderModel(Protocol):
    name: str

    def fit(self, train_df: pd.DataFrame) -> None: ...

    def recommend(self, user_id: int, n_items: int, exclude_items: set[int] | None = None) -> list[int]: ...


class PopularityBaseline:
    name = "popularity"

    def __init__(self) -> None:
        self._popular_items: list[int] = []

    def fit(self, train_df: pd.DataFrame) -> None:
        counts = train_df.groupby("item_id").size().sort_values(ascending=False)
        self._popular_items = [int(i) for i in counts.index.tolist()]

    def recommend(self, user_id: int, n_items: int, exclude_items: set[int] | None = None) -> list[int]:
        exclude = exclude_items or set()
        out: list[int] = []
        for item in self._popular_items:
            if item in exclude:
                continue
            out.append(item)
            if len(out) >= n_items:
                break
        return out


class NmfBaseline:
    name = "nmf_sklearn"

    def __init__(self, n_components: int = 32, random_state: int = 42) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self._user_items: dict[int, set[int]] = {}
        self._popular_items: list[int] = []

    def fit(self, train_df: pd.DataFrame) -> None:
        from sklearn.decomposition import NMF

        pivot = train_df.pivot_table(
            index="user_id",
            columns="item_id",
            values="rating",
            aggfunc="mean",
            fill_value=0.0,
        )
        model = NMF(n_components=self.n_components, random_state=self.random_state, max_iter=200)
        user_factors = model.fit_transform(pivot.values)
        item_factors = model.components_
        self._scores = user_factors @ item_factors
        self._user_index = {int(u): i for i, u in enumerate(pivot.index.tolist())}
        self._item_ids = [int(c) for c in pivot.columns.tolist()]
        counts = train_df.groupby("item_id").size().sort_values(ascending=False)
        self._popular_items = [int(i) for i in counts.index.tolist()]
        self._user_items = train_df.groupby("user_id")["item_id"].apply(lambda s: set(map(int, s))).to_dict()

    def recommend(self, user_id: int, n_items: int, exclude_items: set[int] | None = None) -> list[int]:
        exclude = exclude_items or set()
        seen = self._user_items.get(user_id, set()) | exclude
        if user_id in self._user_index:
            row = self._scores[self._user_index[user_id]]
            order = np.argsort(-row)
            out: list[int] = []
            for idx in order:
                item = self._item_ids[int(idx)]
                if item in seen:
                    continue
                out.append(item)
                if len(out) >= n_items:
                    return out
        return PopularityBaseline().recommend(user_id, n_items, exclude_items=seen)


def create_model(
    backend: Literal["popularity", "nmf", "torch_embedding"],
    **kwargs,
) -> RecommenderModel:
    if backend == "popularity":
        return PopularityBaseline()
    if backend == "nmf":
        return NmfBaseline(
            n_components=int(kwargs.get("nmf_components", 32)),
            random_state=int(kwargs.get("random_state", 42)),
        )
    if backend == "torch_embedding":
        from domains.recommendation.models.embedding_model import TorchEmbeddingRecommender

        return TorchEmbeddingRecommender(
            embedding_dim=int(kwargs.get("embedding_dim", 32)),
            hidden_dim=int(kwargs.get("hidden_dim", 64)),
            dropout=float(kwargs.get("dropout", 0.1)),
            use_item_bias=bool(kwargs.get("use_item_bias", True)),
            n_epochs=int(kwargs.get("n_epochs", 10)),
            batch_size=int(kwargs.get("batch_size", 512)),
            lr=float(kwargs.get("lr", 0.01)),
            random_state=int(kwargs.get("random_state", 42)),
            n_negatives=int(kwargs.get("n_negatives", 4)),
            validation_fraction=float(kwargs.get("validation_fraction", 0.1)),
            early_stopping_patience=int(kwargs.get("early_stopping_patience", 3)),
            early_stopping_min_delta=float(kwargs.get("early_stopping_min_delta", 1e-4)),
            min_positive_rating=float(kwargs.get("min_positive_rating", 0.0)),
        )
    raise ValueError(f"Backend de recomendação desconhecido: {backend!r}")
