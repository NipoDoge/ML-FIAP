"""Modelo de recomendação embedding-based em PyTorch (TC02)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _EmbeddingDotModel(nn.Module):
    """Arquitetura legada mantida para carregar artefatos já treinados."""

    def __init__(self, n_users: int, n_items: int, dim: int) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        return (self.user_emb(user_idx) * self.item_emb(item_idx)).sum(dim=1)


class _EmbeddingMlpModel(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        dim: int,
        hidden_dim: int,
        dropout: float,
        use_item_bias: bool = False,
    ) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.item_bias = nn.Embedding(n_items, 1) if use_item_bias else None
        self.scorer = nn.Sequential(
            nn.Linear(dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        if self.item_bias is not None:
            nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        user_vec = self.user_emb(user_idx)
        item_vec = self.item_emb(item_idx)
        features = torch.cat([user_vec, item_vec, user_vec * item_vec], dim=1)
        score = self.scorer(features).squeeze(-1)
        if self.item_bias is not None:
            score = score + self.item_bias(item_idx).squeeze(-1)
        return score


class TorchEmbeddingRecommender:
    name = "torch_embedding"

    def __init__(
        self,
        embedding_dim: int = 32,
        n_epochs: int = 10,
        batch_size: int = 512,
        lr: float = 0.01,
        random_state: int = 42,
        n_negatives: int = 4,
        validation_fraction: float = 0.1,
        early_stopping_patience: int = 3,
        early_stopping_min_delta: float = 1e-4,
        min_positive_rating: float = 0.0,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        use_item_bias: bool = True,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        self.n_negatives = max(1, n_negatives)
        self.validation_fraction = min(max(validation_fraction, 0.0), 0.5)
        self.early_stopping_patience = max(0, early_stopping_patience)
        self.early_stopping_min_delta = early_stopping_min_delta
        self.min_positive_rating = min_positive_rating
        self.hidden_dim = hidden_dim
        self.dropout = min(max(dropout, 0.0), 0.8)
        self.use_item_bias = use_item_bias
        self._user_map: dict[int, int] = {}
        self._item_map: dict[int, int] = {}
        self._reverse_items: dict[int, int] = {}
        self._model: nn.Module | None = None
        self._user_items: dict[int, set[int]] = {}
        self._user_seen_indices: dict[int, set[int]] = {}
        self._popular_items: list[int] = []
        self.training_summary: dict[str, float | int] = {}

    def fit(self, train_df: pd.DataFrame) -> None:
        if train_df.empty:
            raise ValueError("train_df vazio: não há interações para treinar.")
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        self._index_catalog(train_df)
        positive_pairs = self._positive_pairs(train_df)
        explicit_negative_pairs = self._explicit_negative_pairs(train_df)
        train_pairs, val_pairs = self._split_train_validation(positive_pairs, rng)

        model = _EmbeddingMlpModel(
            len(self._user_map),
            len(self._item_map),
            self.embedding_dim,
            self.hidden_dim,
            self.dropout,
            self.use_item_bias,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()
        val_tensors = self._build_labeled_tensors(
            val_pairs,
            np.random.default_rng(self.random_state + 1),
        )

        best_state = copy.deepcopy(model.state_dict())
        best_val_loss = float("inf")
        best_epoch = 0
        epochs_without_gain = 0

        for epoch in range(1, self.n_epochs + 1):
            train_tensors = self._build_labeled_tensors(train_pairs, rng, explicit_negative_pairs)
            train_loss = self._train_epoch(model, optimizer, loss_fn, train_tensors, epoch)
            val_loss = self._validation_loss(model, loss_fn, val_tensors) or train_loss

            if val_loss < best_val_loss - self.early_stopping_min_delta:
                best_state = copy.deepcopy(model.state_dict())
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_gain = 0
            else:
                epochs_without_gain += 1

            if self.early_stopping_patience and epochs_without_gain >= self.early_stopping_patience:
                break

        model.load_state_dict(best_state)
        self._model = model
        self.training_summary = {
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss),
            "n_positive_interactions": int(len(positive_pairs)),
            "n_explicit_negative_interactions": int(len(explicit_negative_pairs)),
            "n_train_interactions": int(len(train_pairs)),
            "n_validation_interactions": int(len(val_pairs)),
        }

    def _index_catalog(self, train_df: pd.DataFrame) -> None:
        users = sorted(train_df["user_id"].astype(int).unique().tolist())
        items = sorted(train_df["item_id"].astype(int).unique().tolist())
        self._user_map = {user_id: i for i, user_id in enumerate(users)}
        self._item_map = {item_id: i for i, item_id in enumerate(items)}
        self._reverse_items = {i: item_id for item_id, i in self._item_map.items()}
        self._user_items = train_df.groupby("user_id")["item_id"].apply(lambda s: set(map(int, s))).to_dict()
        self._rebuild_seen_indices()
        counts = train_df.groupby("item_id").size().sort_values(ascending=False)
        self._popular_items = [int(i) for i in counts.index.tolist()]

    def _positive_pairs(self, train_df: pd.DataFrame) -> np.ndarray:
        filtered = train_df[train_df["rating"].astype(float) >= self.min_positive_rating]
        if filtered.empty:
            raise ValueError(
                f"Nenhuma interação positiva encontrada com rating >= {self.min_positive_rating}."
            )
        return self._pairs_from_frame(filtered)

    def _explicit_negative_pairs(self, train_df: pd.DataFrame) -> np.ndarray:
        filtered = train_df[train_df["rating"].astype(float) < self.min_positive_rating]
        if filtered.empty:
            return np.empty((0, 2), dtype=np.int64)
        return self._pairs_from_frame(filtered)

    def _pairs_from_frame(self, df: pd.DataFrame) -> np.ndarray:
        user_idx = df["user_id"].astype(int).map(self._user_map).to_numpy(dtype=np.int64)
        item_idx = df["item_id"].astype(int).map(self._item_map).to_numpy(dtype=np.int64)
        return np.column_stack([user_idx, item_idx])

    def _split_train_validation(
        self,
        positive_pairs: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(positive_pairs) < 2 or self.validation_fraction == 0:
            return positive_pairs, np.empty((0, 2), dtype=np.int64)
        order = rng.permutation(len(positive_pairs))
        val_size = int(len(order) * self.validation_fraction)
        val_size = min(max(val_size, 1), len(order) - 1)
        val_idx = order[:val_size]
        train_idx = order[val_size:]
        return positive_pairs[train_idx], positive_pairs[val_idx]

    def _build_labeled_tensors(
        self,
        positive_pairs: np.ndarray,
        rng: np.random.Generator,
        explicit_negative_pairs: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if len(positive_pairs) == 0:
            return None

        pos_users = positive_pairs[:, 0].astype(np.int64)
        pos_items = positive_pairs[:, 1].astype(np.int64)
        neg_users = np.repeat(pos_users, self.n_negatives)
        neg_items = np.array(
            [self._sample_negative_item(int(user_idx), rng) for user_idx in neg_users],
            dtype=np.int64,
        )
        if explicit_negative_pairs is not None and len(explicit_negative_pairs):
            neg_users = np.concatenate([neg_users, explicit_negative_pairs[:, 0].astype(np.int64)])
            neg_items = np.concatenate([neg_items, explicit_negative_pairs[:, 1].astype(np.int64)])
        users = np.concatenate([pos_users, neg_users])
        items = np.concatenate([pos_items, neg_items])
        labels = np.concatenate(
            [
                np.ones(len(pos_users), dtype=np.float32),
                np.zeros(len(neg_users), dtype=np.float32),
            ]
        )
        return (
            torch.from_numpy(users),
            torch.from_numpy(items),
            torch.from_numpy(labels),
        )

    def _sample_negative_item(self, user_idx: int, rng: np.random.Generator) -> int:
        n_items = len(self._item_map)
        seen = self._user_seen_indices.get(user_idx, set())
        if len(seen) >= n_items:
            return int(rng.integers(0, n_items))
        for _ in range(50):
            item_idx = int(rng.integers(0, n_items))
            if item_idx not in seen:
                return item_idx
        candidates = [idx for idx in range(n_items) if idx not in seen]
        return int(rng.choice(candidates))

    def _train_epoch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        epoch: int,
    ) -> float:
        if tensors is None:
            return 0.0
        dataset = TensorDataset(*tensors)
        generator = torch.Generator().manual_seed(self.random_state + epoch)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, generator=generator)
        total_loss = 0.0
        model.train()
        for batch_u, batch_i, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_u, batch_i)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_y)
        return total_loss / max(len(dataset), 1)

    def _validation_loss(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ) -> float | None:
        if tensors is None:
            return None
        model.eval()
        with torch.no_grad():
            logits = model(tensors[0], tensors[1])
            return float(loss_fn(logits, tensors[2]).item())

    def recommend(self, user_id: int, n_items: int, exclude_items: set[int] | None = None) -> list[int]:
        exclude = exclude_items or set()
        seen = self._user_items.get(user_id, set()) | exclude
        if self._model is None or user_id not in self._user_map:
            pop = PopularityFallback(self._popular_items)
            return pop.recommend(user_id, n_items, exclude_items=seen)

        self._model.eval()
        user_idx = self._user_map[user_id]
        item_indices = torch.arange(len(self._item_map), dtype=torch.long)
        user_tensor = torch.full((len(item_indices),), user_idx, dtype=torch.long)
        with torch.no_grad():
            scores = self._model(user_tensor, item_indices).numpy()
        order = np.argsort(-scores)
        out: list[int] = []
        for idx in order:
            item = self._reverse_items[int(idx)]
            if item in seen:
                continue
            out.append(item)
            if len(out) >= n_items:
                break
        return out

    def save(self, prefix: Path) -> None:
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        if self._model is None:
            raise RuntimeError("Modelo torch não treinado.")
        torch.save(self._model.state_dict(), prefix.with_suffix(".pt"))
        meta = {
            "model_type": "embedding_mlp_bce_item_bias" if self.use_item_bias else "embedding_mlp_bce",
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "use_item_bias": self.use_item_bias,
            "n_negatives": self.n_negatives,
            "validation_fraction": self.validation_fraction,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "min_positive_rating": self.min_positive_rating,
            "user_map": self._user_map,
            "item_map": self._item_map,
            "user_items": {str(user): sorted(items) for user, items in self._user_items.items()},
            "popular_items": self._popular_items,
            "training_summary": self.training_summary,
        }
        prefix.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, prefix: Path) -> TorchEmbeddingRecommender:
        prefix = Path(prefix)
        meta = json.loads(prefix.with_suffix(".meta.json").read_text(encoding="utf-8"))
        obj = cls(
            embedding_dim=int(meta["embedding_dim"]),
            hidden_dim=int(meta.get("hidden_dim", 64)),
            dropout=float(meta.get("dropout", 0.1)),
            n_negatives=int(meta.get("n_negatives", 4)),
            validation_fraction=float(meta.get("validation_fraction", 0.1)),
            early_stopping_patience=int(meta.get("early_stopping_patience", 3)),
            early_stopping_min_delta=float(meta.get("early_stopping_min_delta", 1e-4)),
            min_positive_rating=float(meta.get("min_positive_rating", 0.0)),
            use_item_bias=bool(meta.get("use_item_bias", meta.get("model_type") == "embedding_mlp_bce_item_bias")),
        )
        obj._user_map = {int(k): int(v) for k, v in meta["user_map"].items()}
        obj._item_map = {int(k): int(v) for k, v in meta["item_map"].items()}
        obj._reverse_items = {i: it for it, i in obj._item_map.items()}
        obj._popular_items = [int(x) for x in meta["popular_items"]]
        obj._user_items = {
            int(user): set(map(int, items))
            for user, items in (meta.get("user_items") or {}).items()
        }
        obj._rebuild_seen_indices()
        obj.training_summary = dict(meta.get("training_summary") or {})
        model = obj._build_model_from_meta(meta)
        model.load_state_dict(torch.load(prefix.with_suffix(".pt"), map_location="cpu"))
        model.eval()
        obj._model = model
        return obj

    def _build_model_from_meta(self, meta: dict) -> nn.Module:
        if meta.get("model_type") in {
            "embedding_mlp_bce",
            "embedding_mlp_ranking",
            "embedding_mlp_bce_item_bias",
        }:
            return _EmbeddingMlpModel(
                len(self._user_map),
                len(self._item_map),
                self.embedding_dim,
                self.hidden_dim,
                self.dropout,
                self.use_item_bias,
            )
        return _EmbeddingDotModel(len(self._user_map), len(self._item_map), self.embedding_dim)

    def _rebuild_seen_indices(self) -> None:
        self._user_seen_indices = {
            self._user_map[user_id]: {
                self._item_map[item_id]
                for item_id in items
                if item_id in self._item_map
            }
            for user_id, items in self._user_items.items()
            if user_id in self._user_map
        }


class PopularityFallback:
    def __init__(self, popular_items: list[int]) -> None:
        self._popular_items = popular_items

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
