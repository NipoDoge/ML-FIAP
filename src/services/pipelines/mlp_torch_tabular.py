"""
MLP binário tabular em PyTorch (MVP) — uso isolado do restante do pipeline.

**Onde chamar:** preferir :class:`FeatureEngineering` *depois* de ``select_features()``,
com matrizes obtidas pelo *mesmo* ``ColumnTransformer`` que o FE usa nos modelos sklearn
(sem ``SelectKBest``/modelo), para o MLP ver os dados alinhados à etapa de FE.

Exemplo (dentro do FE, após ``select_features``)::

    from sklearn.model_selection import train_test_split
    from services.pipelines.mlp_torch_tabular import train_eval_mlp_binary_tabular

    pre = self._build_preprocess_transformer(self.x_train, groups=self.feature_groups)
    X_all = pre.fit_transform(self.x_train)
    y_all = self.y_train.to_numpy(dtype=np.int64)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=0.15, random_state=self.random_state, stratify=y_all
    )
    X_te = pre.transform(self.x_test)
    y_te = self.y_test.to_numpy(dtype=np.int64)

    mlp_result = train_eval_mlp_binary_tabular(
        X_tr, X_val, X_te, y_tr, y_val, y_te, random_state=self.random_state
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TorchTabularMLPResult:
    """Saída do treino/avaliação do MLP."""

    metrics_test: dict[str, float]
    metrics_val: dict[str, float]
    best_epoch: int
    best_val_loss: float
    state_dict: dict[str, torch.Tensor]


class _MLPBinary(nn.Module):
    """Perceptrão multicamadas para classificação binária (logit único + BCEWithLogitsLoss)."""

    def __init__(self, n_features: int, hidden_dims: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims não pode ser vazio.")
        dims = [n_features, *hidden_dims]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_eval_mlp_binary_tabular(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    *,
    random_state: int = 42,
    hidden_dims: tuple[int, ...] = (64, 32),
    dropout: float = 0.0,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    max_epochs: int = 300,
    early_stopping_patience: int = 20,
    device: str | None = None,
) -> TorchTabularMLPResult:
    """
    Treina um MLP com early stopping pela loss de validação e avalia no teste.

    Parâmetros
    ----------
    X_train, X_val, X_test
        Matrizes numéricas 2D (já pré-processadas, ex.: saída de ``ColumnTransformer``).
        Tipicamente ``float64``; são convertidas para ``float32`` internamente.
    y_train, y_val, y_test
        Alvo binário ``0``/``1`` (array 1D).
    random_state
        Semente para reprodutibilidade (shuffle do DataLoader e operações torch).
    hidden_dims
        Tamanhos das camadas ocultas (ex. ``(64, 32)``).
    dropout
        Dropout após cada ReLU (exceto antes da camada de saída).
    batch_size
        Tamanho do batch; é reduzido automaticamente se o treino for menor que o batch.
    lr, weight_decay
        Hiperparâmetros do AdamW.
    max_epochs
        Limite superior de épocas.
    early_stopping_patience
        Parar se a loss de validação não melhorar neste número de épocas consecutivas.
    device
        ``"cuda"``, ``"cpu"`` ou ``None`` (auto: CUDA se disponível).
    """
    _validate_xy(X_train, y_train, "train")
    _validate_xy(X_val, y_val, "val")
    _validate_xy(X_test, y_test, "test")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(random_state)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(random_state)

    gen = torch.Generator()
    gen.manual_seed(random_state)

    n_features = int(X_train.shape[1])
    model = _MLPBinary(n_features, hidden_dims, dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    X_train_t = torch.as_tensor(X_train, dtype=torch.float32, device=dev)
    y_train_t = torch.as_tensor(y_train, dtype=torch.float32, device=dev)
    X_val_t = torch.as_tensor(X_val, dtype=torch.float32, device=dev)
    y_val_t = torch.as_tensor(y_val, dtype=torch.float32, device=dev)

    bs = max(1, min(batch_size, len(X_train_t)))
    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=bs,
        shuffle=True,
        generator=gen,
        drop_last=False,
    )

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    patience_left = early_stopping_patience

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            v_logits = model(X_val_t)
            v_loss = float(loss_fn(v_logits, y_val_t).item())

        if v_loss < best_val - 1e-6:
            best_val = v_loss
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = early_stopping_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info(
                    "MLP early stopping na época %s (melhor val_loss na época %s).",
                    epoch + 1,
                    best_epoch,
                )
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = max_epochs

    model.load_state_dict(best_state)
    model.to(dev)
    model.eval()

    X_test_t = torch.as_tensor(X_test, dtype=torch.float32, device=dev)
    y_test_np = np.asarray(y_test).astype(np.int64).ravel()
    y_val_np = np.asarray(y_val).astype(np.int64).ravel()

    with torch.no_grad():
        val_logits = model(X_val_t).detach().cpu().numpy()
        test_logits = model(X_test_t).detach().cpu().numpy()

    val_proba = 1.0 / (1.0 + np.exp(-val_logits))
    test_proba = 1.0 / (1.0 + np.exp(-test_logits))
    val_pred = (val_proba >= 0.5).astype(np.int64)
    test_pred = (test_proba >= 0.5).astype(np.int64)

    metrics_val = _classification_metrics(y_val_np, val_pred, val_proba)
    metrics_test = _classification_metrics(y_test_np, test_pred, test_proba)

    return TorchTabularMLPResult(
        metrics_test=metrics_test,
        metrics_val=metrics_val,
        best_epoch=best_epoch,
        best_val_loss=float(best_val),
        state_dict=best_state,
    )


def _validate_xy(X: np.ndarray, y: np.ndarray, name: str) -> None:
    if X.ndim != 2:
        raise ValueError(f"X_{name} deve ser 2D; recebido shape {X.shape}.")
    yv = np.asarray(y).astype(np.int64).ravel()
    if len(yv) != X.shape[0]:
        raise ValueError(
            f"y_{name} length ({len(yv)}) não bate com X_{name}.shape[0] ({X.shape[0]})."
        )
    if set(np.unique(yv)) - {0, 1}:
        raise ValueError(f"y_{name} deve conter apenas 0 e 1.")


def _classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, float]:
    zd = {"zero_division": 0}
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, **zd)),
        "recall": float(recall_score(y_true, y_pred, **zd)),
        "f1": float(f1_score(y_true, y_pred, **zd)),
    }
    if len(np.unique(y_true)) >= 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    else:
        out["roc_auc"] = float("nan")
    return out
