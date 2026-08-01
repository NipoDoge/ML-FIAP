"""Métricas de ranking para recomendação (TC02)."""

from __future__ import annotations

import math


def _dcg(relevances: list[float]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    rec_k = recommended[:k]
    gains = [1.0 if item in relevant else 0.0 for item in rec_k]
    dcg = _dcg(gains)
    ideal = _dcg([1.0] * min(len(relevant), k))
    if ideal == 0:
        return 0.0
    return dcg / ideal


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    rec_k = recommended[:k]
    if not rec_k:
        return 0.0
    hits = sum(1 for item in rec_k if item in relevant)
    return hits / len(rec_k)


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for item in rec_k if item in relevant)
    return hits / len(relevant)


def hit_rate_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    rec_k = recommended[:k]
    return 1.0 if any(item in relevant for item in rec_k) else 0.0


def average_precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    rec_k = recommended[:k]
    hits = 0
    sum_prec = 0.0
    for i, item in enumerate(rec_k, start=1):
        if item in relevant:
            hits += 1
            sum_prec += hits / i
    return sum_prec / min(len(relevant), k)


def aggregate_ranking_metrics(
    recommendations: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    k: int,
) -> dict[str, float]:
    users = [u for u in recommendations if ground_truth.get(u)]
    if not users:
        return {
            "hit_rate": 0.0,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "map_at_k": 0.0,
        }
    hit = prec = rec = ndcg = map_k = 0.0
    for user in users:
        recs = recommendations[user]
        rel = ground_truth[user]
        hit += hit_rate_at_k(recs, rel, k)
        prec += precision_at_k(recs, rel, k)
        rec += recall_at_k(recs, rel, k)
        ndcg += ndcg_at_k(recs, rel, k)
        map_k += average_precision_at_k(recs, rel, k)
    n = float(len(users))
    return {
        "hit_rate": hit / n,
        "precision_at_k": prec / n,
        "recall_at_k": rec / n,
        "ndcg_at_k": ndcg / n,
        "map_at_k": map_k / n,
    }
