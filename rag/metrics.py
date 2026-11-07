"""IR metrics over doc-level rankings. `ranked` is a list of doc ids, `relevant` a set/dict of doc ids."""
import math


def recall_at_k(ranked: list[str], relevant, k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def hit_at_k(ranked: list[str], relevant, k: int) -> float:
    return float(any(d in relevant for d in ranked[:k]))


def mrr_at_k(ranked: list[str], relevant, k: int = 10) -> float:
    for rank, d in enumerate(ranked[:k], start=1):
        if d in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: dict[str, int], k: int = 10) -> float:
    rel = relevant if isinstance(relevant, dict) else {d: 1 for d in relevant}
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0
