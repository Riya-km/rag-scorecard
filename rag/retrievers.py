"""Retrieval strategies. All return chunk-level rankings: list of (chunk_idx, score), best first."""
import json
import pickle
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .config import CFG, Config
from .text import chunk_to_passage, tokenize

Ranking = list[tuple[int, float]]


class IndexStore:
    """Lazily loads chunk store, indexes and models built by `rag.ingest`."""

    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self.dir = cfg.dataset_index_dir
        if not (self.dir / "chunks.jsonl").exists():
            raise FileNotFoundError(f"No index at {self.dir}. Run `python -m rag.ingest` first.")

    @cached_property
    def chunks(self) -> list[dict]:
        with open(self.dir / "chunks.jsonl") as f:
            return [json.loads(line) for line in f]

    @cached_property
    def passages(self) -> list[str]:
        return [chunk_to_passage(c) for c in self.chunks]

    @cached_property
    def bm25(self):
        with open(self.dir / "bm25.pkl", "rb") as f:
            return pickle.load(f)

    @cached_property
    def faiss_index(self):
        import faiss
        return faiss.read_index(str(self.dir / "dense.faiss"))

    @cached_property
    def embedder(self):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(self.cfg.embed_model)

    @cached_property
    def reranker(self):
        from sentence_transformers import CrossEncoder
        return CrossEncoder(self.cfg.rerank_model, max_length=512)


def _top_n(scores: np.ndarray, n: int) -> Ranking:
    n = min(n, len(scores))
    idx = np.argpartition(-scores, n - 1)[:n]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]


@dataclass
class Retriever:
    store: IndexStore
    name: str = "base"

    def search(self, query: str, n: int) -> Ranking:
        return self.search_many([query], n)[0]

    def search_many(self, queries: list[str], n: int) -> list[Ranking]:
        raise NotImplementedError


class BM25Retriever(Retriever):
    def __init__(self, store: IndexStore):
        super().__init__(store, "BM25")

    def search_many(self, queries, n):
        return [_top_n(np.asarray(self.store.bm25.get_scores(tokenize(q))), n) for q in queries]


class DenseRetriever(Retriever):
    def __init__(self, store: IndexStore):
        super().__init__(store, "Dense")

    def search_many(self, queries, n):
        cfg = self.store.cfg
        q = self.store.embedder.encode([cfg.embed_query_prefix + x for x in queries],
                                       normalize_embeddings=True, convert_to_numpy=True,
                                       batch_size=64).astype(np.float32)
        scores, ids = self.store.faiss_index.search(q, n)
        return [[(int(i), float(s)) for i, s in zip(row_i, row_s) if i >= 0]
                for row_i, row_s in zip(ids, scores)]


def rrf_fuse(rankings: list[Ranking], k: int = 60) -> Ranking:
    """Reciprocal Rank Fusion: score(d) = sum_r 1 / (k + rank_r(d))."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (idx, _) in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])


class HybridRetriever(Retriever):
    """BM25 + dense fused with RRF."""

    def __init__(self, store: IndexStore):
        super().__init__(store, "Hybrid (RRF)")
        self.bm25 = BM25Retriever(store)
        self.dense = DenseRetriever(store)

    def search_many(self, queries, n):
        depth = max(n, self.store.cfg.candidate_chunks)
        b = self.bm25.search_many(queries, depth)
        d = self.dense.search_many(queries, depth)
        return [rrf_fuse([x, y], self.store.cfg.rrf_k)[:n] for x, y in zip(b, d)]


class HybridRerankRetriever(Retriever):
    """Hybrid RRF candidates, top `rerank_depth` re-scored by a cross-encoder."""

    def __init__(self, store: IndexStore):
        super().__init__(store, "Hybrid + Rerank")
        self.hybrid = HybridRetriever(store)

    def search_many(self, queries, n):
        depth = self.store.cfg.rerank_depth
        out = []
        for q, cands in zip(queries, self.hybrid.search_many(queries, max(n, depth))):
            head, tail = cands[:depth], cands[depth:]
            scores = self.store.reranker.predict([(q, self.store.passages[i]) for i, _ in head],
                                                 batch_size=32, show_progress_bar=False)
            reranked = sorted(((i, float(s)) for (i, _), s in zip(head, scores)), key=lambda x: -x[1])
            floor = reranked[-1][1] if reranked else 0.0
            out.append((reranked + [(i, floor - 1 - r) for r, (i, _) in enumerate(tail)])[:n])
        return out


STRATEGIES = {
    "bm25": BM25Retriever,
    "dense": DenseRetriever,
    "hybrid": HybridRetriever,
    "hybrid_rerank": HybridRerankRetriever,
}


def build_retrievers(store: IndexStore, names=None) -> dict[str, Retriever]:
    return {k: STRATEGIES[k](store) for k in (names or STRATEGIES)}


def to_doc_ranking(ranking: Ranking, chunks: list[dict]) -> list[tuple[str, float]]:
    """Collapse chunk ranking into doc ranking (max-pool: first chunk hit wins)."""
    seen, docs = set(), []
    for idx, score in ranking:
        did = chunks[idx]["doc_id"]
        if did not in seen:
            seen.add(did)
            docs.append((did, score))
    return docs
