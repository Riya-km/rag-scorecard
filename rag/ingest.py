"""Ingestion: download corpus -> chunk -> build BM25 + dense (FAISS) indexes."""
import argparse
import json
import pickle
import time

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import chunk_corpus
from .config import CFG, Config
from .data import download_beir, load_corpus
from .text import chunk_to_passage, tokenize


def ingest(cfg: Config = CFG, batch_size: int = 64) -> None:
    from sentence_transformers import SentenceTransformer

    path = download_beir(cfg.dataset, cfg.data_dir)
    corpus = load_corpus(path)
    chunks = chunk_corpus(corpus, cfg.chunk_words, cfg.chunk_overlap_sentences)
    out = cfg.dataset_index_dir
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(corpus)} docs -> {len(chunks)} chunks")

    with open(out / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    t = time.time()
    passages = [chunk_to_passage(c) for c in chunks]
    bm25 = BM25Okapi([tokenize(p) for p in passages])
    with open(out / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    print(f"BM25 index built in {time.time() - t:.1f}s")

    t = time.time()
    model = SentenceTransformer(cfg.embed_model)
    emb = model.encode(passages, batch_size=batch_size, normalize_embeddings=True,
                       show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(out / "dense.faiss"))
    print(f"Dense index ({cfg.embed_model}, dim={emb.shape[1]}) built in {time.time() - t:.1f}s")

    (out / "meta.json").write_text(json.dumps({
        "dataset": cfg.dataset, "n_docs": len(corpus), "n_chunks": len(chunks),
        "chunk_words": cfg.chunk_words, "embed_model": cfg.embed_model,
    }, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=CFG.dataset, help="BEIR dataset name (e.g. scifact, nfcorpus, fiqa)")
    p.add_argument("--chunk-words", type=int, default=CFG.chunk_words)
    args = p.parse_args()
    cfg = Config(dataset=args.dataset, chunk_words=args.chunk_words)
    ingest(cfg)


if __name__ == "__main__":
    main()
