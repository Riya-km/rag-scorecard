"""Scorecard: retrieval metrics for every strategy + NLI faithfulness of generated answers.

Writes results/results.{csv,md}, results/per_query.jsonl, results/generations.jsonl and
refreshes the results table in README.md.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import CFG, ROOT, Config  # noqa: E402
from rag.data import load_qrels, load_queries  # noqa: E402
from rag.metrics import mrr_at_k, ndcg_at_k, recall_at_k  # noqa: E402
from rag.retrievers import STRATEGIES, IndexStore, build_retrievers, to_doc_ranking  # noqa: E402

README_START, README_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def evaluate_retrieval(store, retrievers, queries, qrels, ks):
    qids = sorted(qrels)
    texts = [queries[q] for q in qids]
    rows, per_query = [], {}
    for key, r in retrievers.items():
        t = time.perf_counter()
        rankings = r.search_many(texts, max(max(ks), CFG.candidate_chunks))
        ms = (time.perf_counter() - t) * 1000 / len(qids)
        row = {"strategy": r.name, "key": key}
        accum = {f"Recall@{k}": [] for k in ks} | {"MRR@10": [], "nDCG@10": []}
        for qid, ranking in zip(qids, rankings):
            docs = [d for d, _ in to_doc_ranking(ranking, store.chunks)]
            rel = qrels[qid]
            m = {f"Recall@{k}": recall_at_k(docs, rel, k) for k in ks}
            m["MRR@10"] = mrr_at_k(docs, rel, 10)
            m["nDCG@10"] = ndcg_at_k(docs, rel, 10)
            for name, v in m.items():
                accum[name].append(v)
            per_query.setdefault(qid, {"query": queries[qid], "relevant": list(rel)})[key] = {
                "top10": docs[:10], **m}
        row |= {k: sum(v) / len(v) for k, v in accum.items()}
        row["Latency ms/q"] = ms
        rows.append(row)
        print(f"{r.name:<18} nDCG@10={row['nDCG@10']:.3f} MRR@10={row['MRR@10']:.3f} "
              f"R@10={row['Recall@10']:.3f} ({ms:.0f} ms/q)")
    return rows, per_query


def evaluate_faithfulness(cfg, store, retrievers, queries, qrels, n, seed, gen_model, cache_path):
    from rag.faithfulness import NLIFaithfulness
    from rag.generate import Generator

    qids = sorted(qrels)
    random.Random(seed).shuffle(qids)
    qids = qids[:n]
    gen = Generator(gen_model)
    judge = NLIFaithfulness(cfg.nli_model)

    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            d = json.loads(line)
            cache[(d["model"], d["strategy"], d["qid"])] = d
    out = {}
    with open(cache_path, "a") as fh:
        for key, r in retrievers.items():
            scores, abstains, ctx_prec = [], 0, []
            for qid in tqdm(qids, desc=f"faithfulness/{key}"):
                ranking = r.search(queries[qid], cfg.candidate_chunks)[: cfg.context_chunks]
                idxs = [i for i, _ in ranking]
                passages = [store.passages[i] for i in idxs]
                ctx_prec.append(sum(store.chunks[i]["doc_id"] in qrels[qid] for i in idxs) / len(idxs))
                ck = (gen.model, key, qid)
                if ck not in cache or cache[ck]["chunks"] != idxs:
                    answer = gen(queries[qid], passages)
                    cache[ck] = {"model": gen.model, "strategy": key, "qid": qid, "chunks": idxs, "answer": answer}
                    fh.write(json.dumps(cache[ck]) + "\n")
                    fh.flush()
                res = judge(cache[ck]["answer"], passages)
                cache[ck]["faithfulness"] = res.score
                cache[ck]["sentences"] = [s.__dict__ for s in res.sentences]
                if res.abstained:
                    abstains += 1
                else:
                    scores.append(res.score)
            out[key] = {
                "Faithfulness": sum(scores) / len(scores) if scores else float("nan"),
                "Abstain rate": abstains / len(qids),
                f"Context precision@{cfg.context_chunks}": sum(ctx_prec) / len(ctx_prec),
            }
            print(f"{r.name:<18} faithfulness={out[key]['Faithfulness']:.3f} abstain={out[key]['Abstain rate']:.2f}")
    return out, gen.model, {k: v for k, v in cache.items() if k[0] == gen.model and k[2] in qids}


def to_markdown(df: pd.DataFrame, meta: dict) -> str:
    fmt = df.copy()
    for c in fmt.columns:
        if c == "Latency ms/q":
            fmt[c] = fmt[c].map(lambda v: f"{v:.0f}")
        elif fmt[c].dtype.kind == "f":
            fmt[c] = fmt[c].map(lambda v: "-" if pd.isna(v) else f"{v:.3f}")
    for c in df.columns:
        if df[c].dtype.kind != "f" or c in ("Latency ms/q", "Abstain rate") or df[c].isna().all():
            continue
        for i in df.index[df[c].round(3) == round(df[c].max(), 3)]:
            fmt.loc[i, c] = f"**{fmt.loc[i, c]}**"
    lines = [
        f"Dataset: **{meta['dataset']}** ({meta['n_docs']} docs, {meta['n_chunks']} chunks), "
        f"{meta['n_queries']} test queries. Embeddings: `{meta['embed_model']}`, reranker: `{meta['rerank_model']}`.",
    ]
    if meta.get("gen_model"):
        lines.append(f"Faithfulness on {meta['faith_n']} sampled queries, top-{meta['context_chunks']} chunks as context, "
                     f"generator `{meta['gen_model']}`, NLI judge `{meta['nli_model']}`.")
    lines += ["", fmt.to_markdown(index=False)]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=CFG.dataset)
    p.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=list(STRATEGIES))
    p.add_argument("--faithfulness-queries", type=int, default=5, help="0 to skip generation")
    p.add_argument("--gen-model", default=CFG.gen_model)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--faithfulness-only", action="store_true",
                   help="reuse retrieval rows from an existing results/results.csv")
    args = p.parse_args()

    cfg = Config(dataset=args.dataset)
    store = IndexStore(cfg)
    queries, qrels = load_queries(cfg.corpus_dir), load_qrels(cfg.corpus_dir, "test")
    retrievers = build_retrievers(store, args.strategies)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    if args.faithfulness_only:
        prev = pd.read_csv(cfg.results_dir / "results.csv")
        names = {r.name: k for k, r in retrievers.items()}
        prev = prev[prev["strategy"].isin(names)]
        df = prev.assign(key=prev["strategy"].map(names)).set_index("key")
        df = df[[c for c in df.columns if c.startswith(("strategy", "Recall@", "MRR", "nDCG", "Latency"))]]
        per_query = None
    else:
        rows, per_query = evaluate_retrieval(store, retrievers, queries, qrels, cfg.ks)
        df = pd.DataFrame(rows).set_index("key")
    meta = json.loads((store.dir / "meta.json").read_text()) | {
        "n_queries": len(qrels), "rerank_model": cfg.rerank_model, "nli_model": cfg.nli_model,
        "context_chunks": cfg.context_chunks}

    if args.faithfulness_queries > 0:
        faith, gen_model, gens = evaluate_faithfulness(
            cfg, store, retrievers, queries, qrels, args.faithfulness_queries, args.seed, args.gen_model,
            cfg.results_dir / "generations.jsonl")
        for key, m in faith.items():
            for c, v in m.items():
                df.loc[key, c] = v
        meta |= {"gen_model": gen_model, "faith_n": args.faithfulness_queries}
        with open(cfg.results_dir / "faithfulness_detail.jsonl", "w") as f:
            for (_, key, qid), d in sorted(gens.items()):
                f.write(json.dumps({**d, "query": queries[qid]}) + "\n")

    cols = ["strategy"] + [c for c in df.columns if c != "strategy"]
    df = df[cols].reset_index(drop=True)
    df.to_csv(cfg.results_dir / "results.csv", index=False)
    md = to_markdown(df, meta)
    (cfg.results_dir / "results.md").write_text(md + "\n")
    (cfg.results_dir / "run.json").write_text(json.dumps(meta, indent=2))
    if per_query is not None:
        with open(cfg.results_dir / "per_query.jsonl", "w") as f:
            for qid, d in per_query.items():
                f.write(json.dumps({"qid": qid, **d}) + "\n")

    readme = ROOT / "README.md"
    if readme.exists() and README_START in readme.read_text():
        text = readme.read_text()
        head, rest = text.split(README_START, 1)
        _, tail = rest.split(README_END, 1)
        readme.write_text(f"{head}{README_START}\n{md}\n{README_END}{tail}")
    print("\n" + md)


if __name__ == "__main__":
    main()
