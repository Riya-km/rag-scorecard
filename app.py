"""Streamlit UI: compare retrieval strategies side by side, generate a grounded answer, view the scorecard."""
import json

import pandas as pd
import streamlit as st

from rag.config import CFG
from rag.data import load_qrels, load_queries
from rag.retrievers import STRATEGIES, IndexStore, build_retrievers, to_doc_ranking

st.set_page_config(page_title="RAG Scorecard", layout="wide")


@st.cache_resource
def load():
    store = IndexStore(CFG)
    retrievers = build_retrievers(store)
    queries, qrels = load_queries(CFG.corpus_dir), load_qrels(CFG.corpus_dir, "test")
    return store, retrievers, queries, qrels


@st.cache_resource
def load_generator():
    from rag.generate import Generator
    return Generator(CFG.gen_model)


@st.cache_resource
def load_judge():
    from rag.faithfulness import NLIFaithfulness
    return NLIFaithfulness(CFG.nli_model)


store, retrievers, queries, qrels = load()
labels = {k: r.name for k, r in retrievers.items()}

st.title("RAG Scorecard")
st.caption(f"Corpus: BEIR/{CFG.dataset} - {len({c['doc_id'] for c in store.chunks})} docs, "
           f"{len(store.chunks)} chunks. Green = chunk from a gold-relevant document.")

tab_search, tab_score = st.tabs(["Search & answer", "Scorecard"])

with tab_search:
    with st.sidebar:
        st.header("Settings")
        chosen = st.multiselect("Strategies to compare", list(STRATEGIES), default=list(STRATEGIES),
                                format_func=labels.get)
        k = st.slider("Top-k chunks", 1, 20, 5)
        test_qids = sorted(qrels, key=int)
        pick = st.selectbox("Example test query (with gold labels)", ["(custom)"] + test_qids,
                            format_func=lambda q: q if q == "(custom)" else f"{q}: {queries[q][:70]}")
    default = "" if pick == "(custom)" else queries[pick]
    query = st.text_input("Question or claim", value=default,
                          placeholder="e.g. Vitamin D deficiency is associated with multiple sclerosis")
    gold = set(qrels.get(pick, {})) if pick != "(custom)" and query == default else set()

    if query and chosen:
        cols = st.columns(len(chosen))
        results = {}
        for col, key in zip(cols, chosen):
            ranking = retrievers[key].search(query, max(k, CFG.candidate_chunks))
            results[key] = ranking[:k]
            docs = [d for d, _ in to_doc_ranking(ranking, store.chunks)]
            with col:
                st.subheader(labels[key])
                if gold:
                    first = next((i for i, d in enumerate(docs, 1) if d in gold), None)
                    st.caption(f"First relevant doc at rank {first or '>100'}")
                for rank, (idx, score) in enumerate(ranking[:k], 1):
                    c = store.chunks[idx]
                    mark = ":green[**relevant**]" if c["doc_id"] in gold else ""
                    with st.expander(f"{rank}. {c['title'][:60]}  ({score:.3f})", expanded=rank <= 2):
                        st.markdown(f"`doc {c['doc_id']}` {mark}")
                        st.write(c["text"])

        st.divider()
        gen_key = st.selectbox("Generate answer from strategy", chosen, format_func=labels.get)
        if st.button("Generate grounded answer"):
            idxs = [i for i, _ in results[gen_key][: CFG.context_chunks]]
            passages = [store.passages[i] for i in idxs]
            with st.spinner("Generating (local CPU model can take ~10-30s)..."):
                answer = load_generator()(query, passages)
            st.markdown(f"**Answer** ({load_generator().model}):\n\n{answer}")
            with st.spinner("Scoring faithfulness with NLI..."):
                res = load_judge()(answer, passages)
            if res.abstained:
                st.info("Model abstained (no faithfulness score).")
            else:
                st.metric("Faithfulness", f"{res.score:.0%}")
                for s in res.sentences:
                    icon = ":green[supported]" if s.supported else ":red[unsupported]"
                    st.markdown(f"- {icon} ({s.entailment:.2f}) {s.sentence}")

with tab_score:
    csv = CFG.results_dir / "results.csv"
    if not csv.exists():
        st.warning("No results yet. Run `python scripts/eval.py`.")
    else:
        df = pd.read_csv(csv)
        meta = json.loads((CFG.results_dir / "run.json").read_text())
        st.markdown((CFG.results_dir / "results.md").read_text().split("\n\n")[0])
        st.dataframe(df.style.format(precision=3).highlight_max(
            subset=[c for c in df.columns if c not in ("strategy", "Latency ms/q", "Abstain rate")],
            color="#c8f7c5"), use_container_width=True, hide_index=True)
        recall_cols = [c for c in df.columns if c.startswith("Recall@")]
        st.subheader("Recall@k curve")
        st.line_chart(df.set_index("strategy")[recall_cols].T)
        st.subheader("Headline metrics")
        head = [c for c in ["MRR@10", "nDCG@10", "Faithfulness"] if c in df.columns]
        st.bar_chart(df.set_index("strategy")[head])
        detail = CFG.results_dir / "faithfulness_detail.jsonl"
        if detail.exists():
            st.subheader("Generated answers (faithfulness sample)")
            rows = [json.loads(line) for line in detail.read_text().splitlines()]
            st.dataframe(pd.DataFrame(rows)[["strategy", "qid", "query", "answer", "faithfulness"]],
                         use_container_width=True, hide_index=True)
