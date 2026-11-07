# RAG Scorecard

A retrieval-augmented generation system that is **measured**, not just demoed. It compares
three retrieval strategies (plus an ablation) on a real labelled corpus and reports
recall@k, MRR, nDCG and answer faithfulness in a results table.

| Strategy | How it works |
|---|---|
| **BM25** | Lexical retrieval (`rank-bm25`, Okapi BM25, stopword-filtered tokens) |
| **Dense** | `BAAI/bge-small-en-v1.5` embeddings, exact inner-product search in FAISS |
| **Hybrid (RRF)** | BM25 top-100 + dense top-100 fused with Reciprocal Rank Fusion (k=60) - ablation |
| **Hybrid + Rerank** | Hybrid candidates, top-50 re-scored by cross-encoder `ms-marco-MiniLM-L-6-v2` |

## Results

<!-- RESULTS:START -->
Dataset: **scifact** (5183 docs, 13192 chunks), 300 test queries. Embeddings: `BAAI/bge-small-en-v1.5`, reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
Faithfulness on 5 sampled queries, top-4 chunks as context, generator `Qwen/Qwen2.5-1.5B-Instruct`, NLI judge `cross-encoder/nli-deberta-v3-base`.

| strategy        | Recall@1   | Recall@5   | Recall@10   | Recall@20   | Recall@100   | MRR@10    | nDCG@10   |   Latency ms/q | Faithfulness   |   Abstain rate | Context precision@4   |
|:----------------|:-----------|:-----------|:------------|:------------|:-------------|:----------|:----------|---------------:|:---------------|---------------:|:----------------------|
| BM25            | 0.504      | 0.725      | 0.788       | 0.828       | 0.871        | 0.618     | 0.655     |             20 | 0.100          |              0 | 0.150                 |
| Dense           | **0.579**  | 0.782      | 0.833       | 0.878       | 0.935        | 0.686     | 0.716     |             32 | **0.433**      |              0 | **0.300**             |
| Hybrid (RRF)    | 0.557      | 0.780      | 0.841       | 0.884       | **0.952**    | 0.679     | 0.712     |             45 | 0.133          |              0 | **0.300**             |
| Hybrid + Rerank | **0.579**  | **0.785**  | **0.853**   | **0.910**   | **0.952**    | **0.696** | **0.725** |           1795 | 0.200          |              0 | 0.250                 |
<!-- RESULTS:END -->

**Reading the table**

- Retrieval metrics are document-level over all 300 SciFact test queries (chunks are max-pooled to their parent doc).
- **Faithfulness** = share of answer evidence sentences entailed by the retrieved context (NLI), averaged over non-abstaining answers. It measures groundedness, not correctness: an answer can be faithful to the wrong passage. **Context precision@4** shows how much of what the generator saw was actually relevant.
- Latency is CPU-only, per query, for the whole batch of queries (dense query encoding is batched).

## Corpus

[BEIR / SciFact](https://github.com/beir-cellar/beir): 5,183 scientific abstracts, 300 test claims with expert relevance
judgements (e.g. *"Radioiodine treatment of non-toxic multinodular goitre reduces thyroid volume."*). Any BEIR
dataset works via `--dataset` (e.g. `nfcorpus`, `fiqa`).

## Pipeline

```
BEIR zip ─► rag/data.py ─► rag/chunking.py ─► rag/ingest.py ─┬─► index/<ds>/bm25.pkl
                         (sentence-packed ~120-word           ├─► index/<ds>/dense.faiss
                          chunks, 1-sentence overlap,         └─► index/<ds>/chunks.jsonl
                          title prepended)

query ─► rag/retrievers.py (BM25 | Dense | Hybrid RRF | Hybrid+Rerank) ─► top chunks
      ─► rag/generate.py (Verdict + cited evidence)  ─► rag/faithfulness.py (NLI judge)
```

### Faithfulness method

1. The generator gets the top-4 chunks and must reply with `Verdict: ...` and 1-3 cited evidence sentences
   (or `Verdict: Not enough information`, counted as an abstention and reported separately).
2. The verdict line is dropped; each evidence sentence is cleaned of citations, list markers and
   attribution wrappers ("According to [1], ...", "<title> states: ...").
3. `cross-encoder/nli-deberta-v3-base` scores entailment of each sentence against each retrieved chunk;
   a sentence is supported if max P(entailment) >= 0.5.

Generation runs locally on CPU with `Qwen/Qwen2.5-1.5B-Instruct` (no API key needed). If `OPENAI_API_KEY` is set, any
OpenAI-compatible endpoint is used instead (`OPENAI_MODEL`, default `gpt-4o-mini`; `OPENAI_BASE_URL`).
Because CPU generation takes ~40-100 s/answer, faithfulness uses a fixed random sample of queries (`--faithfulness-queries`, default 5, seed 13); answers are cached in `results/generations.jsonl`.

> **Caveat:** with only 5 sampled queries per strategy, the committed faithfulness numbers are indicative, not statistically meaningful. The 1.5B generator also often adds reasoning sentences ("This directly supports the claim that ...") that no passage entails, which pulls faithfulness down for every strategy. Set `OPENAI_API_KEY` and run `python scripts/eval.py --faithfulness-only --faithfulness-queries 300` for a proper estimate.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m rag.ingest                 # download SciFact, chunk, build BM25 + FAISS (~13 min embedding on 8 CPU cores)
python scripts/eval.py               # retrieval metrics on all queries + faithfulness sample -> results/
python scripts/eval.py --faithfulness-queries 0   # retrieval-only, ~10 min
streamlit run app.py                 # UI
pytest -q
```

Outputs in `results/`: `results.md` / `results.csv` (scorecard), `per_query.jsonl` (top-10 + metrics per query and
strategy), `faithfulness_detail.jsonl` (answers with per-sentence entailment scores), `run.json` (config).

## UI

`streamlit run app.py`

- **Search & answer**: pick a test claim (gold-relevant chunks are highlighted) or type your own; see all strategies
  side by side with their rank of the first relevant doc; generate an answer from any strategy and see per-sentence
  faithfulness.
- **Scorecard**: the results table, recall@k curves, headline metrics and every generated answer from the eval.

## Layout

```
rag/config.py        models, chunking and retrieval hyperparameters
rag/data.py          BEIR download + loaders
rag/chunking.py      sentence-aware chunking
rag/ingest.py        ingestion CLI (BM25 + FAISS indexes)
rag/retrievers.py    BM25, Dense, Hybrid (RRF), Hybrid + Rerank
rag/metrics.py       recall@k, MRR@k, nDCG@k
rag/generate.py      prompt, local/API generator, answer parsing
rag/faithfulness.py  NLI faithfulness judge
scripts/eval.py      scorecard
app.py               Streamlit UI
```

## Limitations / next steps

- NLI faithfulness is conservative with paraphrase and multi-passage synthesis; an LLM judge would complement it.
- MiniLM-L6 is the smallest reasonable reranker; `BAAI/bge-reranker-base` should widen the rerank margin at ~5x the cost.
- The faithfulness sample is tiny (CPU generation); with an API key, raise `--faithfulness-queries` to 300.
- `--faithfulness-only` reuses the retrieval rows in `results/results.csv` and only re-runs generation/judging.
