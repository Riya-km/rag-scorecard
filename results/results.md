Dataset: **scifact** (5183 docs, 13192 chunks), 300 test queries. Embeddings: `BAAI/bge-small-en-v1.5`, reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
Faithfulness on 5 sampled queries, top-4 chunks as context, generator `Qwen/Qwen2.5-1.5B-Instruct`, NLI judge `cross-encoder/nli-deberta-v3-base`.

| strategy        | Recall@1   | Recall@5   | Recall@10   | Recall@20   | Recall@100   | MRR@10    | nDCG@10   |   Latency ms/q | Faithfulness   |   Abstain rate | Context precision@4   |
|:----------------|:-----------|:-----------|:------------|:------------|:-------------|:----------|:----------|---------------:|:---------------|---------------:|:----------------------|
| BM25            | 0.504      | 0.725      | 0.788       | 0.828       | 0.871        | 0.618     | 0.655     |             20 | 0.100          |              0 | 0.150                 |
| Dense           | **0.579**  | 0.782      | 0.833       | 0.878       | 0.935        | 0.686     | 0.716     |             32 | **0.433**      |              0 | **0.300**             |
| Hybrid (RRF)    | 0.557      | 0.780      | 0.841       | 0.884       | **0.952**    | 0.679     | 0.712     |             45 | 0.133          |              0 | **0.300**             |
| Hybrid + Rerank | **0.579**  | **0.785**  | **0.853**   | **0.910**   | **0.952**    | **0.696** | **0.725** |           1795 | 0.200          |              0 | 0.250                 |
