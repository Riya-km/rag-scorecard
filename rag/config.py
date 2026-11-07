from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    dataset: str = "scifact"
    data_dir: Path = ROOT / "data"
    index_dir: Path = ROOT / "index"
    results_dir: Path = ROOT / "results"

    chunk_words: int = 120
    chunk_overlap_sentences: int = 1

    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_query_prefix: str = "Represent this sentence for searching relevant passages: "
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    gen_model: str = "Qwen/Qwen2.5-1.5B-Instruct"

    candidate_chunks: int = 100
    rrf_k: int = 60
    rerank_depth: int = 50
    context_chunks: int = 4
    ks: list = field(default_factory=lambda: [1, 5, 10, 20, 100])

    @property
    def corpus_dir(self) -> Path:
        return self.data_dir / self.dataset

    @property
    def dataset_index_dir(self) -> Path:
        return self.index_dir / self.dataset


CFG = Config()
