import pytest

from rag.chunking import chunk_text, split_sentences
from rag.generate import is_abstention, strip_citations
from rag.metrics import hit_at_k, mrr_at_k, ndcg_at_k, recall_at_k
from rag.retrievers import rrf_fuse, to_doc_ranking


def test_recall_and_hit():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, {"b", "z"}, 2) == 0.5
    assert recall_at_k(ranked, {"b", "z"}, 1) == 0.0
    assert hit_at_k(ranked, {"c"}, 3) == 1.0
    assert recall_at_k(ranked, set(), 3) == 0.0


def test_mrr():
    assert mrr_at_k(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)
    assert mrr_at_k(["a", "b", "c"], {"x"}) == 0.0
    assert mrr_at_k(["a", "b", "c"], {"c"}, k=2) == 0.0


def test_ndcg_perfect_and_partial():
    assert ndcg_at_k(["a", "b"], {"a": 1, "b": 1}) == pytest.approx(1.0)
    assert 0 < ndcg_at_k(["x", "a"], {"a": 1}) < 1


def test_rrf_prefers_consensus():
    fused = rrf_fuse([[(1, 9.0), (2, 8.0), (3, 7.0)], [(3, 0.9), (1, 0.8), (4, 0.1)]], k=60)
    assert [i for i, _ in fused][:2] == [1, 3]


def test_doc_ranking_max_pools_chunks():
    chunks = [{"doc_id": "d1"}, {"doc_id": "d2"}, {"doc_id": "d1"}]
    assert [d for d, _ in to_doc_ranking([(2, 3.0), (1, 2.0), (0, 1.0)], chunks)] == ["d1", "d2"]


def test_chunking_respects_budget_and_overlap():
    text = " ".join(f"Sentence number {i} has five words." for i in range(40))
    chunks = chunk_text(text, max_words=30, overlap_sentences=1)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 36 for c in chunks)
    assert split_sentences(chunks[0])[-1] == split_sentences(chunks[1])[0]


def test_citation_and_abstention_helpers():
    assert strip_citations("Mice improved [1, 2]. Rats did not [3].") == "Mice improved. Rats did not."
    assert is_abstention("Not enough information.")
    assert not is_abstention("Vitamin D helps [1].")


def test_parse_answer_and_normalize():
    from rag.generate import normalize_claim, parse_answer
    v, e = parse_answer("Verdict: Supported.\nEvidence: Low vitamin D raises MS risk [1].")
    assert v == "Supported" and e == "Low vitamin D raises MS risk [1]."
    assert is_abstention("Verdict: Not enough information")
    assert not is_abstention("Verdict: Supported\nEvidence: X causes Y in mice [2].")
    assert normalize_claim("According to [1], vitamin D is immunomodulatory [1].") == "Vitamin D is immunomodulatory."
    assert normalize_claim("The study shows that mice improved.") == "Mice improved."


def test_evidence_sentence_cleanup():
    from rag.generate import evidence_sentences, normalize_claim
    ev = "1. Mice improved after treatment [1].\n2. Rats did not respond [2].\n3. The dose was"
    assert evidence_sentences(ev) == ["Mice improved after treatment [1].", "Rats did not respond [2]."]
    assert normalize_claim('Nonlinear Elasticity in Gels states: "Gels stiffen under strain."') == "Gels stiffen under strain."
    assert normalize_claim("CONCLUSIONS A substantial reduction was seen.") == "A substantial reduction was seen."
