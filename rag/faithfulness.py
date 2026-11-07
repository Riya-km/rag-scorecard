"""NLI-based answer faithfulness: share of answer sentences entailed by the retrieved context.

The verdict line is excluded; each evidence sentence (citations and "according to ..." wrappers
stripped) is a hypothesis and each retrieved passage a premise. A sentence counts as supported
if its max entailment probability across passages >= threshold.
"""
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .generate import evidence_sentences, is_abstention, normalize_claim, parse_answer


@dataclass
class SentenceJudgement:
    sentence: str
    entailment: float
    supported: bool


@dataclass
class FaithfulnessResult:
    score: float | None  # None when the model abstained
    abstained: bool
    sentences: list[SentenceJudgement]


class NLIFaithfulness:
    def __init__(self, model: str, threshold: float = 0.5):
        self.model_name = model
        self.threshold = threshold

    @cached_property
    def model(self):
        from sentence_transformers import CrossEncoder
        return CrossEncoder(self.model_name, max_length=512)

    @cached_property
    def entail_idx(self) -> int:
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        return labels["entailment"]

    def __call__(self, answer: str, passages: list[str]) -> FaithfulnessResult:
        if is_abstention(answer):
            return FaithfulnessResult(None, True, [])
        _, evidence = parse_answer(answer)
        sents = [s for s in (normalize_claim(x) for x in evidence_sentences(evidence)) if len(s.split()) >= 3]
        if not sents:
            return FaithfulnessResult(None, True, [])
        premises = passages
        pairs = [(p, s) for s in sents for p in premises]
        logits = np.asarray(self.model.predict(pairs, batch_size=16, show_progress_bar=False))
        probs = np.exp(logits - logits.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        ent = probs[:, self.entail_idx].reshape(len(sents), len(premises)).max(1)
        judged = [SentenceJudgement(s, float(e), bool(e >= self.threshold)) for s, e in zip(sents, ent)]
        return FaithfulnessResult(float(np.mean([j.supported for j in judged])), False, judged)
