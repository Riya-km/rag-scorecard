"""Answer generation. Uses an OpenAI-compatible API if OPENAI_API_KEY is set, else a local HF model on CPU."""
import json
import os
import re
from urllib.request import Request, urlopen

SYSTEM_PROMPT = """You verify claims and answer questions using ONLY the numbered passages.
Reply in exactly this format:
Verdict: <Supported | Contradicted | Answered | Not enough information>
Evidence: 1-3 short sentences. Each sentence restates one specific fact from a passage and ends with its citation, e.g. [2].
Rules: state facts directly ("Low vitamin D levels are linked to higher MS risk [1]." not "Passage 1 says that ...").
No speculation and no outside knowledge. If the passages do not address the input, write "Verdict: Not enough information" and no evidence."""
ABSTAIN = "not enough information"

def build_prompt(query: str, passages: list[str]) -> str:
    ctx = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, start=1))
    return f"Passages:\n{ctx}\n\nQuestion or claim: {query}\n\nAnswer:"


_VERDICT = re.compile(r"^\W*verdict\W*:?\s*(.*)$", re.I | re.M)
_EVIDENCE = re.compile(r"^\W*evidence\W*:?\s*", re.I | re.M)
_ATTRIBUTION = re.compile(
    r"^(according to( passage)?( \[?\d+\]?)?,?|((the )?(passage|study|context|authors?|it)( \[?\d+\]?)? )?"
    r"(says|states|shows|mentions|notes|reports|indicates|suggests|found|finds)( that)?:?)\s*", re.I)


def parse_answer(answer: str) -> tuple[str, str]:
    """Split a generated answer into (verdict, evidence text). Unformatted answers are all evidence."""
    m = _VERDICT.search(answer)
    verdict = m.group(1).strip().rstrip(".") if m else ""
    body = answer[m.end():] if m else answer
    return verdict, _EVIDENCE.sub("", body, count=1).strip()


def is_abstention(answer: str) -> bool:
    verdict, evidence = parse_answer(answer)
    text = (verdict or evidence).lower()
    return text.startswith(ABSTAIN) and (bool(verdict) or len(evidence) < 40) or not (verdict or evidence)


_LIST_MARK = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_QUOTED_ATTRIBUTION = re.compile(
    r"^[^\"]{0,200}?\b(?:states|says|mentions|notes|reports|shows|indicates|found|concludes)(?:\s+that)?\s*:\s*", re.I)
_SECTION_HEADER = re.compile(r"^(?:[A-Z]{3,}(?:\s+[A-Z]{3,})*:?)\s+(?=[A-Z])")


def evidence_sentences(evidence: str) -> list[str]:
    """Split evidence into sentences, one list item/line at a time; drop a truncated final sentence."""
    from .chunking import split_sentences
    out = []
    for line in evidence.splitlines():
        line = _LIST_MARK.sub("", line).strip()
        if line:
            out.extend(split_sentences(line))
    if out and not re.search(r"[.!?\"')\]]$", out[-1].strip()):
        out = out[:-1]
    return out


def normalize_claim(sentence: str) -> str:
    """Drop citations, list marks, section headers and attribution wrappers so NLI judges the content."""
    s = strip_citations(_LIST_MARK.sub("", sentence)).strip()
    s = _QUOTED_ATTRIBUTION.sub("", s)
    s = _ATTRIBUTION.sub("", s).strip().strip('"“”').strip()
    s = _SECTION_HEADER.sub("", s)
    return s[:1].upper() + s[1:]


class Generator:
    def __init__(self, model: str, max_new_tokens: int = 140):
        self.max_new_tokens = max_new_tokens
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if self.api_key:
            self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        else:
            self.model = model
            self._load_local(model)

    def _load_local(self, model: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(os.cpu_count() or 4)
        self.tok = AutoTokenizer.from_pretrained(model)
        self.lm = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32)
        self.lm.eval()

    def __call__(self, query: str, passages: list[str]) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(query, passages)}]
        return self._api(messages) if self.api_key else self._local(messages)

    def _api(self, messages) -> str:
        body = json.dumps({"model": self.model, "messages": messages, "temperature": 0,
                           "max_tokens": self.max_new_tokens}).encode()
        req = Request(f"{self.base_url}/chat/completions", data=body, headers={
            "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        with urlopen(req, timeout=120) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"].strip()

    def _local(self, messages) -> str:
        import torch
        prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        enc = self.tok(prompt, return_tensors="pt")
        with torch.inference_mode():
            out = self.lm.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                   pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def strip_citations(text: str) -> str:
    return re.sub(r"\s*\[\d+(?:\s*[,-]\s*\d+)*\]", "", text).strip()
