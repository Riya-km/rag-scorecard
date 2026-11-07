import re

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def chunk_text(text: str, max_words: int = 120, overlap_sentences: int = 1) -> list[str]:
    """Greedy sentence-packing into ~max_words windows with sentence overlap."""
    sents = split_sentences(text)
    if not sents:
        return []
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        n = len(s.split())
        if cur and cur_len + n > max_words:
            chunks.append(" ".join(cur))
            cur = cur[-overlap_sentences:] if overlap_sentences else []
            cur_len = sum(len(x.split()) for x in cur)
        cur.append(s)
        cur_len += n
    if cur and (not chunks or " ".join(cur) != chunks[-1]):
        tail = " ".join(cur)
        if not chunks or len(cur) > overlap_sentences:
            chunks.append(tail)
    return chunks


def chunk_corpus(corpus: dict[str, dict], max_words: int, overlap_sentences: int) -> list[dict]:
    out = []
    for doc_id, doc in corpus.items():
        pieces = chunk_text(doc["text"], max_words, overlap_sentences) or [doc["text"]]
        for i, piece in enumerate(pieces):
            out.append({
                "chunk_id": f"{doc_id}#{i}",
                "doc_id": doc_id,
                "title": doc["title"],
                "text": piece,
            })
    return out
