import re

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be but by for from has have in is it its of on or that the this to was were "
    "which with not no can do does did been being than then there these those into such".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def chunk_to_passage(chunk: dict) -> str:
    return f"{chunk['title']}. {chunk['text']}" if chunk["title"] else chunk["text"]
