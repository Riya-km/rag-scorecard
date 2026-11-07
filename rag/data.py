"""Download and load BEIR-format datasets (corpus, queries, qrels)."""
import csv
import io
import json
import zipfile
from pathlib import Path
from urllib.request import urlopen

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


def download_beir(name: str, data_dir: Path) -> Path:
    target = data_dir / name
    if (target / "corpus.jsonl").exists():
        return target
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {name} from BEIR ...")
    with urlopen(BEIR_URL.format(name=name)) as resp:
        zipfile.ZipFile(io.BytesIO(resp.read())).extractall(data_dir)
    return target


def load_corpus(path: Path) -> dict[str, dict]:
    corpus = {}
    with open(path / "corpus.jsonl") as f:
        for line in f:
            d = json.loads(line)
            corpus[str(d["_id"])] = {"title": d.get("title", ""), "text": d["text"]}
    return corpus


def load_queries(path: Path) -> dict[str, str]:
    with open(path / "queries.jsonl") as f:
        return {str(d["_id"]): d["text"] for d in map(json.loads, f)}


def load_qrels(path: Path, split: str = "test") -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with open(path / "qrels" / f"{split}.tsv") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for qid, did, score in reader:
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)
    return qrels
