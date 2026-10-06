"""Doc cleaning shared by chunking and clustering: drop per-host boilerplate lines (menus, footers, forum chrome)."""
import json
from collections import Counter, defaultdict
from urllib.parse import urlparse

from ..schemas import Doc


def _read(docs_file):
    seen = set()
    for line in docs_file.open(encoding="utf-8"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:      # crawler may be mid-write on the last line
            continue
        if d["id"] in seen:               # docs.jsonl is append-only; keep the first extraction
            continue
        seen.add(d["id"])
        yield d


def boilerplate_lines(docs_file, min_docs: int = 8, min_frac: float = 0.05) -> dict[str, set[str]]:
    """Lines repeated across many docs of one host are template, not content."""
    per_host: dict[str, Counter] = defaultdict(Counter)
    n_docs: Counter = Counter()
    for d in _read(docs_file):
        host = urlparse(d["url"]).netloc
        n_docs[host] += 1
        per_host[host].update({ln.strip() for ln in d["text"].split("\n") if 0 < len(ln.strip()) < 200})
    return {h: {ln for ln, c in cnt.items() if c >= max(min_docs, min_frac * n_docs[h])}
            for h, cnt in per_host.items()}


def clean_docs(docs_file):
    boiler = boilerplate_lines(docs_file)
    print("boilerplate lines per host:", {h: len(v) for h, v in boiler.items() if v})
    for d in _read(docs_file):
        bl = boiler.get(urlparse(d["url"]).netloc, set())
        d["text"] = "\n".join(ln for ln in d["text"].split("\n") if ln.strip() not in bl).strip()
        if d["text"]:
            yield Doc(**d)
