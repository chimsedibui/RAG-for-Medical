import json
from pathlib import Path

import bm25s
import numpy as np

from ..text import lexical_tokens


class BM25Index:
    def __init__(self, chunk_ids: list[str], retriever: "bm25s.BM25"):
        self.chunk_ids, self.retriever = chunk_ids, retriever

    @classmethod
    def build(cls, chunk_ids: list[str], texts: list[str]) -> "BM25Index":
        toks = [lexical_tokens(t) or ["∅"] for t in texts]
        r = bm25s.BM25()
        r.index(toks, show_progress=False)
        return cls(chunk_ids, r)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        q = lexical_tokens(query)
        if not q:
            return []
        k = min(k, len(self.chunk_ids))
        docs, scores = self.retriever.retrieve([q], k=k, show_progress=False)
        return [(self.chunk_ids[i], float(s)) for i, s in zip(docs[0], scores[0]) if s > 0]

    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(d / "bm25"))
        (d / "bm25_ids.json").write_text(json.dumps(self.chunk_ids))

    @classmethod
    def load(cls, d: Path) -> "BM25Index":
        return cls(json.loads((d / "bm25_ids.json").read_text()), bm25s.BM25.load(str(d / "bm25")))


LANGS = ("vi", "en", "zh")


class LangBM25:
    """One BM25 index per language (BM25 scores are not comparable across indexes, so strata are fused by rank)."""
    def __init__(self, by_lang: dict[str, BM25Index]):
        self.by_lang = by_lang

    @classmethod
    def build(cls, chunk_ids: list[str], texts: list[str], langs: list[str]) -> "LangBM25":
        out = {}
        for lg in LANGS:
            sel = [i for i, x in enumerate(langs) if x == lg]
            if sel:
                out[lg] = BM25Index.build([chunk_ids[i] for i in sel], [texts[i] for i in sel])
        return cls(out)

    def search(self, query: str, k: int, lang: str) -> list[tuple[str, float]]:
        idx = self.by_lang.get(lang)
        return idx.search(query, k) if idx else []

    def save(self, d: Path) -> None:
        for lg, idx in self.by_lang.items():
            idx.retriever.save(str(d / f"bm25_{lg}"))
            (d / f"bm25_{lg}_ids.json").write_text(json.dumps(idx.chunk_ids))

    @classmethod
    def load(cls, d: Path) -> "LangBM25":
        out = {}
        for lg in LANGS:
            if (d / f"bm25_{lg}_ids.json").exists():
                out[lg] = BM25Index(json.loads((d / f"bm25_{lg}_ids.json").read_text()),
                                    bm25s.BM25.load(str(d / f"bm25_{lg}")))
        return cls(out)
