"""Query -> {relevant_docs, relevant_chunks}.

  expand (vi/en/zh) -> per-LANGUAGE stratum: BM25 + dense (filtered) -> RRF inside each stratum
  -> RRF across strata (every language stays represented) -> cross-encoder rerank
  -> doc aggregation -> cross-source consensus (independent hosts/clusters that agree) -> host cap + cut-off
  -> chunk selection/output
"""
from collections import defaultdict

import numpy as np

from ..config import Settings, get_settings
from ..models.embedder import Embedder
from ..models.query_expander import QueryExpander
from ..models.reranker import CrossEncoderReranker, NullReranker
from ..schemas import Chunk
from ..stores.lexical import LANGS, BM25Index, LangBM25
from ..stores.qdrant import QdrantDense
from .consensus import DocFeat, agreement, rescore, select
from .corpus import Corpus
from .diversity import diversity
from .fusion import rrf


class Retriever:
    def __init__(self, s: Settings | None = None, rerank: bool = True, output_unit: str = "child"):
        """output_unit: 'child' | 'parent' - which text goes in chunk_text. Overlap is normalised by the
        reference chunk length, so 'parent' trades chunk precision risk for matching a larger span;
        tune on dev."""
        s = self.s = s or get_settings()
        self.output_unit = output_unit
        self.corpus = Corpus(s)
        self.chunks, self.point_id = self.corpus.chunks, self.corpus.point_id
        self.docs_meta, self.cluster_of = self.corpus.docs_meta, self.corpus.cluster_of
        self.bm25 = BM25Index.load(s.index_dir)                 # global (used when stratify=False)
        self.bm25_lang = LangBM25.load(s.index_dir)             # per-language strata
        self.dense = QdrantDense()
        assert self.dense.count() == len(self.chunks), "Qdrant and chunks.jsonl are out of sync - re-run `index`"
        self.embedder = Embedder(s.embed_model, s.embed_device)
        self._use_rerank = rerank
        self._reranker = None                      # loaded lazily: ~1.2GB VRAM, only if a request needs it
        self.expander = QueryExpander()

    @property
    def reranker(self):
        if not self._use_rerank:
            return NullReranker()
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(self.s.rerank_model, self.s.embed_device)
        return self._reranker

    # -- stage 1+2: candidates ------------------------------------------------
    def candidates(self, query: str, mode: str = "hybrid", stratify: bool | None = None) -> list[tuple[str, float]]:
        s = self.s
        stratify = s.stratify if stratify is None else stratify
        variants = self.expander.expand(query)
        qvecs = self.embedder.encode_queries(list(variants.values()))

        def lists(lang: str | None, k_bm25: int, k_dense: int) -> dict[str, list]:
            out = {}
            for (name, text), qv in zip(variants.items(), qvecs):
                if mode in ("hybrid", "bm25"):
                    out[f"bm25:{name}"] = (self.bm25_lang.search(text, k_bm25, lang) if lang
                                           else self.bm25.search(text, k_bm25))
                if mode in ("hybrid", "dense"):
                    out[f"dense:{name}"] = self.dense.search(qv, k_dense, langs=[lang] if lang else None)
            return out

        weight = lambda r: {k: 2.0 for k in r if k.endswith(":orig")}   # noqa: E731  original vi query counts double
        if not stratify:
            r = lists(None, s.bm25_candidates, s.dense_candidates)
            return rrf(r, s.rrf_k, weight(r))[: s.rerank_candidates]
        strata = {}
        for lg in LANGS:                      # each language gets its own top-K, so a 83%-zh corpus cannot crowd out vi/en
            r = lists(lg, s.per_stratum, s.per_stratum)
            strata[lg] = rrf(r, s.rrf_k, weight(r))[: s.per_stratum]
        # fuse by RANK across strata (scores from different strata are not comparable)
        return rrf({lg: hits for lg, hits in strata.items() if hits}, s.rrf_k)[: s.rerank_candidates]

    # -- stage 3+: rerank, consensus, selection --------------------------------
    def retrieve(self, query: str, mode: str = "hybrid", rerank: bool | None = None, detail: bool = False,
                 stratify: bool | None = None, consensus: bool | None = None, host_cap: int | None = None) -> dict:
        s = self.s
        consensus = s.consensus if consensus is None else consensus
        host_cap = s.host_cap if host_cap is None else host_cap
        if rerank is not None:
            self._use_rerank = rerank
        cands = self.candidates(query, mode, stratify)
        if not cands:
            return {"relevant_docs": [], "relevant_chunks": []}

        by_doc = self._rerank_by_doc(query, cands)
        feats = self._score_docs(by_doc, consensus)
        keep = select(feats, s.doc_min, s.doc_max, s.doc_rel_threshold, host_cap, s.dedupe_clusters)
        out_chunks, view = self._pick_chunks(keep, by_doc)

        out = {"relevant_docs": [int(f.doc_id) for f in keep], "relevant_chunks": out_chunks}
        if detail:
            out["docs"] = view
            out["n_candidates"] = len(cands)
            out["diversity"] = diversity(view)
        return out

    def _rerank_by_doc(self, query: str, cands: list[tuple[str, float]]) -> dict[int, list[tuple[Chunk, float]]]:
        """Score candidate chunks and group them per doc, best chunk first."""
        chunks = [self.chunks[cid] for cid, _ in cands]
        scores = self.reranker.score(query, [c.embed_text for c in chunks])
        if not any(scores):                        # NullReranker: fall back to fusion order
            scores = [fs for _, fs in cands]
        ranked = sorted(zip(chunks, scores), key=lambda x: -x[1])
        by_doc: dict[int, list[tuple[Chunk, float]]] = defaultdict(list)
        for c, sc in ranked:
            by_doc[c.doc_id].append((c, sc))
        return by_doc

    def _score_docs(self, by_doc: dict[int, list[tuple[Chunk, float]]], consensus: bool) -> list[DocFeat]:
        """Doc relevance (best chunk + small bonus for supporting chunks), then optional cross-source consensus."""
        s = self.s
        feats = []
        for d, cs in by_doc.items():
            meta = self.docs_meta.get(d, {})
            rel = cs[0][1] + 0.1 * sum(sc for _, sc in cs[1:3])
            feats.append(DocFeat(doc_id=d, rel=rel, host=meta.get("host", "?"),
                                 cluster=self.cluster_of.get(d, d), lang=meta.get("lang", cs[0][0].lang)))
        feats.sort(key=lambda f: -f.rel)
        if consensus and len(feats) > 1:
            vecs = self.dense.fetch_vectors([self.point_id[by_doc[f.doc_id][0][0].chunk_id] for f in feats])
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
            agreement(feats, vecs @ vecs.T, s.agree_theta, s.agree_rel_min)
            rescore(feats, s.agree_alpha)
        else:
            for f in feats:
                f.final = f.rel
        return feats

    def _pick_chunks(self, keep: list[DocFeat], by_doc: dict[int, list[tuple[Chunk, float]]]) -> tuple[list, list]:
        """Per kept doc, take up to chunk_max_per_doc chunks within chunk_rel_threshold of the doc's best."""
        s = self.s
        out_chunks, view = [], []
        for f in keep:
            d = f.doc_id
            top = by_doc[d][0][1]
            picked = []
            for c, sc in by_doc[d][: s.chunk_max_per_doc]:
                if _rel(sc, top) >= s.chunk_rel_threshold:
                    text = c.parent_text if self.output_unit == "parent" else c.text
                    out_chunks.append({"doc_id": int(d), "chunk_text": text})
                    picked.append({"chunk_id": c.chunk_id, "score": round(float(sc), 4), "context": c.context, "text": text})
            view.append({"doc_id": d, "score": round(float(f.final), 4), "relevance": round(float(f.rel), 4),
                         "support": f.support, "support_hosts": sorted(f.support_hosts), "cluster": f.cluster,
                         **self.docs_meta.get(d, {}), "chunks": picked})
        return out_chunks, view


def _rel(score: float, best: float) -> float:
    """Score relative to the best one. Reranker output is already a probability in [0,1] (sigmoid applied by
    CrossEncoder) and fusion scores are positive, so a plain ratio is right. (A second sigmoid here squashed
    every score into 0.5-0.73 and the cut-off never fired.)"""
    return score / best if best > 0 else 0.0
