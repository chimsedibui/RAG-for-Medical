"""Cross-source consensus + diversification (pure functions, no I/O -> unit-testable).

A doc is *supported* by another doc when (a) the other is from an independent source (different host AND
different near-duplicate cluster), (b) it is itself relevant (>= agree_rel_min x best), and (c) its best chunk
is semantically close to ours (cosine >= theta; works across languages because embeddings are multilingual).
Support is counted in distinct HOSTS, so 50 pages of one site still count as one source."""
import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DocFeat:
    doc_id: int
    rel: float                 # relevance score (reranker prob. or fusion score)
    host: str
    cluster: int               # near-duplicate cluster id (own doc_id if singleton)
    lang: str = "other"
    support_hosts: set = field(default_factory=set)
    final: float = 0.0

    @property
    def support(self) -> int:
        return len(self.support_hosts)


def agreement(docs: list[DocFeat], sim: np.ndarray, theta: float = 0.55, rel_min: float = 0.3) -> None:
    """sim[i][j]: cosine between best chunks of docs i and j. Fills .support_hosts in place."""
    best = max((d.rel for d in docs), default=0.0)
    for i, a in enumerate(docs):
        for j, b in enumerate(docs):
            if i == j or b.host == a.host or b.cluster == a.cluster:
                continue
            if b.rel >= rel_min * best and sim[i][j] >= theta:
                a.support_hosts.add(b.host)


def rescore(docs: list[DocFeat], alpha: float = 0.3) -> None:
    for d in docs:
        d.final = d.rel * (1.0 + alpha * math.log1p(d.support))


def select(docs: list[DocFeat], doc_min: int, doc_max: int, rel_thr: float, host_cap: int,
           dedupe_clusters: bool = False) -> list[DocFeat]:
    """Greedy pick by final score: relative cut-off, per-host cap, optional one-doc-per-cluster."""
    ordered = sorted(docs, key=lambda d: -d.final)
    if not ordered:
        return []
    best, per_host, used_clusters, out = ordered[0].final, {}, set(), []
    for d in ordered:
        if len(out) >= doc_max:
            break
        if per_host.get(d.host, 0) >= host_cap:
            continue
        if dedupe_clusters and d.cluster in used_clusters:
            continue
        if len(out) >= doc_min and (d.final / best if best > 0 else 0.0) < rel_thr:
            break
        out.append(d)
        per_host[d.host] = per_host.get(d.host, 0) + 1
        used_clusters.add(d.cluster)
    return out
