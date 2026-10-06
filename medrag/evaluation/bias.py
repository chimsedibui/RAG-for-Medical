"""Bias report over N random real queries: language/host spread + how often results are single-source."""
import collections
import math

import pandas as pd

from ..config import get_settings


def _entropy(c: collections.Counter) -> float:
    return -sum(v / sum(c.values()) * math.log2(v / sum(c.values())) for v in c.values())


def run(n: int = 40, no_stratify: bool = False, no_consensus: bool = False, host_cap: int | None = None) -> None:
    from ..retrieval.pipeline import Retriever
    s = get_settings()
    r = Retriever(rerank=True)
    qs = pd.read_parquet(s.raw_dir / "query.parquet").sample(n, random_state=1)
    H, L = collections.Counter(), collections.Counter()
    single = corro = nd = n_docs = 0
    for q in qs["query"]:
        out = r.retrieve(q, detail=True, stratify=not no_stratify, consensus=not no_consensus, host_cap=host_cap)
        for d in out["docs"]:
            H[d["host"]] += 1; L[d["lang"]] += 1
        if out["docs"]:
            nd += 1; single += out["diversity"]["single_source"]
            corro += out["diversity"]["corroborated"] > 0; n_docs += len(out["docs"])
    print(f"queries {nd}/{n} | docs/query {n_docs / max(nd, 1):.1f} | single-source {single}/{nd} | corroborated>=1 {corro}/{nd}")
    print("lang share:", {k: f"{v / sum(L.values()):.0%}" for k, v in L.most_common()}, f"| lang entropy {_entropy(L):.2f} bits")
    print("top hosts :", {k: f"{v / sum(H.values()):.0%}" for k, v in H.most_common(5)}, f"| host entropy {_entropy(H):.2f} bits")
