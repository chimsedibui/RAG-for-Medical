"""Near-duplicate clustering of documents (syndicated / re-posted articles).

Why: the same article re-posted on 5 hosts is ONE piece of evidence, not five. Cluster ids let the
consensus step count independent sources instead of raw hits.
simhash (64-bit) over word/char n-gram tokens; LSH banding keeps it ~linear for millions of docs."""
import hashlib
import json
from collections import Counter, defaultdict

import numpy as np

from ..text import lexical_tokens

BITS = 64
_SHIFTS = np.arange(BITS, dtype=np.uint64)
_MIN_TOKENS = 60          # shorter docs: simhash is unreliable -> treated as singletons


def simhash(text: str, max_chars: int = 4000) -> int | None:
    toks = lexical_tokens(text[:max_chars])
    if len(toks) < _MIN_TOKENS:
        return None
    cnt = Counter(toks)
    h = np.array([int.from_bytes(hashlib.blake2b(t.encode(), digest_size=8).digest(), "big") for t in cnt],
                 dtype=np.uint64)
    w = np.array(list(cnt.values()), dtype=np.int64)
    bits = ((h[:, None] >> _SHIFTS) & np.uint64(1)).astype(np.int64) * 2 - 1
    v = (bits * w[:, None]).sum(axis=0)
    return int(sum(1 << i for i in range(BITS) if v[i] > 0))


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def cluster(sig: dict[int, int], max_dist: int = 3, bands: int = 4, max_bucket: int = 500) -> dict[int, int]:
    """sig: doc_id -> simhash. Returns doc_id -> cluster_id (= smallest doc id in the cluster); singletons omitted.
    Candidates share a band (pigeonhole: dist<=bands-1 guarantees one equal band). A doc joins a cluster only if it
    is within max_dist of the cluster's REPRESENTATIVE - no single-linkage chaining, which fuses templated pages
    into giant components."""
    width = BITS // bands
    rep = {}                                   # doc -> representative doc id (default: itself)
    for d in sorted(sig):
        rep[d] = d
    for b in range(bands):
        buckets = defaultdict(list)
        for d in sorted(sig):
            buckets[(sig[d] >> (b * width)) & ((1 << width) - 1)].append(d)
        for ids in buckets.values():
            for i, a in enumerate(ids[:max_bucket]):
                if rep[a] != a:                # already attached to an earlier representative
                    continue
                for c in ids[i + 1:max_bucket]:
                    if rep[c] == c and hamming(sig[a], sig[c]) <= max_dist:
                        rep[c] = a
    groups = defaultdict(list)
    for d, r in rep.items():
        groups[r].append(d)
    return {d: r for r, ds in groups.items() if len(ds) > 1 for d in ds}


def build_clusters(docs_file, out_file, max_dist: int = 3) -> tuple[int, int, int]:
    from ..ingest.clean import clean_docs
    sig, n = {}, 0
    for d in clean_docs(docs_file):            # cluster on cleaned text, or per-host templates dominate the hash
        n += 1
        s = simhash(d.text)
        if s is not None:
            sig[d.id] = s
    cl = cluster(sig, max_dist)
    out_file.write_text(json.dumps({str(k): v for k, v in cl.items()}))
    return n, len(cl), len(set(cl.values()))
