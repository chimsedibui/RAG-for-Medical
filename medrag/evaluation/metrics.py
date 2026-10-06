"""Local re-implementation of the competition metric (02-evaluation.png).

Doc level : P=|D∩G|/|D|, R=|D∩G|/|G|
Chunk level: a submitted chunk c is correct if, for some reference chunk g of the SAME doc_id,
             LCS_tokens(c,g)/|g| >= 0.40.  P = correct/|submitted|, R = refs found/|refs|.
F2 = 5PR/(4P+R); macro-average over queries.
Note overlap is normalised by |g| only -> a longer c that contains g loses nothing at the
matching step; it only costs nothing in precision unless it is a wasted slot.
"""
from collections import defaultdict

from ..text import metric_tokens

OVERLAP_THRESHOLD = 0.40


def f2(p: float, r: float) -> float:
    return 0.0 if (4 * p + r) == 0 else 5 * p * r / (4 * p + r)


def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def overlap(c_tokens: list[str], g_tokens: list[str]) -> float:
    return lcs_len(c_tokens, g_tokens) / len(g_tokens) if g_tokens else 0.0


def doc_prf(pred: list, gold: list) -> tuple[float, float, float]:
    D, G = set(map(str, pred)), set(map(str, gold))
    hit = len(D & G)
    p = hit / len(D) if D else 0.0
    r = hit / len(G) if G else 0.0
    return p, r, f2(p, r)


def chunk_prf(pred: list[dict], gold: list[dict]) -> tuple[float, float, float]:
    gold_by_doc = defaultdict(list)
    for g in gold:
        gold_by_doc[str(g["doc_id"])].append(metric_tokens(g["chunk_text"]))
    found: set[tuple[str, int]] = set()
    correct = 0
    for c in pred:
        did, ct = str(c["doc_id"]), metric_tokens(c["chunk_text"])
        ok = False
        for gi, gt in enumerate(gold_by_doc.get(did, [])):
            if overlap(ct, gt) >= OVERLAP_THRESHOLD:
                ok = True
                found.add((did, gi))
        correct += ok
    p = correct / len(pred) if pred else 0.0
    r = len(found) / len(gold) if gold else 0.0
    return p, r, f2(p, r)


def evaluate(preds: list[dict], golds: list[dict]) -> dict:
    """preds/golds: submission-format records ({id, relevant_docs, relevant_chunks})."""
    gold_by_id = {g["id"]: g for g in golds}
    pred_by_id = {p["id"]: p for p in preds}
    rows = []
    for qid, g in gold_by_id.items():
        p = pred_by_id.get(qid, {"relevant_docs": [], "relevant_chunks": []})
        rows.append((*doc_prf(p["relevant_docs"], g["relevant_docs"]),
                     *chunk_prf(p["relevant_chunks"], g["relevant_chunks"])))
    n = max(len(rows), 1)
    keys = ["doc_P", "doc_R", "doc_F2", "chunk_P", "chunk_R", "chunk_F2"]
    return {k: sum(r[i] for r in rows) / n for i, k in enumerate(keys)} | {"n_queries": len(rows)}
