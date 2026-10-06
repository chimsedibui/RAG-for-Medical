"""Dev-set helpers. `sample_queries` draws a subset of the real queries that mirrors the full set: proportional
allocation over (length quartile x first-person narrative vs generic question), fixed seed -> reproducible."""
import json
import re
from pathlib import Path

import pandas as pd

from ..config import get_settings

_NARRATIVE = re.compile(r"\b(em|tôi|mình|con|bé|cháu|bố|mẹ|chồng|vợ|anh|chị)\b", re.I)


def sample_queries(n: int = 80, seed: int = 42) -> pd.DataFrame:
    q = pd.read_parquet(get_settings().raw_dir / "query.parquet")[["id", "query"]].copy()
    q["len_bin"] = pd.qcut(q["query"].str.split().str.len(), 4, labels=False, duplicates="drop")
    q["narrative"] = q["query"].map(lambda t: bool(_NARRATIVE.search(t)))
    groups = q.groupby(["len_bin", "narrative"])
    quota = (groups["id"].transform("size") * n / len(q)).round().astype(int)   # per-row copy of the group quota
    picked = [g.sample(min(len(g), int(quota.loc[g.index[0]])), random_state=seed) for _, g in groups]
    out = pd.concat(picked)
    if len(out) < n:                                   # rounding shortfall: top up from the rest
        rest = q.drop(out.index)
        out = pd.concat([out, rest.sample(n - len(out), random_state=seed)])
    elif len(out) > n:
        out = out.sample(n, random_state=seed)
    return out.sort_values("id")[["id", "query"]].reset_index(drop=True)


def _strata(q: pd.DataFrame) -> pd.DataFrame:
    q = q.copy()
    q["len_bin"] = pd.qcut(q["query"].str.split().str.len(), 4, labels=False, duplicates="drop")
    q["narrative"] = q["query"].map(lambda t: bool(_NARRATIVE.search(t)))
    return q


def add_replacements(target: int = 80, seed: int = 7) -> list[int]:
    """Top the dev set up to `target` usable queries: every skipped query (no relevant doc in the crawled sample)
    is replaced by an unused query of the SAME stratum, so the dev distribution stays close to the real one."""
    s = get_settings()
    q = _strata(pd.read_parquet(s.raw_dir / "query.parquet")[["id", "query"]])
    dev_file = s.dev_dir / "dev_queries.json"
    dev = json.loads(dev_file.read_text(encoding="utf-8"))
    store = LabelStore(s.dev_dir / "dev_gold.json", s.dev_dir / "dev_progress.json")
    usable = sum(store.status(d["id"]) != "skip" for d in dev)
    need = target - usable
    if need <= 0:
        return []
    skipped = q[q["id"].isin(store.skipped)]
    pool = q[~q["id"].isin({d["id"] for d in dev})]
    want = skipped.sample(need, replace=True, random_state=seed)               # strata ~ proportional to skip rate
    new = []
    for _, r in want.iterrows():
        cand = pool[(pool.len_bin == r.len_bin) & (pool.narrative == r.narrative) & ~pool["id"].isin(new)]
        cand = cand if len(cand) else pool[~pool["id"].isin(new)]
        new.append(int(cand.sample(1, random_state=seed + len(new)).iloc[0]["id"]))
    byid = q.set_index("id")["query"]
    dev += [{"id": i, "query": byid[i]} for i in new]
    dev_file.write_text(json.dumps(dev, ensure_ascii=False, indent=1), encoding="utf-8")
    return new


class LabelStore:
    """Hand-made labels. `gold` file = submission format, only finished queries (feeds `medrag.cli eval`);
    `progress` file keeps the skipped ids and unfinished drafts, so nothing is lost on reload."""
    def __init__(self, gold_file: Path, progress_file: Path):
        self.gold_file, self.progress_file = gold_file, progress_file
        self.gold: dict[int, dict] = {}
        self.skipped: set[int] = set()
        self.drafts: dict[int, dict] = {}
        if gold_file.exists():
            self.gold = {int(r["id"]): r for r in json.loads(gold_file.read_text(encoding="utf-8"))}
        if progress_file.exists():
            prog = json.loads(progress_file.read_text(encoding="utf-8"))
            self.skipped = set(prog.get("skipped", []))
            self.drafts = {int(k): v for k, v in prog.get("drafts", {}).items()}

    def status(self, qid: int) -> str:
        return "done" if qid in self.gold else "skip" if qid in self.skipped else "todo"

    def get(self, qid: int) -> dict:
        if qid in self.gold:
            r = self.gold[qid]
            return {"docs": r["relevant_docs"], "chunks": r["relevant_chunks"], "status": "done"}
        d = self.drafts.get(qid, {})
        return {"docs": d.get("docs", []), "chunks": d.get("chunks", []), "status": self.status(qid)}

    def set(self, qid: int, docs: list[str], chunks: list[dict], status: str) -> None:
        """status: done | skip | todo. A doc is kept only if marked or if one of its chunks is picked.
        `done` without any doc is stored as a draft (an empty gold entry would only add noise to the metric)."""
        self.gold.pop(qid, None); self.skipped.discard(qid); self.drafts.pop(qid, None)
        chunks = [{"doc_id": str(c["doc_id"]), "chunk_text": c["chunk_text"]} for c in chunks]
        docs = list(dict.fromkeys([str(d) for d in docs] + [c["doc_id"] for c in chunks]))
        if status == "done" and docs:
            self.gold[qid] = {"id": qid, "relevant_docs": docs, "relevant_chunks": chunks}
        elif status == "skip":
            self.skipped.add(qid)
        elif docs:
            self.drafts[qid] = {"docs": docs, "chunks": chunks}
        self._save()

    def _save(self) -> None:
        self.gold_file.parent.mkdir(parents=True, exist_ok=True)
        prog = {"skipped": sorted(self.skipped), "drafts": {str(k): self.drafts[k] for k in sorted(self.drafts)}}
        for path, obj in ((self.gold_file, [self.gold[k] for k in sorted(self.gold)]), (self.progress_file, prog)):
            tmp = path.with_suffix(path.suffix + ".tmp")          # atomic: a crash never leaves half a file
            tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(path)


def run(n: int, seed: int, out: str) -> None:
    df = sample_queries(n, seed)
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([{"id": int(r.id), "query": r.query} for r in df.itertuples()],
                            ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(df)} dev queries -> {p}")
