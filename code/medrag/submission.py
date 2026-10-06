import json
import zipfile
from pathlib import Path

import pandas as pd

from .config import get_settings


def load_queries() -> pd.DataFrame:
    return pd.read_parquet(get_settings().raw_dir / "query.parquet")[["id", "query"]]


def build_submission(retriever, out_json: Path, limit: int | None = None, ids: set[int] | None = None) -> Path:
    """Run every query, write submission .json (all queries, empty lists if nothing found) + .zip with the
    single json file at the zip root (04-submission.png)."""
    from tqdm import tqdm
    qs = load_queries()
    if ids is not None:
        qs = qs[qs["id"].isin(ids)]
    if limit:
        qs = qs.head(limit)
    rows = []
    for r in tqdm(qs.itertuples(), total=len(qs)):
        res = retriever.retrieve(r.query)
        rows.append({"id": int(r.id), **res})
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    zp = out_json.with_suffix(".zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out_json, arcname=out_json.name)
    return zp


def validate_submission(path: Path, expected_ids: set[int]) -> list[str]:
    errs = []
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = {d.get("id") for d in data}
    if ids != expected_ids:
        errs.append(f"id mismatch: missing={len(expected_ids - ids)} extra={len(ids - expected_ids)}")
    for d in data:
        docs = set(d["relevant_docs"])
        for c in d["relevant_chunks"]:
            if not c["chunk_text"].strip():
                errs.append(f"q{d['id']}: empty chunk_text")
            if c["doc_id"] not in docs:
                errs.append(f"q{d['id']}: chunk doc_id {c['doc_id']} not in relevant_docs")
    return errs[:50]
