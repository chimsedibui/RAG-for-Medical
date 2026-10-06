"""Dev-set labelling tool:  uv run python -m medrag.cli label   (then open http://localhost:8010)

For each dev query it pools candidates from BM25, dense and hybrid retrieval (pooling avoids labelling only what the
current system already finds), lets you tick relevant docs and pick/highlight answer passages, and autosaves
data/dev/dev_gold.json in submission format -> `medrag.cli eval pred.json data/dev/dev_gold.json`."""
import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..config import get_settings
from ..evaluation.devset import LabelStore
from ..retrieval.pipeline import Retriever

WEB = Path(__file__).parent / "web"
POOL_DEPTH = 40            # chunks taken from each of bm25 / dense / hybrid
MAX_DOCS = 25
state: dict = {}
search_lock = threading.Lock()      # the embedder is not thread-safe


@asynccontextmanager
async def lifespan(_):
    s = get_settings()
    state["queries"] = json.loads((s.dev_dir / "dev_queries.json").read_text(encoding="utf-8"))
    state["store"] = LabelStore(s.dev_dir / "dev_gold.json", s.dev_dir / "dev_progress.json")
    state["retriever"] = Retriever(rerank=False)
    state["offsets"] = _doc_offsets(s.docs_file)
    yield


app = FastAPI(title="R2AI2026 dev-set labeller", lifespan=lifespan)


def _doc_offsets(docs_file: Path) -> dict[int, int]:
    """doc id -> byte offset in docs.jsonl, so the full text of one doc can be read without loading the file."""
    out, pos = {}, 0
    with docs_file.open("rb") as f:
        for line in f:
            try:
                out.setdefault(json.loads(line)["id"], pos)
            except (json.JSONDecodeError, KeyError):
                pass
            pos += len(line)
    return out


def pooled_search(query: str) -> list[dict]:
    r = state["retriever"]
    doc_score: dict[int, float] = {}
    chunk_score: dict[str, float] = {}
    sources: dict[int, set] = {}
    with search_lock:
        for mode in ("bm25", "dense", "hybrid"):
            seen_docs = set()
            for rank, (cid, _) in enumerate(r.candidates(query, mode)[:POOL_DEPTH], 1):
                c = r.chunks[cid]
                chunk_score[cid] = chunk_score.get(cid, 0.0) + 1 / (60 + rank)
                sources.setdefault(c.doc_id, set()).add(mode)
                if c.doc_id not in seen_docs:                 # doc score: best rank per mode, summed over modes
                    seen_docs.add(c.doc_id)
                    doc_score[c.doc_id] = doc_score.get(c.doc_id, 0.0) + 1 / (60 + rank)
    by_doc: dict[int, list] = {}
    for cid, sc in sorted(chunk_score.items(), key=lambda kv: -kv[1]):
        by_doc.setdefault(r.chunks[cid].doc_id, []).append(cid)
    out = []
    for d, _ in sorted(doc_score.items(), key=lambda kv: -kv[1])[:MAX_DOCS]:
        meta = r.docs_meta.get(d, {})
        out.append({"doc_id": str(d), "url": meta.get("url", ""), "title": meta.get("title", ""),
                    "lang": meta.get("lang", ""), "host": meta.get("host", ""), "sources": sorted(sources[d]),
                    "chunks": [{"chunk_id": cid, "context": r.chunks[cid].context, "text": r.chunks[cid].text}
                               for cid in by_doc[d][:3]]})
    return out


class SearchReq(BaseModel):
    query: str


class LabelReq(BaseModel):
    docs: list[str] = []
    chunks: list[dict] = []
    status: str = "done"          # done | skip | todo


@app.get("/")
def index():
    return FileResponse(WEB / "labeler.html")


@app.get("/api/queries")
def queries():
    st = state["store"]
    return [{**q, "status": st.status(q["id"])} for q in state["queries"]]


@app.get("/api/label/{qid}")
def get_label(qid: int):
    return state["store"].get(qid)


@app.post("/api/label/{qid}")
def put_label(qid: int, req: LabelReq):
    if req.status not in ("done", "skip", "todo"):
        raise HTTPException(400, "bad status")
    state["store"].set(qid, req.docs, req.chunks, req.status)
    return {"status": state["store"].status(qid)}


@app.post("/api/search")
def search(req: SearchReq):
    return pooled_search(req.query)


@app.get("/api/doc/{doc_id}")
def get_doc(doc_id: int):
    off = state["offsets"].get(doc_id)
    if off is None:
        raise HTTPException(404, "doc not in docs.jsonl")
    with get_settings().docs_file.open("rb") as f:
        f.seek(off)
        d = json.loads(f.readline())
    return {"doc_id": str(doc_id), "title": d["title"], "url": d["url"], "text": d["text"]}
