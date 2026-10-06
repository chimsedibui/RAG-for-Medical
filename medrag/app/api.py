"""Dev UI backend:  uv run uvicorn medrag.app.api:app --port 8000   (then open http://localhost:8000)"""
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..retrieval.pipeline import Retriever

WEB = Path(__file__).parent / "web"
state: dict = {}


@asynccontextmanager
async def lifespan(_):
    state["retriever"] = Retriever(rerank=False)      # reranker loads lazily on first rerank=true request
    yield


app = FastAPI(title="R2AI2026 medical retrieval", lifespan=lifespan)


class SearchReq(BaseModel):
    query: str
    mode: str = "hybrid"          # bm25 | dense | hybrid
    rerank: bool = False
    output_unit: str = "child"    # child | parent
    stratify: bool = True         # retrieve per language, fuse by rank
    consensus: bool = True        # cross-source agreement re-scoring
    host_cap: int = 2             # max docs per host


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
def health():
    r = state["retriever"]
    return {"chunks": len(r.chunks), "docs": len({c.doc_id for c in r.chunks.values()}),
            "qdrant_points": r.dense.count(), "query_expansion": r.expander.enabled}


@app.post("/api/search")
def search(req: SearchReq):
    r = state["retriever"]
    r.output_unit = req.output_unit
    t = time.perf_counter()
    out = r.retrieve(req.query, mode=req.mode, rerank=req.rerank, detail=True, stratify=req.stratify,
                     consensus=req.consensus, host_cap=req.host_cap)
    out["latency_ms"] = round((time.perf_counter() - t) * 1000)
    return out
