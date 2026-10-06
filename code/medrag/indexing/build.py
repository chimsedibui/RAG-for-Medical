"""Offline build steps after the crawl: docs.jsonl -> chunks.jsonl -> BM25 + Qdrant indexes, and doc clusters."""
import json

from ..config import get_settings
from ..schemas import Chunk


def build_chunks() -> None:
    from ..ingest.clean import clean_docs
    from .chunking import chunk_doc
    s = get_settings()
    n_docs, n_chunks = 0, 0
    with s.chunks_file.open("w", encoding="utf-8") as out:
        for d in clean_docs(s.docs_file):
            n_docs += 1
            for c in chunk_doc(d):
                out.write(json.dumps(c.to_json(), ensure_ascii=False) + "\n"); n_chunks += 1
    print(f"{n_docs} docs -> {n_chunks} chunks")


def build_indexes() -> None:
    from ..models.embedder import Embedder
    from ..stores.lexical import BM25Index, LangBM25
    from ..stores.qdrant import QdrantDense
    s = get_settings()
    chunks = [Chunk(**json.loads(l)) for l in s.chunks_file.open(encoding="utf-8")]
    ids = [c.chunk_id for c in chunks]
    texts = [c.embed_text for c in chunks]
    BM25Index.build(ids, texts).save(s.index_dir)
    LangBM25.build(ids, texts, [c.lang for c in chunks]).save(s.index_dir)
    emb = Embedder(s.embed_model, s.embed_device)
    QdrantDense().build(chunks, emb.encode_docs([c.embed_text for c in chunks]))
    print(f"indexed {len(ids)} chunks: BM25 -> {s.index_dir}, vectors -> Qdrant '{s.qdrant_collection}'")


def build_doc_clusters() -> None:
    from .dedup import build_clusters
    s = get_settings()
    n, in_cl, n_cl = build_clusters(s.docs_file, s.clusters_file)
    print(f"{n} docs: {in_cl} are near-duplicates, in {n_cl} clusters -> {s.clusters_file}")
