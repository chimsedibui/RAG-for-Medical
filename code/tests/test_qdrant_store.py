import numpy as np
from qdrant_client import QdrantClient

from medrag.stores.qdrant import QdrantDense
from medrag.schemas import Chunk


def _chunk(i, lang):
    return Chunk(chunk_id=f"{i}:0", doc_id=i, ordinal=0, text="t", context="", parent_text="t", lang=lang)


def test_build_search_and_lang_filter():
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(40, 16)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    chunks = [_chunk(i, "vi" if i % 2 else "zh") for i in range(40)]
    store = QdrantDense(QdrantClient(":memory:"), "t")
    store.build(chunks, vecs, batch=16)
    assert store.count() == 40
    hits = store.search(vecs[7], 3)
    assert hits[0][0] == "7:0" and hits[0][1] > 0.99
    assert all(int(cid.split(":")[0]) % 2 == 0 for cid, _ in store.search(vecs[7], 5, langs=["zh"]))
