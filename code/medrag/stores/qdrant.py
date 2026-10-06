"""Dense vector store on Qdrant. Point id = ordinal in chunks.jsonl; payload carries chunk_id/doc_id/lang
so the pipeline can later filter by language or doc (e.g. 'only vi+en', 'restrict to these docs')."""
import numpy as np
from qdrant_client import QdrantClient, models

from ..config import get_settings


class QdrantDense:
    def __init__(self, client: QdrantClient | None = None, collection: str | None = None):
        s = get_settings()
        self.client = client or QdrantClient(url=s.qdrant_url, timeout=120)
        self.collection = collection or s.qdrant_collection

    def build(self, chunks: list, vectors: np.ndarray, batch: int = 512, recreate: bool = True) -> None:
        s = get_settings()
        if recreate and self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=vectors.shape[1], distance=models.Distance.COSINE,
                                                   on_disk=True),
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128, on_disk=True),
                quantization_config=models.ScalarQuantization(scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8, quantile=0.99, always_ram=True)) if s.qdrant_quantize else None,
            )
        for lo in range(0, len(chunks), batch):
            sl = slice(lo, lo + batch)
            self.client.upsert(self.collection, wait=True, points=models.Batch(
                ids=list(range(lo, min(lo + batch, len(chunks)))),
                vectors=vectors[sl].tolist(),
                payloads=[{"chunk_id": c.chunk_id, "doc_id": c.doc_id, "lang": c.lang} for c in chunks[sl]],
            ))
        for field, kind in (("doc_id", models.PayloadSchemaType.INTEGER), ("lang", models.PayloadSchemaType.KEYWORD)):
            self.client.create_payload_index(self.collection, field, kind)

    def search(self, qvec: np.ndarray, k: int, langs: list[str] | None = None) -> list[tuple[str, float]]:
        flt = models.Filter(must=[models.FieldCondition(key="lang", match=models.MatchAny(any=langs))]) if langs else None
        res = self.client.query_points(
            self.collection, query=qvec.tolist(), limit=k, query_filter=flt, with_payload=["chunk_id"],
            search_params=models.SearchParams(quantization=models.QuantizationSearchParams(rescore=True)),
        )
        return [(p.payload["chunk_id"], float(p.score)) for p in res.points]

    def fetch_vectors(self, point_ids: list[int]) -> np.ndarray:
        pts = {p.id: p.vector for p in self.client.retrieve(self.collection, ids=point_ids, with_vectors=True)}
        return np.array([pts[i] for i in point_ids], dtype="float32")

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count
