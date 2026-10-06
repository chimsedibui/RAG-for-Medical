"""Dense embedding model wrapper. Vector storage/search lives in stores/qdrant.py."""
import numpy as np


class Embedder:
    """Wraps sentence-transformers. bge-m3 / Qwen3-Embedding both load this way."""
    def __init__(self, model: str, device: str = "cuda", query_prompt: str | None = None):
        from sentence_transformers import SentenceTransformer
        kw = {"torch_dtype": "float16"} if device.startswith("cuda") else {}   # halves VRAM, ~2x faster
        self.model = SentenceTransformer(model, device=device, trust_remote_code=True, model_kwargs=kw)
        self.model.max_seq_length = 512
        self.query_prompt = query_prompt   # e.g. Qwen3-Embedding wants an instruction prefix on queries

    def encode_docs(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 show_progress_bar=True).astype("float32")

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        if self.query_prompt:
            texts = [self.query_prompt + t for t in texts]
        return self.model.encode(texts, normalize_embeddings=True).astype("float32")
