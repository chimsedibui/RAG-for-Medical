class CrossEncoderReranker:
    """bge-reranker-v2-m3 (multilingual, handles vi query vs en/zh passage)."""
    def __init__(self, model: str, device: str = "cuda"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model, device=device, max_length=512, trust_remote_code=True)

    def score(self, query: str, passages: list[str], batch_size: int = 32) -> list[float]:
        if not passages:
            return []
        return [float(x) for x in self.model.predict([(query, p) for p in passages],
                                                     batch_size=batch_size, show_progress_bar=False)]


class NullReranker:
    def score(self, query: str, passages: list[str], **_) -> list[float]:
        return [0.0] * len(passages)
