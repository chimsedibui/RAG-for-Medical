from dataclasses import asdict, dataclass, field


@dataclass
class Doc:
    id: int
    url: str
    lang: str            # vi | en | zh | other
    title: str
    text: str            # extracted main text, markdown-ish (# headings kept)
    source: str = ""     # domain
    extractor: str = ""  # which extractor produced `text`

    def to_json(self) -> dict: return asdict(self)


@dataclass
class Chunk:
    chunk_id: str        # f"{doc_id}:{ordinal}"
    doc_id: int
    ordinal: int
    text: str            # verbatim slice of Doc.text -> goes into submission as chunk_text
    context: str         # heading path / title; used for embedding & rerank only
    parent_text: str     # larger window around `text` (small2big); optional output
    lang: str = "other"

    def to_json(self) -> dict: return asdict(self)

    @property
    def embed_text(self) -> str:
        return f"{self.context}\n{self.text}" if self.context else self.text


@dataclass
class DocHit:
    doc_id: int
    score: float
    chunks: list[tuple[Chunk, float]] = field(default_factory=list)
