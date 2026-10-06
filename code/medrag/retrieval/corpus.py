"""In-memory view of the indexed corpus: chunks, per-doc display metadata and near-duplicate clusters."""
import json
from urllib.parse import urlparse

from ..config import Settings
from ..schemas import Chunk


class Corpus:
    def __init__(self, s: Settings):
        self.chunks: dict[str, Chunk] = {}
        self.point_id: dict[str, int] = {}         # chunk_id -> Qdrant point id (= order in chunks.jsonl)
        for i, line in enumerate(s.chunks_file.open(encoding="utf-8")):
            c = Chunk(**json.loads(line))
            self.chunks[c.chunk_id] = c
            self.point_id[c.chunk_id] = i

        self.docs_meta: dict[int, dict] = {}       # id -> url/title/host for display + source independence
        if s.docs_file.exists():
            for line in s.docs_file.open(encoding="utf-8"):
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.docs_meta.setdefault(d["id"], {"url": d["url"], "title": d["title"], "lang": d["lang"],
                                                    "host": urlparse(d["url"]).netloc})

        self.cluster_of: dict[int, int] = {}
        if s.clusters_file.exists():
            self.cluster_of = {int(k): v for k, v in json.loads(s.clusters_file.read_text()).items()}
