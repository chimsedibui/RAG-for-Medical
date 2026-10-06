from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    hf_token: str = ""
    hf_dataset: str = "AIGuruTinix/ViBioMIR"

    crawl_workers: int = 16
    crawl_per_domain_delay: float = 1.0
    crawl_timeout: int = 30
    crawl_respect_robots: bool = True
    crawl_keep_raw: bool = False      # raw HTML of 4.4M pages would be hundreds of GB

    chunk_max_tokens: int = 220
    chunk_overlap_tokens: int = 40
    parent_max_tokens: int = 700

    qdrant_url: str = "http://localhost:6333"   # ":memory:" works for tests
    qdrant_collection: str = "chunks"
    qdrant_quantize: bool = True              # int8 scalar quantisation, vectors on disk

    embed_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    embed_device: str = "cuda"
    llm_base_url: str = ""
    llm_api_key: str = "EMPTY"
    llm_model: str = "Qwen/Qwen3-8B"

    bm25_candidates: int = 100
    dense_candidates: int = 100
    rrf_k: int = 60
    rerank_candidates: int = 60
    doc_min: int = 1
    doc_max: int = 8
    doc_rel_threshold: float = 0.55
    chunk_max_per_doc: int = 3
    stratify: bool = True            # retrieve per language, then fuse
    per_stratum: int = 30            # candidates kept per language before fusion
    consensus: bool = True           # cross-source agreement re-scoring
    agree_theta: float = 0.62        # cosine(best chunks); = ~99th pct of RANDOM chunk-pair cosine (bge-m3: 0.61), so "agree" means far above chance
    agree_rel_min: float = 0.3       # supporter must score >= this x best
    agree_alpha: float = 0.3         # weight of log(1+support)
    host_cap: int = 2                # max docs per host in the final list
    dedupe_clusters: bool = False    # keep only 1 doc per near-duplicate cluster (may cost recall)
    chunk_rel_threshold: float = 0.6

    # derived paths
    @property
    def raw_dir(self) -> Path: return self.data_dir / "raw"          # HF files
    @property
    def html_dir(self) -> Path: return self.data_dir / "html"        # raw crawled bytes
    @property
    def docs_file(self) -> Path: return self.data_dir / "docs.jsonl"
    @property
    def crawl_log(self) -> Path: return self.data_dir / "crawl_log.jsonl"
    @property
    def chunks_file(self) -> Path: return self.data_dir / "chunks.jsonl"
    @property
    def clusters_file(self) -> Path: return self.data_dir / "doc_clusters.json"
    @property
    def dev_dir(self) -> Path: return self.data_dir / "dev"          # dev queries + hand-made gold labels
    @property
    def index_dir(self) -> Path: return self.data_dir / "index"


@lru_cache
def get_settings() -> Settings:
    return Settings()
