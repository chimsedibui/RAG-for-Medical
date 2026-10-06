"""Step 1: download every file of the competition HF dataset (query.parquet, links_corpus.parquet, ...)."""
import sys

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

from ..config import get_settings


def run() -> None:
    s = get_settings()
    s.raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        path = snapshot_download(
            repo_id=s.hf_dataset, repo_type="dataset",
            token=s.hf_token or None, local_dir=s.raw_dir,
        )
    except (GatedRepoError, RepositoryNotFoundError, HfHubHTTPError) as e:
        sys.exit(
            f"Cannot access {s.hf_dataset}: {e}\n"
            "The dataset answers 401 without credentials. Set HF_TOKEN in .env "
            "(a token of an account that has access) and retry."
        )
    print(f"Dataset downloaded to {path}")
    for p in sorted(s.raw_dir.rglob("*")):
        if p.is_file() and ".cache" not in p.parts:
            print(f"  {p.relative_to(s.raw_dir)}  {p.stat().st_size/1e6:.1f} MB")
