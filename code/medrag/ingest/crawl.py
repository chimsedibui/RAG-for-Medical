"""Step 2: crawl every URL of links_corpus.parquet. Resumable, polite, parallel.

- raw bytes -> data/html/{id}.bin only with --keep-raw (4.4M pages = hundreds of GB)
- extracted docs -> data/docs.jsonl (append-only)
- outcome per id -> data/crawl_log.jsonl (ok | http_<code> | error | empty); `--retry-failed` re-queues non-ok
Resume = skip ids already in crawl_log with status ok (or any status unless --retry-failed).
"""
import json
import threading
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from urllib.parse import urlparse

import pandas as pd
from tqdm import tqdm

from ..config import get_settings
from ..schemas import Doc
from .extract import to_doc
from .net import Robots, Throttle, make_session


def _read_log(path) -> dict[int, str]:
    out: dict[int, str] = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            out[r["id"]] = r["status"]
    return out


def _interleave(links: pd.DataFrame) -> list[tuple[int, str]]:
    """Round-robin across hosts so every host progresses in parallel (per-host throttle is the
    bottleneck) and any partial crawl is a representative sample, not 'all of cnkang first'."""
    links = links.assign(host=links.url.map(lambda u: urlparse(u).netloc))
    links["rank"] = links.groupby("host").cumcount() / links.groupby("host")["id"].transform("size")
    links = links.sort_values(["rank", "host"])
    return list(zip(links.id.astype(int), links.url))


def fetch_doc(sess, throttle: Throttle, robots: Robots, doc_id: int, url: str, timeout: int,
              raw_dir=None) -> tuple[str, Doc | None]:
    """Fetch + extract one URL -> (status, Doc). status: ok | robots_blocked | http_<code> | empty | blocked | garbled | error:<T>."""
    host = urlparse(url).netloc
    try:
        if not robots.allowed(url):
            return "robots_blocked", None
        throttle.wait(host)
        r = sess.get(url, timeout=timeout, allow_redirects=True)
        if r.status_code != 200:
            return f"http_{r.status_code}", None
        data, ctype = r.content, r.headers.get("Content-Type", "").lower()
        if raw_dir is not None:
            (raw_dir / f"{doc_id}.bin").write_bytes(data)
        doc = to_doc(doc_id, url, data, ctype)
        if doc is None:
            return "empty", None
        if doc.extractor in ("blocked", "garbled"):
            return doc.extractor, None
        return "ok", doc
    except Exception as e:  # noqa: BLE001 - one bad URL must not stop the crawl
        return f"error:{type(e).__name__}", None


def run(limit: int | None = None, retry_failed: bool = False, sample_per_host: int | None = None,
        hosts: list[str] | None = None, keep_raw: bool | None = None,
        exclude_hosts: list[str] | None = None) -> None:
    s = get_settings()
    keep_raw = s.crawl_keep_raw if keep_raw is None else keep_raw
    s.data_dir.mkdir(parents=True, exist_ok=True)
    if keep_raw:
        s.html_dir.mkdir(parents=True, exist_ok=True)
    src = s.raw_dir / "links_corpus.parquet"
    if not src.exists():
        raise SystemExit(f"{src} not found - run `python -m medrag.cli fetch` first.")
    links = pd.read_parquet(src)[["id", "url"]]
    if hosts:
        links = links[links.url.map(lambda u: urlparse(u).netloc).isin(hosts)]
    if exclude_hosts:
        links = links[~links.url.map(lambda u: urlparse(u).netloc).isin(exclude_hosts)]
    if sample_per_host:
        links = links.assign(h=links.url.map(lambda u: urlparse(u).netloc)).groupby("h").sample(
            frac=1, random_state=0).groupby("h").head(sample_per_host).drop(columns="h")
    if limit:
        links = links.sample(n=min(limit, len(links)), random_state=0)   # random, not head: head = one site

    done = _read_log(s.crawl_log)
    todo = _interleave(links[[i not in done or (retry_failed and done[i] != "ok") for i in links.id]])
    print(f"{len(links)} urls, {len(links) - len(todo)} already done, {len(todo)} to crawl")

    sess, throttle = make_session(), Throttle(s.crawl_per_domain_delay)
    robots = Robots(sess, s.crawl_respect_robots)
    write_lock = threading.Lock()
    docs_f = s.docs_file.open("a", encoding="utf-8")
    log_f = s.crawl_log.open("a", encoding="utf-8")

    def work(doc_id: int, url: str) -> str:
        status, doc = fetch_doc(sess, throttle, robots, doc_id, url, s.crawl_timeout, s.html_dir if keep_raw else None)
        if doc is not None:
            with write_lock:
                docs_f.write(json.dumps(doc.to_json(), ensure_ascii=False) + "\n")
                docs_f.flush()
        return status

    def record(i: int, u: str, status: str) -> None:
        with write_lock:
            log_f.write(json.dumps({"id": i, "url": u, "status": status}) + "\n")
            log_f.flush()

    # Bounded in-flight window: never materialise millions of futures.
    window = s.crawl_workers * 4
    try:
        with ThreadPoolExecutor(s.crawl_workers) as ex, tqdm(total=len(todo)) as bar:
            it, pending = iter(todo), {}
            while True:
                while len(pending) < window:
                    nxt = next(it, None)
                    if nxt is None:
                        break
                    pending[ex.submit(work, *nxt)] = nxt
                if not pending:
                    break
                fin, _ = wait(pending, return_when=FIRST_COMPLETED)
                for f in fin:
                    i, u = pending.pop(f)
                    record(i, u, f.result()); bar.update(1)
    finally:
        docs_f.close(); log_f.close()
    summarize()


def summarize() -> None:
    s = get_settings()
    st = _read_log(s.crawl_log)
    counts: dict[str, int] = defaultdict(int)
    for v in st.values():
        counts[v] += 1
    print("crawl status:", dict(sorted(counts.items(), key=lambda kv: -kv[1])))
