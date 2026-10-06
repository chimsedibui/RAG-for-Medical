"""Re-fetch docs whose stored text is garbled (pre-fix extractor mis-decoded GBK pages) and rewrite docs.jsonl.

Docs that are still garbled / unreachable after the fix are dropped from docs.jsonl and logged in crawl_log.jsonl.
After a repair run `cluster -> chunk -> index` again (chunk ordinals / Qdrant point ids shift)."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from tqdm import tqdm

from ..config import get_settings
from ..text import is_garbled
from .clean import _read
from .crawl import fetch_doc
from .net import Robots, Throttle, make_session


def find_garbled(docs_file) -> dict[int, str]:
    return {d["id"]: d["url"] for d in _read(docs_file) if is_garbled(d["text"])}


def run(dry_run: bool = False) -> None:
    s = get_settings()
    bad = find_garbled(s.docs_file)
    hosts: dict[str, int] = {}
    for u in bad.values():
        hosts[urlparse(u).netloc] = hosts.get(urlparse(u).netloc, 0) + 1
    print(f"{len(bad)} garbled docs: {hosts}")
    if dry_run or not bad:
        return
    sess, throttle = make_session(), Throttle(s.crawl_per_domain_delay)
    robots = Robots(sess, s.crawl_respect_robots)
    fixed: dict[int, dict | None] = {}
    lock = threading.Lock()

    def work(item):
        i, u = item
        status, doc = fetch_doc(sess, throttle, robots, i, u, s.crawl_timeout)
        with lock:
            fixed[i] = doc.to_json() if doc is not None else None
        return i, u, status

    with ThreadPoolExecutor(s.crawl_workers) as ex:
        results = list(tqdm(ex.map(work, bad.items()), total=len(bad)))

    tmp = s.docs_file.with_suffix(".jsonl.new")
    n_in = n_out = 0
    seen: set[int] = set()
    with s.docs_file.open(encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        for line in src:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d["id"] in seen:               # keep the first record per id, as clean._read does
                continue
            seen.add(d["id"]); n_in += 1
            if d["id"] in fixed:
                d = fixed[d["id"]]
                if d is None:
                    continue
            dst.write(json.dumps(d, ensure_ascii=False) + "\n"); n_out += 1
    ok = sum(v is not None for v in fixed.values())
    assert n_out == n_in - (len(fixed) - ok), "repair would lose unrelated docs - aborting before replacing docs.jsonl"
    tmp.replace(s.docs_file)
    with s.crawl_log.open("a", encoding="utf-8") as log:
        for i, u, status in results:
            if status != "ok":
                log.write(json.dumps({"id": i, "url": u, "status": status}) + "\n")
    print(f"repaired {ok}/{len(bad)}; dropped {len(bad) - ok}; docs.jsonl now {n_out} docs")
