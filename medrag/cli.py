"""python -m medrag.cli <fetch|crawl|repair|stats|cluster|chunk|index|bias|dev-sample|dev-replace|label|eval|submit> ...

Pipeline order: fetch -> crawl -> cluster -> chunk -> index -> submit.  Imports are lazy so cheap
commands (stats, eval) do not pull in torch / qdrant."""
import argparse
import json
from pathlib import Path


def cmd_fetch(_):
    from .ingest.fetch_dataset import run; run()


def cmd_crawl(a):
    from .ingest.crawl import run; run(a.limit, a.retry_failed, a.sample_per_host, a.hosts, a.keep_raw, a.exclude_hosts)


def cmd_repair(a):
    from .ingest.repair import run; run(a.dry_run)


def cmd_stats(_):
    from .ingest.crawl import summarize; summarize()


def cmd_cluster(_):
    from .indexing.build import build_doc_clusters; build_doc_clusters()


def cmd_chunk(_):
    from .indexing.build import build_chunks; build_chunks()


def cmd_index(_):
    from .indexing.build import build_indexes; build_indexes()


def cmd_bias(a):
    from .evaluation.bias import run; run(a.n, a.no_stratify, a.no_consensus, a.host_cap)


def cmd_dev_sample(a):
    from .evaluation.devset import run; run(a.n, a.seed, a.out)


def cmd_dev_replace(a):
    from .evaluation.devset import add_replacements
    new = add_replacements(a.target, a.seed)
    print(f"added {len(new)} replacement queries: {new}")


def cmd_label(a):
    import uvicorn
    uvicorn.run("medrag.app.labeler:app", host="127.0.0.1", port=a.port)


def cmd_eval(a):
    from .evaluation.metrics import evaluate
    pred = json.loads(Path(a.pred).read_text(encoding="utf-8"))
    gold = json.loads(Path(a.gold).read_text(encoding="utf-8"))
    for k, v in evaluate(pred, gold).items():
        print(f"{k:>10}: {v:.4f}" if isinstance(v, float) else f"{k:>10}: {v}")


def cmd_submit(a):
    from .retrieval.pipeline import Retriever
    from .submission import build_submission, load_queries, validate_submission
    r = Retriever(rerank=not a.no_rerank, output_unit=a.output_unit)
    out = Path(a.out)
    only = {int(x["id"]) for x in json.loads(Path(a.queries).read_text(encoding="utf-8"))} if a.queries else None
    zp = build_submission(r, out, a.limit, only)
    ids = (only or set(load_queries()["id"].astype(int))) if not a.limit else None
    errs = validate_submission(out, ids) if ids else []
    print(f"wrote {zp}", "| validation OK" if not errs else f"| {len(errs)} problems: {errs[:5]}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="medrag")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("fetch", help="download the HF dataset")
    c = sub.add_parser("crawl", help="crawl links_corpus.parquet")
    c.add_argument("--limit", type=int); c.add_argument("--retry-failed", action="store_true")
    c.add_argument("--sample-per-host", type=int, help="probe: N random URLs per host")
    c.add_argument("--hosts", nargs="+"); c.add_argument("--exclude-hosts", nargs="+"); c.add_argument("--keep-raw", action="store_true", default=None)
    rp = sub.add_parser("repair", help="re-fetch garbled docs (wrong charset) and rewrite docs.jsonl")
    rp.add_argument("--dry-run", action="store_true")
    sub.add_parser("stats", help="crawl status summary")
    sub.add_parser("cluster", help="near-duplicate doc clusters")
    sub.add_parser("chunk", help="docs.jsonl -> chunks.jsonl")
    sub.add_parser("index", help="chunks.jsonl -> BM25 + Qdrant")
    b = sub.add_parser("bias", help="language/host bias report on random queries")
    b.add_argument("--n", type=int, default=40); b.add_argument("--no-stratify", action="store_true")
    b.add_argument("--no-consensus", action="store_true"); b.add_argument("--host-cap", type=int, default=None)
    d = sub.add_parser("dev-sample", help="draw a dev subset of the real queries (stratified)")
    d.add_argument("--n", type=int, default=80); d.add_argument("--seed", type=int, default=42)
    d.add_argument("--out", default="data/dev/dev_queries.json")
    r = sub.add_parser("dev-replace", help="replace skipped dev queries with unused ones of the same stratum")
    r.add_argument("--target", type=int, default=80); r.add_argument("--seed", type=int, default=7)
    lb = sub.add_parser("label", help="web tool to hand-label the dev queries"); lb.add_argument("--port", type=int, default=8010)
    e = sub.add_parser("eval", help="score a prediction file against a gold file"); e.add_argument("pred"); e.add_argument("gold")
    m = sub.add_parser("submit", help="run all queries, write submission .json + .zip")
    m.add_argument("--out", default="data/submissions/submission.json")
    m.add_argument("--limit", type=int); m.add_argument("--no-rerank", action="store_true")
    m.add_argument("--queries", help="json list of {id,...} (e.g. dev_gold.json): run only these ids")
    m.add_argument("--output-unit", choices=["child", "parent"], default="child")
    return ap


COMMANDS = {"fetch": cmd_fetch, "crawl": cmd_crawl, "repair": cmd_repair, "stats": cmd_stats, "cluster": cmd_cluster, "chunk": cmd_chunk,
            "index": cmd_index, "bias": cmd_bias, "dev-sample": cmd_dev_sample, "dev-replace": cmd_dev_replace, "label": cmd_label, "eval": cmd_eval, "submit": cmd_submit}


def main():
    a = build_parser().parse_args()
    COMMANDS[a.cmd](a)


if __name__ == "__main__":
    main()
