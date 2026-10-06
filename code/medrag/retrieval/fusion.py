def rrf(rankings: dict[str, list[tuple[str, float]]], k: int = 60,
        weights: dict[str, float] | None = None) -> list[tuple[str, float]]:
    """Reciprocal rank fusion over named ranked lists of (chunk_id, score)."""
    acc: dict[str, float] = {}
    for name, hits in rankings.items():
        w = (weights or {}).get(name, 1.0)
        seen: set[str] = set()
        for cid, _ in hits:
            if cid in seen:
                continue
            seen.add(cid)
            acc[cid] = acc.get(cid, 0.0) + w / (k + len(seen))
    return sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))
