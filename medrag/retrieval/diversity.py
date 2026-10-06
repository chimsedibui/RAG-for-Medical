def diversity(view: list[dict]) -> dict:
    """Bias indicators for one result list."""
    hosts = [d.get("host", "?") for d in view]
    langs = [d.get("lang", "?") for d in view]
    return {"n_hosts": len(set(hosts)), "n_langs": len(set(langs)),
            "langs": {l: langs.count(l) for l in set(langs)}, "single_source": len(set(hosts)) <= 1,
            "corroborated": sum(1 for d in view if d.get("support", 0) > 0)}
