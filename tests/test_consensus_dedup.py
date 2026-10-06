import numpy as np

from medrag.indexing.dedup import cluster, hamming, simhash
from medrag.retrieval.consensus import DocFeat, agreement, rescore, select
from medrag.stores.lexical import LangBM25


def test_simhash_clusters_repost_not_other_article():
    a = "Sỏi thận gây đau lưng dữ dội. Cần uống nhiều nước và đi khám khi sốt cao kèm tắc nghẽn niệu quản. " * 12
    b = a + " Nguồn: bài viết được đăng lại từ trang khác."   # typical repost: a credit line appended
    c = "Bệnh lao phổi lây qua đường hô hấp, bệnh nhân cần điều trị đủ phác đồ để tránh kháng thuốc. " * 12
    sa, sb, sc = simhash(a), simhash(b), simhash(c)
    assert hamming(sa, sb) <= 3 < hamming(sa, sc)
    cl = cluster({1: sa, 2: sb, 3: sc})
    assert cl[1] == cl[2] == 1 and 3 not in cl


def _f(i, rel, host, cluster=None, lang="vi"):
    return DocFeat(doc_id=i, rel=rel, host=host, cluster=cluster or i, lang=lang)


def test_agreement_counts_independent_hosts_only():
    docs = [_f(1, 0.9, "a.vn"), _f(2, 0.8, "b.cn", lang="zh"), _f(3, 0.8, "b.cn", lang="zh"),
            _f(4, 0.7, "c.vn", cluster=1), _f(5, 0.05, "d.vn")]
    sim = np.eye(5); sim[0, 1:] = sim[1:, 0] = 0.8
    agreement(docs, sim, theta=0.55, rel_min=0.3)
    assert docs[0].support_hosts == {"b.cn"}     # 2 & 3 same host -> 1 source; 4 is a repost of 1 (same cluster) -> not independent
    rescore(docs, 0.3)
    assert docs[0].final > docs[0].rel
    assert docs[4].final < 0.1                   # similarity to the top doc cannot rescue an irrelevant doc (boost is multiplicative)


def test_select_host_cap_and_cutoff():
    docs = [_f(i, 1.0 - i * 0.05, "x.vn") for i in range(4)] + [_f(9, 0.9, "y.cn", lang="zh"), _f(10, 0.1, "z.vn")]
    for d in docs:
        d.final = d.rel
    got = select(docs, doc_min=1, doc_max=8, rel_thr=0.5, host_cap=2)
    assert [d.doc_id for d in got] == [0, 1, 9]  # x.vn capped at 2, z.vn below cut-off
    assert [d.doc_id for d in select(docs, 1, 8, 0.0, 9, dedupe_clusters=False)][:2] == [0, 1]


def test_lang_bm25_isolates_languages():
    idx = LangBM25.build(["1:0", "2:0"], ["sỏi thận gây tắc nghẽn niệu quản", "肾结石导致输尿管梗阻"], ["vi", "zh"])
    assert [c for c, _ in idx.search("sỏi thận", 5, "vi")] == ["1:0"]
    assert idx.search("sỏi thận", 5, "zh") == []
