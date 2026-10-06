from medrag.evaluation.metrics import chunk_prf, doc_prf, evaluate, f2, lcs_len


def test_lcs_order_kept_non_contiguous():
    assert lcs_len("a b c d".split(), "a x c d".split()) == 3
    assert lcs_len("d c".split(), "c d".split()) == 1


def test_doc_prf():
    p, r, f = doc_prf([1, 2, 3], [2, 3, 4, 5])
    assert (p, r) == (2 / 3, 0.5) and abs(f - f2(p, r)) < 1e-9


def test_chunk_threshold_and_normalisation():
    gold = [{"doc_id": "1", "chunk_text": "Sỏi  thận &amp; niệu quản gây tắc nghẽn"}]
    hit = [{"doc_id": "1", "chunk_text": "SỎI THẬN & niệu quản gây tắc nghẽn đường tiết niệu, cần can thiệp"}]
    miss_doc = [{"doc_id": "2", "chunk_text": hit[0]["chunk_text"]}]
    assert chunk_prf(hit, gold)[:2] == (1.0, 1.0)
    assert chunk_prf(miss_doc, gold)[:2] == (0.0, 0.0)


def test_evaluate_macro_and_missing_query():
    gold = [{"id": 1, "relevant_docs": ["a"], "relevant_chunks": []},
            {"id": 2, "relevant_docs": ["b"], "relevant_chunks": []}]
    pred = [{"id": 1, "relevant_docs": ["a"], "relevant_chunks": []}]
    assert evaluate(pred, gold)["doc_R"] == 0.5
