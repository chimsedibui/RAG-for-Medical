from medrag.evaluation.devset import LabelStore
from medrag.evaluation.metrics import evaluate


def test_label_store_roundtrip_and_eval_format(tmp_path):
    g, p = tmp_path / "gold.json", tmp_path / "prog.json"
    s = LabelStore(g, p)
    s.set(1, ["10"], [{"doc_id": "11", "chunk_text": "đoạn đúng"}], "done")     # chunk's doc is added to docs
    s.set(2, [], [], "skip")
    s.set(3, ["30"], [], "todo")                                                  # draft survives a reload
    s.set(4, [], [], "done")                                                      # no doc -> not written to gold
    s2 = LabelStore(g, p)
    assert s2.get(1) == {"docs": ["10", "11"], "chunks": [{"doc_id": "11", "chunk_text": "đoạn đúng"}], "status": "done"}
    assert (s2.status(2), s2.status(3), s2.status(4)) == ("skip", "todo", "todo") and s2.get(3)["docs"] == ["30"]
    pred = [{"id": 1, "relevant_docs": ["10", "11"], "relevant_chunks": [{"doc_id": "11", "chunk_text": "đoạn đúng"}]}]
    assert evaluate(pred, list(s2.gold.values()))["chunk_F2"] == 1.0              # gold file feeds eval directly
    s2.set(1, ["10"], [], "todo")                                                 # reopening a query replaces its gold
    assert 1 not in s2.gold
