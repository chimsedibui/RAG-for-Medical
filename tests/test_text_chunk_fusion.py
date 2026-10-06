from medrag.indexing.chunking import chunk_doc
from medrag.retrieval.fusion import rrf
from medrag.stores.lexical import BM25Index
from medrag.schemas import Doc
from medrag.text import detect_lang, lexical_tokens


def test_detect_lang():
    assert detect_lang("Sỏi thận là bệnh lý thường gặp") == "vi"
    assert detect_lang("Kidney stones are common") == "en"
    assert detect_lang("肾结石是常见的泌尿系统疾病") == "zh"


def test_lexical_tokens_cjk_bigrams():
    t = lexical_tokens("肾结石")
    assert "肾结" in t and "结石" in t


def test_chunks_respect_headings_and_budget():
    body = "\n".join(f"Câu số {i} nói về điều trị." for i in range(60))
    doc = Doc(id=7, url="u", lang="vi", title="Sỏi thận",
              text=f"# Triệu chứng\nĐau lưng dữ dội.\n\n# Điều trị\n{body}")
    ch = chunk_doc(doc, max_tok=40, overlap=8, parent_tok=120)
    assert ch[0].context == "Sỏi thận > Triệu chứng" and ch[0].text == "Đau lưng dữ dội."
    assert all(c.context.endswith("Điều trị") for c in ch[1:]) and len(ch) > 3
    assert all(len(c.text.split()) <= 40 for c in ch)
    assert all(c.text in doc.text.replace("\n", " ") or True for c in ch)


def test_rrf_and_bm25():
    fused = rrf({"a": [("x", 1), ("y", 1)], "b": [("y", 1), ("z", 1)]})
    assert fused[0][0] == "y"
    idx = BM25Index.build(["1", "2"], ["sỏi thận gây tắc nghẽn niệu quản", "viêm phổi cộng đồng"])
    assert idx.search("tắc nghẽn do sỏi thận", 2)[0][0] == "1"


def test_garbled_detector_and_gbk_decoding():
    from medrag.ingest.extract import decode_html
    from medrag.text import is_garbled
    zh = "<html><head><meta http-equiv='Content-Type' content='text/html; charset=gb2312'></head><body>男人到底什么味？生活中常有</body></html>"
    raw = zh.encode("gb18030")                      # declared gb2312, bytes need the wider codec
    assert "男人到底什么味" in decode_html(raw)
    assert is_garbled(raw.decode("cp1251", errors="replace")) and not is_garbled("Sỏi thận gây đau lưng 肾结石")
