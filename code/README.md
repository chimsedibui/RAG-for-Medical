# R2AI2026 – Truy hồi tài liệu y khoa đa ngôn ngữ

Bối cảnh đề: [../context/01-overview.png](../context/01-overview.png), metric: [02-evaluation.png](../context/02-evaluation.png), dữ liệu: [03-data.png](../context/03-data.png), định dạng nộp: [04-submission.png](../context/04-submission.png).

## Khác biệt cốt lõi so với dự án luật (folder 4)
| | Luật (4) | Cuộc thi này |
|---|---|---|
| Đầu ra | Câu trả lời sinh bởi LLM + trích dẫn | **Không sinh câu trả lời.** Chỉ nộp `relevant_docs` + `relevant_chunks` (chunk_text lấy nguyên văn) |
| Ngôn ngữ | vi | Query vi → tài liệu **vi + en + zh** (cross-lingual) |
| Chấm điểm | Chất lượng câu trả lời | P/R/**F2** (recall nặng gấp 4 lần precision) ở cấp doc và cấp chunk, macro-average |
| Rủi ro chính | Hallucination | Sai doc / sai đoạn; "độ chính xác" = chọn đúng K và đúng ranh giới chunk |
| LLM | tuỳ | Chỉ open-weights ≤15B, phát hành trước 01/08/2026; cấm GPT/Gemini… |

Hệ quả thiết kế: bỏ tầng generation; dồn công sức vào (1) corpus sạch, (2) retrieval đa ngôn ngữ, (3) rerank, (4) **cắt ngưỡng số doc/chunk để tối đa F2**, (5) chunking khớp với chunk tham chiếu.

Chi tiết metric chunk: `overlap = LCS_token(c,g)/|g|` ≥ 0.4 (chuẩn hoá chỉ chia cho độ dài chunk tham chiếu `g`). Nên một chunk *dài hơn* chứa trọn `g` vẫn đạt 1.0; precision chỉ phụ thuộc số chunk nộp. Đây là đòn bẩy: `--output-unit parent` nộp đoạn lớn (small2big). Cần đo trên dev; nếu BTC giới hạn độ dài thì bỏ.

## Kiến trúc
```
links_corpus.parquet ─crawl─▶ data/html/*.bin ─extract─▶ docs.jsonl ─chunk─▶ chunks.jsonl ─index─▶ BM25 + dense
query (vi) ─▶ expand (vi/en/zh, LLM cục bộ, tuỳ chọn) ─▶ BM25+dense mỗi biến thể ─▶ RRF
          ─▶ cross-encoder rerank (bge-reranker-v2-m3) ─▶ gộp theo doc ─▶ ngưỡng thích nghi ─▶ submission.json/zip
```
## Cấu trúc source
```
medrag/
  config.py  schemas.py  text.py     # settings (.env), Doc/Chunk, chuẩn hoá text dùng chung
  ingest/       fetch_dataset · crawl (+ net: session/throttle/robots) · extract · clean
  indexing/     chunking (parent–child) · dedup (simhash) · build (chunk / index / cluster)
  models/       embedder (bge-m3) · reranker (bge-reranker-v2-m3) · query_expander (LLM cục bộ, tuỳ chọn)
  stores/       lexical (BM25 theo ngôn ngữ) · qdrant (vector dense)
  retrieval/    pipeline (Retriever) · corpus · fusion (RRF) · consensus · diversity
  evaluation/   metrics (công thức của BTC, chấm dev) · bias (báo cáo thiên lệch) · devset (lấy mẫu + lưu nhãn)
  submission.py                      # sinh + kiểm tra file nộp
  app/          api.py (dev UI) · labeler.py (web gán nhãn dev set) · web/*.html
  cli.py                             # python -m medrag.cli <lệnh>
scripts/run_pipeline.sh              # cluster -> chunk -> index -> chạy API
tests/
```
- Chunking: parent–child, không cắt qua heading; `context` (Title > H1 > H2) chỉ dùng khi embed/rerank, không đưa vào `chunk_text`.
- Lexical: BM25 với token gồm bigram cho vi/zh. Dense: bge-m3 lưu trong Qdrant. Metric: bản cài lại đúng công thức BTC.
- Thêm bước mới: ingest → `ingest/`, mô hình → `models/`, backend lưu trữ/tìm kiếm → `stores/`, logic xếp hạng/chọn → `retrieval/`; đăng ký lệnh trong `cli.py`.
- Không ghi file log: kết quả in ra terminal (cần thì `| tee`). Chỉ có `data/crawl_log.jsonl` là trạng thái để crawl resume.

## Chạy
```bash
cp .env.example .env            # điền HF_TOKEN
uv sync --group dev --extra ml  # --extra llm nếu dùng query expansion
uv run python -m medrag.cli fetch          # query.parquet, links_corpus.parquet
uv run python -m medrag.cli crawl          # resume được; --limit N để thử; --retry-failed
uv run python -m medrag.cli stats
uv run python -m medrag.cli cluster && uv run python -m medrag.cli chunk && uv run python -m medrag.cli index   # hoặc scripts/run_pipeline.sh
uv run python -m medrag.cli submit --limit 20 --out ../data/submissions/dev.json
uv run python -m medrag.cli eval pred.json gold.json     # khi có nhãn
uv run pytest
uv run uvicorn medrag.app.api:app --port 8000   # dev UI
uv run python -m medrag.cli dev-sample          # 80 query dev -> ../data/dev/dev_queries.json
uv run python -m medrag.cli label               # web gán nhãn :8010 -> ../data/dev/dev_gold.json (định dạng nộp, dùng cho `eval`)
```

## Trạng thái
Đã kiểm thử: metric, chunking, BM25, RRF, crawl+extract (3 URL wiki vi/en/zh thật). **Chưa chạy**: index dense, rerank, query expansion (cần GPU/model/LLM).

## Việc tiếp theo (ưu tiên)
1. Có `HF_TOKEN` → xem `query.parquet`/`links_corpus.parquet`, thống kê domain & ngôn ngữ, crawl toàn bộ, đọc `stats` để xử lý domain bị chặn/JS-render.
2. **Dev set**: nếu dataset không có nhãn, sinh query tổng hợp từ chunk (LLM cục bộ) + tự gán nhãn nhỏ bằng tay ~100 query; không có dev thì không chỉnh được ngưỡng F2.
3. Đo ablation theo thứ tự: BM25 / dense / hybrid → +rerank → +query expansion en/zh → child vs parent → ngưỡng DOC_*/CHUNK_*.
4. Thử embedding khác: Qwen3-Embedding-0.6B/4B/8B, reranker Qwen3-Reranker; fine-tune nhẹ nếu có cặp (query, doc).
5. Nộp: public ≤10 lượt/ngày; private tổng 5 lượt; nhớ viết working-notes paper.
