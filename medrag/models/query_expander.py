"""Query understanding. Queries are Vietnamese; the corpus is vi + en + zh.

expand() returns the original query plus (optionally) LLM translations into English and Chinese and a
medical-term-normalised rewrite. Every variant is retrieved separately and fused (RRF), so a miss
in one language does not sink the query. With no LLM configured it degrades to [query]."""
import json

from ..config import get_settings

_PROMPT = """Bạn là chuyên gia thuật ngữ y khoa. Với câu hỏi y khoa tiếng Việt dưới đây, trả về JSON với các khóa:
"en": bản dịch tiếng Anh dùng đúng thuật ngữ y khoa (MeSH/ICD nếu biết),
"zh": bản dịch tiếng Trung giản thể,
"vi_expanded": câu hỏi viết lại đầy đủ, mở rộng viết tắt và thêm 1-2 từ đồng nghĩa y khoa.
Chỉ trả JSON, không giải thích. Không thêm thông tin không có trong câu hỏi.

Câu hỏi: {q}"""


class QueryExpander:
    def __init__(self):
        s = get_settings()
        self.enabled = bool(s.llm_base_url)
        if self.enabled:
            from openai import OpenAI
            self.client, self.model = OpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key), s.llm_model

    def expand(self, query: str) -> dict[str, str]:
        out = {"orig": query}
        if not self.enabled:
            return out
        try:
            r = self.client.chat.completions.create(
                model=self.model, temperature=0, max_tokens=300,
                messages=[{"role": "user", "content": _PROMPT.format(q=query)}],
                response_format={"type": "json_object"},
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},   # Qwen3: skip <think>
            )
            data = json.loads(r.choices[0].message.content)
            for k in ("en", "zh", "vi_expanded"):
                if isinstance(data.get(k), str) and data[k].strip():
                    out[k] = data[k].strip()
        except Exception:  # noqa: BLE001 - expansion is best-effort
            pass
        return out
