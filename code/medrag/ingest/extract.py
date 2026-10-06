"""Raw bytes -> Doc. Raw responses are kept on disk so extraction can be re-run/compared
(the organisers' reference chunks come from THEIR extraction of the same URLs, so being
able to switch extractors cheaply matters for chunk-level recall)."""
import io
import re
from urllib.parse import urlparse

from ..schemas import Doc
from ..text import detect_lang, is_garbled


def _from_pdf(data: bytes) -> tuple[str, str]:
    from pypdf import PdfReader
    r = PdfReader(io.BytesIO(data))
    pages = [p.extract_text() or "" for p in r.pages]
    title = (r.metadata.title if r.metadata and r.metadata.title else "") or ""
    return title, "\n\n".join(pages)


_CHARSET = re.compile(rb"charset\s*=\s*[\"']?\s*([\w-]+)", re.I)


def decode_html(data: bytes, content_type: str = "") -> str:
    """bytes -> str with an explicit charset policy. Sniffing alone mis-decoded GBK pages (declared `gb2312`, which
    Python's strict codec rejects on extended chars) as Cyrillic. Order: BOM/utf-8 strict, then the declared charset
    (gb2312/gbk widened to gb18030, a superset), then gb18030, then lossy utf-8."""
    if data[:3] == b"\xef\xbb\xbf":
        return data[3:].decode("utf-8", errors="replace")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    m = re.search(r"charset=([\w-]+)", content_type or "", re.I) or _CHARSET.search(data[:4096])
    declared = (m.group(1).decode() if isinstance(m.group(1), bytes) else m.group(1)).lower() if m else ""
    if declared in ("gb2312", "gbk", "gb18030", "x-gbk"):
        declared = "gb18030"
    for enc in (declared, "gb18030"):
        if enc:
            try:
                return data.decode(enc)
            except (LookupError, UnicodeDecodeError):
                continue
    return data.decode("utf-8", errors="replace")


def _from_html(data: bytes, url: str, content_type: str = "") -> tuple[str, str, str]:
    import trafilatura
    html = decode_html(data, content_type)       # pass str, not bytes: charset is decided here, not guessed downstream
    # markdown output keeps headings/lists/tables -> structure-aware chunking downstream
    text = trafilatura.extract(
        html, url=url, output_format="markdown", include_tables=True, include_links=False,
        include_comments=False, favor_recall=True,
    )
    extractor = "trafilatura"
    if not text or len(text) < 200:
        text = re.sub(r"(?s)<(script|style|noscript)\b.*?</\1>", " ", html)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[ \t]+", " ", text)
        extractor = "regex-fallback"
    meta = trafilatura.extract_metadata(html)
    title = (meta.title if meta and meta.title else "") or ""
    return title, text or "", extractor


BLOCK_MARKERS = ("滑动拼图验证", "Just a moment...", "Attention Required! | Cloudflare", "访问验证")


def looks_blocked(text: str) -> bool:
    """Captcha / bot-wall pages come back HTTP 200; never let them into the corpus as 'ok'."""
    return len(text) < 150 or any(m in text[:2000] for m in BLOCK_MARKERS)


def to_doc(doc_id: int, url: str, data: bytes, content_type: str) -> Doc | None:
    if "pdf" in content_type or url.lower().split("?")[0].endswith(".pdf"):
        title, text = _from_pdf(data)
        extractor = "pypdf"
    else:
        title, text, extractor = _from_html(data, url, content_type)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) < 50:
        return None
    if looks_blocked(text):
        return Doc(id=-1, url=url, lang="other", title="", text="", extractor="blocked")
    if is_garbled(text):
        return Doc(id=-1, url=url, lang="other", title="", text="", extractor="garbled")
    return Doc(id=doc_id, url=url, lang=detect_lang(text), title=title.strip(),
               text=text, source=urlparse(url).netloc, extractor=extractor)
