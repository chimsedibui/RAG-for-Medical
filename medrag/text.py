"""Text normalisation shared by chunking, lexical search and the local metric."""
import html
import re
import unicodedata

_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_VI = re.compile(r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", re.I)
_MOJIBAKE = re.compile(r"[\u0400-\u04ff\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]")   # Cyrillic / U+FFFD / control chars
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def detect_lang(text: str) -> str:
    s = text[:4000]
    if not s.strip():
        return "other"
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return "other"
    if len(_CJK.findall(s)) / len(letters) > 0.2:
        return "zh"
    if len(_VI.findall(s)) / len(letters) > 0.03:
        return "vi"
    return "en"


def metric_normalize(text: str) -> str:
    """Official-metric normalisation (overview §3.2): NFKC, unescape HTML, lowercase, collapse spaces."""
    text = unicodedata.normalize("NFKC", html.unescape(text)).lower()
    return re.sub(r"\s+", " ", text).strip()


def metric_tokens(text: str) -> list[str]:
    # Official tokenizer is unspecified; whitespace split after normalisation is the working assumption.
    return metric_normalize(text).split()


def lexical_tokens(text: str) -> list[str]:
    """BM25 tokens. vi/en: words + word bigrams (approximates Vietnamese compound words
    without a segmenter). zh: char unigrams + bigrams. Mixed text handled per run."""
    text = unicodedata.normalize("NFKC", html.unescape(text)).lower()
    out: list[str] = []
    for w in _WORD.findall(text):
        if _CJK.search(w):
            cjk = [c for c in w if _CJK.match(c)]
            out.extend(cjk)
            out.extend(a + b for a, b in zip(cjk, cjk[1:]))
        else:
            out.append(w)
    latin = [t for t in out if not _CJK.match(t[0])]
    out.extend(f"{a}_{b}" for a, b in zip(latin, latin[1:]))
    return out


def approx_tokens(text: str) -> int:
    """Cheap token count: whitespace words + one per CJK char. Good enough for budgets."""
    # len//8 floor: unspaced junk (URLs, base64, table rows) must not count as 1 token
    return max(len(text.split()) + len(_CJK.findall(text)), len(text) // 8)


def is_garbled(text: str, threshold: float = 0.1) -> bool:
    """Wrongly decoded page (e.g. GBK bytes read as Cyrillic) or binary. The corpus is vi/en/zh only, so Cyrillic,
    U+FFFD and control characters should be ~absent; >10% of the first 3000 chars means the text is junk."""
    s = text[:3000]
    return bool(s) and len(_MOJIBAKE.findall(s)) / len(s) > threshold
