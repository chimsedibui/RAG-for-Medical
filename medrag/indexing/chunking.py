"""Structure-aware parent-child chunking (small2big).

Child  = retrieval unit (<= CHUNK_MAX_TOKENS), a verbatim slice of the doc text, never crossing a heading.
Parent = surrounding window (<= PARENT_MAX_TOKENS) inside the same section.
Context= "Title > H1 > H2" path, prepended for embedding/reranking only (never in submitted chunk_text).

Because chunk-level scoring is overlap(c,g)=LCS/|g| (normalised by the reference chunk only), what we
SUBMIT as chunk_text (child vs parent) is a tunable knob: see retrieval/pipeline.py `output_unit`.
"""
import re

from ..config import get_settings
from ..schemas import Chunk, Doc
from ..text import approx_tokens

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_SENT = re.compile(r"(?<=[.!?。！？])\s+|\n+")


def _sections(text: str, title: str):
    """Yield (heading_path, body) per section."""
    path: list[tuple[int, str]] = []
    buf: list[str] = []
    out = []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            out.append((" > ".join([title] * bool(title) + [h for _, h in path]), body))
        buf.clear()

    for line in text.split("\n"):
        m = _HEADING.match(line)
        if m:
            flush()
            lvl = len(m.group(1))
            while path and path[-1][0] >= lvl:
                path.pop()
            path.append((lvl, m.group(2).strip()))
        else:
            buf.append(line)
    flush()
    return out


def _windows(body: str, max_tok: int, overlap: int) -> list[str]:
    """Greedy sentence packing into <=max_tok windows with sentence-level overlap."""
    sents = [x.strip() for x in _SENT.split(body) if x and x.strip()]
    out, cur, cur_tok = [], [], 0
    pieces = []
    for sent in sents:                       # hard-split sentences longer than the budget (tables, unpunctuated text)
        t = approx_tokens(sent)
        if t <= max_tok:
            pieces.append(sent); continue
        step = max(1, len(sent) * max_tok // t)
        i = 0
        while i < len(sent):
            j = min(len(sent), i + step)
            if j < len(sent) and (sp := sent.rfind(" ", i, j)) > i + step // 2:
                j = sp                      # snap to a space for latin text
            pieces.append(sent[i:j].strip()); i = j
    for sent in pieces:
        t = approx_tokens(sent)
        if cur and cur_tok + t > max_tok:
            out.append(" ".join(cur))
            keep, k_tok = [], 0
            for prev in reversed(cur):                # carry trailing sentences as overlap
                pt = approx_tokens(prev)
                if k_tok + pt > overlap:
                    break
                keep.insert(0, prev); k_tok += pt
            cur, cur_tok = keep, k_tok
        cur.append(sent); cur_tok += t
    if cur:
        out.append(" ".join(cur))
    return out


def chunk_doc(doc: Doc, max_tok: int | None = None, overlap: int | None = None,
              parent_tok: int | None = None) -> list[Chunk]:
    s = get_settings()
    max_tok, overlap, parent_tok = (max_tok or s.chunk_max_tokens, overlap if overlap is not None
                                    else s.chunk_overlap_tokens, parent_tok or s.parent_max_tokens)
    chunks: list[Chunk] = []
    for ctx, body in _sections(doc.text, doc.title):
        parent = body if approx_tokens(body) <= parent_tok else None
        for w in _windows(body, max_tok, overlap):
            chunks.append(Chunk(
                chunk_id=f"{doc.id}:{len(chunks)}", doc_id=doc.id, ordinal=len(chunks),
                text=w, context=ctx, parent_text=parent or _parent_window(body, w, parent_tok),
                lang=doc.lang,
            ))
    return chunks


def _parent_window(body: str, child: str, parent_tok: int) -> str:
    """Section too long: take a window of ~parent_tok tokens centred on the child."""
    i = body.find(child[:80])
    if i < 0:
        return child
    words = body.split()
    before = len(body[:i].split())
    half = parent_tok // 2
    lo = max(0, before - half)
    return " ".join(words[lo: lo + parent_tok])
