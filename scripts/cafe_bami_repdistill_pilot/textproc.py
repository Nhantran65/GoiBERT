"""Document -> aligned sequences for the rep-distill pilot (plan §3/§4).

A document is normalised (HTML-unescape, NFC, whitespace), split into paragraphs and sentences, and greedily packed
into chunks of MIN_UNITS..MAX_UNITS letter/digit units whose token count (with specials) is <= MAX_TOKENS under BOTH
tokenizers.  Chunks that would exceed either limit are split at unit boundaries; nothing is silently truncated.
For every chunk we keep both id sequences plus, per token, the index of the unit span it overlaps (-1 for specials
or tokens made only of punctuation/whitespace), computed from the tokenizers' character offsets on the same string.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import common

SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[^\s])|(?<=[.!?…])\s*(?=[A-ZÀ-Ỹ])")
_TOK_S = None
_TOK_T = None


def init_tokenizers() -> None:
    global _TOK_S, _TOK_T
    if _TOK_S is None:
        _TOK_S = common.load_student_tokenizer()
        _TOK_T = common.load_teacher_tokenizer()


@dataclass
class Chunk:
    text: str
    n_units: int
    s_ids: list[int]
    t_ids: list[int]
    s_span: list[int]
    t_span: list[int]
    s_valid: int          # spans with >=1 student subword
    t_valid: int          # spans with >=1 teacher subword


def _encode(text: str) -> tuple[Chunk | None, str]:
    """Encode one candidate piece with both tokenizers; return (chunk, status)."""
    spans = common.unit_spans(text)
    n = len(spans)
    if n < common.MIN_UNITS:
        return None, "too_few_units"
    if n > common.MAX_UNITS:
        return None, "too_many_units"
    es = _TOK_S(text, return_offsets_mapping=True, add_special_tokens=True)
    et = _TOK_T(text, return_offsets_mapping=True, add_special_tokens=True)
    if len(es["input_ids"]) > common.MAX_TOKENS or len(et["input_ids"]) > common.MAX_TOKENS:
        return None, "too_many_tokens"
    s_span = common.span_token_index(es["offset_mapping"], spans)
    t_span = common.span_token_index(et["offset_mapping"], spans)
    s_valid = len({i for i in s_span if i >= 0})
    t_valid = len({i for i in t_span if i >= 0})
    return Chunk(text, n, list(es["input_ids"]), list(et["input_ids"]), s_span, t_span, s_valid, t_valid), "ok"


def _split_units(text: str, max_units: int) -> list[str]:
    """Split text at unit boundaries into pieces of <= max_units units (used for over-long sentences)."""
    spans = common.unit_spans(text)
    pieces = []
    for i in range(0, len(spans), max_units):
        a = spans[i][0]
        b = spans[min(i + max_units, len(spans)) - 1][1]
        pieces.append(text[a:b])
    return pieces


def document_chunks(text: str, stats: dict | None = None) -> list[Chunk]:
    """Greedy sentence packing into aligned chunks; see module docstring."""
    init_tokenizers()
    stats = stats if stats is not None else {}
    out: list[Chunk] = []

    def flush(buf: list[str]) -> None:
        if not buf:
            return
        piece = " ".join(buf)
        chunk, status = _encode(piece)
        if chunk is not None:
            out.append(chunk); stats["chunks"] = stats.get("chunks", 0) + 1
            return
        if status == "too_many_tokens" and len(buf) > 1:
            mid = len(buf) // 2
            flush(buf[:mid]); flush(buf[mid:])
            return
        if status == "too_many_tokens":       # a single sentence with <=96 units but >256 tokens in one tokenizer
            for p in _split_units(piece, max(common.MIN_UNITS, len(common.unit_spans(piece)) // 2)):
                c, st = _encode(p)
                if c is not None:
                    out.append(c); stats["chunks"] = stats.get("chunks", 0) + 1
                else:
                    stats[f"drop_{st}"] = stats.get(f"drop_{st}", 0) + 1
            return
        stats[f"drop_{status}"] = stats.get(f"drop_{status}", 0) + 1

    # sentences are packed greedily across paragraphs of the SAME document (never across documents)
    sents: list[str] = []
    for para in (text or "").split("\n"):
        para = common.normalize(para)
        if not para or not common.is_vietnamese(para):
            stats["para_skipped"] = stats.get("para_skipped", 0) + 1
            continue
        for s in SENT_SPLIT.split(para):
            s = s.strip()
            if not s:
                continue
            if len(common.unit_spans(s)) > common.MAX_UNITS:
                sents.extend(_split_units(s, common.MAX_UNITS))
            else:
                sents.append(s)
    buf: list[str] = []
    n_buf = 0
    for s in sents:
        n = len(common.unit_spans(s))
        if n == 0:
            continue
        if n_buf + n > common.MAX_UNITS:
            flush(buf); buf, n_buf = [], 0
        buf.append(s); n_buf += n
    flush(buf)
    return out
