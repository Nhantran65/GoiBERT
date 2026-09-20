#!/usr/bin/env python3
"""v2 §2 refinement: re-select UGC paragraphs with a tighter reading of "user-voice opinion/review" rules.

The first pass (`ugc_extract.py`) kept any >= 12-unit paragraph carrying a first-person/evaluation cue, which on the
census-whitelisted platforms still let long legal/technical/commercial blocks through (their pages embed a user question
inside scraped legal or product text).  This pass re-decodes the kept register documents, re-splits them into paragraph
sized units (newlines, then sentences for blocks longer than 500 characters) and keeps a unit only when

  * it is 12-120 units long and at most 700 characters,
  * a first-person / experience / evaluation / colloquial cue from `scripts/bami_as_cpt_pilot/fw2_filter.py` appears in its
    first 60 %, or the unit carries at least two cue hits,
  * it has no price list / URL / phone number,
  * and its register-neutral negative markers (legal citations, product specifications, booking/commercial boilerplate —
    a register filter, not a domain marker) are rare relative to the cues.

Outputs a refined register source under `data/mmbert_vi_cpt_pilot/v2/ugc_refined/` in the same layout as `ugc/`, so
`register_pool.py --ugc-dir` can consume it.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v2 as C  # noqa: E402
import ugc_extract as U  # noqa: E402

import numpy as np  # noqa: E402

MAX_CHARS = 700
MIN_UNITS, MAX_UNITS = 12, 120
REGISTER_NEG = re.compile(r"theo quy định|điều \d+|khoản \d+|thông tư|nghị định|bộ luật|án phí|bị cáo|hợp đồng|"
                          r"thông số|cấu hình|bảo hành|khuyến mãi|hotline|liên hệ|quý khách|sản phẩm|giá bán|giá chỉ|"
                          r"đặt hàng|đăng ký|website|fanpage|tư vấn viên", re.I)


def unit_cues(u: str) -> int:
    return sum(bool(rx.search(u)) for rx in (U.FIRST_PERSON, U.EXPERIENCE, U.EVALUATION, U.COLLOQUIAL))


def keep_unit(u: str) -> bool:
    n_units = len(C.common.UNIT.findall(u))
    if not (MIN_UNITS <= n_units <= MAX_UNITS) or len(u) > MAX_CHARS:
        return False
    if U.PRICE_LIST.search(u) or U.URL_RX.search(u) or U.PHONE.search(u):
        return False
    if U.LIST_ITEM.match(u) and len(u) < 200:
        return False
    cues = unit_cues(u)
    if cues == 0:
        return False
    hits = [i for rx in (U.FIRST_PERSON, U.EXPERIENCE, U.EVALUATION, U.COLLOQUIAL) for m in rx.finditer(u) for i in [m.start()]]
    if cues < 2 and hits and min(hits) > 0.6 * len(u):
        return False
    neg = len(REGISTER_NEG.findall(u))
    if neg > 1 and cues < 3:
        return False
    return True


def refine_text(text: str) -> list[str]:
    units: list[str] = []
    for block in re.split(r"\n{1,}", text):
        block = block.strip()
        if not block:
            continue
        if len(block) <= 500:
            pieces = [block]
        else:
            pieces = [s.strip() for s in U.scan_hplt.SEG_SPLIT.split(block) if s.strip()]
        for p in pieces:
            if keep_unit(p):
                units.append(p)
    return units


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ugc-dir", default=str(C.DATA_V2 / "ugc"))
    ap.add_argument("--out-dir", default=str(C.DATA_V2 / "ugc_refined"))
    args = ap.parse_args()
    ugc_dir, out_dir = Path(args.ugc_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokens = np.load(ugc_dir / "ugc.tokens.npy", mmap_mode="r")
    offsets = np.load(ugc_dir / "ugc.offsets.npy")
    docs = [json.loads(l) for l in (ugc_dir / "ugc.docs.jsonl").open(encoding="utf-8")]
    tok = U.C.common.load_tokenizer()
    tok.model_max_length = int(1e9)
    c = collections.Counter()
    out_docs, arrays = [], []
    for i, d in enumerate(docs):
        c["docs"] += 1
        text = tok.decode([int(x) for x in tokens[int(offsets[i]):int(offsets[i + 1])]])
        units = refine_text(text)
        if not units:
            c["drop_no_unit"] += 1
            continue
        joined = "\n".join(units)
        seq = tok(joined, add_special_tokens=False)["input_ids"]
        if len(seq) < 50:
            c["drop_short"] += 1
            continue
        out_docs.append({**{k: v for k, v in d.items() if k != "text"}, "n_paragraphs": len(units), "chars": len(joined), "n_tokens": len(seq)})
        arrays.append(np.asarray(seq, dtype=np.uint32))
        c["kept"] += 1
    flat = np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.uint32)
    np.save(out_dir / "ugc.tokens.npy", flat)
    np.save(out_dir / "ugc.offsets.npy", np.concatenate([[0], np.cumsum([d["n_tokens"] for d in out_docs])]).astype(np.int64) if out_docs else np.zeros(1, np.int64))
    with (out_dir / "ugc.docs.jsonl").open("w", encoding="utf-8") as fh:
        for i, d in enumerate(out_docs):
            fh.write(json.dumps({**d, "i": i}, ensure_ascii=False) + "\n")
    by_host = collections.Counter()
    for d in out_docs:
        by_host[d["host"]] += d["n_tokens"]
    report = {"created_at": C.common.now(), "counters": dict(c), "documents": len(out_docs), "tokens": int(flat.size),
              "rules": {"max_chars": MAX_CHARS, "units": [MIN_UNITS, MAX_UNITS], "cue_position": "first 60 % or >= 2 cue hits",
                        "negatives": "PRICE_LIST|URL|PHONE plus register-neutral legal/commercial markers", "register_neg_pattern": REGISTER_NEG.pattern},
              "tokens_by_host_top": dict(by_host.most_common(30))}
    C.write_json(out_dir / "manifest.json", report)
    print(json.dumps({k: report[k] for k in ("documents", "tokens", "counters", "tokens_by_host_top")}, ensure_ascii=False, indent=1)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
