#!/usr/bin/env python3
"""v2 §2 "vá 3": Wikipedia filter (>= 300 words, no stub patterns) and the 60 M token decision.

Works on the v1 Wikipedia slice (`data/mmbert_vi_cpt_pilot/wiki/`: 128 462 documents, 91.4 M mmBERT tokens, CC BY-SA),
which v1 already decontaminated against the five benchmarks.  ">= 300 words" is approximated by >= 330 mmBERT tokens
(the Gemma tokenizer emits ~0.9 words per token on Vietnamese); stubs are detected with the plan's patterns.  If the
filtered slice keeps fewer than 60 M tokens, the slice is dropped and the plan's fallback (5 % moved to general) applies.
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

STUB = re.compile(r"là một (?:loài|chi|đô thị|xã|thị trấn|huyện|quận|thành phố|bài hát|phim|tiểu hành tinh|ca sĩ|diễn viên|"
                  r"cầu thủ|nhà văn|chính trị gia|thực vật|côn trùng|bướm|ốc|cá|chim|động vật|ngôi làng|tỉnh|bang|quần đảo)|"
                  r"thị trấn thống kê|là một trong những loài|là một họ|là một nhóm|là một chi bướm|"
                  r"^\{\{DISPLAYTITLE|được phát hiện ngày|là một đơn vị hành chính", re.I)
MIN_TOKENS = 330
MIN_KEEP_TOKENS = 60_000_000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wiki-dir", default=str(C.DATA_V1 / "wiki"))
    ap.add_argument("--out-dir", default=str(C.ART_V2))
    args = ap.parse_args()
    wiki = Path(args.wiki_dir)
    excerpts = {}
    with (wiki / "wiki_audit_excerpts.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            excerpts[r["id"]] = r.get("excerpt", "")
    c = collections.Counter()
    kept = []
    with (wiki / "wiki.docs.jsonl").open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            c["docs"] += 1
            text = excerpts.get(r["id"], "")
            if r["n_tokens"] < MIN_TOKENS:
                c["drop_too_short"] += 1
                continue
            if STUB.search(text[:600]):
                c["drop_stub"] += 1
                continue
            kept.append({"i": i, "id": r["id"], "n_tokens": int(r["n_tokens"]), "chars": int(r["chars"]), "url": r.get("url")})
            c["kept"] += 1
    tokens = sum(k["n_tokens"] for k in kept)
    verdict = "keep" if tokens >= MIN_KEEP_TOKENS else "drop_and_move_5pct_to_general"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "wiki_filtered.jsonl").open("w", encoding="utf-8") as fh:
        for k in kept:
            fh.write(json.dumps(k) + "\n")
    report = {"created_at": C.common.now(), "counters": dict(c), "documents": len(kept), "tokens": tokens,
              "min_tokens_rule": MIN_TOKENS, "stub_pattern": STUB.pattern, "min_keep_tokens": MIN_KEEP_TOKENS,
              "verdict": verdict, "source": {"dir": str(wiki.relative_to(C.ROOT)), "license": C.common.WIKI_LICENSE,
                                             "dataset_revision": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"},
              "note": ">= 300 words approximated by >= 330 mmBERT tokens"}
    C.write_json(out_dir / "wiki_filter.json", report)
    print(json.dumps({"documents": len(kept), "tokens": tokens, "verdict": verdict, "counters": dict(c)}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
