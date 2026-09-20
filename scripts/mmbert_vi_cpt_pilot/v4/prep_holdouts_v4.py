#!/usr/bin/env python3
"""v4 §2 holdouts: the four MLM diagnostic sets (HPLT held-out, FineWeb-2 local, ViSoBERT held-out, v3 review CC-BY)."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v4 as C4  # noqa: E402

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

SEQ = 512
EOS = 1
MAX_SEQUENCES = 4000


def pack(arrays: list[np.ndarray], max_sequences: int) -> np.ndarray:
    out = np.zeros((max_sequences, SEQ), dtype=np.uint32)
    buf = np.empty(SEQ * 2, dtype=np.uint32)
    filled = emitted = 0
    for toks in arrays:
        pos = 0
        while pos < toks.size and emitted < max_sequences:
            take = min(SEQ - filled, toks.size - pos)
            buf[filled:filled + take] = toks[pos:pos + take]
            filled += take
            pos += take
            if filled == SEQ:
                out[emitted] = buf[:SEQ]
                emitted += 1
                filled = 0
        if emitted >= max_sequences:
            break
        if filled == SEQ:
            out[emitted] = buf[:SEQ]
            emitted += 1
            filled = 0
        buf[filled] = EOS
        filled += 1
    return out[:emitted]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(C4.ART_V4 / "holdouts"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"created_at": C4.common.now(), "masking": "fixed mask seed 1234, 15 % masked, 80/10/10", "seq_len": SEQ}
    # (a) HPLT held-out and (b) FineWeb-2: reuse the v2 holdout sets (same pool, unchanged)
    for name, src in (("a_hlpt", C4.C2.ART_V1 / "holdouts" / "a_hlpt.npy"), ("b_fineweb2", C4.C2.ART_V1 / "holdouts" / "b_fineweb2.npy")):
        arr = np.load(src)
        np.save(out / f"{name}.npy", arr)
        manifest[name] = {"source": str(src.relative_to(C4.ROOT)), "sequences": int(arr.shape[0]), "tokens": int(arr.size),
                          "sha256": C4.common.sha256_array(arr)}
    # (c) ViSoBERT held-out comments (NC, research-only)
    vh = np.load(C4.DATA_V4 / "visobert" / "visobert_holdout.npy")
    np.save(out / "c_visobert.npy", vh)
    manifest["c_visobert"] = {"sequences": int(vh.shape[0]), "tokens": int(vh.size), "sha256": C4.common.sha256_array(vh),
                              "comments": 2000, "license": C4.LICENSE_NOTE, "research_only": True}
    # (d) v3 review sentences (CC BY 4.0) — holdout only, never trained on
    tok = C4.common.load_tokenizer()
    tok.model_max_length = int(1e9)
    pf = pq.ParquetFile(str(C4.C2.V3_PARQUET))
    texts = []
    for rg in range(pf.num_row_groups):
        tb = pf.read_row_group(rg, columns=["language", "sentence"])
        for lang, sent in zip(tb.column("language").to_pylist(), tb.column("sentence").to_pylist()):
            if lang == "vi" and sent and len(sent) > 20 and C4.common.is_vietnamese(sent):
                texts.append(sent)
    arrays = [np.asarray(tok(t, add_special_tokens=False)["input_ids"], dtype=np.uint32) for t in texts]
    d = pack(arrays, MAX_SEQUENCES)
    np.save(out / "d_v3review.npy", d)
    manifest["d_v3review"] = {"sequences": int(d.shape[0]), "tokens": int(d.size), "sentences": len(texts),
                              "sha256": C4.common.sha256_array(d), "license": "CC BY 4.0 (phamluan/vn_msa_sar_v3)",
                              "role": "holdout only; never used as training data in this pilot"}
    C4.write_json(out / "holdout_manifest.json", manifest)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("sequences", "tokens")} for k, v in manifest.items() if isinstance(v, dict)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
