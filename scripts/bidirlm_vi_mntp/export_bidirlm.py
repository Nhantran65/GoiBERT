#!/usr/bin/env python3
"""Stage 1 §4 — export the MNTP checkpoint back into the shipped BidirLM format.

The fast-path state dict (`model.*` + tied `lm_head`) is loaded into a fresh custom `BidirLMForMaskedLM`
from the pinned snapshot and saved with the custom code files and the pinned tokenizer, so the fine-tune
screen loads P_vi through exactly the same path as arm O.  Verification: reload with ``AutoModel``
(``trust_remote_code``), parameter count 596 049 920, bidirectional read on a **padded** batch, and hidden
equivalence with the training model on one batch (≤ 1e-3).  sha256 of every exported file is recorded.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import common as C  # noqa: E402
import mntp_train as M  # noqa: E402

EXPORT_FILES = ("configuration_bidirlm.py", "modeling_bidirlm.py", "vocab.json", "merges.txt",
                "tokenizer.json", "tokenizer_config.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--state", default=str(M.MNTP_DIR / "final_state.pt"))
    ap.add_argument("--out-dir", default=str(C.P_VI_EXPORT_DIR))
    ap.add_argument("--tokens", default=str(M.DATA_DIR / "tokens.npy"))
    ap.add_argument("--heldout", default=str(M.DATA_DIR / "heldout_indices.npy"))
    ap.add_argument("--report", default=str(M.MNTP_DIR / "export_report.json"))
    args = ap.parse_args()
    device = torch.device(args.device)
    started = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = C.pinned_snapshot("bidirlm_base")

    from transformers import AutoModel, AutoModelForMaskedLM

    state = torch.load(args.state, map_location="cpu", weights_only=False)
    custom = AutoModelForMaskedLM.from_pretrained(str(snapshot), trust_remote_code=True, local_files_only=True,
                                                  attn_implementation="sdpa", dtype=torch.float32)
    missing, unexpected = custom.load_state_dict(state["model"], strict=True)
    print(f"[export] loaded fast-path state into custom BidirLMForMaskedLM: missing {len(missing)} unexpected {len(unexpected)}",
          flush=True)
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing {missing} unexpected {unexpected}")
    custom.config.attention_dropout = 0.0
    custom.save_pretrained(str(out_dir), safe_serialization=True, max_shard_size="5GB")
    for name in EXPORT_FILES:
        source = snapshot / name
        if source.is_file():
            shutil.copyfile(source, out_dir / name)
    print(f"[export] wrote {out_dir} ({sorted(p.name for p in out_dir.iterdir())})", flush=True)

    # ---- verification ---------------------------------------------------------------------------
    reloaded, info = AutoModel.from_pretrained(str(out_dir), trust_remote_code=True, local_files_only=True,
                                               attn_implementation="sdpa", dtype=torch.float32, output_loading_info=True)
    reloaded.eval().to(device)
    parameters = int(sum(p.numel() for p in reloaded.parameters()))
    checks: dict = {"reload_class": type(reloaded).__name__, "parameters": parameters,
                    "reload_missing_keys": list(info.get("missing_keys", [])),
                    "reload_unexpected_keys": list(info.get("unexpected_keys", []))}

    tokens = np.load(args.tokens, mmap_mode="r")
    indices = np.load(args.heldout)
    rows = np.asarray(tokens[indices[:2]]).astype(np.int64)
    short, long = rows[0][:37], rows[1]
    width = max(len(short), len(long))
    padded = torch.zeros(2, width, dtype=torch.long, device=device)
    mask = torch.zeros(2, width, dtype=torch.long, device=device)
    for i, row in enumerate((short, long)):
        padded[i, : len(row)] = torch.from_numpy(row)
        mask[i, : len(row)] = 1
    with torch.inference_mode():
        base = reloaded(input_ids=padded, attention_mask=mask, return_dict=True).last_hidden_state.float()
        perturbed = padded.clone()
        perturbed[1, len(long) - 1] = int(long[0]) if int(long[0]) != int(long[-1]) else 4242
        changed = reloaded(input_ids=perturbed, attention_mask=mask, return_dict=True).last_hidden_state.float()
    direction_delta = float((base[1, 0] - changed[1, 0]).abs().max().item())
    checks["padded_batch_bidirectional_delta"] = direction_delta

    fast, build_info = M.build_fast_model(attn_implementation="sdpa", dtype=torch.float32)
    fast.load_state_dict(state["model"], strict=True)  # same weights that were exported (training model)
    fast.eval().to(device)
    x = torch.from_numpy(rows).to(device)
    ids = torch.cat([torch.full((2, 1), M.BOS_ID, dtype=x.dtype, device=device), x], dim=1)
    with torch.inference_mode():
        hidden_train = fast.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state.float()
        hidden_export = reloaded(input_ids=x, attention_mask=torch.ones_like(x), return_dict=True).last_hidden_state.float()
    hidden_delta = float((hidden_train[:, : M.SEQ] - hidden_export[:, : M.SEQ]).abs().max().item())
    checks["hidden_export_vs_train_max_abs_delta"] = hidden_delta

    files = {p.name: C.sha256_file(p.resolve()) for p in sorted(out_dir.iterdir()) if p.resolve().is_file()}
    report = {"protocol": "bidirlm-vi-mntp-stage1-export-v1", "created_at": C.now(), "out_dir": str(out_dir),
              "source_state": args.state, "source_step": state.get("step"), "files": files,
              "checks": checks, "elapsed_seconds": time.time() - started}
    report["passed"] = bool(parameters == C.EXPECTED_TOTAL_PARAMETERS and direction_delta > 0.0 and hidden_delta <= 1e-3
                            and not checks["reload_missing_keys"] and not checks["reload_unexpected_keys"])
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
    print(f"[export] params {parameters} direction {direction_delta:.4f} hidden_delta {hidden_delta:.2e} "
          f"-> {'PASSED' if report['passed'] else 'FAILED'} ({path})", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
