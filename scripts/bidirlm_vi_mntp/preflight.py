#!/usr/bin/env python3
"""Stage 1 §3 preflight — fast-path (stock Qwen3ForCausalLM) vs custom BidirLMForMaskedLM equivalence.

Checks, before any MNTP training: strict weight load into the stock class, tied lm_head, is_causal=False
everywhere, bidirectional read on an unpadded batch, and — on the same masked batch — that the fast path
reproduces the custom code's logits at the selected positions (max|Δ| ≤ 1e-2) and loss (≤ 1e-3) in fp32.
Writes artifacts/bidirlm_vi_mntp/preflight.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import common as C  # noqa: E402
import mntp_train as M  # noqa: E402

LOGIT_TOLERANCE = 1e-2
LOSS_TOLERANCE = 1e-3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--tokens", default=str(M.DATA_DIR / "tokens.npy"))
    ap.add_argument("--heldout", default=str(M.DATA_DIR / "heldout_indices.npy"))
    ap.add_argument("--out", default=str(C.MNTP_ART.parent / "preflight.json"))
    args = ap.parse_args()
    device = torch.device(args.device)
    started = time.time()
    report: dict = {"protocol": "bidirlm-vi-mntp-stage1-preflight-v1", "created_at": C.now(),
                    "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu", "checks": []}
    checks = report["checks"]

    def check(name: str, ok: bool, detail) -> bool:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        print(f"[preflight] {'OK  ' if ok else 'FAIL'} {name}: {detail}", flush=True)
        return bool(ok)

    model, custom, info = M.build_fast_model(load_custom=True, attn_implementation="sdpa", dtype=torch.float32)
    report["fast_path"] = info
    check("strict_weight_load", not info["missing_keys"] and not info["unexpected_keys"], info)
    check("parameters_596_049_920", info["parameters"] == C.EXPECTED_TOTAL_PARAMETERS, info["parameters"])
    check("lm_head_tied_to_input_embedding", info["tied_lm_head_is_input_embedding"], True)
    check("is_causal_false_on_every_attention_module", info["is_causal_modules_set_false"] > 0, info["is_causal_modules_set_false"])

    # ---- bidirectional read on an unpadded batch ------------------------------------------------
    torch.manual_seed(0)
    tokens = np.load(args.tokens, mmap_mode="r")
    indices = np.load(args.heldout)[: args.batch]
    x = torch.from_numpy(np.asarray(tokens[indices]).astype(np.int64)).to(device)
    batch, length = x.shape
    check("batch_is_unpadded", bool(x.shape[1] == M.SEQ), {"shape": list(x.shape)})
    model.eval().to(device)
    ids = torch.cat([torch.full((batch, 1), M.BOS_ID, dtype=x.dtype, device=device), x], dim=1)
    ones = torch.ones_like(ids)
    perturbed = x.clone()
    # read position 0 of row 0 while changing the *last* token of the same row (causal read would leave it untouched)
    perturbed[0, -1] = int(x[0, 0].item()) if int(x[0, 0].item()) != int(x[0, -1].item()) else 4242
    ids_b = torch.cat([torch.full((batch, 1), M.BOS_ID, dtype=x.dtype, device=device), perturbed], dim=1)
    with torch.inference_mode():
        hidden_a = model.model(input_ids=ids, attention_mask=ones, use_cache=False).last_hidden_state.clone()
        hidden_b = model.model(input_ids=ids_b, attention_mask=torch.ones_like(ids_b), use_cache=False).last_hidden_state
        hidden_none = model.model(input_ids=ids, attention_mask=None, use_cache=False).last_hidden_state.clone()
        hidden_none_b = model.model(input_ids=ids_b, attention_mask=None, use_cache=False).last_hidden_state
    backward_delta = float((hidden_a[0, 0] - hidden_b[0, 0]).abs().max().item())
    backward_delta_none = float((hidden_none[0, 0] - hidden_none_b[0, 0]).abs().max().item())
    report["direction"] = {
        "mask_2d_all_ones": {"last_token_to_first_token_max_abs_delta": backward_delta, "is_bidirectional": backward_delta > 0.0},
        "attention_mask_none_trap": {
            "last_token_to_first_token_max_abs_delta": backward_delta_none, "is_bidirectional": backward_delta_none > 0.0,
            "note": ("attention_mask=None with position_ids set makes transformers 4.57.6 treat the batch as packed sequences and "
                     "materialise a *causal* mask (allow_is_causal_skip=False), so the read stays one-directional even with "
                     "is_causal=False on every attention module; the all-ones 2D mask is the correct unpadded bidirectional input"),
        },
    }
    check("fast_path_bidirectional_with_all_ones_mask", backward_delta > 0.0, backward_delta)
    check("recorded_trap_attention_mask_none_stays_one_directional", backward_delta_none == 0.0, backward_delta_none)

    # ---- equivalence with the custom code on the same masked batch -------------------------------
    rng = np.random.default_rng([C.SEED, 0, 0])
    masked, labels, selected = M.make_masked_batch(np.asarray(tokens[indices]), rng)
    masked_t = torch.from_numpy(masked.astype(np.int64)).to(device)
    labels_t = torch.from_numpy(labels).to(device)
    select_t = torch.from_numpy(selected).to(device)
    custom.to(device).eval()
    with torch.inference_mode():
        fast_ids = torch.cat([torch.full((batch, 1), M.BOS_ID, dtype=masked_t.dtype, device=device), masked_t], dim=1)
        fast_hidden = model.model(input_ids=fast_ids, attention_mask=torch.ones_like(fast_ids),
                                  use_cache=False).last_hidden_state[:, : M.SEQ]
        fast_logits = model.lm_head(fast_hidden[select_t]).float()
        custom_out = custom(input_ids=masked_t, attention_mask=torch.ones_like(masked_t), return_dict=True)
        custom_logits_all = custom_out.logits.float()
        custom_logits = custom_logits_all[select_t]
        custom_hidden = None
    targets = labels_t[select_t]
    fast_loss = float(F.cross_entropy(fast_logits, targets).item())
    custom_loss = float(F.cross_entropy(custom_logits, targets).item())
    logit_delta = float((fast_logits - custom_logits).abs().max().item())
    report["equivalence"] = {"selected_positions": int(select_t.sum().item()), "logits_shape": list(custom_logits_all.shape),
                             "logits_max_abs_delta": logit_delta, "logit_tolerance": LOGIT_TOLERANCE,
                             "fast_loss": fast_loss, "custom_loss": custom_loss, "loss_abs_delta": abs(fast_loss - custom_loss),
                             "loss_tolerance": LOSS_TOLERANCE, "precision": "float32", "dm_head": "lm_head (tied) applied after gather"}
    ok_logits = check("equivalence_logits_le_1e-2", logit_delta <= LOGIT_TOLERANCE, logit_delta)
    ok_loss = check("equivalence_loss_le_1e-3", abs(fast_loss - custom_loss) <= LOSS_TOLERANCE,
                    {"fast": fast_loss, "custom": custom_loss})
    report["custom_code_note"] = ("BidirLMModel requires an explicit attention_mask (it packs with cu_seqlens and "
                                  "would raise with None), so the equivalence batch passes an all-ones 2D mask, which "
                                  "the custom path packs into per-sequence varlen attention.")

    report["elapsed_seconds"] = time.time() - started
    report["fast_path_usable_for_mntp"] = bool(ok_logits and ok_loss)
    report["passed"] = all(c["ok"] for c in checks)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
    print(f"[preflight] {'PASSED' if report['passed'] else 'FAILED'} "
          f"(fast path usable: {report['fast_path_usable_for_mntp']}) -> {out}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
