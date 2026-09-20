#!/usr/bin/env python3
"""Alignment gate (§4) and objective/gradient preflight (§5) before the 30M-token runs.

Checks written to artifacts/cafe_bami_repdistill_pilot/preflight.json (+ alignment_cases.txt for the manual review):
  * student RobertaForMaskedLM loads cleanly with pretrained/tied MLM head; teacher CafeBERT loads cleanly, frozen, eval
  * on >= 500 holdout passages: fraction of unit spans mapped in BOTH models >= 95 %; every token's span lies inside its own
    sequence (no cross-document/sequence pooling: batch-invariance test); hidden states finite; teacher has no grad,
    student receives non-zero encoder gradient from L_rel alone
  * objective: hand-computed KL on a 3x3 case, KL(T,T)=0, diagonal/pad excluded (objective.py tests)
  * gradient ratio grad_norm(L_rel)/grad_norm(L_MLM) on the encoder for 4 fixed minibatches must lie in [0.05, 20]
  * batch/mask determinism: two independent collates + generators give identical hashes (M and G will see the same data)
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import data as D  # noqa: E402
import objective as O  # noqa: E402
import cpt_train as T  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

N_PASSAGES = 600
N_CASES = 30


def main() -> int:
    device = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(common.SEED)
    autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # noqa: E731
    report: dict = {"protocol": common.PROTOCOL, "created_at": common.now(), "device": str(device), "source_hash": common.source_hash()}

    student, s_checks = common.load_student_mlm(); student.to(device)
    teacher, t_checks = common.load_teacher(); teacher.to(device)
    tok_s, tok_t = common.load_student_tokenizer(), common.load_teacher_tokenizer()
    report["student_checks"] = s_checks; report["teacher_checks"] = t_checks; report["tokenizers"] = common.tokenizer_manifest()
    report["teacher_special_ids"] = sorted(set(tok_t.all_special_ids)); report["student_special_ids"] = sorted(set(tok_s.all_special_ids))

    # ---- alignment coverage on holdout passages
    hold = D.Corpus(split="holdout")
    texts = {}
    with (common.DATA_DIR / "texts.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["split"] == "holdout":
                texts[r["i"]] = r["text"]
    rng = np.random.default_rng(common.SEED)
    sel = np.sort(rng.choice(len(hold), size=min(N_PASSAGES, len(hold)), replace=False))
    both_valid = total = 0; per_passage = []; cross = 0; s_only = t_only = 0
    for i in sel:
        r = hold.sequence(int(i))
        n = r["n_units"]
        sv = set(int(x) for x in r["s_span"] if x >= 0); tv = set(int(x) for x in r["t_span"] if x >= 0)
        if any(x >= n for x in sv | tv):
            cross += 1
        b = len(sv & tv); both_valid += b; total += n; per_passage.append(b / n)
        s_only += len(sv - tv); t_only += len(tv - sv)
    report["alignment"] = {"passages": int(len(sel)), "spans_total": total, "spans_valid_both": both_valid, "coverage_both": both_valid / total,
                           "coverage_min_passage": float(min(per_passage)), "passages_below_95": int(sum(c < 0.95 for c in per_passage)),
                           "spans_student_only": s_only, "spans_teacher_only": t_only, "span_index_out_of_range": cross, "pass_95": both_valid / total >= 0.95}

    # ---- manual cases: diacritics / teencode / punctuation / digits
    pats = {"diacritics": re.compile(r"[ăâđêôơư]", re.I), "teencode": re.compile(r"(?<![\w])(ko|dc|đc|hok|bt|ntn|mn|ae|k|j|z|w|lun|hem)(?![\w])", re.I),
            "punct_digits": re.compile(r"\d+[.,:/%-]\d+|\w+[-/.]\w+|[!?…]{2,}|\d+k\b", re.I)}
    lines = []; n_cases = 0; picked: dict[str, int] = {}
    for i in sel:
        if n_cases >= N_CASES:
            break
        text = texts[int(i)]
        kinds = [k for k, p in pats.items() if p.search(text)]
        want = None
        for k in ("teencode", "punct_digits", "diacritics"):
            if k in kinds and picked.get(k, 0) < (12 if k == "diacritics" else 9):
                want = k; break
        if want is None:
            continue
        picked[want] = picked.get(want, 0) + 1; n_cases += 1
        r = hold.sequence(int(i)); spans = common.unit_spans(text)
        lines.append(f"=== case {n_cases} [{want}] holdout#{int(i)} units={r['n_units']}\nTEXT: {text}")
        for name, tok, ids, sp in (("bami", tok_s, r["s_ids"], r["s_span"]), ("xlmr", tok_t, r["t_ids"], r["t_span"])):
            pieces = tok.convert_ids_to_tokens([int(x) for x in ids])
            pairs = [f"{p}→{int(s)}:{text[spans[s][0]:spans[s][1]] if s >= 0 else '-'}" for p, s in zip(pieces, sp)]
            lines.append(f"  {name}: " + " | ".join(pairs))
    (common.ART_DIR / "alignment_cases.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report["manual_cases"] = {"file": "artifacts/cafe_bami_repdistill_pilot/alignment_cases.txt", "count": n_cases, "by_kind": picked, "verdict": "pending manual review"}

    # ---- hidden finite / grads / batch invariance on a fixed batch
    batch = hold.collate(sel[:16])
    s_ids, s_am, s_sp = batch["s_ids"].to(device), batch["s_am"].to(device), batch["s_span"].to(device)
    t_ids, t_am, t_sp = batch["t_ids"].to(device), batch["t_am"].to(device), batch["t_span"].to(device)
    N = int(batch["n_spans"])
    with autocast():
        s_out = student.roberta(input_ids=s_ids, attention_mask=s_am, output_hidden_states=True)
    with torch.no_grad(), autocast():
        t_out = teacher(input_ids=t_ids, attention_mask=t_am, output_hidden_states=True)
    finite = {f"student_layer{l}": bool(torch.isfinite(s_out.hidden_states[l]).all()) for _, l in common.LAYER_PAIRS}
    finite.update({f"teacher_layer{l}": bool(torch.isfinite(t_out.hidden_states[l]).all()) for l, _ in common.LAYER_PAIRS})
    l_rel, info = T.relational_loss(student, teacher, batch, device, autocast)
    student.zero_grad(set_to_none=True); l_rel.backward()
    enc_grad = T.encoder_grad_norm(student)
    report["gradients"] = {"hidden_finite": finite, "rel_loss_value": float(l_rel), "rel_rows": info["rel_rows"], "rel_by_layer": info["rel_by_layer"],
                           "teacher_requires_grad_any": any(p.requires_grad for p in teacher.parameters()), "teacher_grad_any": any(p.grad is not None for p in teacher.parameters()),
                           "student_encoder_grad_norm_from_rel": enc_grad, "student_grad_nonzero": enc_grad > 0, "teacher_training_mode": teacher.training}
    # batch invariance: pooled vectors of sequence 0 identical when batched with different neighbours
    with torch.no_grad(), autocast():
        h_full = student.roberta(input_ids=s_ids, attention_mask=s_am, output_hidden_states=True).hidden_states[12]
        b2 = hold.collate(np.concatenate([sel[:1], sel[20:23]]))
        h_alt = student.roberta(input_ids=b2["s_ids"].to(device), attention_mask=b2["s_am"].to(device), output_hidden_states=True).hidden_states[12]
    p_full, v_full = O.span_pool(h_full, s_sp, N); p_alt, v_alt = O.span_pool(h_alt, b2["s_span"].to(device), int(b2["n_spans"]))
    n0 = int(hold.n_units[sel[0]])
    diff = (p_full[0, :n0] - p_alt[0, :n0]).abs().max().item()
    report["batch_invariance"] = {"max_abs_diff_seq0_pooled": diff, "ok": diff < 5e-2, "note": "bf16 autocast with different padding lengths; differences are rounding only"}

    # ---- objective unit tests (mirrors the smoke run)
    t = torch.tensor([[[1., 0., 0.], [0., 1., 0.], [1., 1., 0.]]]); s = torch.tensor([[[1., 0., 0.], [1., 0.1, 0.], [0., 0., 1.]]]); valid = torch.tensor([[True, True, True]])
    def hand(v):
        z = torch.nn.functional.normalize(v[0], dim=-1); C = z @ z.T; P = []
        for i in range(3):
            cols = [j for j in range(3) if j != i]; p = torch.softmax(torch.tensor([C[i, j] / common.REL_TEMPERATURE for j in cols]), 0); P.append(dict(zip(cols, p.tolist())))
        return P
    PT, PS = hand(t), hand(s)
    kl_hand = sum(sum(PT[i][j] * math.log(PT[i][j] / PS[i][j]) for j in PT[i]) for i in range(3)) / 3
    kl_impl = float(O.relational_kl(t, valid, s, valid, common.REL_TEMPERATURE)[0])
    t4 = torch.cat([t, torch.randn(1, 1, 3)], 1); s4 = torch.cat([s, torch.randn(1, 1, 3)], 1); v4 = torch.tensor([[True, True, True, False]])
    report["objective_tests"] = {"kl_hand": kl_hand, "kl_impl": kl_impl, "hand_matches": abs(kl_hand - kl_impl) < 1e-5, "kl_tt": float(O.relational_kl(t, valid, t, valid, common.REL_TEMPERATURE)[0]),
                                 "pad_invariant": abs(float(O.relational_kl(t4, v4, s4, v4, common.REL_TEMPERATURE)[0]) - kl_impl) < 1e-6,
                                 "diag_excluded": bool((~O.relation_log_probs(t, valid, common.REL_TEMPERATURE)[1].diagonal(dim1=1, dim2=2)).all())}
    # teacher self-relation on real data ~ 0
    tp, tv = O.span_pool(t_out.hidden_states[24], t_sp, N)
    report["objective_tests"]["kl_teacher_self_real_batch"] = float(O.relational_kl(tp, tv, tp, tv, common.REL_TEMPERATURE)[0])

    # ---- gradient ratio on 4 fixed train minibatches (steps 0..3, micro 0, 32 sequences)
    train = D.Corpus(split="train")
    special_ids = torch.tensor(sorted(set(tok_s.all_special_ids)))
    ratios = []
    student.train()
    for step in range(4):
        idx = D.global_batch_indices(step)[:32]
        b = train.collate(idx)
        masked, labels = O.mask_tokens(b["s_ids"], b["s_am"], special_ids, tok_s.mask_token_id, len(tok_s), common.MASK_PROB, D.mask_generator(step, 0))
        with autocast():
            l_mlm, n_m = O.mlm_loss(student, masked.to(device), b["s_am"].to(device), labels.to(device))
        l_rel, info = T.relational_loss(student, teacher, b, device, autocast)
        student.zero_grad(set_to_none=True); l_mlm.backward(); g_mlm = T.encoder_grad_norm(student)
        student.zero_grad(set_to_none=True); l_rel.backward(); g_rel = T.encoder_grad_norm(student)
        student.zero_grad(set_to_none=True)
        ratios.append({"step": step, "mlm": float(l_mlm), "rel": float(l_rel), "n_masked": n_m, "grad_norm_mlm": g_mlm, "grad_norm_rel": g_rel, "ratio": g_rel / max(g_mlm, 1e-12), "finite": math.isfinite(g_mlm) and math.isfinite(g_rel)})
    rs = [r["ratio"] for r in ratios]
    report["grad_ratio"] = {"per_batch": ratios, "min": min(rs), "max": max(rs), "in_range_0.05_20": all(0.05 <= r <= 20 for r in rs), "all_finite": all(r["finite"] for r in ratios)}

    # ---- determinism of batches and masks
    det = {}
    for step in range(4):
        idx = D.global_batch_indices(step)[:32]
        h1, h2 = D.batch_hash(train.collate(idx)), D.batch_hash(train.collate(idx))
        b = train.collate(idx)
        m1 = O.mask_tokens(b["s_ids"], b["s_am"], special_ids, tok_s.mask_token_id, len(tok_s), common.MASK_PROB, D.mask_generator(step, 0))[0]
        m2 = O.mask_tokens(b["s_ids"], b["s_am"], special_ids, tok_s.mask_token_id, len(tok_s), common.MASK_PROB, D.mask_generator(step, 0))[0]
        det[str(step)] = {"batch_hash": h1, "batch_repeatable": h1 == h2, "mask_hash": common.sha256_array(m1.numpy()), "mask_repeatable": bool(torch.equal(m1, m2))}
    report["determinism"] = det
    report["gate"] = {"alignment_pass_95": report["alignment"]["pass_95"], "no_out_of_range_spans": cross == 0, "hidden_finite": all(finite.values()),
                      "teacher_frozen": not report["gradients"]["teacher_requires_grad_any"] and not report["gradients"]["teacher_grad_any"], "student_grad_from_rel": enc_grad > 0,
                      "objective_tests_ok": report["objective_tests"]["hand_matches"] and report["objective_tests"]["kl_tt"] == 0 and report["objective_tests"]["pad_invariant"] and report["objective_tests"]["diag_excluded"],
                      "grad_ratio_ok": report["grad_ratio"]["in_range_0.05_20"] and report["grad_ratio"]["all_finite"], "determinism_ok": all(v["batch_repeatable"] and v["mask_repeatable"] for v in det.values()),
                      "manual_cases_reviewed": False}
    report["gate"]["all_automatic_ok"] = all(v for k, v in report["gate"].items() if k != "manual_cases_reviewed")
    common.write_json(common.ART_DIR / "preflight.json", report)
    print(json.dumps({k: report[k] for k in ("alignment", "gradients", "batch_invariance", "objective_tests", "grad_ratio", "gate")}, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
