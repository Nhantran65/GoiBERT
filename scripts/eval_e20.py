#!/usr/bin/env python3
"""E20 evaluation: pair F1 + error-decomposition groups + fork contamination
+ composite battery, per (task, arm), written to docs/E20_RESULT.md.

Arms: e19 (artifacts/e19/<task>/seed_42, no rerun), d1_composite, cti
(artifacts/e20/<arm>/<task>/seed_42).  Tolerant of missing arms/tasks: any
missing checkpoint/prediction file is skipped with a note in the report
instead of crashing, so this can be run at any point (including on E19-only
checkpoints before E20 has produced anything).

Group-table logic (k=1, k>=2 same/mixed polarity, freq/rare pair, missed/
wrong/spurious) is a direct, non-printing port of the scratchpad
``err_decomp.py`` used earlier in this research loop.  Fork contamination
follows ``scripts/independent_review/fork_test.py`` part (ii) (masked-CLS
counterfactual); pairs come from ``mechanism_battery.build_pairs`` (seed
2026, 200 pairs).  Nothing here modifies src/, models/, or artifacts/e1..e19.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import types
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

ROOT = Path("/home/kiettqa/GoiBERT")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "independent_review"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from goibert import train as tr  # noqa: E402
from goibert_e2.modeling import build_e2_teacher  # noqa: E402
from mechanism_battery import build_pairs, predictions, scored_aspects, MAX_LEN  # noqa: E402

ENCODER_DIR = ROOT / "models" / "goibert-b-dapt"
TASKS = ("absa_restaurant", "absa_hotel")
ARMS = ("e19", "d1_composite", "cti")
RARE_PAIR_THRESHOLD = 5
N_FORK_PAIRS = 200
FORK_SEED = 2026


def _artifact_dir(arm: str, task: str, seed: int = 42) -> Path:
    """Resolve the trial dir for (arm, task, seed).

    Tries the plain ``seed_<seed>`` dir first; if that dir has no
    ``status.json`` (e.g. E20's pre-``_train_any`` composites_manifest.json
    write used to bump ``_unique_dir`` to ``__rerun1`` before the runner fix),
    falls back to ``seed_<seed>__rerun1``. Falls back to the plain dir's path
    (even if missing) when neither has a status.json, so callers still get a
    path to report as missing.
    """

    if arm == "e19":
        base = ROOT / "artifacts" / "e19" / task / f"seed_{seed}"
    else:
        base = ROOT / "artifacts" / "e20" / arm / task / f"seed_{seed}"
    if (base / "status.json").is_file():
        return base
    rerun = base.with_name(f"{base.name}__rerun1")
    if (rerun / "status.json").is_file():
        return rerun
    return base


def _checkpoint_path(arm: str, task: str, seed: int = 42) -> Path:
    return _artifact_dir(arm, task, seed) / "best_checkpoint.pt"


def _predictions_path(arm: str, task: str, seed: int = 42) -> Path:
    return _artifact_dir(arm, task, seed) / "validation_predictions.json"


def _metrics_path(arm: str, task: str, seed: int = 42) -> Path:
    return _artifact_dir(arm, task, seed) / "best_validation_metrics.json"


def _metadata_path(arm: str, task: str, seed: int = 42) -> Path:
    return _artifact_dir(arm, task, seed) / "metadata.json"


# ---------------------------------------------------------------------------
# Error decomposition (verbatim port of scratchpad err_decomp.py::decomp,
# non-printing; returns the group table instead)
# ---------------------------------------------------------------------------
def train_pair_frequency(task: str, scored: Sequence[int] | None = None) -> dict[tuple[int, int], int]:
    """Train-split co-occurrence frequency of *scored* aspect pairs.

    ``scored`` defaults to ``mechanism_battery.scored_aspects(task, bundle)``,
    which is every aspect for plain ABSA tasks and every aspect but ViSFD's
    binary, polarity-less OTHERS slot. Group metrics must never key off
    OTHERS: it is detection-only and, in the dumped prediction rows, ``gold``
    carries it in raw code space (0/4) while ``prediction`` carries it in
    model code space (0/1) -- see ``error_decomposition_groups`` below.
    """

    bundle = tr.load_task_bundle(task, ROOT, include_test=False)
    train_split = tr.split_field(bundle, "train")
    Ytr = np.array([list(row) for row in tr.split_labels(train_split)])
    num_aspects = Ytr.shape[1]
    if scored is None:
        scored = scored_aspects(task, bundle)
    pairs: Counter = Counter()
    for y in Ytr:
        present = [a for a in scored if y[a] > 0]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                pairs[(present[i], present[j])] += 1
    return dict(pairs)


def error_decomposition_groups(
    pred_path: Path, pair_freq: dict[tuple[int, int], int], scored: Sequence[int] | None = None
) -> dict[str, Any]:
    """Per-row k/mixed/rare-pair grouping and missed/wrong/spurious tallies.

    Restricted to ``scored`` aspects (OTHERS excluded for ViSFD -- see
    ``train_pair_frequency``) so k, mixed-polarity, and rare-pair membership,
    and the missed/wrong/spurious counts, are never computed off a slot whose
    ``gold``/``prediction`` codings disagree (raw vs. model space).
    """

    rows = json.load(open(pred_path))["rows"]
    num_aspects = len(rows[0]["gold"])
    if scored is None:
        scored = list(range(num_aspects))
    groups: dict[str, dict[str, float]] = {}
    for row in rows:
        g = np.array(row["gold"])
        p = np.array(row["prediction"])
        present = [a for a in scored if g[a] > 0]
        k = len(present)
        mixed = len(set(g[a] for a in present)) > 1
        minpair = min(
            (pair_freq.get((present[i], present[j]), 0) for i in range(k) for j in range(i + 1, k)),
            default=None,
        )
        keys = [f"k={min(k, 3)}"]
        if k >= 2:
            keys.append("k>=2|mixed" if mixed else "k>=2|same")
            keys.append("k>=2|rare_pair(<=5)" if minpair is not None and minpair <= RARE_PAIR_THRESHOLD else "k>=2|freq_pair")
        for key in keys:
            d = groups.setdefault(key, {"rows": 0, "gold": 0, "missed": 0, "wrong": 0, "spurious": 0, "tp": 0})
            d["rows"] += 1
            for a in scored:
                if g[a] > 0:
                    d["gold"] += 1
                    if p[a] == 0:
                        d["missed"] += 1
                    elif p[a] != g[a]:
                        d["wrong"] += 1
                    else:
                        d["tp"] += 1
                elif p[a] > 0:
                    d["spurious"] += 1
    report: dict[str, Any] = {}
    for key, d in groups.items():
        report[key] = {
            "rows": d["rows"],
            "gold": d["gold"],
            "missed_pct": round(100 * d["missed"] / max(d["gold"], 1), 1),
            "wrong_pct": round(100 * d["wrong"] / max(d["gold"], 1), 1),
            "spurious_per_row": round(d["spurious"] / max(d["rows"], 1), 3),
        }
    return report


# ---------------------------------------------------------------------------
# Checkpoint loading (E2TeacherClassifier over goibert-b-dapt, strict load)
# ---------------------------------------------------------------------------
def load_checkpoint(task: str, ckpt_path: Path, device: torch.device):
    bundle = tr.load_task_bundle(task, ROOT, include_test=False)
    num_aspects = tr.task_num_aspects(bundle)
    counts = tr.task_aspect_class_counts(bundle) if num_aspects else None
    model, _ = build_e2_teacher(
        ENCODER_DIR,
        num_classes=tr.task_num_classes(bundle),
        num_aspects=num_aspects,
        aspect_class_counts=counts,
        attention_implementation="sdpa",
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    result = model.load_state_dict(state["model"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Strict load failed for {ckpt_path}: {result}")
    model.to(device).eval()
    return model, bundle, counts


# ---------------------------------------------------------------------------
# Masked-CLS fork test (verbatim port of independent_review/fork_test.py)
# ---------------------------------------------------------------------------
class MaskedCLS:
    def __init__(self, model):
        self.model = model
        self.on = False
        self.mask2d = None
        self.keep = None
        self._orig = {}
        for idx, layer in enumerate(model.encoder.encoder.layer):
            sa = layer.attention.self
            self._orig[idx] = sa.forward
            sa.forward = types.MethodType(self._fwd(), sa)

    def restore(self):
        for idx, f in self._orig.items():
            self.model.encoder.encoder.layer[idx].attention.self.forward = f

    def _fwd(self):
        st = self

        def forward(sa, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                    encoder_attention_mask=None, past_key_value=None, output_attentions=False):
            B, N, _ = hidden_states.shape
            H, d = sa.num_attention_heads, sa.attention_head_size
            with torch.autocast(device_type=hidden_states.device.type, enabled=False):
                x = hidden_states.float()
                q = sa.query(x).view(B, N, H, d).transpose(1, 2)
                k = sa.key(x).view(B, N, H, d).transpose(1, 2)
                v = sa.value(x).view(B, N, H, d).transpose(1, 2)
                valid = st.mask2d.to(x.device).bool()
                s = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)
                s = s.masked_fill(~valid[:, None, None, :], float("-inf"))
                if st.on:
                    keep = st.keep.to(x.device).bool()
                    s_cls = s[:, :, 0, :].masked_fill(~keep[:, None, :], float("-inf"))
                    s = s.clone()
                    s[:, :, 0, :] = s_cls
                probs = torch.softmax(s, dim=-1)
                ctx = torch.matmul(probs, v).transpose(1, 2).reshape(B, N, H * d)
            return (ctx.to(hidden_states.dtype),)

        return forward


@torch.inference_mode()
def cls_of(model, tok, texts, mc: MaskedCLS, device, keep_len=None):
    outs = []
    for i in range(0, len(texts), 32):
        enc = tok(texts[i : i + 32], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        mc.mask2d = enc["attention_mask"]
        if keep_len is not None:
            K = torch.zeros_like(enc["input_ids"], dtype=torch.bool)
            for b in range(enc["input_ids"].shape[0]):
                K[b, : keep_len[i + b]] = True
                K[b, int(enc["attention_mask"][b].sum()) - 1] = True
            mc.keep = K
        with tr.make_autocast("bf16", device):
            h = model.encoder(
                input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device), return_dict=True
            ).last_hidden_state
        outs.append(h[:, 0].float().cpu())
    return torch.cat(outs, 0)


@torch.inference_mode()
def head_pred(model, cls, counts, device):
    head = model.head
    x = head.output(head.activation(head.dense(cls.to(device))))
    if model.num_aspects and len(set(model.aspect_class_counts)) == 1:
        x = x.reshape(x.shape[0], model.num_aspects, -1)
    return np.array(predictions(x.float().cpu(), counts))


def ab_report(P, pairs) -> dict[str, float]:
    return {
        "A_correct": round(100 * float(np.mean([P[i, A["aspect"]] == A["pol"] for i, (A, B) in enumerate(pairs)])), 1),
        "A_present": round(100 * float(np.mean([P[i, A["aspect"]] > 0 for i, (A, B) in enumerate(pairs)])), 1),
        "B_correct": round(100 * float(np.mean([P[i, B["aspect"]] == B["pol"] for i, (A, B) in enumerate(pairs)])), 1),
        "B_present": round(100 * float(np.mean([P[i, B["aspect"]] > 0 for i, (A, B) in enumerate(pairs)])), 1),
        "both_correct": round(
            100 * float(np.mean([(P[i, A["aspect"]] == A["pol"]) and (P[i, B["aspect"]] == B["pol"]) for i, (A, B) in enumerate(pairs)])),
            1,
        ),
    }


@torch.inference_mode()
def full_forward_logits(model, tok, texts, device):
    outs = []
    for i in range(0, len(texts), 32):
        enc = tok(texts[i : i + 32], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        with tr.make_autocast("bf16", device):
            logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        outs.append(logits.detach().float().cpu())
    return torch.cat(outs, 0)


def fork_and_battery(model, tok, bundle, counts, task: str, device: torch.device) -> dict[str, Any]:
    scored = scored_aspects(task, bundle)
    singles, pairs = build_pairs(task, bundle, scored, tok, N_FORK_PAIRS, FORK_SEED, max_tok=48)
    va_comp = [A["text"] + " " + B["text"] for A, B in pairs]

    # (ii) masked-CLS: CLS sees only the A-part (all layers); B alone; A alone
    mc = MaskedCLS(model)
    a_len = [len(tok(A["text"], truncation=True, max_length=MAX_LEN)["input_ids"]) - 1 for A, B in pairs]
    mc.on = True
    X_mask_a = cls_of(model, tok, va_comp, mc, device, keep_len=a_len)
    mc.on = False
    X_a_alone = cls_of(model, tok, [A["text"] for A, B in pairs], mc, device)
    mc.restore()

    fork_report = {
        "n_pairs": len(pairs),
        "n_singles_pool": len(singles),
        "cls_sees_only_A": ab_report(head_pred(model, X_mask_a, counts, device), pairs),
        "A_alone": ab_report(head_pred(model, X_a_alone, counts, device), pairs),
    }

    # Composite battery: full (unmasked) A+B forward
    logits = full_forward_logits(model, tok, va_comp, device)
    preds = np.array(predictions(logits, counts))
    battery_report = ab_report(preds, pairs)

    return {"fork_test": fork_report, "composite_battery": battery_report}


# ---------------------------------------------------------------------------
# Per-(task, arm) evaluation, tolerant of missing files
# ---------------------------------------------------------------------------
def evaluate_arm_task(
    task: str,
    arm: str,
    pair_freq: dict[tuple[int, int], int],
    device: torch.device,
    seed: int = 42,
    scored: Sequence[int] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"task": task, "arm": arm, "seed": seed, "status": "ok", "notes": []}
    metrics_path = _metrics_path(arm, task, seed)
    pred_path = _predictions_path(arm, task, seed)
    ckpt_path = _checkpoint_path(arm, task, seed)
    meta_path = _metadata_path(arm, task, seed)

    if not metrics_path.is_file() or not pred_path.is_file():
        result["status"] = "skipped"
        result["notes"].append(f"missing metrics/predictions: {metrics_path} / {pred_path}")
        return result

    metrics = json.load(open(metrics_path))
    result["pair_f1"] = metrics.get("selection_f1")
    result["detection_micro_f1"] = metrics.get("detection_micro_f1")
    if meta_path.is_file():
        meta = json.load(open(meta_path))
        result["best_epoch"] = meta.get("best_epoch")
        result["elapsed_seconds"] = meta.get("elapsed_seconds")

    try:
        result["error_groups"] = error_decomposition_groups(pred_path, pair_freq, scored=scored)
    except Exception as exc:  # noqa: BLE001
        result["notes"].append(f"error_decomposition failed: {exc!r}")

    if not ckpt_path.is_file():
        result["notes"].append(f"missing checkpoint (fork test / battery skipped): {ckpt_path}")
        return result

    try:
        tok = AutoTokenizer.from_pretrained(ENCODER_DIR, local_files_only=True, use_fast=True)
        model, bundle, counts = load_checkpoint(task, ckpt_path, device)
        result["fork_and_battery"] = fork_and_battery(model, tok, bundle, counts, task, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        result["notes"].append(f"fork_and_battery failed: {exc!r}")

    return result


# ---------------------------------------------------------------------------
# E19 5-task summary (vs E2 teacher CE / E14 BamiBERT CE references)
# ---------------------------------------------------------------------------
def e19_summary() -> dict[str, Any]:
    protocol_path = ROOT / "configs" / "e19" / "protocol.json"
    references = {}
    if protocol_path.is_file():
        protocol = json.load(open(protocol_path))
        references = protocol.get("references_no_rerun", {})
    e19_tasks = ("vsfc_sentiment", "vsfc_topic", "visfd", "absa_restaurant", "absa_hotel")
    rows = []
    for task in e19_tasks:
        status_path = ROOT / "artifacts" / "e19" / task / "seed_42" / "status.json"
        metrics_path = ROOT / "artifacts" / "e19" / task / "seed_42" / "best_validation_metrics.json"
        row: dict[str, Any] = {"task": task}
        if status_path.is_file():
            status = json.load(open(status_path))
            row["status"] = status.get("status")
        else:
            row["status"] = "not_started"
        if metrics_path.is_file():
            metrics = json.load(open(metrics_path))
            row["e19_selection_f1"] = metrics.get("selection_f1")
        e2_ref = references.get("e2_teacher_ce", {}).get(task)
        e14_ref = references.get("e14_bamibert_ce", {}).get(task)
        row["e2_teacher_ce_pct"] = e2_ref
        row["e14_bamibert_ce_pct"] = e14_ref
        if "e19_selection_f1" in row and row["e19_selection_f1"] is not None and e2_ref is not None:
            row["delta_vs_e2_teacher_pts"] = round(100 * row["e19_selection_f1"] - e2_ref, 2)
        rows.append(row)
    return {"rows": rows}


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def format_group_table(groups: dict[str, Any] | None) -> str:
    if not groups:
        return "(no group data)\n"
    order = ["k=1", "k=2", "k=3", "k>=2|same", "k>=2|mixed", "k>=2|freq_pair", "k>=2|rare_pair(<=5)"]
    lines = ["| group | rows | gold | missed% | wrong% | spurious/row |", "|---|---|---|---|---|---|"]
    seen = set()
    for key in order:
        if key in groups:
            d = groups[key]
            lines.append(f"| {key} | {d['rows']} | {d['gold']} | {d['missed_pct']} | {d['wrong_pct']} | {d['spurious_per_row']} |")
            seen.add(key)
    for key, d in groups.items():
        if key not in seen:
            lines.append(f"| {key} | {d['rows']} | {d['gold']} | {d['missed_pct']} | {d['wrong_pct']} | {d['spurious_per_row']} |")
    return "\n".join(lines) + "\n"


def gate_verdict(task_results: dict[str, dict[str, Any]]) -> str:
    d1 = task_results.get("d1_composite", {})
    cti = task_results.get("cti", {})
    e19 = task_results.get("e19", {})
    if d1.get("status") != "ok" or cti.get("status") != "ok" or e19.get("status") != "ok":
        return "INCOMPLETE (one or more arms missing/skipped)"
    d1_rare = (d1.get("error_groups") or {}).get("k>=2|rare_pair(<=5)", {}).get("missed_pct")
    cti_rare = (cti.get("error_groups") or {}).get("k>=2|rare_pair(<=5)", {}).get("missed_pct")
    e19_bpresent = ((e19.get("fork_and_battery") or {}).get("fork_test") or {}).get("cls_sees_only_A", {}).get("B_present")
    cti_bpresent = ((cti.get("fork_and_battery") or {}).get("fork_test") or {}).get("cls_sees_only_A", {}).get("B_present")
    d1_f1 = d1.get("pair_f1")
    cti_f1 = cti.get("pair_f1")
    if None in (d1_rare, cti_rare, e19_bpresent, cti_bpresent, d1_f1, cti_f1):
        return "INCOMPLETE (missing metric for gate computation)"
    rare_improve = d1_rare - cti_rare  # missed% lower is better
    bpresent_reduced = cti_bpresent < e19_bpresent
    f1_not_down = (100 * cti_f1) >= (100 * d1_f1) - 0.3
    passed = rare_improve >= 5.0 and bpresent_reduced and f1_not_down
    verdict = "PASS" if passed else "FAIL"
    return (
        f"{verdict} -- cti vs d1 rare-pair missed improvement = {rare_improve:.1f} pts (need >=5); "
        f"B_present cti={cti_bpresent} vs e19={e19_bpresent} (need reduced); "
        f"F1 cti={100*cti_f1:.2f} vs d1={100*d1_f1:.2f} (need not down >0.3 pts)"
    )


def write_report(out_path: Path, all_results: dict[str, dict[str, Any]], e19_summary_data: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# E20 Result -- CTI (Compositional Token-state Invariance) screening\n")
    lines.append(f"Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}. See `docs/E20_PLAN.md` (locked spec) and `docs/CTI_PROPOSAL.md` (motivation).\n")

    for task in TASKS:
        lines.append(f"\n## {task}\n")
        task_results = all_results.get(task, {})

        lines.append("\n### Validation pair F1 (best epoch)\n")
        lines.append("| arm | pair F1 | detection micro F1 | best epoch | elapsed (s) | status |")
        lines.append("|---|---|---|---|---|---|")
        for arm in ARMS:
            r = task_results.get(arm, {"status": "not_run"})
            f1 = r.get("pair_f1")
            f1_str = f"{100*f1:.2f}" if isinstance(f1, (int, float)) else "-"
            det = r.get("detection_micro_f1")
            det_str = f"{100*det:.2f}" if isinstance(det, (int, float)) else "-"
            lines.append(f"| {arm} | {f1_str} | {det_str} | {r.get('best_epoch','-')} | {r.get('elapsed_seconds','-')} | {r.get('status')} |")

        for arm in ARMS:
            r = task_results.get(arm, {"status": "not_run"})
            lines.append(f"\n#### {arm} -- error decomposition groups\n")
            if r.get("status") != "ok":
                lines.append(f"(skipped: {'; '.join(r.get('notes', ['not run']))})\n")
                continue
            lines.append(format_group_table(r.get("error_groups")))

        lines.append("\n### Fork contamination (masked-CLS, part (ii)) and composite battery\n")
        lines.append("| arm | A_correct (mask A) | B_present (mask A, contamination) | A_correct (A alone) | battery A_correct | battery B_correct | battery both_correct |")
        lines.append("|---|---|---|---|---|---|---|")
        for arm in ARMS:
            r = task_results.get(arm, {"status": "not_run"})
            fb = r.get("fork_and_battery")
            if not fb:
                lines.append(f"| {arm} | - | - | - | - | - | - |")
                continue
            ft = fb["fork_test"]
            bat = fb["composite_battery"]
            lines.append(
                f"| {arm} | {ft['cls_sees_only_A']['A_correct']} | {ft['cls_sees_only_A']['B_present']} | "
                f"{ft['A_alone']['A_correct']} | {bat['A_correct']} | {bat['B_correct']} | {bat['both_correct']} |"
            )

        lines.append(f"\n### Gate verdict ({task})\n")
        lines.append(gate_verdict(task_results) + "\n")

    lines.append("\n## Preflight (from artifacts/e20/preflight.json)\n")
    preflight_path = ROOT / "artifacts" / "e20" / "preflight.json"
    if preflight_path.is_file():
        pf = json.load(open(preflight_path))
        lines.append(f"- status: **{pf.get('status')}**, elapsed {pf.get('elapsed_seconds'):.1f}s\n")
        lines.append(f"- checks: `{json.dumps(pf.get('checks'))}`\n")
        for task, row in pf.get("functional", {}).get("composite_assert", {}).items():
            c = row["coverage"]
            lines.append(
                f"- composites/{task}: built {c['n_composites_built']}/{c['n_composites_target']}, "
                f"valid pairs {c['n_valid_pairs']}, rare pairs(<=5) {c['n_rare_pairs_le5']} "
                f"(covered {c['n_rare_pairs_covered']}), distinct pairs used {c['distinct_pairs_used']}\n"
            )
        vram = pf.get("functional", {}).get("vram_step_time", {})
        for task in TASKS:
            ratio = vram.get(f"{task}/ratio_cti_to_d1")
            if ratio is not None:
                lines.append(f"- cti/d1 step-time ratio ({task}): {ratio:.2f}x\n")
    else:
        lines.append("(preflight.json not found)\n")

    lines.append("\n## E19 summary (5 tasks vs E2 teacher CE / E14 BamiBERT CE references)\n")
    lines.append("| task | status | E19 selection F1 | E2 teacher CE % | E14 BamiBERT CE % | delta vs E2 teacher (pts) |")
    lines.append("|---|---|---|---|---|---|")
    for row in e19_summary_data["rows"]:
        f1 = row.get("e19_selection_f1")
        f1_str = f"{100*f1:.2f}" if isinstance(f1, (int, float)) else "-"
        lines.append(
            f"| {row['task']} | {row['status']} | {f1_str} | {row.get('e2_teacher_ce_pct','-')} | "
            f"{row.get('e14_bamibert_ce_pct','-')} | {row.get('delta_vs_e2_teacher_pts','-')} |"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "E20_RESULT.md")
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    all_results: dict[str, dict[str, Any]] = {}
    for task in args.tasks:
        task_bundle = tr.load_task_bundle(task, ROOT, include_test=False)
        scored = scored_aspects(task, task_bundle)
        pair_freq = train_pair_frequency(task, scored=scored)
        all_results[task] = {}
        for arm in args.arms:
            all_results[task][arm] = evaluate_arm_task(task, arm, pair_freq, device, scored=scored)

    e19_data = e19_summary()
    write_report(args.out, all_results, e19_data)

    raw_path = ROOT / "artifacts" / "e20" / "eval_results.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", encoding="utf-8") as handle:
        json.dump({"results": all_results, "e19_summary": e19_data}, handle, indent=2, default=str)

    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
