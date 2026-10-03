#!/usr/bin/env python3
"""Merge the 5 teacher OOF folds into one soft-label file per task and report teacher-vs-gold statistics."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max(-1, keepdims=True))
    return z / z.sum(-1, keepdims=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=C.TASKS, required=True)
    args = ap.parse_args()
    bundle = C.e1_train.load_task_bundle(args.task, C.ROOT, include_test=False)
    train = bundle.splits["train"]
    counts = list(C.label_widths(bundle))
    single = not C.e1_train.task_aspect_class_counts(bundle)
    ids, logits = [], []
    for k in range(C.N_FOLDS):
        z = np.load(C.ART / "teacher" / args.task / f"fold_{k}" / "oof_logits.npz", allow_pickle=False)
        assert list(z["aspect_class_counts"]) == counts
        ids.extend(str(i) for i in z["ids"])
        logits.append(z["logits"])
    logits = np.concatenate(logits)
    if len(ids) != len(set(ids)) or set(ids) != set(train.ids):
        raise RuntimeError(f"OOF coverage mismatch: {len(ids)} ids, {len(set(ids))} unique, train {len(train.ids)}")
    pos = {i: n for n, i in enumerate(ids)}
    order = [pos[i] for i in train.ids]
    logits = logits[order]
    (C.ART / "soft").mkdir(parents=True, exist_ok=True)
    np.savez(C.ART / "soft" / f"{args.task}.npz", ids=np.array(train.ids), logits=logits, aspect_class_counts=np.array(counts))

    # Teacher OOF statistics against gold, in model-label space.
    gold = np.array(bundle.labels_for_model("train"))
    segs = C.slot_segments(counts)
    preds = np.stack([logits[:, a:b].argmax(-1) for a, b in segs], 1)
    if single:
        preds = preds[:, 0]
    ent = np.concatenate([(-(p := softmax(logits[:, a:b])) * np.log(p + 1e-12)).sum(-1) / np.log(b - a) for a, b in segs])
    maxp = np.concatenate([softmax(logits[:, a:b]).max(-1) for a, b in segs])
    raw_gold = [ex.label for ex in train.examples]
    raw_pred = [int(r) for r in preds] if single else [list(map(int, r)) for r in preds]  # model space, as run_validation
    metrics = C.e1_train.evaluate_predictions(args.task, bundle, raw_gold, raw_pred, split="train")
    stats = {
        "task": args.task, "train_examples": len(train.ids), "slots": len(counts),
        "teacher_oof_selection_f1_vs_train_gold": C.e1_train.selection_score(metrics),
        "slot_disagreement_rate_vs_gold": float((preds != gold).mean()),
        "mean_normalized_entropy": float(ent.mean()), "mean_max_prob": float(maxp.mean()),
        "share_slots_max_prob_ge_0.99": float((maxp >= 0.99).mean()),
    }
    C.write_json(C.ART / "soft" / f"{args.task}_stats.json", stats)
    print(stats, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
