#!/usr/bin/env python3
"""v4 test evaluation of the validation-selected ("locked") checkpoints — opened on explicit user instruction.

The handed-over plan says "không test" and the pilot gate failed; the user explicitly asked for the test numbers, so this
script evaluates **the checkpoint that validation already selected** (best epoch on validation, `best_checkpoint.pt`) for
every v4 fine-tune plus the reused baseline O seed 42 and the new O seed 43.  No test-based selection happens here: the
model, epoch and hyper-parameters are exactly the ones fixed on validation.

For each run it writes `<ft dir>/test_eval/{test_metrics.json,test_predictions.json,test_evaluation.json}` (the training
metadata is left untouched) and checks that the reloaded checkpoint reproduces its predictions.  `--summary` merges the
per-run files into `artifacts/mmbert_vi_cpt_pilot/v4/test_summary.json` + a markdown table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v4 as C4  # noqa: E402

import torch  # noqa: E402

from goibert import train as e1_train  # noqa: E402


def arm_dir(arm: str) -> str:
    return arm + (C4.NC_SUFFIX if arm in C4.ARMS_NC else "")


def run_dir(arm: str, task: str, seed: int) -> Path:
    if arm == "O" and seed == 42:
        p = {"absa_restaurant": "absa_restaurant/lr_5e-05", "visfd": "visfd/lr_5e-05", "vsfc_topic": "vsfc_topic/lr_2e-05"}[task]
        return C4.ROOT / "artifacts/base_encoder_baseline/mmbert_base" / p
    return C4.ART_V4 / "ft" / arm_dir(arm) / task / f"seed_{seed}"


def encoder_for(arm: str) -> str:
    return "O" if arm == "O" else str(C4.ART_V4 / "cpt" / arm_dir(arm) / "hf_encoder")


def evaluate_one(arm: str, task: str, seed: int, device_name: str) -> dict:
    d = run_dir(arm, task, seed)
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    ckpt = d / "best_checkpoint.pt"
    out_dir = d / "test_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = e1_train.resolve_device(device_name)
    precision = e1_train.resolve_precision("bf16", device)
    bundle = e1_train.load_task_bundle(task, C4.ROOT, include_test=True)
    test_split = e1_train.split_field(bundle, "test")
    from transformers import ModernBertModel
    src = encoder_for(arm)
    kwargs = {"local_files_only": True, "attn_implementation": "sdpa", "output_loading_info": True}
    encoder, info = (ModernBertModel.from_pretrained(C4.common.MODEL_REPO, revision=C4.common.MODEL_COMMIT, **kwargs)
                     if src == "O" else ModernBertModel.from_pretrained(src, **kwargs))
    unexpected = [k for k in info.get("unexpected_keys", []) if not ("head" in k or "decoder" in k or "pooler" in k)]
    if info.get("missing_keys") or unexpected or info.get("mismatched_keys"):
        raise RuntimeError(f"encoder reload not clean for {arm}: {info}")
    from goibert_e14.modeling import BamiBERTClassifier
    model = BamiBERTClassifier(encoder, num_labels=e1_train.task_num_classes(bundle), num_aspects=e1_train.task_num_aspects(bundle),
                               aspect_class_counts=e1_train.task_aspect_class_counts(bundle), head_hidden_size=int(encoder.config.hidden_size))
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state["model"], strict=True)
    model.to(device)
    tokenizer = C4.common.load_tokenizer()
    loader = e1_train._make_loader(tokenizer, test_split, max_length=256, batch_size=32, shuffle=False, seed=seed,
                                   labels=e1_train.model_labels(bundle, "test", test_split), gold_labels=e1_train.split_labels(test_split))
    t0 = time.time()
    metrics, rows = e1_train.run_test(model, loader, device, precision, task, bundle)
    metrics2, rows2 = e1_train.run_test(model, loader, device, precision, task, bundle)
    same_preds = [r["prediction"] for r in rows] == [r["prediction"] for r in rows2]
    same_ids = [r["id"] for r in rows] == [r["id"] for r in rows2]
    recomputed = e1_train.selection_score(e1_train.evaluate_predictions(task, bundle, [r["gold"] for r in rows],
                                                                       [r["prediction"] for r in rows], split="test"))
    payload = {"protocol": "mmbert-vi-cpt-pilot-v4-test", "arm": arm, "task_id": task, "seed": seed,
               "checkpoint": str(ckpt.relative_to(C4.ROOT)), "checkpoint_sha256": C4.common.sha256_file(ckpt),
               "validation_selected_epoch": meta.get("best_epoch"), "validation_selection_f1": meta.get("best_selection_f1"),
               "learning_rate": meta.get("learning_rate"), "test_metrics": metrics, "test_selection_f1": e1_train.selection_score(metrics),
               "selection_f1_recomputed_from_predictions": recomputed,
               "test_count": len(test_split), "predictions_rows": len(rows),
               "reload_prediction_repeatable": bool(same_preds and same_ids),
               "load_missing_keys": list(missing), "load_unexpected_keys": list(unexpected),
               "encoder_source": src, "license": C4.LICENSE_NOTE if arm in C4.ARMS_NC else "no NC data",
               "research_only": arm in C4.ARMS_NC, "elapsed_s": round(time.time() - t0, 1),
               "note": "test opened by explicit user instruction after the v4 pilot gate failed; checkpoint is the validation-selected one (no test-based selection)"}
    C4.write_json(out_dir / "test_metrics.json", metrics)
    C4.write_json(out_dir / "test_predictions.json", {"rows": rows})
    C4.write_json(out_dir / "test_evaluation.json", payload)
    print(f"[test] {arm}/{task}/seed{seed}: {100*payload['test_selection_f1']:.2f} (val-selected epoch {payload['validation_selected_epoch']})", flush=True)
    return payload


def summarize() -> dict:
    import statistics
    TASKS = ("absa_restaurant", "visfd", "vsfc_topic")
    rows: dict[tuple[str, str, int], dict] = {}
    for arm in C4.ARMS + ("O",):
        for task in TASKS:
            seeds = (42, 43) if (task == "absa_restaurant") else (42,)
            for seed in seeds:
                p = run_dir(arm, task, seed) / "test_eval" / "test_evaluation.json"
                if p.is_file():
                    rows[(arm, task, seed)] = json.loads(p.read_text(encoding="utf-8"))

    def f1(arm, task, seed):
        r = rows.get((arm, task, seed))
        return r["test_selection_f1"] if r else None

    def pair(arm, task):
        vals = [f1(arm, task, s) for s in ((42, 43) if task == "absa_restaurant" else (42,))]
        vals = [v for v in vals if v is not None]
        return statistics.mean(vals) if vals else None

    table = ["| Arm | Restaurant (42 / 43 / pair-mean) | ViSFD (42) | Topic (42) |", "|---|---|---|---|"]
    for arm in ("O", "P_web", "P_inf", "C_inf"):
        r42, r43 = f1(arm, "absa_restaurant", 42), f1(arm, "absa_restaurant", 43)
        pm = pair(arm, "absa_restaurant")
        cells = " / ".join("n/a" if v is None else f"{100*v:.2f}" for v in (r42, r43))
        table.append(f"| {arm} | {cells}" + (f" / **{100*pm:.2f}**" if pm else "") + " | "
                     + (f"{100*f1(arm,'visfd',42):.2f}" if f1(arm, 'visfd', 42) else "n/a") + " | "
                     + (f"{100*f1(arm,'vsfc_topic',42):.2f}" if f1(arm, 'vsfc_topic', 42) else "n/a") + " |")
    deltas = {q: {t: (pair(a, t) - pair(b, t)) if (pair(a, t) is not None and pair(b, t) is not None) else None for t in TASKS}
              for q, (a, b) in {"Q1": ("P_web", "O"), "Q2": ("P_inf", "P_web"), "Q3": ("C_inf", "P_inf")}.items()}
    table.append("")
    table.append("| Delta | Restaurant | ViSFD | Topic |")
    table.append("|---|---|---|---|")
    for q, dd in deltas.items():
        table.append(f"| {q} | " + " | ".join("n/a" if dd[t] is None else f"{100*dd[t]:+.2f}" for t in TASKS) + " |")
    summary = {"created_at": C4.common.now(), "runs": {f"{k[0]}/{k[1]}/{k[2]}": v for k, v in rows.items()},
               "pair_means": {a: {t: pair(a, t) for t in TASKS} for a in ("O", "P_web", "P_inf", "C_inf")}, "deltas": deltas,
               "note": "test numbers are descriptive only: the test split was opened by explicit user instruction after the pilot gate failed; checkpoints were selected on validation"}
    (C4.ART_V4 / "test_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    (C4.ART_V4 / "test_tables.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print("\n".join(table))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=C4.ARMS + ("O",))
    ap.add_argument("--task", choices=("absa_restaurant", "visfd", "vsfc_topic"))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summarize()
        return 0
    if not (args.arm and args.task and args.seed is not None):
        raise SystemExit("need --arm --task --seed (or --summary)")
    evaluate_one(args.arm, args.task, args.seed, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
