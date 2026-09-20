"""Cache arm-TD train logits (fp32, train split only, keyed by stable example ID) and run the M1 alignment checks.

Adapted from scripts/cdbk_pilot/cache_teacher_logits.py (same cache schema / checks); differences: teacher = TD built by
common_d1.build_model("cafebert_dapt"), outputs under artifacts/cdbk_dapt_pilot/{teacher_cache,checks}/.  Also records the
teacher's train-set confidence at T=2 split by gold present/absent slots, for both the M1 T cache and the new TD cache.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_d1  # noqa: E402

common_d1.install()
import common  # noqa: E402  (patched M1 module)
from common import MAX_LENGTH, ROOT, SEED, TASK_ID, e1_train  # noqa: E402

import torch  # noqa: E402

from goibert_e2.cache import build_teacher_logit_cache, load_teacher_logit_cache, text_hash  # noqa: E402


def confidence_by_presence(logits: torch.Tensor, labels: list, temperature: float) -> dict:
    gold = torch.tensor(labels)
    p = torch.softmax(logits / temperature, dim=-1).max(-1).values
    absent, present = gold == 0, gold != 0
    return {"temperature": temperature, "mean_max_prob_all": float(p.mean()), "mean_max_prob_absent_slots": float(p[absent].mean()),
            "mean_max_prob_present_slots": float(p[present].mean()), "n_absent_slots": int(absent.sum()), "n_present_slots": int(present.sum()),
            "argmax_agreement_with_gold": float((logits.argmax(-1) == gold).float().mean())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher-run", default=str(common_d1.run_dir("TD")))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    teacher_run = Path(args.teacher_run)
    checkpoint = teacher_run / "best_checkpoint.pt"
    teacher_meta = common.read_json(teacher_run / "metadata.json")
    if teacher_meta.get("arm") != "TD" or teacher_meta.get("status") != "completed":
        raise RuntimeError(f"teacher run is not a completed arm TD: {teacher_meta.get('arm')} / {teacher_meta.get('status')}")
    teacher_hash = common.sha256_file(checkpoint)
    if teacher_hash != teacher_meta.get("checkpoint_sha256"):
        raise RuntimeError("teacher checkpoint sha256 differs from the run metadata")

    e1_train.seed_everything(SEED)
    device = e1_train.resolve_device(args.device)
    contract = common.task_contract()
    bundle, train_split = contract["bundle"], contract["train"]
    train_ids = [str(v) for v in e1_train.split_ids(train_split)]
    train_texts = list(e1_train.split_texts(train_split))
    train_labels = list(e1_train.model_labels(bundle, "train", train_split))

    tokenizer, tokenizer_manifest = common.load_tokenizer("cafebert_dapt")
    teacher, teacher_spec = common.build_model("cafebert_dapt", contract)
    state = common.load_model_checkpoint(checkpoint, teacher)
    teacher.to(device).eval()
    loader = common.ordered_train_loader(tokenizer, contract, args.batch_size)

    cache_path = common.CACHE_ROOT / f"{TASK_ID}_train_logits_TD_seed{SEED}.pt"
    build_teacher_logit_cache(
        cache_path, teacher=teacher, loader=loader, device=device, precision="fp32", task_id=TASK_ID,
        source_ids=train_ids, source_texts=train_texts, tokenizer_hashes=tokenizer_manifest["files"], max_length=MAX_LENGTH,
        teacher_hash=teacher_hash, aspect_class_counts=contract["aspect_class_counts"], num_aspects=contract["num_aspects"],
        num_labels=contract["num_classes"], e1_dependency_source_hash=common.source_hash(),
        data_provenance=e1_train.bundle_provenance(bundle, ROOT), inference_precision="fp32",
        source_hashes={"teacher_run": str(teacher_run.relative_to(ROOT)), "teacher_checkpoint_sha256": teacher_hash,
                       "teacher_best_epoch": str(teacher_meta.get("best_epoch")), "teacher_best_selection_f1": str(teacher_meta.get("best_selection_f1")),
                       "teacher_backbone": str(teacher_spec.get("backbone")), "dapt_weight_sha256": str(teacher_spec.get("weight_file_sha256"))},
    )
    cache = load_teacher_logit_cache(cache_path)
    checks: dict[str, object] = {"cache_path": str(cache_path.relative_to(ROOT)), "cache_sha256": common.sha256_file(cache_path),
                                 "logits_shape": list(cache.logits.shape), "teacher_checkpoint_sha256": teacher_hash, "checkpoint_epoch": state.get("epoch")}
    checks["ids_equal_ordered_train_ids"] = [str(v) for v in cache.ids] == train_ids
    checks["ids_unique"] = len(set(train_ids)) == len(train_ids)
    checks["text_hash_matches_train"] = cache.metadata["source_text_hash"] == text_hash(train_texts)
    checks["shape_is_N_12_4"] = tuple(cache.logits.shape) == (len(train_ids), 12, 4)
    checks["all_finite"] = bool(torch.isfinite(cache.logits).all())

    student_tok, _ = common.load_tokenizer("bamibert")
    student_loader, _ = common.make_loaders(student_tok, contract, 32)
    pos = {i: k for k, i in enumerate(train_ids)}
    label_by_id = {i: tuple(int(x) for x in l) for i, l in zip(train_ids, train_labels)}
    lookup_ok, label_ok, seen = True, True, 0
    for b_index, batch in enumerate(student_loader):
        if b_index >= 5:
            break
        ids = [str(v) for v in batch["ids"]]
        got = cache.lookup(batch["ids"]); direct = cache.logits[[pos[i] for i in ids]]
        lookup_ok &= bool(torch.equal(got, direct))
        label_ok &= all(tuple(int(x) for x in row) == label_by_id[i] for i, row in zip(ids, batch["labels"].tolist()))
        seen += len(ids)
    checks["student_loader_lookup_equals_direct_index"] = lookup_ok
    checks["student_loader_labels_match_split_by_id"] = label_ok
    checks["student_loader_rows_checked"] = seen

    rng = random.Random(SEED)
    sample = sorted(rng.sample(range(len(train_ids)), 64))
    with torch.inference_mode():
        enc = tokenizer([train_texts[i] for i in sample], add_special_tokens=True, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt")
        out = teacher(enc["input_ids"].to(device), enc["attention_mask"].to(device)).float().cpu()
    diff = (out - cache.logits[sample]).abs().max().item()
    checks["reforward_max_abs_diff"] = diff
    checks["reforward_matches_cache"] = diff < 1e-3

    preds = e1_train._predictions(cache.logits, aspect_class_counts=contract["aspect_class_counts"])
    checks["teacher_train_pair_micro_f1_from_cache"] = float(e1_train.evaluate_predictions(TASK_ID, bundle, train_labels, preds, split="train")["selection_f1"])
    checks["TD_train_confidence"] = confidence_by_presence(cache.logits, train_labels, common.KD_TEMPERATURE)
    checks["TD_train_confidence_T1"] = confidence_by_presence(cache.logits, train_labels, 1.0)
    m1_cache = ROOT / "artifacts" / "cdbk_pilot" / "teacher_cache" / f"{TASK_ID}_train_logits_T_seed{SEED}.pt"
    if m1_cache.is_file():
        t_cache = load_teacher_logit_cache(m1_cache)
        checks["T_train_confidence"] = confidence_by_presence(t_cache.logits, train_labels, common.KD_TEMPERATURE)
        checks["T_train_confidence_T1"] = confidence_by_presence(t_cache.logits, train_labels, 1.0)
        checks["TD_vs_T_train_argmax_agreement"] = float((cache.logits.argmax(-1) == t_cache.logits.argmax(-1)).float().mean())

    _, valid_loader = common.make_loaders(tokenizer, contract, args.batch_size)
    val_metrics, _ = e1_train.run_validation(teacher, valid_loader, device, "fp32", TASK_ID, bundle)
    checks["teacher_validation_f1_fp32_reload"] = float(val_metrics["selection_f1"])
    checks["teacher_validation_f1_recorded_bf16"] = float(teacher_meta["best_selection_f1"])
    checks["teacher_validation_f1_abs_diff"] = abs(checks["teacher_validation_f1_fp32_reload"] - checks["teacher_validation_f1_recorded_bf16"])

    flags = {k: v for k, v in checks.items() if isinstance(v, bool)}
    checks["all_ok"] = all(flags.values()); checks["checked_flags"] = sorted(flags); checks["created_at"] = common.now()
    common.write_json(common.CHECK_ROOT / "cache_checks_TD.json", checks)
    for k, v in checks.items():
        print(f"{k}: {v}")
    return 0 if checks["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
