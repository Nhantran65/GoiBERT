"""Preflight for the aspect-pair pilot: data-conversion guarantees, encoding, numerics, accumulation weighting,
init provenance, checkpoint reload and a GPU profile of one 384-pair optimizer step.

Writes artifacts/cafebert_aspect_pair/checks/preflight.json.  Data checks guarantee expand -> regroup is lossless;
they say nothing about model quality.
"""

from __future__ import annotations

import argparse
import copy
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import ARMS, ASPECT_TEXTS, EFFECTIVE_BATCH_REVIEWS, NUM_ASPECTS, NUM_CLASSES, SEED, e1_train  # noqa: E402

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from goibert.data import RESTAURANT_ASPECT_NAMES  # noqa: E402


def check_mapping() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in ARMS:
        table = common.mapping_table(arm)
        out[arm] = {"entries": len(table), "texts": [row["text"] for row in table], "unique_texts": len({row["text"] for row in table}) == NUM_ASPECTS}
    out["A_is_muc_01_12"] = list(ASPECT_TEXTS["A"]) == [f"mục {i:02d}" for i in range(1, 13)]
    out["slot_order_equals_loader"] = [row["aspect"] for row in common.mapping_table("B")] == list(RESTAURANT_ASPECT_NAMES)
    out["ok"] = bool(out["A_is_muc_01_12"] and out["slot_order_equals_loader"] and all(out[a]["unique_texts"] and out[a]["entries"] == 12 for a in ARMS))
    return out


def check_expand_regroup(tokenizer: Any, contract: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("train", "validation"):
        reviews = common.reviews_of(contract[split])
        for arm in ARMS:
            pairs = common.encode_pairs(tokenizer, reviews, arm)
            flat = [p for g in pairs for p in g]
            keys = [(p.id, p.slot) for p in flat]
            regrouped = common.regroup(reviews, {(p.id, p.slot): p.label for p in flat})
            original = [list(r.gold) for r in reviews]
            # shuffled-order regroup must also be exact
            perm = torch.randperm(len(flat), generator=torch.Generator().manual_seed(7)).tolist()
            regrouped_shuffled = common.regroup(reviews, {(flat[i].id, flat[i].slot): flat[i].label for i in perm})
            ids_in_order = [r.id for r in reviews] == [str(v) for v in e1_train.split_ids(contract[split])]
            # fail-closed behaviour
            try:
                common.regroup(reviews, {k: 0 for k in keys[:-1]}); missing_raises = False
            except RuntimeError:
                missing_raises = True
            try:
                common.regroup(reviews, {**{k: 0 for k in keys}, (reviews[0].id, 99): 0}); extra_raises = False
            except RuntimeError:
                extra_raises = True
            absent = sum(p.label == 0 for p in flat)
            out[f"{split}/{arm}"] = {
                "reviews": len(reviews), "pairs": len(flat), "pairs_per_review_all_12": all(len(g) == NUM_ASPECTS for g in pairs),
                "unique_keys": len(set(keys)) == len(flat), "regroup_equals_gold": regrouped == original,
                "regroup_shuffled_equals_gold": regrouped_shuffled == original, "ids_in_split_order": ids_in_order,
                "missing_key_raises": missing_raises, "extra_key_raises": extra_raises,
                "label_histogram": {c: sum(p.label == c for p in flat) for c in range(NUM_CLASSES)}, "absent_pairs": absent,
            }
    expected = {"train": (7028, 84336), "validation": (771, 9252)}
    out["counts_match_plan"] = all(out[f"{s}/{a}"]["reviews"] == expected[s][0] and out[f"{s}/{a}"]["pairs"] == expected[s][1] for s in expected for a in ARMS)
    out["ok"] = bool(out["counts_match_plan"] and all(
        v["pairs_per_review_all_12"] and v["unique_keys"] and v["regroup_equals_gold"] and v["regroup_shuffled_equals_gold"]
        and v["ids_in_split_order"] and v["missing_key_raises"] and v["extra_key_raises"] for k, v in out.items() if "/" in k))
    return out


def check_pair_encoding(tokenizer: Any, contract: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"special": {"cls": tokenizer.cls_token_id, "sep": tokenizer.sep_token_id, "pad": tokenizer.pad_token_id, "unk": tokenizer.unk_token_id},
                           "model_input_names": list(tokenizer.model_input_names)}
    structure_ok, checked = True, 0
    for split in ("train", "validation"):
        reviews = common.reviews_of(contract[split])
        for arm in ARMS:
            pairs = common.encode_pairs(tokenizer, reviews, arm)
            flat = [p for g in pairs for p in g]
            lengths = [len(p.input_ids) for p in flat]
            aspect_lengths = {}
            for slot in range(NUM_ASPECTS):
                ids = tokenizer(ASPECT_TEXTS[arm][slot], add_special_tokens=False)["input_ids"]
                aspect_lengths[ASPECT_TEXTS[arm][slot]] = {"tokens": len(ids), "ids": ids, "has_unk": tokenizer.unk_token_id in ids,
                                                          "pieces": tokenizer.convert_ids_to_tokens(ids)}
            out[f"{split}/{arm}"] = {
                "truncated_pairs": sum(p.truncated for p in flat), "max_len": max(lengths), "mean_len": statistics.fmean(lengths),
                "p95_len": sorted(lengths)[int(0.95 * len(lengths))], "any_over_max_length": any(l > common.MAX_LENGTH for l in lengths),
                "aspect_token_lengths": {k: v["tokens"] for k, v in aspect_lengths.items()}, "aspect_has_unk": any(v["has_unk"] for v in aspect_lengths.values()),
            }
            if split == "train":
                out[f"aspect_pieces/{arm}"] = {k: v["pieces"] for k, v in aspect_lengths.items()}
            # structural check on a sample: <s> review </s></s> aspect </s>
            sample = reviews[:50] + reviews[-50:]
            for review, group in zip(sample, common.encode_pairs(tokenizer, sample, arm)):
                review_ids = tokenizer(review.text, add_special_tokens=False)["input_ids"]
                for pair in group:
                    aspect_ids = tokenizer(ASPECT_TEXTS[arm][pair.slot], add_special_tokens=False)["input_ids"]
                    expected = [tokenizer.cls_token_id, *review_ids, tokenizer.sep_token_id, tokenizer.sep_token_id, *aspect_ids, tokenizer.sep_token_id]
                    if list(pair.input_ids) != expected:
                        structure_ok = False
                    checked += 1
    out["structure_checked_pairs"] = checked
    out["structure_ok"] = structure_ok
    out["example"] = {arm: tokenizer.convert_ids_to_tokens(list(common.encode_pairs(tokenizer, common.reviews_of(contract["validation"])[:1], arm)[0][3].input_ids)) for arm in ARMS}
    out["ok"] = bool(structure_ok and all(v["truncated_pairs"] == 0 and not v["any_over_max_length"] and not v["aspect_has_unk"] for k, v in out.items() if "/" in k and not k.startswith("aspect_pieces")))
    return out


def check_review_order(contract: dict[str, Any]) -> dict[str, Any]:
    n = len(contract["train"])
    # the reference run's loader: DataLoader(dataset, batch_size=32, shuffle=True, generator=Generator().manual_seed(42))
    class _Idx(torch.utils.data.Dataset):
        def __len__(self): return n
        def __getitem__(self, i): return i
    ref = DataLoader(_Idx(), batch_size=EFFECTIVE_BATCH_REVIEWS, shuffle=True, generator=torch.Generator().manual_seed(SEED), num_workers=0, collate_fn=lambda rows: list(rows))
    ours = common.review_loader([[i] for i in range(n)], reviews_per_group=EFFECTIVE_BATCH_REVIEWS, shuffle=True, seed=SEED)
    ref_orders = [[i for b in ref for i in b] for _ in range(3)]
    our_orders = [[i for b in ours for i in b] for _ in range(3)]
    sizes = [len(b) for b in ours]
    ids = [str(v) for v in e1_train.split_ids(contract["train"])]
    return {
        "groups_per_epoch": len(sizes), "group_sizes_reviews": {"first": sizes[0], "last": sizes[-1], "count_32": sizes.count(32)},
        "last_group_pairs": sizes[-1] * NUM_ASPECTS, "epochs_compared": 3,
        "matches_reference_sampler": ref_orders == our_orders, "epoch_orders_distinct": len({tuple(o) for o in our_orders}) == 3,
        "each_epoch_is_permutation": all(sorted(o) == list(range(n)) for o in our_orders),
        "epoch1_order_sha256": common.sha256_text("\n".join(ids[i] for i in our_orders[0]) + "\n"),
        "ok": ref_orders == our_orders and all(sorted(o) == list(range(n)) for o in our_orders) and len(sizes) == 220 and sizes[-1] == 20,
    }


def check_init(device: torch.device) -> dict[str, Any]:
    e1_train.seed_everything(SEED)
    model1, spec1 = common.build_model()
    e1_train.seed_everything(SEED)
    model2, spec2 = common.build_model()
    enc1 = common.state_dict_hash({k: v for k, v in model1.state_dict().items() if k.startswith("encoder.")})
    enc2 = common.state_dict_hash({k: v for k, v in model2.state_dict().items() if k.startswith("encoder.")})
    # must not be the fine-tuned T checkpoint's encoder
    ref_ckpt = torch.load(common.REFERENCE_RUN / "best_checkpoint.pt", map_location="cpu", weights_only=False)["model"]
    ref_enc = common.state_dict_hash({k: v for k, v in ref_ckpt.items() if k.startswith("encoder.")})
    ref_keys = {k for k in ref_ckpt if k.startswith("encoder.")}
    our_keys = {k for k in model1.state_dict() if k.startswith("encoder.")}
    del ref_ckpt
    # head must differ from a differently seeded build
    e1_train.seed_everything(SEED + 1)
    model3, spec3 = common.build_model()
    out = {
        "head_init_sha256": spec1["head_init_sha256"], "head_init_reproducible": spec1["head_init_sha256"] == spec2["head_init_sha256"],
        "head_init_changes_with_seed": spec1["head_init_sha256"] != spec3["head_init_sha256"],
        "encoder_state_sha256": enc1, "encoder_reproducible": enc1 == enc2, "encoder_differs_from_finetuned_T": enc1 != ref_enc,
        "encoder_keys_equal_T": ref_keys == our_keys, "reference_T_encoder_sha256": ref_enc,
        "head_parameters": spec1["head_parameters"], "total_parameters": spec1["total_parameters"], "trainable_parameters": spec1["trainable_parameters"],
        "weight_file_sha256": spec1["weight_file_sha256"], "reference_T_weight_file_sha256": common.read_json(common.REFERENCE_RUN / "resolved_config.json")["model"]["weight_file_sha256"],
    }
    out["same_pretrained_file_as_T"] = out["weight_file_sha256"] == out["reference_T_weight_file_sha256"]
    out["ok"] = all(out[k] for k in ("head_init_reproducible", "head_init_changes_with_seed", "encoder_reproducible", "encoder_differs_from_finetuned_T", "encoder_keys_equal_T", "same_pretrained_file_as_T"))
    del model2, model3
    return out, model1


def check_numerics(model: nn.Module, tokenizer: Any, contract: dict[str, Any], device: torch.device) -> dict[str, Any]:
    reviews = common.reviews_of(contract["train"])[:3]
    pairs = [p for g in common.encode_pairs(tokenizer, reviews, "B") for p in g]  # 36 pairs
    model.to(device)
    out: dict[str, Any] = {}
    # (1) bf16 forward/backward on a 16-pair micro-batch in train mode
    model.train()
    moved = common.pad_microbatch(tokenizer, pairs[:16], device)
    with e1_train.make_autocast("bf16", device):
        logits = model(moved["input_ids"], moved["attention_mask"])
        ce = nn.functional.cross_entropy(logits.float(), moved["labels"])
    ce.backward()
    grads = [p.grad for p in model.parameters()]
    out["bf16_train"] = {"logits_shape": list(logits.shape), "logits_dtype": str(logits.dtype), "loss": float(ce), "loss_finite": bool(torch.isfinite(ce)),
                         "all_params_have_grad": all(g is not None for g in grads), "all_grads_finite": all(bool(torch.isfinite(g).all()) for g in grads if g is not None),
                         "grad_norm": float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9))}
    model.zero_grad(set_to_none=True)
    # (2) accumulation weighting, fp32, dropout off: sum_micro CE_mean_micro * n_micro / n_group == CE_mean_group, grads too
    model.eval()
    n_group = len(pairs)
    with torch.autocast(device_type="cuda", enabled=False):
        moved = common.pad_microbatch(tokenizer, pairs, device)
        full = nn.functional.cross_entropy(model(moved["input_ids"], moved["attention_mask"]).float(), moved["labels"])
        full.backward()
        full_grads = [p.grad.detach().clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        acc, micro_sizes = 0.0, []
        for chunk in common.iter_microbatches(pairs, 16):
            m = common.pad_microbatch(tokenizer, chunk, device)
            ce = nn.functional.cross_entropy(model(m["input_ids"], m["attention_mask"]).float(), m["labels"])
            (ce * (len(chunk) / n_group)).backward()
            acc += float(ce) * len(chunk) / n_group
            micro_sizes.append(len(chunk))
        acc_grads = [p.grad.detach().clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        # per-pair reference: mean of individual CE terms
        per_pair = []
        for chunk in common.iter_microbatches(pairs, 1):
            m = common.pad_microbatch(tokenizer, chunk, device)
            per_pair.append(float(nn.functional.cross_entropy(model(m["input_ids"], m["attention_mask"]).float(), m["labels"])))
    diff = max(float((a - b).abs().max()) for a, b in zip(full_grads, acc_grads))
    num = math_sqrt(sum(float(((a - b) ** 2).sum()) for a, b in zip(full_grads, acc_grads)))
    den = math_sqrt(sum(float((a ** 2).sum()) for a in full_grads))
    out["accumulation_fp32_eval"] = {
        "n_group": n_group, "micro_sizes": micro_sizes, "full_batch_ce": float(full), "accumulated_weighted_ce": acc, "mean_of_per_pair_ce": statistics.fmean(per_pair),
        "ce_abs_diff_full_vs_accumulated": abs(float(full) - acc), "ce_abs_diff_full_vs_per_pair": abs(float(full) - statistics.fmean(per_pair)),
        "grad_max_abs_diff": diff, "grad_rel_l2_diff": num / max(den, 1e-30), "grad_l2_full": den,
    }
    out["accumulation_ok"] = out["accumulation_fp32_eval"]["ce_abs_diff_full_vs_accumulated"] < 1e-4 and out["accumulation_fp32_eval"]["grad_rel_l2_diff"] < 1e-3
    out["ok"] = bool(out["bf16_train"]["loss_finite"] and out["bf16_train"]["all_params_have_grad"] and out["bf16_train"]["all_grads_finite"] and out["accumulation_ok"])
    return out


def math_sqrt(x: float) -> float:
    return x ** 0.5


def check_checkpoint(model: nn.Module, tokenizer: Any, contract: dict[str, Any], device: torch.device, scratch: Path) -> dict[str, Any]:
    reviews = common.reviews_of(contract["validation"])[:4]
    pairs = common.encode_pairs(tokenizer, reviews, "A")
    model.to(device)
    before = common.predict_pairs(model, tokenizer, pairs, device, "bf16", micro=16)
    path = scratch / "ckpt_test.pt"
    common.save_model_checkpoint(path, model, epoch=1, global_step=1, metadata={"protocol": common.PROTOCOL, "arm": "A"})
    fresh, _ = common.build_model()
    extra = common.load_model_checkpoint(path, fresh)
    fresh.to(device)
    after = common.predict_pairs(fresh, tokenizer, pairs, device, "bf16", micro=16)
    regrouped = common.regroup(reviews, after)
    state = torch.load(path, map_location="cpu", weights_only=False)
    out = {"predictions_identical_after_reload": before == after, "format": state.get("format"), "keys": sorted(state), "epoch": extra.get("epoch"),
           "model_only": set(state) == {"format", "model", "epoch", "global_step", "metadata"}, "regrouped_shape": [len(regrouped), len(regrouped[0])], "bytes": path.stat().st_size}
    out["ok"] = out["predictions_identical_after_reload"] and out["model_only"] and out["regrouped_shape"] == [4, 12]
    path.unlink()
    fresh.cpu(); del fresh
    return out


def profile(tokenizer: Any, contract: dict[str, Any], device: torch.device, micros: list[int], eval_micro: int) -> dict[str, Any]:
    e1_train.seed_everything(SEED)
    model, _ = common.build_model()
    model.to(device)
    optimizer = e1_train.make_optimizer(model, common.LEARNING_RATE)
    train_reviews = common.reviews_of(contract["train"])
    loader = common.review_loader(common.encode_pairs(tokenizer, train_reviews, "B"), reviews_per_group=EFFECTIVE_BATCH_REVIEWS, shuffle=True, seed=SEED)
    groups = []
    for i, g in enumerate(loader):
        groups.append(g)
        if i >= 5:
            break
    out: dict[str, Any] = {"device": str(device), "device_name": torch.cuda.get_device_name(device), "arm_profiled": "B (longest prompts)", "groups_per_epoch": len(loader)}
    for micro in micros:
        torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
        times = []
        for step, group in enumerate(groups):
            torch.cuda.synchronize(device); t0 = time.time()
            model.train(); optimizer.zero_grad(set_to_none=True)
            n_group = len(group)
            for chunk in common.iter_microbatches(group, micro):
                m = common.pad_microbatch(tokenizer, chunk, device)
                with e1_train.make_autocast("bf16", device):
                    ce = nn.functional.cross_entropy(model(m["input_ids"], m["attention_mask"]).float(), m["labels"])
                    (ce * (len(chunk) / n_group)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            if step >= 2:
                times.append(time.time() - t0)
        sec = statistics.fmean(times)
        out[f"micro_{micro}"] = {"seconds_per_step_384_pairs": sec, "steps_timed": len(times), "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30, "train_minutes_per_epoch_est": sec * len(loader) / 60}
    valid_reviews = common.reviews_of(contract["validation"])
    valid_pairs = common.encode_pairs(tokenizer, valid_reviews, "B")
    torch.cuda.synchronize(device); t0 = time.time()
    common.predict_pairs(model, tokenizer, valid_pairs, device, "bf16", micro=eval_micro)
    torch.cuda.synchronize(device)
    out["validation"] = {"eval_micro": eval_micro, "pairs": sum(len(g) for g in valid_pairs), "seconds": time.time() - t0}
    model.cpu(); del model, optimizer
    torch.cuda.empty_cache()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--micros", default="16,32,48,64,96")
    parser.add_argument("--eval-micro", type=int, default=128)
    parser.add_argument("--scratch", default=str(common.CHECK_ROOT))
    args = parser.parse_args()
    common.CHECK_ROOT.mkdir(parents=True, exist_ok=True)
    scratch = Path(args.scratch); scratch.mkdir(parents=True, exist_ok=True)
    device = e1_train.resolve_device(args.device)
    report: dict[str, Any] = {"protocol": common.PROTOCOL, "started_at": common.now(), "source_hash": common.source_hash(), "device": str(device)}
    contract = common.task_contract()
    tokenizer, tok_manifest = common.load_tokenizer()
    report["tokenizer"] = tok_manifest
    report["mapping"] = check_mapping(); print("mapping ok:", report["mapping"]["ok"], flush=True)
    report["expand_regroup"] = check_expand_regroup(tokenizer, contract); print("expand/regroup ok:", report["expand_regroup"]["ok"], flush=True)
    report["pair_encoding"] = check_pair_encoding(tokenizer, contract); print("pair encoding ok:", report["pair_encoding"]["ok"], flush=True)
    report["review_order"] = check_review_order(contract); print("review order ok:", report["review_order"]["ok"], flush=True)
    report["init"], model = check_init(device); print("init ok:", report["init"]["ok"], flush=True)
    report["numerics"] = check_numerics(model, tokenizer, contract, device); print("numerics ok:", report["numerics"]["ok"], report["numerics"]["accumulation_fp32_eval"], flush=True)
    report["checkpoint"] = check_checkpoint(model, tokenizer, contract, device, scratch); print("checkpoint ok:", report["checkpoint"]["ok"], flush=True)
    model.cpu(); del model; torch.cuda.empty_cache()
    report["profile"] = profile(tokenizer, contract, device, [int(v) for v in args.micros.split(",")], args.eval_micro)
    for k, v in report["profile"].items():
        if k.startswith("micro_"):
            print(f"profile {k}: {v['seconds_per_step_384_pairs']:.3f} s/step, {v['train_minutes_per_epoch_est']:.1f} min/epoch train, peak {v['peak_reserved_gib']:.1f} GiB", flush=True)
    print("validation pass:", report["profile"]["validation"], flush=True)
    report["all_checks_ok"] = all(report[k]["ok"] for k in ("mapping", "expand_regroup", "pair_encoding", "review_order", "init", "numerics", "checkpoint"))
    report["finished_at"] = common.now()
    common.write_json(common.CHECK_ROOT / "preflight.json", report)
    print("ALL CHECKS OK:", report["all_checks_ok"], flush=True)
    return 0 if report["all_checks_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
