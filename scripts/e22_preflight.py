#!/usr/bin/env python3
"""E22 preflight: minimal checks for the run_e20.py extension (task ``visfd``,
arm ``d1_uniform``) ahead of the 12-trial ``scripts/e22_launch.sh`` queue.

This is deliberately small: a handful of CPU/GPU micro-checks, not a training
run. It never uses cuda:0/cuda:1 (the two GPUs that host other people's small
processes and are off-limits to this agent's own dispatch; the eligibility
guard below would exclude them anyway given >5GiB used on both right now, but
we also refuse them explicitly as a second guard). Writes
``artifacts/e22/preflight.json``. Never modifies ``src/``, ``models/``, or any
existing artifact.

Checks:

1. ``visfd`` composite construction (the actual arm/sampling used in the e22
   queue, ``d1_composite``/``rare_pair_weighted``) -- 0 assert failures out of
   a 200-composite sample, plus pair-coverage stats
   (``run_e20._build_composites``).
2. ``d1_uniform`` composite manifest for ``absa_restaurant`` (the only task it
   runs on in the e22 queue) has a materially more uniform pair-use
   distribution (lower stdev of per-pair use counts) than
   ``rare_pair_weighted`` on the same task/seed.
3. One NC+composite training step (``d1_composite`` arm, mu=0) on ``visfd``
   produces a finite total loss and a finite clipped grad norm.
4. Strict reload: ``build_e20_model("visfd", ...)``'s ``state_dict()`` loads
   strictly (no missing/unexpected keys) into an independently constructed
   ``build_e2_teacher("visfd", ...)`` -- confirms the E20 wrapper still adds
   no parameters for ViSFD's variable-width (incl. binary OTHERS) head.
5. Natural-row loader-label check: for one ``visfd`` train row with exactly
   one scored aspect present and OTHERS raw-absent, the loader's model-space
   label (``goibert.train.model_labels``) equals a manual raw->model
   conversion of the row's raw label (``goibert.train.split_labels``), the
   label vector length matches ``tr.task_aspect_class_counts(bundle)``
   (11 slots), and that counts vector is (4,)*10 with a 2 at the OTHERS
   index -- i.e. exactly the "variable slot widths incl. the binary OTHERS
   slot" the task asked to confirm.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(SRC))

import torch  # noqa: E402

import run_e20 as R  # noqa: E402
from goibert import train as tr  # noqa: E402
from goibert import data as data_mod  # noqa: E402
from goibert_e2 import train as e2_train  # noqa: E402
from goibert_e2.modeling import build_e2_teacher  # noqa: E402
from goibert_e7 import optim as e7_optim  # noqa: E402

OUT_PATH = ROOT / "artifacts" / "e22" / "preflight.json"
BANNED_DEVICES = {"cuda:0", "cuda:1"}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _pick_device() -> torch.device:
    snapshot = R._gpu_snapshot()
    eligible = [d for d in R._eligible_devices(snapshot) if d not in BANNED_DEVICES]
    if not eligible:
        raise RuntimeError(
            f"No eligible GPU outside cuda:0/cuda:1 for E22 preflight (snapshot={snapshot})"
        )
    return torch.device(eligible[0])


# ---------------------------------------------------------------------------
# 1. visfd composite construction (0 assert failures + pair coverage stats)
# ---------------------------------------------------------------------------
def check_visfd_composites(tokenizer: Any) -> dict[str, Any]:
    task_id = "visfd"
    contract = R._original_task_contract(task_id)
    composites, coverage = R._build_composites(
        task_id, contract, tokenizer, seed=R.SEED, max_length=R.MAX_LENGTH, sampling="rare_pair_weighted"
    )
    bos_id, eos_id = tokenizer.bos_token_id, tokenizer.eos_token_id
    sample = composites[: min(200, len(composites))]
    assert_failures = 0
    for c in sample:
        a_ids = tokenizer(c["a_text"], add_special_tokens=False)["input_ids"]
        b_ids_space = tokenizer(" " + c["b_text"], add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(c["text"], truncation=True, max_length=R.MAX_LENGTH)["input_ids"]
        boundary = c["boundary"]
        ok = (
            full_ids[0] == bos_id
            and full_ids[-1] == eos_id
            and full_ids[1:boundary] == a_ids
            and full_ids[boundary:-1] == b_ids_space
            and boundary == 1 + len(a_ids)
        )
        if not ok:
            assert_failures += 1
    # OTHERS must never appear as aspect_a/aspect_b, and every composite
    # label must keep OTHERS (index 5) at 0.
    others_idx = data_mod.VISFD_OTHERS_ASPECT_INDEX
    others_never_paired = all(c["aspect_a"] != others_idx and c["aspect_b"] != others_idx for c in composites)
    others_always_absent_in_label = all(int(c["label"][others_idx]) == 0 for c in composites)
    return {
        "task_id": task_id,
        "sampling": "rare_pair_weighted",
        "coverage": coverage,
        "n_asserted": len(sample),
        "assert_failures": assert_failures,
        "others_never_paired": others_never_paired,
        "others_always_absent_in_label": others_always_absent_in_label,
        "checks": {
            "assert_pass": assert_failures == 0,
            "composites_built_at_target": coverage["n_composites_built"]
            >= coverage["n_composites_target"] - 5,
            "eligible_aspects_exclude_others": others_idx not in coverage["eligible_aspects"],
            "others_never_paired": others_never_paired,
            "others_always_absent_in_label": others_always_absent_in_label,
        },
    }


# ---------------------------------------------------------------------------
# 2. d1_uniform manifest (absa_restaurant) shows uniform pair frequencies
# ---------------------------------------------------------------------------
def check_uniform_sampling(tokenizer: Any) -> dict[str, Any]:
    task_id = "absa_restaurant"
    contract = R._original_task_contract(task_id)
    rows: dict[str, Any] = {}
    for sampling in ("rare_pair_weighted", "uniform"):
        composites, coverage = R._build_composites(
            task_id, contract, tokenizer, seed=R.SEED, max_length=R.MAX_LENGTH, sampling=sampling
        )
        counts = list(coverage["pair_use_count"].values())
        rows[sampling] = {
            "n_composites_built": coverage["n_composites_built"],
            "n_composites_target": coverage["n_composites_target"],
            "dropped_assert_failures": coverage["dropped_assert_failures"],
            "n_valid_pairs": coverage["n_valid_pairs"],
            "pair_use_min": min(counts),
            "pair_use_max": max(counts),
            "pair_use_mean": round(statistics.mean(counts), 3),
            "pair_use_stdev": round(statistics.pstdev(counts), 3),
        }
    # A handful of dropped/retried attempts is the designed behaviour of the
    # boundary assert inside _build_composites (retried until n_composites
    # reaches target); it is not itself a failure. The visfd_composites check
    # above separately re-verifies a 200-composite *sample* of the built set
    # with an independent re-tokenization and requires that to be 0/200.
    checks = {
        "both_reached_target": rows["rare_pair_weighted"]["n_composites_built"]
        == rows["rare_pair_weighted"]["n_composites_target"]
        and rows["uniform"]["n_composites_built"] == rows["uniform"]["n_composites_target"],
        "both_low_drop_rate": (rows["rare_pair_weighted"]["dropped_assert_failures"] / max(rows["rare_pair_weighted"]["n_composites_target"], 1)) < 0.05
        and (rows["uniform"]["dropped_assert_failures"] / max(rows["uniform"]["n_composites_target"], 1)) < 0.05,
        "same_valid_pairs_and_target": rows["rare_pair_weighted"]["n_valid_pairs"] == rows["uniform"]["n_valid_pairs"]
        and rows["rare_pair_weighted"]["n_composites_target"] == rows["uniform"]["n_composites_target"],
        "uniform_has_lower_stdev": rows["uniform"]["pair_use_stdev"] < rows["rare_pair_weighted"]["pair_use_stdev"],
    }
    return {"task_id": task_id, "by_sampling": rows, "checks": checks}


# ---------------------------------------------------------------------------
# 3. one NC+composite step finite on visfd (d1_composite arm, mu=0)
# ---------------------------------------------------------------------------
def check_nc_composite_step(tokenizer: Any, device: torch.device) -> dict[str, Any]:
    task_id = "visfd"
    contract = R._original_task_contract(task_id)
    composites, _coverage = R._build_composites(
        task_id, contract, tokenizer, seed=R.SEED, max_length=R.MAX_LENGTH, sampling="rare_pair_weighted"
    )
    new_bundle, new_train_split = R._inject_composites(contract["bundle"], composites)
    composite_by_id = {c["id"]: c for c in composites}

    train_loader, _, _ = e2_train._loader_set(
        tokenizer,
        new_train_split,
        contract["validation"],
        new_bundle,
        seed=R.SEED,
        micro_batch_size=R.MICRO_BATCH_SIZE,
        max_length=R.MAX_LENGTH,
    )
    clean_cpu = None
    for batch in train_loader:
        if any(str(eid) in composite_by_id for eid in batch["ids"]):
            clean_cpu = batch
            break
    if clean_cpu is None:
        clean_cpu = next(iter(train_loader))
    text_by_id = {
        str(eid): text for eid, text in zip(e2_train.split_ids(new_train_split), e2_train.split_texts(new_train_split))
    }
    noisy_cpu, noise_stats = e2_train._make_noisy_batch(
        tokenizer, clean_cpu["ids"], text_by_id, max_length=R.MAX_LENGTH, seed=R.SEED, epoch=1, device=torch.device("cpu")
    )
    counts = contract["aspect_class_counts"]
    num_aspects = contract["num_aspects"]
    nc_fn = e2_train._load_loss_api()[1]

    R.e1_train.seed_everything(R.SEED)
    model, spec = R.build_e20_model(
        R.ENCODER_DIR,
        num_classes=contract["num_classes"],
        num_aspects=num_aspects,
        aspect_class_counts=counts if num_aspects else None,
        attention_implementation=R.ATTENTION_IMPLEMENTATION,
        arm="d1_composite",
        mu=0.0,
        tokenizer=tokenizer,
        max_length=R.MAX_LENGTH,
    )
    model.to(device).train()
    optimizer = e7_optim.build_layerwise_optimizer(
        model, R.TASK_LRS[task_id], gamma=R.GAMMA, num_layers=R.NUM_LAYERS, weight_decay=R.WEIGHT_DECAY
    )

    clean = {key: value.to(device) for key, value in clean_cpu.items() if isinstance(value, torch.Tensor)}
    noisy = {key: value.to(device) for key, value in noisy_cpu.items()}

    model._pending = [
        (
            i,
            composite_by_id[str(eid)]["boundary"],
            composite_by_id[str(eid)]["a_text"],
            composite_by_id[str(eid)]["b_text_standalone"],
        )
        for i, eid in enumerate(clean_cpu["ids"])
        if str(eid) in composite_by_id
    ]
    n_composite_rows_in_batch = len(model._pending)

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        logits = model(clean["input_ids"], clean["attention_mask"])
        ce = e2_train.compute_loss(logits, clean["labels"], num_aspects=num_aspects, aspect_class_counts=counts)
        noisy_logits = model(noisy["input_ids"], noisy["attention_mask"])
        consistency = nc_fn(logits, noisy_logits, temperature=1.0, slot_class_counts=counts or None)
        total = ce + R.NC_LAMBDAS[task_id] * consistency
    total_finite = bool(torch.isfinite(total).item())
    total.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    result = {
        "task_id": task_id,
        "arm": "d1_composite",
        "mu": 0.0,
        "nc_lambda": R.NC_LAMBDAS[task_id],
        "n_composite_rows_in_batch": n_composite_rows_in_batch,
        "input_shape": list(clean["input_ids"].shape),
        "ce_loss": float(ce.detach().cpu().item()),
        "consistency_loss": float(consistency.detach().cpu().item()),
        "total_loss": float(total.detach().cpu().item()),
        "grad_norm": float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm),
        "inv_loss_computed": model.last_inv_loss is not None,
        "noise_change_rate": noise_stats["changed_count"] / max(noise_stats["count"], 1),
        "checks": {
            "total_finite": total_finite,
            "grad_norm_finite": math.isfinite(float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm)),
            "n_composite_rows_in_batch_gt_0": n_composite_rows_in_batch > 0,
            "mu_zero_never_computes_inv": model.last_inv_loss is None,
        },
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# 4. strict reload: E20 wrapper adds no parameters for visfd's variable head
# ---------------------------------------------------------------------------
def check_strict_reload(tokenizer: Any, device: torch.device) -> dict[str, Any]:
    task_id = "visfd"
    contract = R._original_task_contract(task_id)
    counts = contract["aspect_class_counts"]
    num_aspects = contract["num_aspects"]

    R.e1_train.seed_everything(R.SEED)
    wrapper, spec = R.build_e20_model(
        R.ENCODER_DIR,
        num_classes=contract["num_classes"],
        num_aspects=num_aspects,
        aspect_class_counts=counts if num_aspects else None,
        attention_implementation=R.ATTENTION_IMPLEMENTATION,
        arm="d1_composite",
        mu=0.0,
        tokenizer=tokenizer,
        max_length=R.MAX_LENGTH,
    )
    wrapper.to(device).eval()

    plain, _ = build_e2_teacher(
        R.ENCODER_DIR,
        num_classes=contract["num_classes"],
        num_aspects=num_aspects,
        aspect_class_counts=counts if num_aspects else None,
        attention_implementation=R.ATTENTION_IMPLEMENTATION,
    )
    plain.to(device)
    result = plain.load_state_dict(wrapper.state_dict(), strict=True)

    texts = list(e2_train.split_texts(contract["validation"]))[:16]
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=R.MAX_LENGTH, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    plain.eval()
    with torch.no_grad():
        logits_wrapper = wrapper(input_ids, attention_mask)
        logits_plain = plain(input_ids, attention_mask)
    max_diff = float((logits_wrapper.float() - logits_plain.float()).abs().max().item())

    out = {
        "task_id": task_id,
        "model_spec": spec.to_dict(),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "max_abs_logit_diff_after_reload": max_diff,
        "checks": {
            "strict_reload_ok": not result.missing_keys and not result.unexpected_keys,
            "added_parameter_count_zero": spec.added_parameter_count == 0,
            "reload_parity_le_1e4": max_diff <= 1e-4,
        },
    }
    del wrapper, plain
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
# 5. natural-row loader-label check (union-label code space, OTHERS width)
# ---------------------------------------------------------------------------
def check_natural_row_label() -> dict[str, Any]:
    task_id = "visfd"
    bundle = tr.load_task_bundle(task_id, ROOT, include_test=False)
    train_split = tr.split_field(bundle, "train")
    num_aspects = tr.task_num_aspects(bundle)
    counts = tr.task_aspect_class_counts(bundle)
    others_idx = data_mod.VISFD_OTHERS_ASPECT_INDEX

    raw_labels = list(tr.split_labels(train_split))
    loader_labels = list(tr.model_labels(bundle, "train", train_split))
    scored = [a for a in range(num_aspects) if a != others_idx]

    row_idx = next(
        i
        for i, lab in enumerate(raw_labels)
        if sum(1 for a in scored if int(lab[a]) > 0) == 1 and int(lab[others_idx]) == 0
    )
    raw_row = list(raw_labels[row_idx])
    loader_row = list(loader_labels[row_idx])
    manual = list(raw_row)
    manual[others_idx] = int(raw_row[others_idx] == data_mod.VISFD_OTHERS_CODE)

    return {
        "task_id": task_id,
        "row_index": row_idx,
        "row_id": tr.split_ids(train_split)[row_idx],
        "num_aspects": num_aspects,
        "aspect_class_counts": list(counts),
        "raw_label": raw_row,
        "loader_label": loader_row,
        "manual_conversion_of_raw": manual,
        "checks": {
            "loader_label_matches_manual_conversion": loader_row == manual,
            "label_vector_length_matches_aspect_class_counts": len(loader_row) == len(counts),
            "aspect_class_counts_others_is_binary": int(counts[others_idx]) == 2,
            "aspect_class_counts_other_slots_are_4way": all(
                int(c) == 4 for i, c in enumerate(counts) if i != others_idx
            ),
            "others_slot_binary_in_loader_space": loader_row[others_idx] in (0, 1),
        },
    }


def main() -> int:
    started = time.time()
    device = _pick_device()
    tokenizer = R._tokenizer()

    functional = {
        "visfd_composites": check_visfd_composites(tokenizer),
        "uniform_sampling_absa_restaurant": check_uniform_sampling(tokenizer),
        "nc_composite_step_visfd": check_nc_composite_step(tokenizer, device),
        "strict_reload_visfd": check_strict_reload(tokenizer, device),
        "natural_row_label_visfd": check_natural_row_label(),
    }

    checks = {
        "visfd_composites_all_pass": all(functional["visfd_composites"]["checks"].values()),
        "uniform_sampling_all_pass": all(functional["uniform_sampling_absa_restaurant"]["checks"].values()),
        "nc_composite_step_all_pass": all(functional["nc_composite_step_visfd"]["checks"].values()),
        "strict_reload_all_pass": all(functional["strict_reload_visfd"]["checks"].values()),
        "natural_row_label_all_pass": all(functional["natural_row_label_visfd"]["checks"].values()),
    }
    payload = {
        "protocol": "E22-composite-extension-preflight-v1",
        "created_at": _now(),
        "elapsed_seconds": time.time() - started,
        "device": str(device),
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "functional": functional,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(OUT_PATH.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=R._json_default)
        handle.write("\n")
    tmp.replace(OUT_PATH)
    print(json.dumps({"status": payload["status"], "checks": checks, "out": str(OUT_PATH)}, indent=2))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
