"""Pre-launch checks for the seed-confirmation round: prove the new seed changes head init and the review
permutation, that the T loader and the A loader visit reviews in the same order under that seed, and record
the expected hashes so the finished runs can be verified against them (confirm_summarize.py).

Writes <root>/checks/pre_launch.json.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as ap  # noqa: E402  (aspect-pair common; ap.cdbk is scripts/cdbk_pilot/common.py loaded by path)
from common import EFFECTIVE_BATCH_REVIEWS, e1_train  # noqa: E402

import torch  # noqa: E402

ROOT = ap.ROOT
CONFIRM_ROOT = ROOT / "artifacts" / "cafebert_aspect_pair_confirmation"


def head_hash_t(seed: int, contract) -> str:
    e1_train.seed_everything(seed)
    model, _ = ap.cdbk.build_model("cafebert", contract)
    return ap.state_dict_hash({k: v for k, v in model.state_dict().items() if k.startswith("head.")})


def head_hash_a(seed: int) -> str:
    e1_train.seed_everything(seed)
    _, spec = ap.build_model()
    return spec["head_init_sha256"]


def order_hashes_t(tokenizer, contract, seed: int, epochs: int, micro: int) -> tuple[list[str], int]:
    saved = ap.cdbk.SEED
    ap.cdbk.SEED = seed
    try:
        train_loader, _ = ap.cdbk.make_loaders(tokenizer, contract, micro)
    finally:
        ap.cdbk.SEED = saved
    hashes = []
    for _ in range(epochs):
        digest = hashlib.sha256()
        for batch in train_loader:
            digest.update(("\n".join(str(v) for v in batch["ids"]) + "\n").encode("utf-8"))
        hashes.append(digest.hexdigest())
    return hashes, len(train_loader)


def order_hashes_a(ids: list[str], seed: int, epochs: int) -> tuple[list[str], int]:
    loader = ap.review_loader([[i] for i in ids], reviews_per_group=EFFECTIVE_BATCH_REVIEWS, shuffle=True, seed=seed)
    hashes = []
    for _ in range(epochs):
        digest = hashlib.sha256()
        for group in loader:
            digest.update(("\n".join(group) + "\n").encode("utf-8"))
        hashes.append(digest.hexdigest())
    return hashes, len(loader)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--baseline-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    seed, base = args.seed, args.baseline_seed
    contract = ap.task_contract()
    tokenizer, _ = ap.load_tokenizer()
    ids = [str(v) for v in e1_train.split_ids(contract["train"])]
    report: dict[str, Any] = {"seed": seed, "baseline_seed": base, "generated_at": ap.now()}

    # head init: seed changes it; method validated against the recorded A seed42 hash
    a42_recorded = ap.read_json(ap.run_dir("A") / "metadata.json")["model"]["head_init_sha256"]
    head = {"T": {base: head_hash_t(base, contract), seed: head_hash_t(seed, contract)}, "A": {base: head_hash_a(base), seed: head_hash_a(seed)}}
    report["head_init"] = {
        "expected": {arm: {str(s): h for s, h in d.items()} for arm, d in head.items()},
        "A_seed42_expected_equals_recorded_run": head["A"][base] == a42_recorded, "A_seed42_recorded": a42_recorded,
        "T_changes_with_seed": head["T"][base] != head["T"][seed], "A_changes_with_seed": head["A"][base] != head["A"][seed],
        "T_reproducible": head_hash_t(seed, contract) == head["T"][seed], "A_reproducible": head_hash_a(seed) == head["A"][seed],
    }
    # review order: T loader (reference code path) vs A loader under the new seed, vs the seed42 A run
    t_hashes, t_batches = order_hashes_t(tokenizer, contract, seed, args.epochs, 32)
    a_hashes, a_groups = order_hashes_a(ids, seed, args.epochs)
    a42_curves = ap.read_json(ap.run_dir("A") / "epoch_curves.json")["epochs"]
    a42_hashes = [e["review_order_sha256"] for e in a42_curves]
    report["review_order"] = {
        "expected_per_epoch": a_hashes, "expected_all_epochs": hashlib.sha256("".join(a_hashes).encode("ascii")).hexdigest(),
        "T_loader_equals_A_loader_all_epochs": t_hashes == a_hashes, "batches_per_epoch": {"T": t_batches, "A": a_groups},
        "differs_from_seed42_every_epoch": all(h != h42 for h, h42 in zip(a_hashes, a42_hashes)),
        "epochs_distinct": len(set(a_hashes)) == args.epochs,
    }
    report["ok"] = all([
        report["head_init"]["A_seed42_expected_equals_recorded_run"], report["head_init"]["T_changes_with_seed"], report["head_init"]["A_changes_with_seed"],
        report["head_init"]["T_reproducible"], report["head_init"]["A_reproducible"], report["review_order"]["T_loader_equals_A_loader_all_epochs"],
        report["review_order"]["differs_from_seed42_every_epoch"], report["review_order"]["epochs_distinct"], t_batches == 220 and a_groups == 220,
    ])
    out = CONFIRM_ROOT / f"seed{seed}" / "checks" / "pre_launch.json"
    ap.write_json(out, report)
    print({k: v for k, v in report["head_init"].items() if k != "expected"})
    print({k: v for k, v in report["review_order"].items() if k != "expected_per_epoch"})
    print("PRE-LAUNCH OK:", report["ok"], "->", out)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
