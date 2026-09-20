"""Seed-confirmation entrypoint: run the unchanged T (scripts/cdbk_pilot/finetune.py) or A
(scripts/cafebert_aspect_pair/finetune.py) loop with a different seed, without editing either script.

Both loops read the seed only through module globals (``finetune.SEED`` for seed_everything / metadata,
``common.SEED`` for the loader sampler), so the adapter loads the two modules by path, rebinds both globals,
asserts the rebinding took, and calls the original ``run(args)``.  For T, which does not record head-init or
review-order hashes, two spies are installed: ``common.build_model`` is wrapped to hash the freshly initialised
head, and the train loader's collate is wrapped to record the review IDs of every batch so per-epoch order
hashes can be written in the same format as arm A's ``epoch_curves.json`` (sha256 over "\n".join(ids)+"\n"
per group).  Outputs go to --out-dir plus ``seed_checks.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CDBK = ROOT / "scripts" / "cdbk_pilot"
ASPECT = ROOT / "scripts" / "cafebert_aspect_pair"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _epoch_order_hashes(batches_ids: list[list[str]], batches_per_epoch: int) -> list[str]:
    hashes = []
    for start in range(0, len(batches_ids), batches_per_epoch):
        digest = hashlib.sha256()
        for ids in batches_ids[start : start + batches_per_epoch]:
            digest.update(("\n".join(ids) + "\n").encode("utf-8"))
        hashes.append(digest.hexdigest())
    return hashes


def run_t(seed: int, device: str, out_dir: Path, micro_batch: int, smoke: int = 0) -> dict[str, Any]:
    # scripts/cdbk_pilot/finetune.py does `import common` -> must resolve to scripts/cdbk_pilot/common.py
    sys.path.insert(0, str(CDBK))
    common = _load("common", CDBK / "common.py")
    finetune = _load("cdbk_finetune", CDBK / "finetune.py")
    common.SEED = seed
    finetune.SEED = seed
    assert finetune.SEED == seed and sys.modules["common"].SEED == seed and finetune.common.SEED == seed

    import torch
    head_hashes: list[str] = []
    original_build = common.build_model

    def spied_build(init, contract):
        model, spec_dict = original_build(init, contract)
        head = {k: v for k, v in model.state_dict().items() if k.startswith("head.")}
        head_hashes.append(_state_hash(head))
        return model, spec_dict

    common.build_model = spied_build
    batches_ids: list[list[str]] = []
    original_make_loaders = common.make_loaders

    def spied_make_loaders(tokenizer, contract, micro):
        train_loader, valid_loader = original_make_loaders(tokenizer, contract, micro)
        inner = train_loader.collate_fn

        def recording_collate(rows):
            batch = inner(rows)
            batches_ids.append([str(v) for v in batch["ids"]])
            return batch

        train_loader.collate_fn = recording_collate
        spied_make_loaders.batches_per_epoch = len(train_loader)
        return train_loader, valid_loader

    spied_make_loaders.batches_per_epoch = None
    common.make_loaders = spied_make_loaders

    args = SimpleNamespace(arm="T", device=device, micro_batch=micro_batch, teacher_cache=None, out_dir=str(out_dir), smoke=smoke)
    metadata = finetune.run(args)
    n_epochs = int(metadata["epochs"])
    per_epoch = spied_make_loaders.batches_per_epoch
    order_hashes = _epoch_order_hashes(batches_ids, per_epoch)
    checks = {
        "arm": "T", "seed_requested": seed, "seed_in_finetune_module": finetune.SEED, "seed_in_common_module": common.SEED,
        "seed_in_metadata": metadata["seed"], "head_init_sha256": head_hashes[0] if head_hashes else None, "build_model_calls": len(head_hashes),
        "batches_recorded": len(batches_ids), "batches_per_epoch": per_epoch, "epochs": n_epochs,
        "batches_recorded_equals_epochs_x_per_epoch": len(batches_ids) == n_epochs * per_epoch,
        "review_order_sha256_per_epoch": order_hashes,
        "review_order_sha256_all_epochs": hashlib.sha256("".join(order_hashes).encode("ascii")).hexdigest(),
        "reviews_per_epoch_unique": all(len({i for b in batches_ids[s : s + per_epoch] for i in b}) == sum(len(b) for b in batches_ids[s : s + per_epoch]) for s in range(0, len(batches_ids), per_epoch)),
    }
    _write_json(out_dir / "seed_checks.json", checks)
    return checks


def _state_hash(state) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        t = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8")); digest.update(str(tuple(t.shape)).encode("ascii")); digest.update(str(t.dtype).encode("ascii")); digest.update(t.numpy().tobytes())
    return digest.hexdigest()


def run_a(seed: int, device: str, out_dir: Path, micro: int, eval_micro: int, log_every: int, smoke: int = 0) -> dict[str, Any]:
    sys.path.insert(0, str(ASPECT))
    common = _load("common", ASPECT / "common.py")
    finetune = _load("aspect_finetune", ASPECT / "finetune.py")
    common.SEED = seed
    finetune.SEED = seed
    assert finetune.SEED == seed and sys.modules["common"].SEED == seed and finetune.common.SEED == seed
    args = SimpleNamespace(arm="A", device=device, micro=micro, eval_micro=eval_micro, out_dir=str(out_dir), smoke=smoke, log_every=log_every)
    metadata = finetune.run(args)
    curves = json.loads((out_dir / "epoch_curves.json").read_text(encoding="utf-8"))["epochs"]
    checks = {
        "arm": "A", "seed_requested": seed, "seed_in_finetune_module": finetune.SEED, "seed_in_common_module": common.SEED,
        "seed_in_metadata": metadata["seed"], "head_init_sha256": metadata["model"]["head_init_sha256"],
        "review_order_sha256_per_epoch": [e["review_order_sha256"] for e in curves],
        "review_order_sha256_all_epochs": metadata["review_order_sha256_all_epochs"], "epochs": metadata["epochs"],
    }
    _write_json(out_dir / "seed_checks.json", checks)
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, choices=("T", "A"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--micro", type=int, default=None, help="T: micro-batch reviews (32); A: micro-batch pairs (64)")
    parser.add_argument("--eval-micro", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=55)
    parser.add_argument("--smoke", type=int, default=0, help="debug only: N batches/groups, 1 epoch (never for the real runs)")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        existing = sorted(p.name for p in out_dir.iterdir())
        if not args.smoke and any(n in existing for n in ("epoch_curves.json", "best_checkpoint.pt", "metadata.json")):
            print(f"refusing to overwrite {out_dir}: {existing}", file=sys.stderr)
            return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[confirm] arm={args.arm} seed={args.seed} device={args.device} out={out_dir} pid={os.getpid()}", flush=True)
    try:
        if args.arm == "T":
            checks = run_t(args.seed, args.device, out_dir, args.micro or 32, args.smoke)
        else:
            checks = run_a(args.seed, args.device, out_dir, args.micro or 64, args.eval_micro, args.log_every, args.smoke)
        print(f"[confirm] seed_checks: seed={checks['seed_in_metadata']} head_init={checks['head_init_sha256'][:12]} order_all={checks['review_order_sha256_all_epochs'][:12]}", flush=True)
        return 0
    except Exception as error:  # noqa: BLE001
        traceback.print_exc()
        _write_json(out_dir / "status.json", {"status": "failed", "arm": args.arm, "seed": args.seed, "pid": os.getpid(), "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "error": repr(error), "traceback": traceback.format_exc()})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
