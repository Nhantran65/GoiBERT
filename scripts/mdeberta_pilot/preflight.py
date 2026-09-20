"""Tiny five-schema GPU preflight for the mDeBERTa pilot."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.mdeberta_pilot import common  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--micro-batch", type=int, default=2)
    args = parser.parse_args()
    started = time.time()
    common.e1_train.seed_everything(common.SEED)
    device = common.e1_train.resolve_device(args.device)
    precision = common.e1_train.resolve_precision(common.PRECISION, device)
    tokenizer, tokenizer_manifest = common.load_tokenizer()
    result = {
        "protocol": common.PROTOCOL,
        "status": "running",
        "seed": common.SEED,
        "model": {"repo": common.MODEL_REPO, "commit": common.MODEL_COMMIT, "path": str(common.MODEL_DIR.relative_to(common.ROOT))},
        "runtime": common.runtime(device),
        "gpu_snapshot": common.gpu_snapshot(),
        "tokenizer": tokenizer_manifest,
        "tasks": {},
        "source_hash": common.source_hash(),
        "started_at": common.now(),
    }
    common.write_json(common.ARTIFACT_ROOT / "preflight.json", result)
    for task_id in common.TASKS:
        contract = common.task_contract(task_id)
        train_loader, valid_loader = common.make_loaders(tokenizer, contract, args.micro_batch)
        valid_ids = [example_id for batch in valid_loader for example_id in batch["ids"]]
        expected_ids = list(common.e1_train.split_ids(contract["validation"]))
        if valid_ids != expected_ids:
            raise RuntimeError(f"{task_id}: validation IDs changed by loader")
        model, spec = common.build_model(contract)
        model.to(device)
        batch = next(iter(train_loader))
        moved = common.e1_train._move_batch(batch, device)
        model.train()
        model.zero_grad(set_to_none=True)
        with common.e1_train.make_autocast(precision, device):
            logits = model(moved["input_ids"], moved["attention_mask"])
            loss = common.e1_train.compute_loss(
                logits,
                moved["labels"],
                num_aspects=contract["num_aspects"],
                aspect_class_counts=contract["aspect_class_counts"],
            )
        loss.backward()
        finite_logits = bool(torch.isfinite(logits.detach().float()).all())
        finite_loss = bool(torch.isfinite(loss.detach().float()))
        finite_grads = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad.detach().float()).all())
            for parameter in model.parameters()
        )
        expected_shape = (
            [args.micro_batch, contract["num_aspects"], contract["aspect_class_counts"][0]]
            if contract["num_aspects"] and len(set(contract["aspect_class_counts"])) == 1
            else ([args.micro_batch, sum(contract["aspect_class_counts"])] if contract["num_aspects"] else [args.micro_batch, contract["num_classes"]])
        )
        # This adapter intentionally keeps flat logits; E1's loss/evaluator
        # split variable-width ViSFD slots and equal-width ABSA slots.
        expected_flat_shape = [args.micro_batch, contract["num_classes"]]
        if list(logits.shape) != expected_flat_shape or not (finite_logits and finite_loss and finite_grads):
            raise RuntimeError(
                f"{task_id}: output/gradient failure shape={list(logits.shape)} expected={expected_flat_shape} "
                f"finite={(finite_logits, finite_loss, finite_grads)}"
            )
        result["tasks"][task_id] = {
            "train_count": len(contract["train"]),
            "validation_count": len(contract["validation"]),
            "num_classes": contract["num_classes"],
            "num_aspects": contract["num_aspects"],
            "aspect_class_counts": list(contract["aspect_class_counts"]),
            "validation_ids_exact": True,
            "first_batch_ids": list(batch["ids"]),
            "logits_shape": list(logits.shape),
            "expected_flat_logits_shape": expected_flat_shape,
            "semantic_slot_shape": expected_shape,
            "finite_logits": finite_logits,
            "finite_loss": finite_loss,
            "finite_gradients": finite_grads,
            "model": spec,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        common.write_json(common.ARTIFACT_ROOT / "preflight.json", result)
        print(f"preflight {task_id}: shape={list(logits.shape)} finite=({finite_logits},{finite_loss},{finite_grads}) IDs=exact", flush=True)
    result.update({"status": "passed", "finished_at": common.now(), "elapsed_seconds": time.time() - started, "gpu_snapshot_end": common.gpu_snapshot()})
    common.write_json(common.ARTIFACT_ROOT / "preflight.json", result)
    print("preflight passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
