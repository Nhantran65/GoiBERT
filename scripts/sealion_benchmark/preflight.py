"""SEA-LION five-task preflight: model/tokenizer, finite gradients, IDs/checkpoint."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.sealion_benchmark import common  # noqa: E402

import torch  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--micro-batch", type=int, default=2)
    args = parser.parse_args()
    started = time.time()
    common.e1_train.seed_everything(common.SEED)
    device = common.e1_train.resolve_device(args.device)
    precision = common.e1_train.resolve_precision(common.PRECISION, device)
    tokenizer, tokenizer_manifest = common.load_sealion_tokenizer()
    result = {
        "protocol": common.PROTOCOL, "status": "running", "seed": common.SEED,
        "model": {"repo": common.MODEL_REPO, "commit": common.MODEL_COMMIT, "path": str(common.MODEL_DIR.relative_to(common.ROOT))},
        "runtime": common.runtime(device), "gpu_snapshot": common.gpu_snapshot(),
        "tokenizer": tokenizer_manifest, "tasks": {}, "source_hash": common.source_hash(), "started_at": common.now(),
    }
    common.write_json(common.ARTIFACT_ROOT / "preflight.json", result)
    for task_id in common.TASKS:
        contract = common.task_contract(task_id)
        loader, valid_loader = common.make_loaders(tokenizer, contract, args.micro_batch)
        # Check the complete validation ID stream independently from model code.
        valid_ids = [example_id for batch in valid_loader for example_id in batch["ids"]]
        expected_ids = list(common.e1_train.split_ids(contract["validation"]))
        if valid_ids != expected_ids:
            raise RuntimeError(f"{task_id}: validation IDs changed by loader")
        model, spec = common.build_model("sealion", contract)
        model.to(device)
        batch = next(iter(loader))
        moved = common.e1_train._move_batch(batch, device)
        model.train(); model.zero_grad(set_to_none=True)
        with common.e1_train.make_autocast(precision, device):
            logits = model(moved["input_ids"], moved["attention_mask"])
            loss = common.e1_train.compute_loss(logits, moved["labels"], num_aspects=contract["num_aspects"], aspect_class_counts=contract["aspect_class_counts"])
        loss.backward()
        finite_logits = bool(torch.isfinite(logits.detach().float()).all())
        finite_loss = bool(torch.isfinite(loss.detach().float()))
        finite_grads = all(p.grad is None or bool(torch.isfinite(p.grad.detach().float()).all()) for p in model.parameters())
        expected_shape = ([args.micro_batch, contract["num_aspects"], contract["aspect_class_counts"][0]] if contract["num_aspects"] and len(set(contract["aspect_class_counts"])) == 1 else ([args.micro_batch, sum(contract["aspect_class_counts"])] if contract["num_aspects"] else [args.micro_batch, contract["num_classes"]]))
        if list(logits.shape) != expected_shape or not (finite_logits and finite_loss and finite_grads):
            raise RuntimeError(f"{task_id}: preflight output/gradient failure shape={list(logits.shape)} expected={expected_shape} finite={(finite_logits, finite_loss, finite_grads)}")
        # A real serialization round trip on the representative first task
        # catches key/schema issues before the expensive queue starts.
        checkpoint_ok = None
        if task_id == common.TASKS[0]:
            path = common.ARTIFACT_ROOT / "preflight_checkpoint.pt"
            torch.save({"model": model.state_dict(), "task_id": task_id, "ids": valid_ids[:8]}, path)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            checkpoint_ok = saved["task_id"] == task_id and saved["ids"] == valid_ids[:8] and set(saved["model"]) == set(model.state_dict())
            path.unlink()
            if not checkpoint_ok:
                raise RuntimeError("preflight checkpoint/ID round trip failed")
        result["tasks"][task_id] = {
            "train_count": len(contract["train"]), "validation_count": len(contract["validation"]),
            "num_classes": contract["num_classes"], "num_aspects": contract["num_aspects"],
            "aspect_class_counts": list(contract["aspect_class_counts"]), "validation_ids_exact": True,
            "first_batch_ids": list(batch["ids"]), "logits_shape": list(logits.shape),
            "expected_logits_shape": expected_shape, "finite_logits": finite_logits, "finite_loss": finite_loss,
            "finite_gradients": finite_grads, "model": spec, "checkpoint_id_roundtrip": checkpoint_ok,
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
