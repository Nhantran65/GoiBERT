"""Freeze validation winners and evaluate SEA-LION/CafeBERT on test once.

This is deliberately separate from the training source digest.  It refuses to
read test data until every requested validation run is complete and writes the
selection manifest before loading any test split.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.sealion_benchmark import common  # noqa: E402

import torch  # noqa: E402


def read_run(path: Path) -> dict[str, Any]:
    metadata = common.read_json(path / "metadata.json")
    status = common.read_json(path / "status.json") if (path / "status.json").is_file() else {}
    if metadata.get("status") != "completed" or status.get("status") != "completed":
        raise RuntimeError(f"Run is not cleanly completed: {path} metadata={metadata.get('status')} status={status.get('status')}")
    return {"path": path, "metadata": metadata}


def candidates(model_kind: str, task_id: str) -> list[dict[str, Any]]:
    if model_kind == "sealion":
        paths = [common.run_dir(model_kind, task_id, lr) for lr in common.LEARNING_RATES]
    elif task_id == "absa_restaurant":
        paths = [common.CAFE_EXISTING_RUN]
    else:
        paths = [common.run_dir("cafebert", task_id, 2e-5)]
    return [read_run(p) for p in paths]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--micro-batch", type=int, default=32)
    args = parser.parse_args()
    started = time.time()
    common.e1_train.seed_everything(common.SEED)
    device = common.e1_train.resolve_device(args.device)
    precision = common.e1_train.resolve_precision(common.PRECISION, device)
    all_candidates: dict[str, dict[str, list[dict[str, Any]]]] = {"sealion": {}, "cafebert": {}}
    selected: dict[str, dict[str, Any]] = {"sealion": {}, "cafebert": {}}
    for model_kind in ("sealion", "cafebert"):
        for task_id in common.TASKS:
            values = candidates(model_kind, task_id)
            all_candidates[model_kind][task_id] = [{"path": str(v["path"].relative_to(common.ROOT)), "learning_rate": v["metadata"].get("learning_rate"), "best_epoch": v["metadata"].get("best_epoch"), "validation_selection_f1": v["metadata"].get("best_selection_f1"), "checkpoint_sha256": v["metadata"].get("checkpoint_sha256"), "source_hash": v["metadata"].get("source_hash")} for v in values]
            # Ties use the smaller LR, then earlier best epoch.  Test data is
            # not loaded until this whole manifest is written.
            winner = sorted(values, key=lambda v: (-float(v["metadata"]["best_selection_f1"]), float(v["metadata"].get("learning_rate", 2e-5)), int(v["metadata"].get("best_epoch", 10**9))))[0]
            selected[model_kind][task_id] = {"path": str(winner["path"].relative_to(common.ROOT)), "learning_rate": winner["metadata"].get("learning_rate"), "best_epoch": winner["metadata"].get("best_epoch"), "validation_selection_f1": winner["metadata"].get("best_selection_f1"), "checkpoint_sha256": winner["metadata"].get("checkpoint_sha256"), "source_hash": winner["metadata"].get("source_hash"), "selection_rule": "max validation selection_f1; tie smaller learning rate then earliest best epoch"}
    manifest = {"protocol": common.PROTOCOL, "selection": "validation_only", "seed": common.SEED, "created_at": common.now(), "candidates": all_candidates, "selected": selected, "test_status": "locked_after_validation_manifest"}
    out_root = common.ARTIFACT_ROOT / "test"
    common.write_json(common.ARTIFACT_ROOT / "selection_manifest.json", manifest)
    common.write_json(out_root / "selection_manifest.json", manifest)
    for model_kind in ("sealion", "cafebert"):
        tokenizer = common.load_sealion_tokenizer() if model_kind == "sealion" else common.load_cafe_tokenizer()
        tokenizer_obj, tokenizer_manifest = tokenizer
        for task_id in common.TASKS:
            selected_path = common.ROOT / selected[model_kind][task_id]["path"]
            contract = common.task_contract(task_id, include_test=True)
            model, model_spec = common.build_model(model_kind, contract)
            checkpoint = selected_path / "best_checkpoint.pt"
            state = common.load_model_checkpoint(checkpoint, model)
            model.to(device)
            loader = common.test_loader(tokenizer_obj, contract, args.micro_batch)
            expected_ids = list(common.e1_train.split_ids(common.e1_train.split_field(contract["bundle"], "test")))
            test_ids = [example_id for batch in loader for example_id in batch["ids"]]
            if test_ids != expected_ids:
                raise RuntimeError(f"{model_kind}/{task_id}: test IDs changed by loader")
            metrics, rows = common.e1_train.run_test(model, loader, device, precision, task_id, contract["bundle"])
            destination = out_root / model_kind / task_id
            common.write_json(destination / "test_metrics.json", {"model_kind": model_kind, "task_id": task_id, "metrics": metrics, "selected": selected[model_kind][task_id]})
            common.write_json(destination / "test_predictions.json", {"model_kind": model_kind, "task_id": task_id, "rows": rows})
            common.write_json(destination / "evaluation_metadata.json", {"protocol": common.PROTOCOL, "model_kind": model_kind, "task_id": task_id, "selected_checkpoint": str(checkpoint.relative_to(common.ROOT)), "checkpoint_sha256": common.sha256_file(checkpoint), "checkpoint_state_metadata": state.get("metadata"), "model": model_spec, "tokenizer": tokenizer_manifest, "test_count": len(test_ids), "test_ids_exact": True, "precision": precision, "runtime": common.runtime(device), "evaluated_at": common.now()})
            print(f"test {model_kind}/{task_id}: selection_f1={metrics.get('selection_f1')}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    manifest["test_status"] = "completed"
    manifest["test_evaluated_at"] = common.now()
    manifest["elapsed_seconds"] = time.time() - started
    common.write_json(common.ARTIFACT_ROOT / "selection_manifest.json", manifest)
    common.write_json(out_root / "selection_manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
