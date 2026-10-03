#!/usr/bin/env python3
"""Lock P_vi's five validation-selected checkpoints, then evaluate each test split once.

The lock command reads validation artifacts only. Test data is loaded exclusively by
--test-task after the five-row lock has been written and verified.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

from goibert import train as e1_train  # noqa: E402


ROOT = common.ART.parent / "test" / "P_vi"
LOCK = ROOT / "locked_selection.json"
TASKS = common.TASKS
LR = 1e-5
SEED = 42
CHECKS = ("reload_matches_best", "prediction_file_matches_best",
          "reload_predictions_identical", "ids_stable_and_complete")
SOURCE_FILES = (Path(__file__), Path(common.__file__), Path(__file__).with_name("finetune.py"))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def source_hashes():
    return {str(path.resolve().relative_to(common.ROOT)): common.sha256_file(path) for path in SOURCE_FILES}


def selection(task: str):
    run_dir = common.arm_dir("P_vi", task, LR, SEED)
    metadata_path = run_dir / "metadata.json"
    checkpoint_path = run_dir / "best_checkpoint.pt"
    predictions_path = run_dir / "validation_predictions.json"
    for path in (metadata_path, checkpoint_path, predictions_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    meta = read_json(metadata_path)
    expected = {"arm": "P_vi", "task_id": task, "seed": SEED,
                "learning_rate": LR, "epochs": 30, "micro_batch_size": 16,
                "max_length": 256, "include_test": False, "status": "completed"}
    for key, value in expected.items():
        if meta.get(key) != value:
            raise RuntimeError(f"{task}: metadata {key}={meta.get(key)!r}, expected {value!r}")
    if not all(meta.get("postrun_checks", {}).get(key) for key in CHECKS):
        raise RuntimeError(f"{task}: validation postrun checks incomplete")
    if not 1 <= int(meta["best_epoch"]) <= 30:
        raise RuntimeError(f"{task}: invalid best epoch")
    if abs(float(meta["best_validation_metrics"]["selection_f1"]) - float(meta["best_selection_f1"])) > 1e-12:
        raise RuntimeError(f"{task}: validation score mismatch")
    return {"task_id": task, "run_dir": str(run_dir.resolve()), "seed": SEED,
            "learning_rate": LR, "best_epoch": int(meta["best_epoch"]),
            "validation_f1": float(meta["best_selection_f1"]),
            "checkpoint_sha256": common.sha256_file(checkpoint_path),
            "metadata_sha256": common.sha256_file(metadata_path),
            "validation_predictions_sha256": common.sha256_file(predictions_path)}


def lock():
    if LOCK.exists():
        raise FileExistsError(LOCK)
    if ROOT.exists() and any(ROOT.glob("*/test_metrics.json")):
        raise RuntimeError("Test outputs exist before the selection lock")
    rows = [selection(task) for task in TASKS]
    manifest = {"protocol": "bidirlm-p-vi-5task-test-v1", "locked": True,
                "locked_at": common.now(), "test_accessed_before_lock": False,
                "selection_rule": "P_vi; LR 1e-5 fixed from Restaurant grid; seed 42; best epoch per task by validation F1",
                "source_hashes": source_hashes(), "selections": rows}
    ROOT.mkdir(parents=True, exist_ok=True)
    with LOCK.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Locked {len(rows)} P_vi checkpoints at {LOCK}", flush=True)


def checked_lock(task: str):
    manifest = read_json(LOCK)
    if manifest.get("protocol") != "bidirlm-p-vi-5task-test-v1" or manifest.get("locked") is not True:
        raise RuntimeError("Invalid or unlocked test manifest")
    if manifest.get("test_accessed_before_lock") is not False:
        raise RuntimeError("Test accessed before lock")
    if tuple(row.get("task_id") for row in manifest.get("selections", [])) != TASKS:
        raise RuntimeError("Test manifest must lock exactly the five tasks")
    if manifest.get("source_hashes") != source_hashes():
        raise RuntimeError("Source changed after test lock")
    row = next(row for row in manifest["selections"] if row["task_id"] == task)
    if row != selection(task):
        raise RuntimeError(f"{task}: validation checkpoint or metadata changed after lock")
    return row


def evaluate(task: str, device_name: str):
    row = checked_lock(task)
    out_dir = ROOT / task
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{task}: test outputs already exist at {out_dir}")

    import torch

    device = e1_train.resolve_device(device_name)
    precision = e1_train.resolve_precision("bf16", device)
    bundle = e1_train.load_task_bundle(task, common.ROOT, include_test=True)
    test_split = e1_train.split_field(bundle, "test")
    if test_split is None:
        raise RuntimeError(f"{task}: missing test split")
    train_split = e1_train.split_field(bundle, "train")
    if len(train_split) != int(read_json(Path(row["run_dir"]) / "metadata.json")["train_count"]):
        raise RuntimeError(f"{task}: train split changed since validation")
    model, spec = common.build_model("P_vi", num_classes=e1_train.task_num_classes(bundle),
                                     num_aspects=e1_train.task_num_aspects(bundle),
                                     aspect_class_counts=e1_train.task_aspect_class_counts(bundle))
    state = torch.load(Path(row["run_dir"]) / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    if int(state.get("epoch", -1)) != row["best_epoch"]:
        raise RuntimeError(f"{task}: checkpoint epoch differs from lock")
    model.load_state_dict(state["model"], strict=True)
    model.to(device)
    tokenizer = common.load_tokenizer("P_vi")
    loader = e1_train._make_loader(tokenizer, test_split, max_length=common.MAX_LENGTH,
                                   batch_size=16, shuffle=False, seed=SEED,
                                   labels=e1_train.model_labels(bundle, "test", test_split),
                                   gold_labels=e1_train.split_labels(test_split))
    started = time.time()
    metrics, rows = e1_train.run_test(model, loader, device, precision, task, bundle)
    recomputed = e1_train.evaluate_predictions(task, bundle, [r["gold"] for r in rows],
                                                [r["prediction"] for r in rows], split="test")
    if len(rows) != len(test_split) or len({str(r["id"]) for r in rows}) != len(rows):
        raise RuntimeError(f"{task}: incomplete or duplicate test IDs")
    if abs(e1_train.selection_score(metrics) - e1_train.selection_score(recomputed)) > 1e-12:
        raise RuntimeError(f"{task}: test metric recomputation failed")
    result = {"protocol": "bidirlm-p-vi-5task-test-v1", "arm": "P_vi", "task_id": task,
              "seed": SEED, "learning_rate": LR, "best_epoch": row["best_epoch"],
              "validation_f1": row["validation_f1"], "test_f1": e1_train.selection_score(metrics),
              "test_count": len(rows), "checkpoint_sha256": row["checkpoint_sha256"],
              "model_spec": spec, "elapsed_seconds": time.time() - started,
              "test_metrics": metrics, "recomputed_from_predictions": True, "finished_at": common.now()}
    out_dir.mkdir(parents=True, exist_ok=False)
    e1_train.write_json(out_dir / "test_metrics.json", metrics)
    e1_train.write_json(out_dir / "test_predictions.json", {"rows": rows})
    e1_train.write_json(out_dir / "test_evaluation.json", result)
    print(f"P_vi/{task}: validation {100*row['validation_f1']:.2f}, test {100*result['test_f1']:.2f}", flush=True)


def summarize():
    manifest = read_json(LOCK)
    rows = []
    for item in manifest["selections"]:
        task = item["task_id"]
        evaluation = read_json(ROOT / task / "test_evaluation.json")
        if evaluation["checkpoint_sha256"] != item["checkpoint_sha256"]:
            raise RuntimeError(f"{task}: test result checkpoint differs from lock")
        rows.append({"task_id": task, "validation_f1": item["validation_f1"],
                     "test_f1": evaluation["test_f1"], "test_count": evaluation["test_count"]})
    e1_train.write_json(ROOT / "summary.json", {"protocol": manifest["protocol"], "results": rows})
    for row in rows:
        print(f"{row['task_id']:18} val {100*row['validation_f1']:.2f} test {100*row['test_f1']:.2f} n={row['test_count']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--lock", action="store_true")
    mode.add_argument("--test-task", choices=TASKS)
    mode.add_argument("--summary", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.lock:
        lock()
    elif args.test_task:
        evaluate(args.test_task, args.device)
    else:
        summarize()


if __name__ == "__main__":
    main()
