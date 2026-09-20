#!/usr/bin/env python3
"""Evaluate exactly the three dev-locked winners on the frozen test split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_intent import (
    BENCHMARK,
    TEST_PATH,
    IntentDataset,
    build_dataset_rows,
    collate_examples,
    evaluate,
    load_locked_config,
    load_tokenizer_and_model,
    metric_report,
    pinned_provenance,
    resolve_device,
    seed_everything,
)


ROOT = Path(__file__).resolve().parents[2]
SELECTION = BENCHMARK / "results/selection.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, default=BENCHMARK / "results")
    args = parser.parse_args()
    output_dir = args.output if args.output.is_absolute() else ROOT / args.output
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    if selection.get("selection_status") != "locked" or selection.get("criteria", {}).get("test_used_for_selection"):
        raise RuntimeError("selection is not locked before test evaluation")
    selected = selection.get("selected", [])
    if len(selected) != 3:
        raise RuntimeError(f"expected exactly three selected winners, found {len(selected)}")
    config = load_locked_config()
    seed_everything(int(config["seed"]))
    device = resolve_device(args.gpu)
    test_rows = build_dataset_rows(TEST_PATH)
    if len(test_rows) != 77:
        raise AssertionError(f"frozen test count mismatch: {len(test_rows)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = []
    for winner in selected:
        model_name = winner["model_name"]
        run_dir = ROOT / winner["run_dir"]
        checkpoint = run_dir / "best.pt"
        if sha256(checkpoint) != winner["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint changed after selection: {checkpoint}")
        tokenizer, model, loading = load_tokenizer_and_model(model_name, device, config)
        test_dataset = IntentDataset(test_rows, tokenizer, config["labels"], int(config["max_length"]))
        loader = DataLoader(
            test_dataset,
            batch_size=int(config["batch_size"]),
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=lambda examples: collate_examples(examples, tokenizer.pad_token_id),
        )
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint_data["model"], strict=True)
        metrics, predictions = evaluate(model, loader, device, config["labels"], "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32")
        gold = [row["gold_id"] for row in predictions]
        predicted = [row["pred_id"] for row in predictions]
        recomputed = metric_report(gold, predicted, config["labels"])
        recomputed["loss"] = metrics["loss"]
        if recomputed["macro_f1"] != metrics["macro_f1"] or recomputed["accuracy"] != metrics["accuracy"]:
            raise AssertionError(f"saved test metric mismatch for {model_name}")
        prediction_path = output_dir / f"test_predictions_{model_name}.jsonl"
        prediction_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in predictions), encoding="utf-8")
        result = {
            "model_name": model_name,
            "run_dir": winner["run_dir"],
            "learning_rate": winner["learning_rate"],
            "best_epoch": winner["best_epoch"],
            "device": str(device),
            "precision": "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32",
            "test_metrics": metrics,
            "recomputed_from_saved_predictions": recomputed,
            "prediction_path": str(prediction_path.relative_to(ROOT)),
            "prediction_sha256": sha256(prediction_path),
            "checkpoint_sha256": winner["checkpoint_sha256"],
            "loading": loading,
            "provenance": pinned_provenance(config),
        }
        (output_dir / f"test_metrics_{model_name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        aggregate.append(result)
        del model, tokenizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    payload = {
        "selection_path": str(SELECTION.relative_to(ROOT)),
        "selection_sha256": sha256(SELECTION),
        "test_sha256": sha256(TEST_PATH),
        "evaluated_count": len(aggregate),
        "models": aggregate,
    }
    (output_dir / "test_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
