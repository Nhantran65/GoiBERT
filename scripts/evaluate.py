#!/usr/bin/env python3
"""Evaluate a fine-tuned checkpoint on the test split.

Reads ``config.json`` and ``best_checkpoint.pt`` from a run directory written
by ``scripts/finetune.py`` and writes ``test_metrics.json`` and
``test_predictions.json`` next to them.

    python scripts/evaluate.py runs/finetune/large/absa_restaurant/seed_42
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from goibert import training as T
from goibert.data import extra, family, uit
from goibert.modeling import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    cfg = T.read_json(args.run_dir / "config.json")
    task, kind, micro = cfg["task"], cfg["kind"], int(cfg["micro_batch_size"])
    device = T.resolve_device(args.device)
    precision = T.resolve_precision(cfg["precision"], device)
    tokenizer = AutoTokenizer.from_pretrained(cfg["backbone"], use_fast=True)
    state = torch.load(args.run_dir / "best_checkpoint.pt", map_location="cpu", weights_only=True)

    if family(task) == "uit":
        bundle = uit.load_task(task, include_test=True)
        num_labels, num_aspects, counts = T.uit_task_shape(bundle)
        model = build_model(cfg["backbone"], kind=kind, task_type="sequence", num_labels=num_labels,
                            num_aspects=num_aspects, aspect_class_counts=counts if num_aspects else None)
        model.load_state_dict(state["model"], strict=True)
        model.to(device)
        test = bundle.splits["test"]
        loader = T.make_uit_loader(tokenizer, test, labels=bundle.labels_for_model("test"), max_length=int(cfg["max_length"]),
                                   batch_size=micro, shuffle=False, seed=42)
        metrics, rows = T.validate_uit(model, loader, device, precision, task, bundle, split="test")
    else:
        data = extra.load_task(task, include_test=True)
        labels = list(data.label_names)
        model = build_model(cfg["backbone"], kind=kind, task_type="token" if task == "phoner" else "sequence", num_labels=len(labels))
        model.load_state_dict(state["model"], strict=True)
        model.to(device)
        loader, _ = T.make_extra_loader(data.test, task, tokenizer, labels, micro, 42, False, int(cfg["max_length"]))
        metrics, rows = T.validate_extra(model, loader, task, labels, device, precision, split="test")

    T.write_json(args.run_dir / "test_metrics.json", {"checkpoint_epoch": state["epoch"], **metrics})
    T.write_json(args.run_dir / "test_predictions.json", {"rows": rows})
    print({k: v for k, v in metrics.items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
