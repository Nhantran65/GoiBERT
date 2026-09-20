#!/usr/bin/env python3
"""Locked eight-intent classifier training and evaluation implementation.

The module is intentionally self-contained and offline: it reads only the frozen
intent_benchmark JSONL files and local backbone/tokenizer directories.  It supports
the one-batch smoke path and one full trial; detached orchestration is provided by
the companion controller script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_scheduler


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "intent_benchmark"
CONFIG_PATH = BENCHMARK / "training" / "training_config.json"
LABEL_MAP_PATH = BENCHMARK / "label_map.json"
TRAIN_PATH = BENCHMARK / "train.jsonl"
DEV_PATH = BENCHMARK / "dev.jsonl"
TEST_PATH = BENCHMARK / "test.jsonl"
CORRECTED_PATH = BENCHMARK / "corrected_dataset.jsonl"
PROVENANCE_PATH = BENCHMARK / "provenance.json"
IMPLEMENTATION_PATH = Path(__file__).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_locked_config() -> dict[str, Any]:
    config = load_json(CONFIG_PATH)
    labels = load_json(LABEL_MAP_PATH)["order"]
    if config["labels"] != labels:
        raise AssertionError("training labels disagree with frozen label_map order")
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # This dataset is small and the locked protocol prefers repeatability.  The
    # L40S kernels used here support deterministic execution for this path.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(gpu: int | None = None) -> torch.device:
    if gpu is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; training protocol requires an authorized CUDA GPU")
        return torch.device(f"cuda:{gpu}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_context(device: torch.device, precision: str = "bf16"):
    if device.type == "cuda" and precision == "bf16" and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _tokenize_segment(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]


@dataclass
class EncodedExample:
    input_ids: list[int]
    label: int
    sample_id: str
    original_length: int
    history_tokens_dropped: int
    current_tokens_dropped: int


def serialize_and_encode(tokenizer: Any, row: dict[str, Any], max_length: int) -> EncodedExample:
    """Encode customer history/current turn with current-first truncation.

    The serialized text has explicit history/current markers.  History is flattened
    oldest-first and dropped from its beginning until the current segment fits.  If
    the current segment itself exceeds the model budget, its tail is truncated and
    the count is recorded in the example metadata.
    """

    messages = row["model_input"]["messages"]
    if not messages or messages[-1]["role"] != "customer":
        raise AssertionError(f"invalid model input messages for {row['sample_id']}")
    history_segments: list[list[int]] = []
    history_prefix = "[HISTORY] customer: "
    current_prefix = "[CURRENT] customer: "
    newline_ids = _tokenize_segment(tokenizer, "\n")
    for message in messages[:-1]:
        if message["role"] != "customer":
            raise AssertionError(f"non-customer history message for {row['sample_id']}")
        history_segments.append(_tokenize_segment(tokenizer, history_prefix + message["text"]))
    current_ids = _tokenize_segment(tokenizer, current_prefix + messages[-1]["text"])
    history_ids: list[int] = []
    for index, segment in enumerate(history_segments):
        if index:
            history_ids.extend(newline_ids)
        history_ids.extend(segment)
    special_count = len(tokenizer.build_inputs_with_special_tokens([]))
    budget = max_length - special_count
    if budget <= 0:
        raise ValueError("max_length is too small for tokenizer special tokens")
    current_dropped = max(0, len(current_ids) - budget)
    if len(current_ids) > budget:
        history_kept: list[int] = []
        current_kept = current_ids[:budget]
        history_dropped = len(history_ids)
    else:
        # Reserve the separator that is inserted between retained history and
        # the current segment; otherwise a long history can produce max_length+1.
        separator_budget = len(newline_ids) if history_ids else 0
        history_budget = max(0, budget - len(current_ids) - separator_budget)
        history_dropped = max(0, len(history_ids) - history_budget)
        history_kept = history_ids[history_dropped:]
        current_kept = current_ids
    content = history_kept + (newline_ids if history_kept else []) + current_kept
    # Keep the current marker separated after history trimming.  The newline
    # before current is counted in content budget and is harmless if absent.
    input_ids = tokenizer.build_inputs_with_special_tokens(content)
    if len(input_ids) > max_length:
        raise AssertionError(f"encoded sequence exceeds max_length for {row['sample_id']}")
    return EncodedExample(
        input_ids=input_ids,
        label=int(row["label_id"]),
        sample_id=row["sample_id"],
        original_length=special_count + len(history_ids) + (len(newline_ids) if history_ids else 0) + len(current_ids),
        history_tokens_dropped=history_dropped,
        current_tokens_dropped=current_dropped,
    )


class IntentDataset(Dataset[EncodedExample]):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, labels: list[str], max_length: int):
        self.rows = []
        label_to_id = {label: index for index, label in enumerate(labels)}
        for row in rows:
            if row["label"] not in label_to_id:
                raise AssertionError(f"unknown reviewed label {row['label']}")
            if "expectedIntent" in json.dumps(row.get("model_input", {}), ensure_ascii=False):
                raise AssertionError("source expectedIntent leaked into model input")
            copied = dict(row)
            copied["label_id"] = label_to_id[row["label"]]
            self.rows.append(serialize_and_encode(tokenizer, copied, max_length))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> EncodedExample:
        return self.rows[index]


def collate_examples(examples: list[EncodedExample], pad_token_id: int) -> dict[str, Any]:
    ids = [torch.tensor(example.input_ids, dtype=torch.long) for example in examples]
    input_ids = pad_sequence(ids, batch_first=True, padding_value=pad_token_id)
    attention_mask = input_ids.ne(pad_token_id).long()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": torch.tensor([example.label for example in examples], dtype=torch.long),
        "sample_ids": [example.sample_id for example in examples],
        "history_tokens_dropped": [example.history_tokens_dropped for example in examples],
        "current_tokens_dropped": [example.current_tokens_dropped for example in examples],
    }


def build_dataset_rows(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    for row in rows:
        if "model_input" not in row or "label" not in row:
            raise AssertionError(f"invalid training row {row.get('sample_id')}")
        forbidden = ("expectedIntent", "assertions", "fixtures", "primaryIntent", "source_ref", "sourceRef")
        model_text = json.dumps(row["model_input"], ensure_ascii=False)
        if any(field in model_text for field in forbidden):
            raise AssertionError(f"metadata leakage in {row['sample_id']}")
    return rows


def load_tokenizer_and_model(model_name: str, device: torch.device, config: dict[str, Any] | None = None) -> tuple[Any, "IntentClassifier", dict[str, Any]]:
    config = config or load_locked_config()
    model_cfg = config["models"][model_name]
    model_path = ROOT / model_cfg["path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    load_kwargs: dict[str, Any] = {"local_files_only": True, "output_loading_info": True, "add_pooling_layer": False}
    sdpa_requested = device.type == "cuda"
    if sdpa_requested:
        load_kwargs["attn_implementation"] = "sdpa"
    try:
        backbone, loading_info = AutoModel.from_pretrained(model_path, **load_kwargs)
        attention_impl = "sdpa" if sdpa_requested else "default"
    except (TypeError, ValueError) as error:
        if not sdpa_requested:
            raise
        load_kwargs.pop("attn_implementation", None)
        backbone, loading_info = AutoModel.from_pretrained(model_path, **load_kwargs)
        attention_impl = f"fallback_default:{type(error).__name__}"
    allowed_missing = set(config["backbone_loading"]["allowed_missing_patterns"])
    missing = list(loading_info.get("missing_keys", []))
    unexpected = list(loading_info.get("unexpected_keys", []))
    mismatched = list(loading_info.get("mismatched_keys", []))
    bad_missing = [key for key in missing if key not in allowed_missing]
    bad_unexpected = [
        key
        for key in unexpected
        if not (
            key.startswith("lm_head.")
            or key.startswith("cls.predictions.")
            or key.startswith("pooler.")
            or key.startswith("roberta.pooler.")
        )
    ]
    if bad_missing or bad_unexpected or mismatched:
        raise RuntimeError(
            "backbone loading mismatch: "
            + json.dumps({"missing": bad_missing, "unexpected": bad_unexpected, "mismatched": mismatched}, sort_keys=True)
        )
    classifier = IntentClassifier(backbone, num_labels=len(config["labels"]), dropout=float(config["head_dropout"]))
    classifier.to(device)
    info = {
        "model_name": model_name,
        "model_path": str(model_path.relative_to(ROOT)),
        "model_files": {
            "config_sha256": sha256(model_path / "config.json"),
            "weights_sha256": sha256(model_path / "model.safetensors"),
            "tokenizer_sha256": sha256(model_path / "tokenizer.json"),
        },
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": int(tokenizer.vocab_size),
        "hidden_size": int(backbone.config.hidden_size),
        "backbone_class": type(backbone).__name__,
        "attention_implementation": attention_impl,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "mismatched_keys": mismatched,
        "allowed_missing_only": True,
        "allowed_unexpected_only": True,
        "trainable_parameters": sum(parameter.numel() for parameter in classifier.parameters() if parameter.requires_grad),
        "total_parameters": sum(parameter.numel() for parameter in classifier.parameters()),
    }
    return tokenizer, classifier, info


class IntentClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_labels: int, dropout: float):
        super().__init__()
        self.backbone = backbone
        hidden_size = int(backbone.config.hidden_size)
        self.head_dropout1 = nn.Dropout(dropout)
        self.head_dense = nn.Linear(hidden_size, hidden_size)
        self.head_activation = nn.Tanh()
        self.head_dropout2 = nn.Dropout(dropout)
        self.head_out = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        hidden = self.head_dropout1(cls)
        hidden = self.head_activation(self.head_dense(hidden))
        hidden = self.head_dropout2(hidden)
        return self.head_out(hidden)


def optimizer_for(model: nn.Module, learning_rate: float, config: dict[str, Any]) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lower = name.lower()
        if lower.endswith(".bias") or "layernorm.weight" in lower or "layer_norm.weight" in lower or lower.endswith("norm.weight"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(config["weight_decay"])},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=tuple(config["adam_betas"]),
        eps=float(config["adam_eps"]),
    )


def make_loaders(
    tokenizer: Any,
    config: dict[str, Any],
    batch_size: int | None = None,
    seed: int = 42,
) -> tuple[DataLoader[dict[str, Any]], DataLoader[dict[str, Any]], dict[str, Any]]:
    train_rows = build_dataset_rows(TRAIN_PATH)
    dev_rows = build_dataset_rows(DEV_PATH)
    dataset_train = IntentDataset(train_rows, tokenizer, config["labels"], int(config["max_length"]))
    dataset_dev = IntentDataset(dev_rows, tokenizer, config["labels"], int(config["max_length"]))
    if len(dataset_train) != 353 or len(dataset_dev) != 77:
        raise AssertionError(f"frozen split count mismatch: {len(dataset_train)}/{len(dataset_dev)}")
    if batch_size is not None and int(batch_size) != int(config["batch_size"]):
        raise ValueError("the locked protocol uses effective batch size 16; no non-accumulated batch override is permitted")
    effective_batch = int(config["batch_size"])
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        dataset_train,
        batch_size=effective_batch,
        shuffle=True,
        generator=generator,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda examples: collate_examples(examples, tokenizer.pad_token_id),
    )
    dev_loader = DataLoader(
        dataset_dev,
        batch_size=effective_batch,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda examples: collate_examples(examples, tokenizer.pad_token_id),
    )
    truncation = {
        "train_examples": len(dataset_train),
        "dev_examples": len(dataset_dev),
        "train_examples_with_history_drop": sum(example.history_tokens_dropped > 0 for example in dataset_train.rows),
        "dev_examples_with_history_drop": sum(example.history_tokens_dropped > 0 for example in dataset_dev.rows),
        "train_current_truncated": sum(example.current_tokens_dropped > 0 for example in dataset_train.rows),
        "dev_current_truncated": sum(example.current_tokens_dropped > 0 for example in dataset_dev.rows),
        "max_length": int(config["max_length"]),
    }
    return train_loader, dev_loader, truncation


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "input_ids": batch["input_ids"].to(device, non_blocking=True),
        "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
        "labels": batch["labels"].to(device, non_blocking=True),
        "sample_ids": batch["sample_ids"],
    }


def metric_report(gold: list[int], predictions: list[int], labels: list[str]) -> dict[str, Any]:
    if len(gold) != len(predictions):
        raise AssertionError("metric length mismatch")
    per_intent = {}
    confusion = [[0 for _ in labels] for _ in labels]
    for target, prediction in zip(gold, predictions):
        confusion[target][prediction] += 1
    f1s = []
    for index, label in enumerate(labels):
        tp = confusion[index][index]
        fp = sum(confusion[row][index] for row in range(len(labels))) - tp
        fn = sum(confusion[index]) - tp
        support = sum(confusion[index])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        per_intent[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return {
        "accuracy": sum(target == prediction for target, prediction in zip(gold, predictions)) / max(1, len(gold)),
        "macro_f1": sum(f1s) / len(labels),
        "per_intent": per_intent,
        "confusion_matrix": confusion,
        "support": len(gold),
    }


@torch.no_grad()
def evaluate(model: IntentClassifier, loader: DataLoader[dict[str, Any]], device: torch.device, labels: list[str], precision: str = "bf16") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    gold, predictions, prediction_rows = [], [], []
    total_loss = 0.0
    total_items = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        with autocast_context(device, precision):
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = nn.functional.cross_entropy(logits.float(), batch["labels"])
        probs = torch.softmax(logits.float(), dim=-1).cpu()
        predicted = logits.argmax(dim=-1)
        total_loss += float(loss.item()) * len(raw_batch["sample_ids"])
        total_items += len(raw_batch["sample_ids"])
        gold.extend(int(value) for value in batch["labels"].cpu())
        predictions.extend(int(value) for value in predicted.cpu())
        for sample_id, target, prediction, probability in zip(raw_batch["sample_ids"], batch["labels"].cpu(), predicted.cpu(), probs):
            prediction_rows.append({
                "sample_id": sample_id,
                "gold_id": int(target),
                "gold_label": labels[int(target)],
                "pred_id": int(prediction),
                "pred_label": labels[int(prediction)],
                "probabilities": [float(value) for value in probability],
            })
    metrics = metric_report(gold, predictions, labels)
    metrics["loss"] = total_loss / max(1, total_items)
    return metrics, prediction_rows


def pinned_provenance(config: dict[str, Any]) -> dict[str, Any]:
    paths = {
        "corrected_dataset": CORRECTED_PATH,
        "train": TRAIN_PATH,
        "dev": DEV_PATH,
        "test": TEST_PATH,
        "label_map": LABEL_MAP_PATH,
        "training_config": CONFIG_PATH,
        "benchmark_provenance": PROVENANCE_PATH,
    }
    result = {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for name, path in paths.items()}
    result["source_bundle"] = load_json(PROVENANCE_PATH)
    result["torch_version"] = torch.__version__
    result["transformers_version"] = __import__("transformers").__version__
    result["python_version"] = sys.version
    result["training_implementation"] = {
        "path": str(IMPLEMENTATION_PATH.relative_to(ROOT)),
        "sha256": sha256(IMPLEMENTATION_PATH),
    }
    return result


def run_smoke_model(model_name: str, gpu: int, output_dir: Path) -> dict[str, Any]:
    config = load_locked_config()
    seed_everything(int(config["seed"]))
    device = resolve_device(gpu)
    tokenizer, model, loading = load_tokenizer_and_model(model_name, device, config)
    train_loader, _, truncation = make_loaders(tokenizer, config, seed=int(config["seed"]))
    batch = next(iter(train_loader))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    moved = move_batch(batch, device)
    optimizer = optimizer_for(model, float(config["learning_rates"][0]), config)
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, "bf16"):
        logits = model(moved["input_ids"], moved["attention_mask"])
        loss = nn.functional.cross_entropy(logits.float(), moved["labels"])
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"])).item())
    finite_gradients = all(parameter.grad is None or torch.isfinite(parameter.grad).all().item() for parameter in model.parameters())
    backbone_gradients = [parameter.grad for name, parameter in model.named_parameters() if name.startswith("backbone.") and parameter.grad is not None]
    head_gradients = [parameter.grad for name, parameter in model.named_parameters() if name.startswith("head_") and parameter.grad is not None]
    backbone_nonzero_gradients = sum(bool(torch.count_nonzero(gradient).item()) for gradient in backbone_gradients)
    head_nonzero_gradients = sum(bool(torch.count_nonzero(gradient).item()) for gradient in head_gradients)
    finite_loss = bool(torch.isfinite(loss).item())
    optimizer.step()
    peak_allocated = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    result = {
        "model_name": model_name,
        "device": str(device),
        "batch_size": len(batch["sample_ids"]),
        "batch_max_sequence_length": int(batch["input_ids"].shape[1]),
        "loss": float(loss.item()),
        "finite_loss": finite_loss,
        "gradient_norm_before_clip": grad_norm,
        "finite_gradients": finite_gradients,
        "backbone_gradient_tensors": len(backbone_gradients),
        "backbone_nonzero_gradient_tensors": backbone_nonzero_gradients,
        "head_gradient_tensors": len(head_gradients),
        "head_nonzero_gradient_tensors": head_nonzero_gradients,
        "trainable_parameters": loading["trainable_parameters"],
        "total_parameters": loading["total_parameters"],
        "peak_memory_allocated_bytes": peak_allocated,
        "peak_memory_reserved_bytes": peak_reserved,
        "loading": loading,
        "truncation": truncation,
        "provenance": pinned_provenance(config),
        "smoke_pass": (
            finite_loss
            and finite_gradients
            and backbone_nonzero_gradients > 0
            and head_nonzero_gradients > 0
            and loading["allowed_missing_only"]
            and loading["allowed_unexpected_only"]
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{model_name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    del optimizer, model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_trial(model_name: str, learning_rate: float, seed: int, gpu: int, run_dir: Path, batch_size: int | None = None) -> dict[str, Any]:
    config = load_locked_config()
    seed_everything(seed)
    device = resolve_device(gpu)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "RUNNING").write_text(json.dumps({"pid": os.getpid(), "started_at": time.time()}) + "\n", encoding="utf-8")
    tokenizer, model, loading = load_tokenizer_and_model(model_name, device, config)
    train_loader, dev_loader, truncation = make_loaders(tokenizer, config, batch_size=batch_size, seed=seed)
    optimizer = optimizer_for(model, learning_rate, config)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * int(config["max_epochs"])
    warmup_steps = int(round(total_steps * float(config["warmup_ratio"])))
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    precision = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    best_metric = -math.inf
    best_epoch = None
    stale_epochs = 0
    history = []
    best_predictions: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, precision):
                logits = model(batch["input_ids"], batch["attention_mask"])
                loss = nn.functional.cross_entropy(logits.float(), batch["labels"])
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
            optimizer.step()
            scheduler.step()
            size = len(raw_batch["sample_ids"])
            total_loss += float(loss.item()) * size
            total_items += size
        dev_metrics, dev_predictions = evaluate(model, dev_loader, device, config["labels"], precision)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, total_items),
            "dev": dev_metrics,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(json.dumps({"event": "epoch", "model": model_name, "lr": learning_rate, **row}, ensure_ascii=False), flush=True)
        metric = float(dev_metrics["macro_f1"])
        if metric > best_metric + float(config["min_delta"]):
            best_metric = metric
            best_epoch = epoch
            stale_epochs = 0
            best_predictions = dev_predictions
            torch.save({"model": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "model_name": model_name, "learning_rate": learning_rate, "seed": seed, "epoch": epoch}, run_dir / "best.pt")
        else:
            stale_epochs += 1
        if stale_epochs >= int(config["early_stopping_patience"]):
            break
    peak_allocated = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    result = {
        "status": "completed",
        "model_name": model_name,
        "learning_rate": learning_rate,
        "seed": seed,
        "gpu": gpu,
        "device": str(device),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_dev_macro_f1": best_metric,
        "best_dev_metrics": next((row["dev"] for row in history if row["epoch"] == best_epoch), None),
        "history": history,
        "peak_memory_allocated_bytes": peak_allocated,
        "peak_memory_reserved_bytes": peak_reserved,
        "trainable_parameters": loading["trainable_parameters"],
        "total_parameters": loading["total_parameters"],
        "loading": loading,
        "truncation": truncation,
        "precision": precision,
        "config": config,
        "provenance": pinned_provenance(config),
    }
    (run_dir / "config.json").write_text(json.dumps({"model_name": model_name, "learning_rate": learning_rate, "seed": seed, "gpu": gpu, "locked_config": config}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "loading_info.json").write_text(json.dumps(loading, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "dev_predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in best_predictions), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "COMPLETED").write_text(json.dumps({"finished_at": time.time(), "best_epoch": best_epoch, "best_dev_macro_f1": best_metric}) + "\n", encoding="utf-8")
    (run_dir / "RUNNING").unlink(missing_ok=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--model", required=True, choices=["goi_dapt", "goi_student", "bamibert"])
    smoke.add_argument("--gpu", type=int, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    trial = sub.add_parser("trial")
    trial.add_argument("--model", required=True, choices=["goi_dapt", "goi_student", "bamibert"])
    trial.add_argument("--lr", type=float, required=True)
    trial.add_argument("--seed", type=int, default=42)
    trial.add_argument("--gpu", type=int, required=True)
    trial.add_argument("--run-dir", type=Path, required=True)
    trial.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    if args.command == "smoke":
        result = run_smoke_model(args.model, args.gpu, args.output)
    else:
        result = run_trial(args.model, args.lr, args.seed, args.gpu, args.run_dir, args.batch_size)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
