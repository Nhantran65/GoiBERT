"""Train one architecture/dataset/mode combination with a plain PyTorch loop."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
import transformers
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data import DATASET_NAMES, DatasetSplits, load_dataset
from modeling import build_model


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
MAX_LENGTH = 256
TEMPERATURE = 2.0


class EncodedDataset(Dataset[dict[str, torch.Tensor]]):
    """In-memory token tensors plus stable row indices for teacher-logit lookup."""

    def __init__(self, texts: list[str], labels: list[Any], tokenizer: Any) -> None:
        encoded = tokenizer(texts, max_length=MAX_LENGTH, truncation=True)
        self.input_ids = [torch.tensor(ids, dtype=torch.long) for ids in encoded["input_ids"]]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "labels": self.labels[index],
            "index": torch.tensor(index, dtype=torch.long),
        }


def collate_pad(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    # Pad to the longest sequence in the batch (pad_token_id=1) instead of a fixed
    # MAX_LENGTH; identical model outputs via attention_mask, ~an order of magnitude
    # less compute on short-text datasets.
    longest = max(item["input_ids"].shape[0] for item in items)
    input_ids = torch.full((len(items), longest), 1, dtype=torch.long)
    attention_mask = torch.zeros((len(items), longest), dtype=torch.long)
    for row, item in enumerate(items):
        ids = item["input_ids"]
        input_ids[row, : ids.shape[0]] = ids
        attention_mask[row, : ids.shape[0]] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": torch.stack([item["labels"] for item in items]),
        "index": torch.stack([item["index"] for item in items]),
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def vsfc_metrics(gold: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(f1_score(gold, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(gold, predicted, average="weighted", zero_division=0)),
    }


def _absa_counts(gold: np.ndarray, predicted: np.ndarray) -> tuple[int, int, int, int]:
    gold_present = gold > 0
    predicted_present = predicted > 0
    detection_tp = int(np.logical_and(gold_present, predicted_present).sum())
    classification_tp = int(np.logical_and(gold_present, predicted == gold).sum())
    return (
        detection_tp,
        classification_tp,
        int(predicted_present.sum()),
        int(gold_present.sum()),
    )


def _prf(tp: int, predicted_count: int, gold_count: int) -> tuple[float, float, float]:
    precision = _safe_ratio(tp, predicted_count)
    recall = _safe_ratio(tp, gold_count)
    f1 = _safe_ratio(2 * tp, predicted_count + gold_count)
    return precision, recall, f1


def absa_metrics(gold: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    detection_tp, classification_tp, predicted_count, gold_count = _absa_counts(gold, predicted)
    detection = _prf(detection_tp, predicted_count, gold_count)
    classification = _prf(classification_tp, predicted_count, gold_count)
    detection_by_aspect: list[tuple[float, float, float]] = []
    classification_by_aspect: list[tuple[float, float, float]] = []
    for aspect_index in range(gold.shape[1]):
        det_tp, cls_tp, pred_n, gold_n = _absa_counts(
            gold[:, aspect_index], predicted[:, aspect_index]
        )
        detection_by_aspect.append(_prf(det_tp, pred_n, gold_n))
        classification_by_aspect.append(_prf(cls_tp, pred_n, gold_n))
    detection_macro = np.asarray(detection_by_aspect).mean(axis=0)
    classification_macro = np.asarray(classification_by_aspect).mean(axis=0)
    return {
        "detection_micro_precision": detection[0],
        "detection_micro_recall": detection[1],
        "detection_micro_f1": detection[2],
        "detection_macro_precision": float(detection_macro[0]),
        "detection_macro_recall": float(detection_macro[1]),
        "detection_macro_f1": float(detection_macro[2]),
        "classification_micro_precision": classification[0],
        "classification_micro_recall": classification[1],
        "classification_micro_f1": classification[2],
        "classification_macro_precision": float(classification_macro[0]),
        "classification_macro_recall": float(classification_macro[1]),
        "classification_macro_f1": float(classification_macro[2]),
    }


def supervised_loss(logits: torch.Tensor, labels: torch.Tensor, is_absa: bool) -> torch.Tensor:
    if not is_absa:
        return functional.cross_entropy(logits, labels)
    per_cell = functional.cross_entropy(
        logits.reshape(-1, 4), labels.reshape(-1), reduction="none"
    ).reshape(labels.shape)
    return per_cell.mean(dim=0).sum()


def distillation_loss(
    student: torch.Tensor, teacher: torch.Tensor, is_absa: bool
) -> torch.Tensor:
    student_log_probs = functional.log_softmax(student / TEMPERATURE, dim=-1)
    teacher_log_probs = functional.log_softmax(teacher / TEMPERATURE, dim=-1)
    per_class = student_log_probs.exp() * (student_log_probs - teacher_log_probs)
    if is_absa:
        return per_class.sum(dim=-1).mean()
    return per_class.sum(dim=-1).mean()


def evaluate(
    model: nn.Module, loader: DataLoader[dict[str, torch.Tensor]], device: torch.device, is_absa: bool
) -> dict[str, float]:
    model.eval()
    gold_batches: list[torch.Tensor] = []
    predicted_batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in loader:
            logits = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device)
            )
            predicted_batches.append(logits.argmax(dim=-1).cpu())
            gold_batches.append(batch["labels"])
    gold = torch.cat(gold_batches).numpy()
    predicted = torch.cat(predicted_batches).numpy()
    return absa_metrics(gold, predicted) if is_absa else vsfc_metrics(gold, predicted)


def selection_score(metrics: dict[str, float], is_absa: bool) -> float:
    if not is_absa:
        return metrics["macro_f1"]
    return (metrics["detection_micro_f1"] + metrics["classification_micro_f1"]) / 2


def _load_teacher_logits(path: Path, splits: DatasetSplits) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(
            f"Teacher logits are required for distillation but missing: {path}. "
            "Run the matching baseline finetune first."
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    logits = payload["logits"]
    if payload.get("aspect_names", []) != splits.aspect_names:
        raise ValueError("Teacher-logit aspect order does not match the current dataset")
    if logits.shape[0] != len(splits.train_texts):
        raise ValueError(
            f"Teacher logits have {logits.shape[0]} rows; expected {len(splits.train_texts)}"
        )
    return logits


def _cache_teacher_logits(
    model: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    path: Path,
    aspect_names: list[str],
) -> None:
    model.eval()
    batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in loader:
            batches.append(
                model(batch["input_ids"].to(device), batch["attention_mask"].to(device)).cpu()
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"logits": torch.cat(batches), "aspect_names": aspect_names}, path)


def _device_details(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return platform.processor() or "CPU"


def run(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    repo_root = args.repo_root.resolve()
    results_dir = EXPERIMENT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    splits = load_dataset(args.dataset, repo_root, smoke=args.smoke)
    tokenizer = AutoTokenizer.from_pretrained(
        repo_root / "models" / "bamibert", local_files_only=True
    )
    train_data = EncodedDataset(splits.train_texts, splits.train_labels, tokenizer)
    valid_data = EncodedDataset(splits.valid_texts, splits.valid_labels, tokenizer)
    test_data = EncodedDataset(splits.test_texts, splits.test_labels, tokenizer)
    batch_size = 8 if args.smoke else 32
    epochs = 1 if args.smoke else 10
    generator = torch.Generator().manual_seed(args.seed)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=pin_memory,
        collate_fn=collate_pad,
    )
    train_ordered_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=False, collate_fn=collate_pad
    )
    valid_loader = DataLoader(
        valid_data, batch_size=batch_size, shuffle=False, collate_fn=collate_pad
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_pad
    )
    model, arch, encoder_count, head_count = build_model(
        args.model, args.mode, len(splits.aspect_names), repo_root, EXPERIMENT_ROOT
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    learning_rate = 2e-5 if args.mode == "finetune" else 5e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(total_steps * 0.06),
        num_training_steps=total_steps,
    )
    # fp16 autocast produces NaN losses on ROCm/MI250X (observed on LUMI, torch
    # 2.5.1+rocm6.2, 2026-08-30) - default AMP off there; GOITURN_AMP=1/0 overrides.
    amp_env = os.environ.get("GOITURN_AMP")
    is_rocm = getattr(torch.version, "hip", None) is not None
    use_amp = device.type == "cuda" and (
        amp_env == "1" if amp_env is not None else not is_rocm
    )
    if device.type == "cuda":
        print(f"== AMP fp16: {use_amp} (rocm={is_rocm})", flush=True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    teacher_path = results_dir / "teacher_logits" / f"{args.dataset}.pt"
    teacher_logits = (
        _load_teacher_logits(teacher_path, splits) if args.mode == "distill" else None
    )
    best_score = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_total = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                logits = model(input_ids, attention_mask)
                task_loss = supervised_loss(logits, labels, splits.is_absa)
                if teacher_logits is None:
                    loss = task_loss
                else:
                    teacher_batch = teacher_logits[batch["index"]].to(device)
                    kd_loss = distillation_loss(logits, teacher_batch, splits.is_absa)
                    loss = 0.5 * task_loss + 0.5 * TEMPERATURE**2 * kd_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_total += float(loss.detach().cpu())
        metrics = evaluate(model, valid_loader, device, splits.is_absa)
        epoch_record = {
            "epoch": epoch,
            "train_loss": loss_total / len(train_loader),
            "metrics": metrics,
        }
        history.append(epoch_record)
        score = selection_score(metrics, splits.is_absa)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        metrics_text = " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        print(
            f"epoch={epoch}/{epochs} loss={epoch_record['train_loss']:.4f} {metrics_text}",
            flush=True,
        )
    train_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    model.to(device)
    test_metrics = evaluate(model, test_loader, device, splits.is_absa)
    if args.mode == "finetune":
        checkpoint_path = results_dir / "checkpoints" / f"{args.dataset}__baseline.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, checkpoint_path)
        _cache_teacher_logits(model, train_ordered_loader, device, teacher_path, splits.aspect_names)
    full_arch = {
        **arch,
        "vocab_size": 20_481,
        "type_vocab_size": 1,
        "pad_token_id": 1,
        "bos_token_id": 0,
        "eos_token_id": 2,
        "position_embedding_type": "absolute",
    }
    result = {
        "dataset": args.dataset,
        "model": args.model,
        "mode": args.mode,
        "smoke": args.smoke,
        "seed": args.seed,
        "config": full_arch,
        "aspect_names": splits.aspect_names,
        "lr": learning_rate,
        "epochs_run": epochs,
        "best_epoch": best_epoch,
        "per_epoch_validation": history,
        "test_metrics": test_metrics,
        "param_counts": {"encoder_and_embeddings": encoder_count, "head": head_count},
        "wall_clock_train_seconds": train_seconds,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device_name": _device_details(device),
    }
    output_path = results_dir / f"{args.dataset}__{args.model}__{args.mode}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(f"Wrote {output_path}", flush=True)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_NAMES, required=True)
    parser.add_argument("--model", choices=("baseline", "variant_a", "variant_b"), required=True)
    parser.add_argument("--mode", choices=("finetune", "scratch", "distill"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
