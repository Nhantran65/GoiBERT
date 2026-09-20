"""Shared data, metrics, and BamiBERT baseline training for the family harness."""

from __future__ import annotations

import json
import math
import os
import platform
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
import transformers
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data import DatasetSplits, load_dataset
from modeling import build_model, parameter_counts
from noise import corrupt


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
MAX_LENGTH = 256
FULL_BATCH_SIZE = 32
VSFC_CLASS_NAMES = ("negative", "neutral", "positive")
NC_WEIGHT = 1.0


class EncodedDataset(Dataset[dict[str, Any]]):
    """In-memory token tensors with stable row indices."""

    def __init__(self, texts: list[str], labels: list[Any], tokenizer: Any) -> None:
        encoded = tokenizer(texts, max_length=MAX_LENGTH, truncation=True)
        self.input_ids = [torch.tensor(ids, dtype=torch.long) for ids in encoded["input_ids"]]
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.texts = texts

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "input_ids": self.input_ids[index],
            "labels": self.labels[index],
            "index": torch.tensor(index, dtype=torch.long),
            "text": self.texts[index],
        }


def collate_pad(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Dynamically pad one batch with the BamiBERT pad token."""
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
        "texts": [item["text"] for item in items],
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def classification_metrics(
    gold: list[int], predicted: list[int], class_names: list[str] | tuple[str, ...]
) -> dict[str, Any]:
    label_ids = list(range(len(class_names)))
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        gold, predicted, labels=label_ids, zero_division=0
    )
    macro_precision, macro_recall, _, _ = precision_recall_fscore_support(
        gold, predicted, average="macro", labels=label_ids, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(
            f1_score(gold, predicted, average="macro", labels=label_ids, zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(gold, predicted, average="weighted", labels=label_ids, zero_division=0)
        ),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(per_class_f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(class_names)
        },
    }


def vsfc_metrics(gold: list[int], predicted: list[int]) -> dict[str, Any]:
    return classification_metrics(gold, predicted, VSFC_CLASS_NAMES)


def _absa_counts(gold: list[int], predicted: list[int]) -> tuple[int, int, int, int]:
    detection_tp = sum(gold_value > 0 and predicted_value > 0 for gold_value, predicted_value in zip(gold, predicted))
    classification_tp = sum(
        gold_value > 0 and predicted_value == gold_value
        for gold_value, predicted_value in zip(gold, predicted)
    )
    return (
        detection_tp,
        classification_tp,
        sum(value > 0 for value in predicted),
        sum(value > 0 for value in gold),
    )


def _prf(tp: int, predicted_count: int, gold_count: int) -> tuple[float, float, float]:
    return (
        _safe_ratio(tp, predicted_count),
        _safe_ratio(tp, gold_count),
        _safe_ratio(2 * tp, predicted_count + gold_count),
    )


def absa_metrics(gold: list[list[int]], predicted: list[list[int]]) -> dict[str, float]:
    flat_gold = [value for row in gold for value in row]
    flat_predicted = [value for row in predicted for value in row]
    det_tp, cls_tp, pred_count, gold_count = _absa_counts(flat_gold, flat_predicted)
    detection = _prf(det_tp, pred_count, gold_count)
    classification = _prf(cls_tp, pred_count, gold_count)
    n_aspects = len(gold[0])
    detection_by_aspect: list[tuple[float, float, float]] = []
    classification_by_aspect: list[tuple[float, float, float]] = []
    for aspect in range(n_aspects):
        aspect_gold = [row[aspect] for row in gold]
        aspect_predicted = [row[aspect] for row in predicted]
        det_tp, cls_tp, pred_count, gold_count = _absa_counts(aspect_gold, aspect_predicted)
        detection_by_aspect.append(_prf(det_tp, pred_count, gold_count))
        classification_by_aspect.append(_prf(cls_tp, pred_count, gold_count))
    detection_macro = [
        sum(metric[index] for metric in detection_by_aspect) / n_aspects for index in range(3)
    ]
    classification_macro = [
        sum(metric[index] for metric in classification_by_aspect) / n_aspects
        for index in range(3)
    ]
    return {
        "detection_micro_precision": detection[0],
        "detection_micro_recall": detection[1],
        "detection_micro_f1": detection[2],
        "detection_macro_precision": detection_macro[0],
        "detection_macro_recall": detection_macro[1],
        "detection_macro_f1": detection_macro[2],
        "classification_micro_precision": classification[0],
        "classification_micro_recall": classification[1],
        "classification_micro_f1": classification[2],
        "classification_macro_precision": classification_macro[0],
        "classification_macro_recall": classification_macro[1],
        "classification_macro_f1": classification_macro[2],
    }


def supervised_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2:
        return functional.cross_entropy(logits, labels)
    return functional.cross_entropy(logits.reshape(-1, 4), labels.reshape(-1))


def symmetric_kl(clean_logits: torch.Tensor, corrupted_logits: torch.Tensor) -> torch.Tensor:
    """Average KL in both directions over the classifier's final dimension."""
    clean_log_probabilities = functional.log_softmax(clean_logits.float(), dim=-1)
    corrupted_log_probabilities = functional.log_softmax(
        corrupted_logits.float(), dim=-1
    )
    clean_probabilities = clean_log_probabilities.exp()
    corrupted_probabilities = corrupted_log_probabilities.exp()
    clean_to_corrupted = functional.kl_div(
        corrupted_log_probabilities, clean_probabilities, reduction="none"
    ).sum(dim=-1)
    corrupted_to_clean = functional.kl_div(
        clean_log_probabilities, corrupted_probabilities, reduction="none"
    ).sum(dim=-1)
    return 0.5 * (clean_to_corrupted + corrupted_to_clean).mean()


def tokenize_corrupted_view(
    texts: list[str],
    tokenizer: Any,
    seed: int,
    epoch: int,
    batch_index: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Corrupt raw texts with a batch-specific deterministic RNG, then tokenize."""
    rng = random.Random(f"{seed}:{epoch}:{batch_index}")
    corrupted_texts = [corrupt(text, rng) for text in texts]
    encoded = tokenizer(
        corrupted_texts,
        max_length=MAX_LENGTH,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


def configure_output_head(
    model: nn.Module,
    counts: dict[str, int],
    splits: DatasetSplits,
) -> tuple[nn.Module, dict[str, int]]:
    """Expand the default three-class head for non-ABSA datasets such as PhoATIS."""
    if splits.is_absa or splits.output_size == 3:
        return model, counts
    model.head.output = nn.Linear(model.head.output.in_features, splits.output_size)
    updated_counts = parameter_counts(model)
    print(
        f"Configured {splits.output_size}-class head: "
        f"head={updated_counts['head']:,}; total={updated_counts['model_total']:,}",
        flush=True,
    )
    return model, updated_counts


def selection_score(metrics: dict[str, Any], is_absa: bool) -> float:
    if not is_absa:
        return float(metrics["macro_f1"])
    return float(
        (metrics["detection_micro_f1"] + metrics["classification_micro_f1"]) / 2
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    is_absa: bool,
    class_names: list[str] | tuple[str, ...] = VSFC_CLASS_NAMES,
) -> dict[str, Any]:
    model.eval()
    gold: list[Any] = []
    predicted: list[Any] = []
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            if not isinstance(logits, torch.Tensor):
                raise TypeError("Evaluation expected classifier logits")
            predicted.extend(logits.argmax(dim=-1).cpu().tolist())
            gold.extend(batch["labels"].tolist())
    return (
        absa_metrics(gold, predicted)
        if is_absa
        else classification_metrics(gold, predicted, class_names)
    )


def device_details(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return platform.processor() or "CPU"


def amp_enabled(device: torch.device) -> bool:
    """Apply the proven LUMI ROCm guard while preserving the environment override."""
    amp_environment = os.environ.get("GOITURN_AMP")
    is_rocm = getattr(torch.version, "hip", None) is not None
    enabled = device.type == "cuda" and (
        amp_environment == "1" if amp_environment is not None else not is_rocm
    )
    if device.type == "cuda":
        print(f"AMP fp16: {enabled} (rocm={is_rocm})", flush=True)
    return enabled


def make_loaders(
    splits: DatasetSplits,
    tokenizer: Any,
    seed: int,
    smoke: bool,
) -> tuple[
    DataLoader[dict[str, Any]],
    DataLoader[dict[str, Any]],
    DataLoader[dict[str, Any]],
]:
    train_data = EncodedDataset(splits.train_texts, splits.train_labels, tokenizer)
    valid_data = EncodedDataset(splits.valid_texts, splits.valid_labels, tokenizer)
    test_data = EncodedDataset(splits.test_texts, splits.test_labels, tokenizer)
    batch_size = 8 if smoke else FULL_BATCH_SIZE
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_pad,
    )
    valid_loader = DataLoader(
        valid_data, batch_size=batch_size, shuffle=False, collate_fn=collate_pad
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_pad
    )
    return train_loader, valid_loader, test_loader


def lr_slug(learning_rate: float) -> str:
    return f"{learning_rate:.0e}"


def seed_suffix(seed: int) -> str:
    return "" if seed == 42 else f"__seed{seed}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _runtime_metadata(device: torch.device) -> dict[str, str]:
    return {
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device_name": device_details(device),
    }


def train_baseline_trial(
    dataset: str,
    learning_rate: float,
    seed: int,
    smoke: bool,
    repo_root: Path,
    init_dir: Path | None = None,
    noise_consistency: bool = False,
) -> dict[str, Any]:
    """Train one baseline-family LR trial and stop before test evaluation."""
    base_run_name = "goibert_b_dapt" if init_dir is not None else "baseline"
    run_name = f"{base_run_name}_nc" if noise_consistency else base_run_name
    seed_everything(seed)
    splits = load_dataset(dataset, repo_root, smoke=smoke)
    tokenizer = AutoTokenizer.from_pretrained(
        repo_root / "models" / "bamibert", local_files_only=True
    )
    train_loader, valid_loader, _ = make_loaders(splits, tokenizer, seed, smoke)
    model, architecture, counts = build_model(
        "goibert_b",
        len(splits.aspect_names),
        repo_root,
        EXPERIMENT_ROOT,
        pretrained=True,
        init_dir=init_dir,
    )
    model, counts = configure_output_head(model, counts, splits)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    epochs = 1 if smoke else 10
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(total_steps * 0.06),
        num_training_steps=total_steps,
    )
    use_amp = amp_enabled(device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_score = -1.0
    best_epoch = 0
    best_metrics: dict[str, Any] = {}
    best_state: dict[str, torch.Tensor] = {}
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_total = 0.0
        consistency_total = 0.0
        for batch_index, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids, attention_mask)
                if not isinstance(logits, torch.Tensor):
                    raise TypeError("Baseline training expected classifier logits")
                loss = supervised_loss(logits, labels)
                consistency_loss = None
                if noise_consistency:
                    corrupted_input_ids, corrupted_attention_mask = tokenize_corrupted_view(
                        batch["texts"], tokenizer, seed, epoch, batch_index, device
                    )
                    corrupted_logits = model(
                        corrupted_input_ids, corrupted_attention_mask
                    )
                    if not isinstance(corrupted_logits, torch.Tensor):
                        raise TypeError("Noise-consistency training expected classifier logits")
                    consistency_loss = symmetric_kl(logits, corrupted_logits)
                    loss = loss + NC_WEIGHT * consistency_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_total += float(loss.detach().cpu())
            if consistency_loss is not None:
                consistency_total += float(consistency_loss.detach().cpu())
        metrics = evaluate(
            model, valid_loader, device, splits.is_absa, splits.class_names
        )
        score = selection_score(metrics, splits.is_absa)
        if not math.isfinite(score):
            raise RuntimeError(f"Non-finite validation score at epoch {epoch}")
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = metrics
            best_state = _clone_state(model)
        epoch_result: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": loss_total / len(train_loader),
            "metrics": metrics,
        }
        if noise_consistency:
            epoch_result["consistency_loss"] = consistency_total / len(train_loader)
        history.append(epoch_result)
        print(
            f"{run_name} lr={learning_rate:g} epoch={epoch}/{epochs} "
            f"loss={history[-1]['train_loss']:.4f} val_score={score:.4f}",
            flush=True,
        )
    checkpoint_path = (
        EXPERIMENT_ROOT
        / "results"
        / "checkpoints"
        / ".candidates"
        / f"{dataset}__{run_name}__lr_{lr_slug(learning_rate)}{seed_suffix(seed)}.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "config": architecture,
            "aspect_names": splits.aspect_names,
            "best_epoch": best_epoch,
            "lr": learning_rate,
            "seed": seed,
        },
        checkpoint_path,
    )
    result_path = (
        EXPERIMENT_ROOT
        / "results"
        / f"{dataset}__{run_name}__lr_{lr_slug(learning_rate)}{seed_suffix(seed)}.json"
    )
    description = (
        "DAPT GoiBERT-B, fine-tuned"
        if init_dir is not None
        else "BamiBERT baseline, fine-tuned"
    )
    if noise_consistency:
        description += ", noise consistency"
    result: dict[str, Any] = {
        "dataset": dataset,
        "model": run_name,
        "description": description,
        "mode": "finetune",
        "smoke": smoke,
        "seed": seed,
        "config": architecture,
        "aspect_names": splits.aspect_names,
        "lr": learning_rate,
        "epochs_run": epochs,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "best_validation_metrics": best_metrics,
        "per_epoch_validation": history,
        "test_metrics": None,
        "test_evaluated": False,
        "param_counts": counts,
        "wall_clock_train_seconds": time.perf_counter() - started,
        "candidate_checkpoint": str(checkpoint_path),
        **_runtime_metadata(device),
    }
    if noise_consistency:
        result["noise_consistency"] = {"enabled": True, "weight": NC_WEIGHT}
    if init_dir is not None:
        result["initialization_checkpoint"] = str(init_dir)
    write_json(result_path, result)
    result["result_path"] = str(result_path)
    print(f"Wrote validation-only trial {result_path}", flush=True)
    return result


def finalize_baseline_trial(
    result: dict[str, Any], repo_root: Path
) -> dict[str, Any]:
    """Evaluate the validation-selected baseline once on test and save its checkpoint."""
    seed_everything(int(result["seed"]))
    splits = load_dataset(result["dataset"], repo_root, smoke=bool(result["smoke"]))
    tokenizer = AutoTokenizer.from_pretrained(
        repo_root / "models" / "bamibert", local_files_only=True
    )
    _, _, test_loader = make_loaders(splits, tokenizer, int(result["seed"]), bool(result["smoke"]))
    init_dir = (
        Path(result["initialization_checkpoint"])
        if "initialization_checkpoint" in result
        else None
    )
    model, _, counts = build_model(
        "goibert_b",
        len(splits.aspect_names),
        repo_root,
        EXPERIMENT_ROOT,
        pretrained=True,
        init_dir=init_dir,
    )
    model, counts = configure_output_head(model, counts, splits)
    candidate_path = Path(result["candidate_checkpoint"])
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=True)
    model.load_state_dict(candidate["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    result["test_metrics"] = evaluate(
        model, test_loader, device, splits.is_absa, splits.class_names
    )
    result["test_evaluated"] = True
    result["selected_by"] = "best validation score across the LR grid"
    result["param_counts"] = counts
    final_checkpoint = (
        EXPERIMENT_ROOT
        / "results"
        / "checkpoints"
        / f"{result['dataset']}__{result['model']}{seed_suffix(int(result['seed']))}.pt"
    )
    final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": candidate["model_state"],
            "config": result["config"],
            "aspect_names": splits.aspect_names,
            "best_epoch": result["best_epoch"],
            "lr": result["lr"],
            "seed": result["seed"],
        },
        final_checkpoint,
    )
    result["checkpoint"] = str(final_checkpoint)
    result.pop("candidate_checkpoint", None)
    result_path = Path(result.pop("result_path"))
    write_json(result_path, result)
    best_path = (
        EXPERIMENT_ROOT
        / "results"
        / f"{result['dataset']}__{result['model']}__best{seed_suffix(int(result['seed']))}.json"
    )
    write_json(best_path, result)
    candidate_path.unlink()
    print(f"Saved validation-selected {result['model']} checkpoint {final_checkpoint}", flush=True)
    return result
