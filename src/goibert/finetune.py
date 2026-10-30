"""Task fine-tuning for students, teachers and baselines.

One loop covers the three roles used in the paper:

* **student**: a GoiBERT backbone, layer-wise LR decay (``optimizer="llrd"``)
  and, optionally, knowledge distillation from out-of-fold CafeBERT soft
  labels (``soft_labels=...``);
* **teacher**: CafeBERT trained on four of five folds of the training split
  (``fold=k``); after training it writes logits for the held-out fold;
* **baseline**: BamiBERT with plain AdamW and cross-entropy.

The task loss is ``alpha * CE + (1 - alpha) * T^2 * KL(teacher_T || student_T)``
with ``alpha = 0.5`` and ``T = 2``.  For ABSA the KL term is averaged over
aspect slots; for NER it is applied to the first subword of every word.

Every run selects the epoch with the best validation ``selection_f1``,
saves that checkpoint, reloads it and checks that the reloaded model
reproduces the saved validation predictions.
"""

from __future__ import annotations

import dataclasses
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from . import training as T
from .data import extra, family, uit
from .modeling import CAFEBERT_TOKENIZER_REPO, CAFEBERT_TOKENIZER_REVISION, build_model, count_parameters
from .optim import build_adamw, build_layerwise_optimizer

KD_ALPHA = 0.5
KD_TEMPERATURE = 2.0
N_FOLDS = 5
FOLD_SEED = 42


@dataclass
class RunConfig:
    task: str
    kind: str  # "student", "bamibert" or "cafebert"
    out: Path
    learning_rate: float
    seed: int = 42
    backbone: str | None = None  # local directory or Hub ID (student, bamibert)
    optimizer: str = "llrd"  # "llrd" or "adamw"
    soft_labels: Path | None = None  # enables knowledge distillation
    fold: int | None = None  # teacher run on 4/5 of train; writes held-out logits
    epochs: int = 30
    batch_size: int = 32
    micro_batch_size: int = 32
    device: str = "cuda:0"
    precision: str = "bf16"
    limit: int | None = None  # smoke tests: use only the first N training examples
    local_files_only: bool = False
    keep_checkpoint: bool = True


# ------------------------------------------------------------------ folds


def fold_assignment(n: int) -> np.ndarray:
    """Fold index (0-4) for each training position, from a seeded permutation."""

    order = np.random.default_rng(FOLD_SEED).permutation(n)
    fold = np.empty(n, dtype=np.int64)
    fold[order] = np.arange(n) % N_FOLDS
    return fold


def _uit_fold_bundle(bundle: uit.TaskDataset, fold: int) -> uit.TaskDataset:
    train = bundle.splits["train"]
    assign = fold_assignment(len(train))
    splits = dict(bundle.splits)
    splits["train"] = uit.TaskSplit("train", tuple(ex for ex, f in zip(train.examples, assign) if f != fold))
    splits["oof"] = uit.TaskSplit("oof", tuple(ex for ex, f in zip(train.examples, assign) if f == fold))
    return dataclasses.replace(bundle, splits=splits)


def _extra_folds(split: extra.Split, fold: int) -> tuple[extra.Split, extra.Split]:
    assign = fold_assignment(len(split.ids))
    return (
        split.subset([i for i, f in enumerate(assign) if f != fold]),
        split.subset([i for i, f in enumerate(assign) if f == fold]),
    )


# ----------------------------------------------------------- soft labels


def _slot_segments(widths) -> list[tuple[int, int]]:
    out, offset = [], 0
    for width in widths:
        out.append((offset, offset + int(width)))
        offset += int(width)
    return out


def kd_term(student: torch.Tensor, teacher: torch.Tensor, segments, temperature: float = KD_TEMPERATURE) -> torch.Tensor:
    """Mean over slots of ``T^2 * KL(softmax(teacher/T) || softmax(student/T))``."""

    losses = []
    for a, b in segments:
        s = F.log_softmax(student[:, a:b] / temperature, -1)
        t = F.softmax(teacher[:, a:b] / temperature, -1)
        losses.append(F.kl_div(s, t, reduction="batchmean") * temperature**2)
    return torch.stack(losses).mean()


def load_soft_labels(path: Path, task: str, train_ids: list[str]) -> dict[str, Any]:
    """Map training ID -> teacher logits (NER: ``(word logits, has_logit mask)``)."""

    z = np.load(path, allow_pickle=False)
    ids = [str(i) for i in z["ids"]]
    if task == "phoner":
        if ids != train_ids:
            raise ValueError("Soft-label IDs do not match the training split")
        offsets, logits, mask = z["offsets"], z["logits"], z["has_logit"]
        return {sid: (logits[offsets[i] : offsets[i + 1]], mask[offsets[i] : offsets[i + 1]]) for i, sid in enumerate(ids)}
    table = {sid: z["logits"][i] for i, sid in enumerate(ids)}
    missing = set(train_ids) - set(table)
    if missing:
        raise ValueError(f"{len(missing)} training examples have no teacher logits")
    return table


def _ner_pairs(logits: torch.Tensor, batch: dict[str, Any], soft: dict[str, Any]):
    student, teacher = [], []
    for i, sid in enumerate(batch["ids"]):
        word_logits, mask = soft[sid]
        if len(word_logits) != batch["word_count"][i]:
            raise ValueError("Soft-label word count mismatch")
        for pos, wid in batch["positions"][i]:
            if mask[wid]:
                student.append(logits[i, pos].float())
                teacher.append(torch.as_tensor(word_logits[wid], device=logits.device))
    if not student:
        return None, None
    return torch.stack(student), torch.stack(teacher)


# --------------------------------------------------------------- the run


def _tokenizer(cfg: RunConfig):
    if cfg.kind == "cafebert":
        return AutoTokenizer.from_pretrained(
            CAFEBERT_TOKENIZER_REPO, revision=CAFEBERT_TOKENIZER_REVISION, use_fast=True, local_files_only=cfg.local_files_only
        )
    return AutoTokenizer.from_pretrained(cfg.backbone, use_fast=True, local_files_only=cfg.local_files_only)


def run(cfg: RunConfig) -> dict[str, Any]:
    if cfg.kind not in ("student", "bamibert", "cafebert"):
        raise ValueError(f"Unknown model kind {cfg.kind!r}")
    if cfg.kind != "cafebert" and not cfg.backbone:
        raise ValueError("--backbone is required for student and bamibert runs")
    if cfg.batch_size % cfg.micro_batch_size:
        raise ValueError("micro batch size must divide the batch size")
    if cfg.soft_labels and cfg.micro_batch_size != cfg.batch_size:
        raise ValueError("Knowledge distillation runs without gradient accumulation")
    out = Path(cfg.out)
    if (out / "metadata.json").exists():
        raise FileExistsError(f"{out} already holds a finished run")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    task_family = family(cfg.task)
    T.seed_everything(cfg.seed)
    device = T.resolve_device(cfg.device)
    precision = T.resolve_precision(cfg.precision, device)
    tokenizer = _tokenizer(cfg)
    accumulation = cfg.batch_size // cfg.micro_batch_size
    micro = cfg.micro_batch_size

    held = None
    if task_family == "uit":
        bundle = uit.load_task(cfg.task)
        if cfg.fold is not None:
            bundle = _uit_fold_bundle(bundle, cfg.fold)
            held = bundle.splits["oof"]
        train_split, valid_split = bundle.splits["train"], bundle.splits["validation"]
        if cfg.limit:
            train_split = uit.TaskSplit("train", train_split.examples[: cfg.limit])
            bundle = dataclasses.replace(bundle, splits={**bundle.splits, "train": train_split})
        lengths = T.inspect_train_lengths(tokenizer, list(train_split.texts))
        # Students use the length chosen from the training split; teachers and
        # baselines use 256 tokens.
        max_length = lengths.max_length if cfg.kind == "student" else 256
        num_labels, num_aspects, counts = T.uit_task_shape(bundle)
        model = build_model(
            cfg.backbone or "", kind=cfg.kind, task_type="sequence", num_labels=num_labels,
            num_aspects=num_aspects, aspect_class_counts=counts if num_aspects else None,
            local_files_only=cfg.local_files_only,
        )
        label_names = list(bundle.label_names)
        segments = _slot_segments(counts if counts else (num_labels,))
    else:
        data = extra.load_task(cfg.task)
        label_names = list(data.label_names)
        train_split, valid_split = data.train, data.validation
        if cfg.fold is not None:
            train_split, held = _extra_folds(data.train, cfg.fold)
        max_length = 512 if cfg.task == "phoner" else 256
        num_labels, num_aspects, counts = len(label_names), 0, ()
        model = build_model(
            cfg.backbone or "", kind=cfg.kind, task_type="token" if cfg.task == "phoner" else "sequence",
            num_labels=num_labels, local_files_only=cfg.local_files_only,
        )
        segments = _slot_segments((num_labels,))
    model.to(device)

    if cfg.optimizer == "llrd":
        optimizer = build_layerwise_optimizer(model, cfg.learning_rate, num_layers=model.encoder.config.num_hidden_layers)
    elif cfg.optimizer == "adamw":
        optimizer = build_adamw(model, cfg.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer {cfg.optimizer!r}")

    if task_family == "uit":
        train_loader = T.make_uit_loader(tokenizer, train_split, labels=bundle.labels_for_model("train"), max_length=max_length, batch_size=micro, shuffle=True, seed=cfg.seed)
        valid_loader = T.make_uit_loader(tokenizer, valid_split, labels=bundle.labels_for_model("validation"), max_length=max_length, batch_size=micro, shuffle=False, seed=cfg.seed)
        length_stats = lengths.to_dict()
    else:
        train_loader, train_stats = T.make_extra_loader(train_split, cfg.task, tokenizer, label_names, micro, cfg.seed, True, max_length, cfg.limit)
        valid_loader, valid_stats = T.make_extra_loader(valid_split, cfg.task, tokenizer, label_names, micro, cfg.seed, False, max_length)
        length_stats = {"train": train_stats, "validation": valid_stats}

    total_steps = math.ceil(len(train_loader) / accumulation) * cfg.epochs
    warmup_steps = round(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    soft = load_soft_labels(cfg.soft_labels, cfg.task, list(train_split.ids)) if cfg.soft_labels else None

    def validate(m):
        if task_family == "uit":
            return T.validate_uit(m, valid_loader, device, precision, cfg.task, bundle)
        return T.validate_extra(m, valid_loader, cfg.task, label_names, device, precision)

    config = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in dataclasses.asdict(cfg).items()},
        "family": task_family, "precision": precision, "max_length": max_length,
        "gradient_accumulation_steps": accumulation, "total_steps": total_steps, "warmup_steps": warmup_steps,
        "kd_alpha": KD_ALPHA if soft else None, "kd_temperature": KD_TEMPERATURE if soft else None,
        "labels": label_names, "parameters": count_parameters(model), "length_stats": length_stats,
    }
    T.write_json(out / "config.json", config)

    best, best_epoch, curves = -math.inf, 0, []
    n_batches = len(train_loader)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        t0, loss_sum, kd_sum, seen = time.time(), 0.0, 0.0, 0
        for index, batch in enumerate(train_loader):
            moved = T.move_batch(batch, device)
            with T.make_autocast(precision, device):
                logits = model(moved["input_ids"], moved["attention_mask"], token_type_ids=moved.get("token_type_ids"))
                if cfg.task == "phoner":
                    ce = F.cross_entropy(logits.reshape(-1, num_labels), moved["labels"].reshape(-1), ignore_index=-100)
                else:
                    ce = T.compute_loss(logits, moved["labels"], num_aspects=num_aspects, aspect_class_counts=counts)
                if accumulation > 1:
                    # Weight each micro batch by its share of the accumulation window.
                    start = (index // accumulation) * accumulation
                    end = min(start + accumulation, n_batches)
                    window = sum(min(micro, len(train_loader.dataset) - j * micro) for j in range(start, end))
                    scaled = ce * (len(batch["ids"]) / window)
                else:
                    scaled = ce
            loss = scaled
            if soft is not None:
                if cfg.task == "phoner":
                    s, t = _ner_pairs(logits, batch, soft)
                    kd = kd_term(s, t, segments) if s is not None else logits.sum() * 0
                else:
                    s = logits.float().reshape(len(batch["ids"]), -1)
                    t = torch.as_tensor(np.stack([soft[str(i)] for i in batch["ids"]]), dtype=torch.float32, device=device)
                    kd = kd_term(s, t, segments)
                loss = KD_ALPHA * ce + (1 - KD_ALPHA) * kd
                kd_sum += float(kd.detach()) * len(batch["ids"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {index}")
            loss.backward()
            loss_sum += float(ce.detach()) * len(batch["ids"])
            seen += len(batch["ids"])
            if accumulation == 1 or (index + 1) % accumulation == 0 or index + 1 == n_batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        metrics, rows = validate(model)
        score = T.selection_score(metrics)
        curves.append({"epoch": epoch, "train_ce": loss_sum / seen, "train_kd": kd_sum / seen if soft else None, "selection_f1": score, "seconds": time.time() - t0})
        T.write_json(out / "epoch_curves.json", {"epochs": curves})
        if score > best:
            best, best_epoch = score, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch}, out / "best_checkpoint.pt")
            T.write_json(out / "validation_predictions.json", {"epoch": epoch, "rows": rows})
            T.write_json(out / "best_validation_metrics.json", {"epoch": epoch, **metrics})
        print(f"[{cfg.kind}/{cfg.task}/seed{cfg.seed}] epoch {epoch:2d} val {100 * score:.2f} best {100 * best:.2f}@{best_epoch} ({time.time() - t0:.0f}s)", flush=True)

    # Reload the best checkpoint and check it reproduces the saved predictions.
    del model, optimizer
    reloaded = _rebuild(cfg, task_family, num_labels, num_aspects, counts)
    reloaded.load_state_dict(torch.load(out / "best_checkpoint.pt", map_location="cpu", weights_only=True)["model"], strict=True)
    reloaded.to(device)
    reload_metrics, reload_rows = validate(reloaded)
    saved = T.read_json(out / "validation_predictions.json")["rows"]
    checks = {
        "reload_matches_best": abs(T.selection_score(reload_metrics) - best) < 1e-9,
        "reload_predictions_identical": [r["prediction"] for r in reload_rows] == [r["prediction"] for r in saved],
        "ids_complete": [r["id"] for r in reload_rows] == list(valid_split.ids),
    }
    if held is not None:
        _write_oof_logits(reloaded, cfg, task_family, tokenizer, held, bundle if task_family == "uit" else None, label_names, max_length, device, precision, out)
    status = "completed" if all(checks.values()) else "completed_with_check_failure"
    metadata = {**config, "status": status, "best_epoch": best_epoch, "best_selection_f1": best, "checks": checks, "elapsed_seconds": time.time() - started}
    T.write_json(out / "metadata.json", metadata)
    if status == "completed" and not cfg.keep_checkpoint:
        (out / "best_checkpoint.pt").unlink()
    return metadata


def _rebuild(cfg: RunConfig, task_family: str, num_labels: int, num_aspects: int, counts) -> torch.nn.Module:
    task_type = "token" if cfg.task == "phoner" else "sequence"
    return build_model(
        cfg.backbone or "", kind=cfg.kind, task_type=task_type, num_labels=num_labels, num_aspects=num_aspects,
        aspect_class_counts=counts if num_aspects else None, local_files_only=cfg.local_files_only,
    )


def _write_oof_logits(model, cfg, task_family, tokenizer, held, bundle, label_names, max_length, device, precision, out: Path) -> None:
    """Teacher logits for the held-out fold, saved as ``oof_logits.npz``."""

    model.eval()
    ids: list[str] = []
    values: list[np.ndarray] = []
    if task_family == "uit":
        loader = T.make_uit_loader(tokenizer, held, labels=bundle.labels_for_model("oof"), max_length=256, batch_size=64, shuffle=False, seed=42)
        with torch.inference_mode():
            for batch in loader:
                moved = T.move_batch(batch, device)
                with T.make_autocast(precision, device):
                    logits = model(moved["input_ids"], moved["attention_mask"], token_type_ids=moved.get("token_type_ids"))
                values.append(logits.float().reshape(len(batch["ids"]), -1).cpu().numpy())
                ids.extend(batch["ids"])
        np.savez(out / "oof_logits.npz", ids=np.array(ids), logits=np.concatenate(values), aspect_class_counts=np.array(list(bundle.aspect_class_counts) or [len(bundle.label_names)]))
        return
    loader, _ = T.make_extra_loader(held, cfg.task, tokenizer, label_names, cfg.micro_batch_size, 42, False, max_length)
    masks: list[np.ndarray] = []
    offsets = [0]
    with torch.inference_mode():
        for batch in loader:
            moved = T.move_batch(batch, device)
            with T.make_autocast(precision, device):
                logits = model(moved["input_ids"], moved["attention_mask"], token_type_ids=moved.get("token_type_ids"))
            logits = logits.float().cpu().numpy()
            ids.extend(batch["ids"])
            if cfg.task == "phoner":
                for i, positions in enumerate(batch["positions"]):
                    n = batch["word_count"][i]
                    word_logits = np.full((n, len(label_names)), np.nan, dtype=np.float32)
                    mask = np.zeros(n, dtype=bool)
                    for pos, wid in positions:
                        word_logits[wid] = logits[i, pos]
                        mask[wid] = True
                    values.append(word_logits)
                    masks.append(mask)
                    offsets.append(offsets[-1] + n)
            else:
                values.append(logits)
    if cfg.task == "phoner":
        np.savez_compressed(out / "oof_logits.npz", ids=np.asarray(ids), offsets=np.asarray(offsets), logits=np.concatenate(values), has_logit=np.concatenate(masks))
    else:
        np.savez_compressed(out / "oof_logits.npz", ids=np.asarray(ids), logits=np.concatenate(values))


def merge_soft_labels(task: str, fold_dirs: list[Path], out_path: Path) -> dict[str, Any]:
    """Combine five held-out folds into one soft-label file in training order."""

    if len(fold_dirs) != N_FOLDS:
        raise ValueError(f"Expected {N_FOLDS} fold directories")
    if family(task) == "uit":
        train = uit.load_task(task).splits["train"]
        train_ids, label_names, widths = list(train.ids), None, None
    else:
        data = extra.load_task(task)
        train_ids, label_names = list(data.train.ids), list(data.label_names)
    position = {sid: i for i, sid in enumerate(train_ids)}
    rows: list[Any] = [None] * len(train_ids)
    masks: list[Any] = [None] * len(train_ids)
    for directory in fold_dirs:
        z = np.load(Path(directory) / "oof_logits.npz", allow_pickle=False)
        ids = [str(i) for i in z["ids"]]
        for j, sid in enumerate(ids):
            if sid not in position or rows[position[sid]] is not None:
                raise ValueError(f"Unexpected or duplicate example {sid}")
            if task == "phoner":
                a, b = int(z["offsets"][j]), int(z["offsets"][j + 1])
                rows[position[sid]], masks[position[sid]] = z["logits"][a:b], z["has_logit"][a:b]
            else:
                rows[position[sid]] = z["logits"][j]
        if family(task) == "uit":
            widths = z["aspect_class_counts"]
    if any(row is None for row in rows):
        raise ValueError("Some training examples are not covered by any fold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if task == "phoner":
        offsets = np.asarray([0] + list(np.cumsum([len(r) for r in rows])), dtype=np.int64)
        np.savez_compressed(out_path, ids=np.asarray(train_ids), offsets=offsets, logits=np.concatenate(rows).astype(np.float32), has_logit=np.concatenate(masks).astype(bool), label_names=np.asarray(label_names))
    elif family(task) == "uit":
        np.savez(out_path, ids=np.array(train_ids), logits=np.stack(rows), aspect_class_counts=np.asarray(widths))
    else:
        np.savez_compressed(out_path, ids=np.asarray(train_ids), logits=np.stack(rows).astype(np.float32), label_names=np.asarray(label_names))
    return {"task": task, "examples": len(train_ids), "path": str(out_path)}
