#!/usr/bin/env python3
"""Run the isolated E6 student MLM continuation pilot.

The runner consumes ``artifacts/e6/mlm/corpus.jsonl`` from
``e6_build_corpus.py``.  It performs a 500-step head-only warmup followed by
2,000 joint encoder/head steps, evaluates a fixed train-only holdout at the
declared checkpoints, and exports only the trained encoder and tokenizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, RobertaModel, get_linear_schedule_with_warmup

from goibert_e6.mlm import (
    E6RobertaMLM,
    assert_tied_decoder,
    encoder_state_hash,
    fixed_masked_batch,
    mask_batch,
    masked_loss,
)


SEED = 42
MAX_LENGTH = 256
MLM_PROBABILITY = 0.15
HEAD_STEPS = 500
JOINT_STEPS = 2000
HEAD_LR = 5e-4
JOINT_LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.10
EFFECTIVE_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 32
FIXED_HOLDOUT_SEED = 1729
CHECKPOINT_STEPS = (500, 1000, 1500, 2000)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=json_default)
        handle.write("\n")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(repo_root: Path, paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(str(path.relative_to(repo_root)).encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def source_hash(repo_root: Path) -> str:
    paths = [
        repo_root / "src" / "goibert_e6" / "__init__.py",
        repo_root / "src" / "goibert_e6" / "mlm.py",
        repo_root / "scripts" / "e6_build_corpus.py",
        repo_root / "scripts" / "e6_run_mlm.py",
        repo_root / "configs" / "e6" / "mlm_protocol.json",
    ]
    return tree_hash(repo_root, paths)


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_corpus(corpus_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = read_json(corpus_dir / "corpus_manifest.json")
    corpus_path = corpus_dir / "corpus.jsonl"
    if manifest.get("corpus_jsonl_sha256") != sha256_file(corpus_path):
        raise RuntimeError("E6 corpus JSONL hash differs from its manifest")
    records: list[dict[str, Any]] = []
    with corpus_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("Each E6 corpus JSONL row must be an object")
            records.append(value)
    if len(records) != int(manifest.get("unique_text_count_after_token_filter", -1)):
        raise RuntimeError("E6 corpus row count differs from its manifest")
    if not records:
        raise RuntimeError("E6 corpus is empty")
    return records, manifest


def tokenizer_hashes(model_dir: Path) -> dict[str, str]:
    return {
        name: sha256_file(model_dir / name)
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
    }


class TextDataset(Dataset[dict[str, Any]]):
    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        return {"uid": str(row["uid"]), "text": str(row["text"])}


def make_dynamic_collate(tokenizer: Any):
    def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            [row["text"] for row in rows],
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )
        corrupted, labels, selected = mask_batch(
            encoded["input_ids"],
            encoded["attention_mask"],
            mask_token_id=int(tokenizer.mask_token_id),
            vocab_size=int(tokenizer.vocab_size),
            special_ids=tuple(int(value) for value in tokenizer.all_special_ids),
            probability=MLM_PROBABILITY,
        )
        return {
            "input_ids": corrupted,
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
            "masked_positions": selected,
            "uids": [row["uid"] for row in rows],
        }

    return collate


def make_train_loader(tokenizer: Any, records: Sequence[Mapping[str, Any]]) -> DataLoader:
    dataset = TextDataset(records)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        collate_fn=make_dynamic_collate(tokenizer),
    )


def make_fixed_holdout(
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    fixed_path: Path,
) -> list[dict[str, Any]]:
    if fixed_path.exists():
        existing: list[dict[str, Any]] = []
        with fixed_path.open("r", encoding="utf-8") as handle:
            existing = [json.loads(line) for line in handle if line.strip()]
        if len(existing) != len(records):
            raise RuntimeError("Existing fixed holdout does not match corpus holdout count")
        if [row.get("uid") for row in existing] != [row.get("uid") for row in records]:
            raise RuntimeError("Existing fixed holdout UID order differs from corpus")
        return existing

    generator = torch.Generator(device="cpu").manual_seed(FIXED_HOLDOUT_SEED)
    special_ids = tuple(int(value) for value in tokenizer.all_special_ids)
    output: list[dict[str, Any]] = []
    for record in records:
        ids = list(
            tokenizer.encode(
                str(record["text"]),
                add_special_tokens=True,
                truncation=True,
                max_length=MAX_LENGTH,
            )
        )
        input_ids = torch.tensor(ids, dtype=torch.long)
        attention = torch.ones_like(input_ids)
        eligible = attention.bool()
        for token_id in special_ids:
            eligible &= input_ids.ne(token_id)
        selected = (torch.rand(input_ids.shape, generator=generator) < MLM_PROBABILITY) & eligible
        if not selected.any():
            first = eligible.nonzero(as_tuple=False)
            if first.numel() == 0:
                raise RuntimeError(f"Holdout row {record['uid']} has no maskable token")
            selected[int(first[0].item())] = True
        choices = torch.rand(input_ids.shape, generator=generator)
        corrupted = input_ids.clone()
        replace_mask = selected & choices.lt(0.8)
        replace_random = selected & choices.ge(0.8) & choices.lt(0.9)
        corrupted[replace_mask] = int(tokenizer.mask_token_id)
        if replace_random.any():
            corrupted[replace_random] = torch.randint(
                0,
                int(tokenizer.vocab_size),
                (int(replace_random.sum().item()),),
                generator=generator,
            )
        output.append(
            {
                "uid": str(record["uid"]),
                "input_ids": input_ids.tolist(),
                "corrupted_ids": corrupted.tolist(),
                "mask_positions": selected.tolist(),
                "masked_token_count": int(selected.sum().item()),
            }
        )
    fixed_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = fixed_path.with_suffix(fixed_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(fixed_path)
    return output


def fixed_batches(records: Sequence[Mapping[str, Any]], batch_size: int, pad_id: int):
    for start in range(0, len(records), batch_size):
        rows = records[start : start + batch_size]
        length = max(len(row["input_ids"]) for row in rows)
        input_ids = torch.full((len(rows), length), int(pad_id), dtype=torch.long)
        corrupted = input_ids.clone()
        attention = torch.zeros_like(input_ids)
        selected = torch.zeros_like(input_ids, dtype=torch.bool)
        for index, row in enumerate(rows):
            row_length = len(row["input_ids"])
            input_ids[index, :row_length] = torch.tensor(row["input_ids"], dtype=torch.long)
            corrupted[index, :row_length] = torch.tensor(row["corrupted_ids"], dtype=torch.long)
            attention[index, :row_length] = 1
            selected[index, :row_length] = torch.tensor(row["mask_positions"], dtype=torch.bool)
        fixed_corrupted, fixed_labels, fixed_selected = fixed_masked_batch(
            input_ids, attention, selected, corrupted
        )
        yield fixed_corrupted, fixed_labels, fixed_selected, attention


def make_optimizer(model: E6RobertaMLM, learning_rate: float) -> AdamW:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith(".bias") or "layer_norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    if not decay and not no_decay:
        raise RuntimeError("No trainable parameters available for MLM optimizer")
    return AdamW(
        [
            {"params": decay, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", dtype=torch.bfloat16)


def run_fixed_eval(
    model: E6RobertaMLM,
    fixed_records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for original, labels, selected, attention in fixed_batches(
            fixed_records, MICRO_BATCH_SIZE, int(tokenizer.pad_token_id)
        ):
            moved_ids = original.to(device, non_blocking=True)
            moved_labels = labels.to(device, non_blocking=True)
            moved_selected = selected.to(device, non_blocking=True)
            moved_attention = attention.to(device, non_blocking=True)
            with autocast_context(device):
                logits = model.forward_masked(moved_ids, moved_attention, moved_selected)
            loss_sum, count = masked_loss(logits.float(), moved_labels)
            total_loss += float(loss_sum.detach().cpu())
            total_tokens += count
    if total_tokens <= 0:
        raise RuntimeError("Fixed MLM holdout contained no masked tokens")
    return {
        "mlm_loss": total_loss / total_tokens,
        "masked_tokens": total_tokens,
        "rows": len(fixed_records),
    }


def contract_check(
    model: E6RobertaMLM,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    """Check tying, protected masking, and masked-only/full-head algebra."""

    assert_tied_decoder(model)
    rows = list(records[:2])
    encoded = tokenizer(
        [str(row["text"]) for row in rows],
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    )
    seed_everything(SEED)
    corrupted, labels, selected = mask_batch(
        encoded["input_ids"],
        encoded["attention_mask"],
        mask_token_id=int(tokenizer.mask_token_id),
        vocab_size=int(tokenizer.vocab_size),
        special_ids=tuple(int(value) for value in tokenizer.all_special_ids),
        probability=MLM_PROBABILITY,
    )
    original = encoded["input_ids"]
    original_special = torch.zeros_like(selected)
    for token_id in tokenizer.all_special_ids:
        original_special |= original.eq(int(token_id))
    if bool((selected & original_special).any()):
        raise AssertionError("Dynamic masking selected an original special/pad token")
    if int(selected.sum().item()) <= 0:
        raise AssertionError("Dynamic masking selected no token in contract batch")
    model.eval()
    with torch.inference_mode():
        full = model.forward_full(corrupted.to(device), encoded["attention_mask"].to(device))
        masked = model.forward_masked(
            corrupted.to(device),
            encoded["attention_mask"].to(device),
            selected.to(device),
        )
        full_loss = nn.functional.cross_entropy(
            full.float().reshape(-1, full.shape[-1]),
            labels.to(device).reshape(-1),
            ignore_index=-100,
        )
        masked_loss_value, count = masked_loss(masked.float(), labels.to(device))
        masked_loss_value = masked_loss_value / count
    if not torch.allclose(full_loss, masked_loss_value, atol=2e-5, rtol=2e-5):
        raise AssertionError(
            f"Masked-only/full-head loss mismatch: {full_loss.item()} vs {masked_loss_value.item()}"
        )
    return {
        "tied_decoder": True,
        "masked_tokens": int(selected.sum().item()),
        "full_head_loss": float(full_loss.detach().cpu()),
        "masked_only_loss": float(masked_loss_value.detach().cpu()),
        "protected_special_count": int((selected & original_special).sum().item()),
    }


def append_log(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=False, default=json_default) + "\n")


def set_stage_trainability(model: E6RobertaMLM, *, encoder_trainable: bool) -> None:
    for parameter in model.roberta.parameters():
        parameter.requires_grad_(encoder_trainable)
    for parameter in model.lm_head.parameters():
        # Shared decoder.weight aliases an encoder embedding parameter and is
        # therefore controlled by the encoder flag.  Other head parameters
        # are always trainable.
        if parameter is model.roberta.embeddings.word_embeddings.weight:
            parameter.requires_grad_(encoder_trainable)
        else:
            parameter.requires_grad_(True)


def train_stage(
    model: E6RobertaMLM,
    loader: DataLoader,
    loader_iter: Any,
    *,
    steps: int,
    stage_name: str,
    device: torch.device,
    precision: str,
    optimizer: AdamW,
    scheduler: Any,
    output_dir: Path,
    starting_step: int,
    stage_step_offset: int = 0,
    log_every: int = 25,
) -> tuple[Any, list[dict[str, Any]]]:
    model.train()
    if stage_name == "head_warmup":
        model.roberta.eval()
        model.lm_head.train()
    rows: list[dict[str, Any]] = []
    running_loss = 0.0
    running_tokens = 0
    for local_step in range(1, steps + 1):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        optimizer.zero_grad(set_to_none=True)
        moved_ids = batch["input_ids"].to(device, non_blocking=True)
        moved_attention = batch["attention_mask"].to(device, non_blocking=True)
        moved_labels = batch["labels"].to(device, non_blocking=True)
        moved_selected = batch["masked_positions"].to(device, non_blocking=True)
        with autocast_context(device):
            logits = model.forward_masked(moved_ids, moved_attention, moved_selected)
        loss_sum, count = masked_loss(logits.float(), moved_labels)
        loss = loss_sum / count
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite MLM loss at {stage_name} step {local_step}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        running_loss += float(loss_sum.detach().cpu())
        running_tokens += count
        global_step = starting_step + local_step
        if local_step == 1 or local_step % log_every == 0 or local_step == steps:
            row = {
                "stage": stage_name,
                "stage_step": stage_step_offset + local_step,
                "global_step": global_step,
                "train_mlm_loss": running_loss / max(running_tokens, 1),
                "masked_tokens": running_tokens,
                "learning_rate": scheduler.get_last_lr()[0],
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            append_log(output_dir / "train_steps.jsonl", row)
            write_json(output_dir / "status.json", {"status": "running", **row})
            rows.append(row)
            running_loss = 0.0
            running_tokens = 0
    return loader_iter, rows


def runtime_info(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "precision": "bf16" if device.type == "cuda" else "cpu_bf16_autocast",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def run(repo_root: Path, *, device_name: str, precision: str) -> dict[str, Any]:
    output_dir = repo_root / "artifacts" / "e6" / "mlm"
    corpus_dir = output_dir
    export_dir = repo_root / "models" / "goiturn-s-gd-mlm-e6"
    if export_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing E6 export: {export_dir}")
    corpus_records, corpus_manifest = load_corpus(corpus_dir)
    train_records = [row for row in corpus_records if row.get("split") == "mlm_train"]
    holdout_records = [row for row in corpus_records if row.get("split") == "mlm_holdout"]
    if not train_records or not holdout_records:
        raise RuntimeError("Corpus must contain both mlm_train and mlm_holdout records")
    backbone_dir = repo_root / "models" / "goiturn-s-gd"
    tokenizer = AutoTokenizer.from_pretrained(backbone_dir, local_files_only=True, use_fast=True)
    current_tokenizer_hashes = tokenizer_hashes(backbone_dir)
    if current_tokenizer_hashes != corpus_manifest.get("tokenizer_hashes"):
        raise RuntimeError("Tokenizer changed after E6 corpus construction")
    fixed_path = output_dir / "fixed_holdout.jsonl"
    fixed_records = make_fixed_holdout(holdout_records, tokenizer, fixed_path)
    fixed_hash = sha256_file(fixed_path)

    seed_everything(SEED)
    load_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "add_pooling_layer": False,
        "attn_implementation": "sdpa",
        "output_loading_info": True,
    }
    loaded = RobertaModel.from_pretrained(backbone_dir, **load_kwargs)
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise RuntimeError("Expected RobertaModel loading info for strict E6 encoder check")
    encoder, loading_info = loaded
    if loading_info.get("missing_keys") or loading_info.get("unexpected_keys") or loading_info.get("mismatched_keys"):
        raise RuntimeError(f"E6 encoder load is not strict: {loading_info}")
    model = E6RobertaMLM(encoder)
    assert_tied_decoder(model)
    initial_encoder_hash = encoder_state_hash(model)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if device.type == "cuda":
        # PyTorch 2.6 can reject a torch.device object here when the process
        # is launched with a remapped CUDA_VISIBLE_DEVICES list; set the
        # remapped current device and use the no-argument API instead.
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
    model.to(device)
    checks = contract_check(model, tokenizer, train_records, device)
    # The contract check intentionally exercises random masking; reset every
    # RNG before creating the real loader so it cannot alter training order.
    seed_everything(SEED)
    loader = make_train_loader(tokenizer, train_records)
    loader_iter = iter(loader)
    initial_eval = run_fixed_eval(model, fixed_records, tokenizer, device, precision)

    # Stage A: only the new head is trainable, while the encoder is in eval.
    set_stage_trainability(model, encoder_trainable=False)
    trainable_head_names = [name for name, value in model.named_parameters() if value.requires_grad]
    if any(name.startswith("roberta.") for name in trainable_head_names):
        raise AssertionError(f"Stage-A encoder parameters unexpectedly trainable: {trainable_head_names}")
    optimizer_a = make_optimizer(model, HEAD_LR)
    scheduler_a = get_linear_schedule_with_warmup(
        optimizer_a, round(HEAD_STEPS * WARMUP_FRACTION), HEAD_STEPS
    )
    stage_a_started = time.time()
    loader_iter, stage_a_log = train_stage(
        model,
        loader,
        loader_iter,
        steps=HEAD_STEPS,
        stage_name="head_warmup",
        device=device,
        precision=precision,
        optimizer=optimizer_a,
        scheduler=scheduler_a,
        output_dir=output_dir,
        starting_step=0,
    )
    post_head_encoder_hash = encoder_state_hash(model)
    if post_head_encoder_hash != initial_encoder_hash:
        raise AssertionError("Encoder changed during Stage-A head-only warmup")
    head_eval = run_fixed_eval(model, fixed_records, tokenizer, device, precision)
    evaluations: list[dict[str, Any]] = [
        {"checkpoint": "initial", **initial_eval},
        {"checkpoint": "after_head_warmup_500", **head_eval},
    ]

    # Stage B: fresh optimizer and scheduler for all encoder/head parameters.
    set_stage_trainability(model, encoder_trainable=True)
    optimizer_b = make_optimizer(model, JOINT_LR)
    scheduler_b = get_linear_schedule_with_warmup(
        optimizer_b, round(JOINT_STEPS * WARMUP_FRACTION), JOINT_STEPS
    )
    stage_b_log: list[dict[str, Any]] = []
    completed_joint_steps = 0
    for chunk_end in CHECKPOINT_STEPS:
        chunk = chunk_end - completed_joint_steps
        if chunk <= 0:
            continue
        loader_iter, chunk_log = train_stage(
            model,
            loader,
            loader_iter,
            steps=chunk,
            stage_name="joint_mlm",
            device=device,
            precision=precision,
            optimizer=optimizer_b,
            scheduler=scheduler_b,
            output_dir=output_dir,
            starting_step=HEAD_STEPS + completed_joint_steps,
            stage_step_offset=completed_joint_steps,
        )
        stage_b_log.extend(chunk_log)
        completed_joint_steps = chunk_end
        eval_row = run_fixed_eval(model, fixed_records, tokenizer, device, precision)
        evaluations.append({"checkpoint": f"joint_step_{chunk_end}", **eval_row})
        write_json(output_dir / "mlm_evaluations.json", {"evaluations": evaluations})

    if completed_joint_steps != JOINT_STEPS:
        raise AssertionError(
            f"E6 Stage-B optimizer step contract failed: {completed_joint_steps} != {JOINT_STEPS}"
        )

    final_encoder_hash = encoder_state_hash(model)
    if final_encoder_hash == initial_encoder_hash:
        raise RuntimeError("Joint MLM stage left encoder byte-identical to initialization")
    model.eval()
    model.roberta.cpu()
    export_dir.mkdir(parents=False, exist_ok=False)
    model.roberta.save_pretrained(export_dir, safe_serialization=True)
    tokenizer.save_pretrained(export_dir)
    exported = RobertaModel.from_pretrained(
        export_dir,
        local_files_only=True,
        add_pooling_layer=False,
        attn_implementation="sdpa",
    )
    exported_hash = encoder_state_hash(E6RobertaMLM(exported))
    if exported_hash != final_encoder_hash:
        raise AssertionError("Reloaded E6 encoder hash differs from in-memory final encoder")

    metadata = {
        "protocol": "E6-student-MLM-continuation-v1",
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_backbone": str(backbone_dir),
        "export_backbone": str(export_dir),
        "corpus_manifest_sha256": sha256_file(output_dir / "corpus_manifest.json"),
        "corpus_jsonl_sha256": corpus_manifest["corpus_jsonl_sha256"],
        "fixed_holdout_sha256": fixed_hash,
        "corpus_counts": {
            "unique_total": len(corpus_records),
            "mlm_train": len(train_records),
            "mlm_holdout": len(holdout_records),
            "fixed_holdout_masked_tokens": sum(int(row["masked_token_count"]) for row in fixed_records),
        },
        "dataset_ids": {
            "train_uids": [str(row["uid"]) for row in train_records],
            "holdout_uids": [str(row["uid"]) for row in holdout_records],
        },
        "seed": SEED,
        "max_length": MAX_LENGTH,
        "mask_probability": MLM_PROBABILITY,
        "replacement": {"mask": 0.8, "random": 0.1, "unchanged": 0.1},
        "special_ids_protected": [int(value) for value in tokenizer.all_special_ids],
        "fixed_holdout_seed": FIXED_HOLDOUT_SEED,
        "stage_a": {
            "optimizer_steps": HEAD_STEPS,
            "learning_rate": HEAD_LR,
            "weight_decay": WEIGHT_DECAY,
            "warmup_steps": round(HEAD_STEPS * WARMUP_FRACTION),
            "encoder_initial_hash": initial_encoder_hash,
            "encoder_after_hash": post_head_encoder_hash,
            "encoder_unchanged": post_head_encoder_hash == initial_encoder_hash,
            "trainable_parameters": trainable_head_names,
            "elapsed_seconds": time.time() - stage_a_started,
        },
        "stage_b": {
            "optimizer_steps": JOINT_STEPS,
            "learning_rate": JOINT_LR,
            "weight_decay": WEIGHT_DECAY,
            "warmup_steps": round(JOINT_STEPS * WARMUP_FRACTION),
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "gradient_clip_norm": 1.0,
        },
        "head": "standard RobertaLMHead; dense/layer_norm/decoder bias newly initialized with config initializer range; decoder weight tied to encoder word embedding",
        "contract_checks": checks,
        "mlm_evaluations": evaluations,
        "source_hash": source_hash(repo_root),
        "backbone_hashes_initial": {
            str(path.relative_to(backbone_dir)): sha256_file(path)
            for path in sorted(backbone_dir.rglob("*"))
            if path.is_file()
        },
        "export_hash": exported_hash,
        "export_files": {
            str(path.relative_to(export_dir)): sha256_file(path)
            for path in sorted(export_dir.rglob("*"))
            if path.is_file()
        },
        "runtime": runtime_info(device),
        "peak_gpu_memory_allocated_bytes": (
            torch.cuda.max_memory_allocated() if device.type == "cuda" else None
        ),
        "peak_gpu_memory_reserved_bytes": (
            torch.cuda.max_memory_reserved() if device.type == "cuda" else None
        ),
        "architecture_unchanged": True,
        "mlm_head_exported": False,
    }
    write_json(output_dir / "mlm_metadata.json", metadata)
    write_json(output_dir / "status.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2, default=json_default))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", default="bf16")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    run(args.repo_root.resolve(), device_name=args.device, precision=args.precision)


if __name__ == "__main__":
    main()
