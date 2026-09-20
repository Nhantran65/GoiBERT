#!/usr/bin/env python3
"""Extract layer-wise frozen representations for the D1 bottleneck diagnostic.

The extractor is deliberately independent of all task heads and training
code.  It opens only train/dev through ``load_task(..., include_test=False)``;
the selected train rows and every validation row are tokenized once and the
same tensors are passed to both the local student and DAPT teacher.

See ``docs/D1_CACHE_CONTRACT.md`` for the cache schema.  The default command
extracts all five tasks for both encoders.  ``--task`` and ``--model`` are
available for a controlled rerun or a smoke-like debugging pass, but they do
not alter the fixed 2,000-row sampling contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# The CUDA backend and allocator environment must be selected before torch is
# imported.  This mirrors the E1/E2 runtime contract.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import Tensor
from transformers import AutoTokenizer, RobertaModel

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goibert.data import (  # noqa: E402
    TASK_IDS,
    TaskDataset,
    TaskSplit,
    example_label_key,
    load_task,
    sha256_file,
    visfd_to_model_labels,
)


TASKS = tuple(TASK_IDS)
MODEL_IDS = ("student", "teacher")
MODEL_DIRS = {
    "student": ROOT / "models" / "goiturn-s-gd",
    "teacher": ROOT / "models" / "goibert-b-dapt",
}
MODEL_LINEAGE = {
    "student": "goiturn-s-gd general-distilled student",
    "teacher": "goibert-b-dapt DAPT derivative of BamiBERT; not the original BamiBERT checkpoint",
}
SELECTION_SEED = 42
TRAIN_SAMPLE_SIZE = 2000
MAX_LENGTH = 256
TRAIN_BATCH_SIZE = 32
TEACHER_BATCH_SIZE = 16
PRECISION = "bf16"
ATTENTION_IMPLEMENTATION = "sdpa"
EXPECTED_STUDENT_LAYERS = 11  # embedding + 10 transformer layers
EXPECTED_STUDENT_HIDDEN = 384
EXPECTED_TEACHER_LAYERS = 13  # embedding + 12 transformer layers
EXPECTED_TEACHER_HIDDEN = 768


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _save_npy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _sha256_bytes(chunks: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def _directory_hashes(root: Path) -> dict[str, str]:
    allowed = {".json", ".safetensors", ".txt", ".model", ".vocab"}
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in allowed
    }


def _tokenizer_hashes(root: Path) -> dict[str, str]:
    """Hash tokenizer assets without mixing in encoder weights/config."""

    tokenizer_names = {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "spiece.model",
        "sentencepiece.bpe.model",
    }
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name in tokenizer_names
    }


def _directory_digest(hashes: Mapping[str, str]) -> str:
    return _sha256_bytes(
        f"{name}\0{hashes[name]}\n".encode("utf-8") for name in sorted(hashes)
    )


def _json_digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _label_json(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_label_json(item) for item in value]
    if isinstance(value, list):
        return [_label_json(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _support_counts(labels: Sequence[Any], class_schema: Mapping[str, Any]) -> Any:
    if labels and isinstance(labels[0], (tuple, list)):
        widths = [len(values) for values in class_schema.get("slot_label_names", [])]
        value_ids = class_schema.get("slot_label_ids") or [list(range(width)) for width in widths]
        counts = [[0 for _ in range(width)] for width in widths]
        for row in labels:
            if len(row) != len(counts):
                raise ValueError("ABSA label row has a different number of slots")
            for slot, value in enumerate(row):
                ids = [int(item) for item in value_ids[slot]]
                if int(value) not in ids:
                    raise ValueError(f"Label {value} is outside schema for slot {slot}")
                counts[slot][ids.index(int(value))] += 1
        return counts
    names = class_schema.get("label_names", [])
    counts = [0 for _ in names]
    for value in labels:
        if not 0 <= int(value) < len(counts):
            raise ValueError(f"Label {value} is outside scalar class schema")
        counts[int(value)] += 1
    return counts


def _class_schema(task: TaskDataset) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "label_names": list(task.label_names),
        "aspect_names": list(task.aspect_names),
        "slot_label_names": [list(values) for values in task.slot_label_names],
        "slot_label_ids": [
            [0, 4] if task.task_id == "visfd" and index == 5 else list(range(len(values)))
            for index, values in enumerate(task.slot_label_names)
        ],
        "pair_scored_aspects": (
            list(task.pair_scored_aspects) if task.pair_scored_aspects is not None else None
        ),
        "label_policy": task.label_policy,
    }


def _select_train_indices(count: int) -> np.ndarray:
    size = min(TRAIN_SAMPLE_SIZE, count)
    rng = np.random.default_rng(SELECTION_SEED)
    return np.sort(rng.choice(count, size=size, replace=False).astype(np.int64))


def _raw_and_model_labels(task: TaskDataset, split: TaskSplit, indices: Sequence[int]) -> tuple[list[Any], list[Any]]:
    raw = [_label_json(split[index].label) for index in indices]
    if task.task_id == "visfd":
        model = [_label_json(value) for value in visfd_to_model_labels([split[index].label for index in indices])]
    else:
        model = raw
    return raw, model


@dataclass(frozen=True)
class PreparedSplit:
    name: str
    ids: tuple[str, ...]
    source_indices: tuple[int, ...]
    texts: tuple[str, ...]
    raw_labels: tuple[Any, ...]
    model_labels: tuple[Any, ...]
    input_ids: tuple[tuple[int, ...], ...]
    attention_masks: tuple[tuple[int, ...], ...]
    text_hashes: tuple[str, ...]


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _input_digest(split: PreparedSplit) -> str:
    chunks: list[bytes] = []
    for ids, mask in zip(split.input_ids, split.attention_masks):
        chunks.append(np.asarray(ids, dtype="<i8").tobytes())
        chunks.append(b"\0")
        chunks.append(np.asarray(mask, dtype="<i1").tobytes())
        chunks.append(b"\n")
    return _sha256_bytes(chunks)


def _tokenizer_digest(tokenizer_hashes: Mapping[str, str]) -> str:
    return _directory_digest(tokenizer_hashes)


def _prepare_split(
    task: TaskDataset,
    split_name: str,
    indices: Sequence[int],
    tokenizer: Any,
) -> PreparedSplit:
    split = task.splits[split_name]
    examples = [split[index] for index in indices]
    texts = tuple(example.text for example in examples)
    encoded = tokenizer(
        list(texts),
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        return_attention_mask=True,
    )
    input_ids = tuple(tuple(int(item) for item in row) for row in encoded["input_ids"])
    masks = tuple(tuple(int(item) for item in row) for row in encoded["attention_mask"])
    if len(input_ids) != len(examples) or len(masks) != len(examples):
        raise RuntimeError(f"Tokenizer returned the wrong number of rows for {task.task_id}/{split_name}")
    if any(len(ids) != len(mask) or not ids for ids, mask in zip(input_ids, masks)):
        raise RuntimeError(f"Tokenizer returned malformed rows for {task.task_id}/{split_name}")
    raw, model = _raw_and_model_labels(task, split, indices)
    return PreparedSplit(
        name=split_name,
        ids=tuple(example.id for example in examples),
        source_indices=tuple(int(index) for index in indices),
        texts=texts,
        raw_labels=tuple(raw),
        model_labels=tuple(model),
        input_ids=input_ids,
        attention_masks=masks,
        text_hashes=tuple(_hash_text(text) for text in texts),
    )


def _pad_batch(
    input_ids: Sequence[Sequence[int]],
    attention_masks: Sequence[Sequence[int]],
    pad_token_id: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    width = max(len(row) for row in input_ids)
    ids = torch.full((len(input_ids), width), int(pad_token_id), dtype=torch.long, device=device)
    masks = torch.zeros((len(input_ids), width), dtype=torch.long, device=device)
    for row, (tokens, mask) in enumerate(zip(input_ids, attention_masks)):
        length = len(tokens)
        ids[row, :length] = torch.as_tensor(tokens, dtype=torch.long, device=device)
        masks[row, :length] = torch.as_tensor(mask, dtype=torch.long, device=device)
    return ids, masks


def _content_mask(
    input_ids: Tensor,
    attention_mask: Tensor,
    special_ids: Sequence[int],
) -> Tensor:
    mask = attention_mask.to(dtype=torch.bool)
    special = torch.zeros_like(mask)
    for token_id in special_ids:
        special |= input_ids.eq(int(token_id))
    return mask & ~special


def _extract_split(
    model: RobertaModel,
    prepared: PreparedSplit,
    *,
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    cls_parts: list[Tensor] = []
    mean_parts: list[Tensor] = []
    empty_rows = 0
    special_ids = tuple(int(value) for value in tokenizer.all_special_ids)
    with torch.inference_mode():
        for start in range(0, len(prepared.ids), batch_size):
            stop = min(start + batch_size, len(prepared.ids))
            input_ids, attention_mask = _pad_batch(
                prepared.input_ids[start:stop],
                prepared.attention_masks[start:stop],
                int(tokenizer.pad_token_id),
                device,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    output_attentions=False,
                    return_dict=True,
                )
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("RobertaModel did not return hidden states")
            stacked_cls = torch.stack(tuple(state[:, 0, :] for state in hidden_states), dim=0)
            content = _content_mask(input_ids, attention_mask, special_ids)
            denominator = content.sum(dim=1)
            empty = denominator.eq(0)
            empty_rows += int(empty.sum().item())
            weights = content.to(dtype=hidden_states[0].dtype)
            stacked_mean = torch.stack(
                tuple(
                    (state * weights.unsqueeze(-1)).sum(dim=1)
                    / denominator.clamp_min(1).to(dtype=state.dtype).unsqueeze(-1)
                    for state in hidden_states
                ),
                dim=0,
            )
            if bool(empty.any()):
                stacked_mean[:, empty, :] = stacked_cls[:, empty, :]
            cls_parts.append(stacked_cls.float().cpu())
            mean_parts.append(stacked_mean.float().cpu())
    if not cls_parts:
        raise RuntimeError(f"No rows extracted for {prepared.name}")
    cls = torch.cat(cls_parts, dim=1).numpy()
    mean = torch.cat(mean_parts, dim=1).numpy()
    if not np.isfinite(cls).all() or not np.isfinite(mean).all():
        raise FloatingPointError(f"Non-finite representation in {prepared.name}")
    return cls, mean, empty_rows


def _load_encoder(model_id: str, device: torch.device) -> tuple[RobertaModel, dict[str, Any]]:
    model_dir = MODEL_DIRS[model_id]
    loaded = RobertaModel.from_pretrained(
        model_dir,
        local_files_only=True,
        add_pooling_layer=False,
        attn_implementation=ATTENTION_IMPLEMENTATION,
        output_loading_info=True,
    )
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise RuntimeError("Expected Transformers to return (model, loading_info)")
    model, loading_info = loaded
    if not isinstance(model, RobertaModel):
        raise RuntimeError(f"Unexpected model type for {model_id}: {type(model)!r}")
    if not isinstance(loading_info, Mapping):
        raise RuntimeError("Transformers loading info has an unexpected shape")
    missing = list(loading_info.get("missing_keys", []))
    unexpected = list(loading_info.get("unexpected_keys", []))
    mismatched = list(loading_info.get("mismatched_keys", []))
    # A pooler is intentionally excluded by add_pooling_layer=False.  Any
    # other missing/unexpected/mismatched weight is a hard failure: this
    # diagnostic must never probe a silently random model.
    missing_non_pooler = [key for key in missing if not key.startswith("pooler.")]
    unexpected_non_unused = [
        key for key in unexpected if not key.startswith("lm_head.")
    ]
    if missing_non_pooler or unexpected_non_unused or mismatched:
        raise RuntimeError(
            f"Unsafe {model_id} checkpoint load: missing={missing!r}, "
            f"unexpected={unexpected!r}, mismatched={mismatched!r}"
        )
    model.to(device)
    model.eval()
    expected_layers, expected_hidden = (
        (EXPECTED_STUDENT_LAYERS, EXPECTED_STUDENT_HIDDEN)
        if model_id == "student"
        else (EXPECTED_TEACHER_LAYERS, EXPECTED_TEACHER_HIDDEN)
    )
    if int(model.config.num_hidden_layers) + 1 != expected_layers:
        raise RuntimeError(f"Unexpected {model_id} layer count: {model.config.num_hidden_layers}")
    if int(model.config.hidden_size) != expected_hidden:
        raise RuntimeError(f"Unexpected {model_id} hidden size: {model.config.hidden_size}")
    return model, {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "mismatched_keys": mismatched,
    }


def _dataset_metadata(
    task: TaskDataset,
    prepared: Mapping[str, PreparedSplit],
    *,
    tokenizer_hashes: Mapping[str, str],
) -> dict[str, Any]:
    schema = _class_schema(task)
    splits: dict[str, Any] = {}
    for split_name in ("train", "validation"):
        split = prepared[split_name]
        splits[split_name] = {
            "count": len(split.ids),
            "source_indices": list(split.source_indices),
            "ids": list(split.ids),
            "text_sha256": list(split.text_hashes),
            "raw_gold_labels": [_label_json(value) for value in split.raw_labels],
            "model_gold_labels": [_label_json(value) for value in split.model_labels],
            "support_counts_raw": _support_counts(split.raw_labels, schema),
            "support_counts_model": _support_counts(
                split.model_labels,
                {
                    **schema,
                    "slot_label_names": [
                        ["absent", "positive", "negative", "neutral"] if task.task_id == "visfd" and index != 5
                        else ["absent", "others"] if task.task_id == "visfd" and index == 5
                        else values
                        for index, values in enumerate(schema.get("slot_label_names", []))
                    ],
                    "slot_label_ids": [
                        [0, 1] if task.task_id == "visfd" and index == 5
                        else list(range(len(values)))
                        for index, values in enumerate(schema.get("slot_label_names", []))
                    ],
                },
            ),
            "sequence_lengths": [len(row) for row in split.input_ids],
            "input_sha256": _input_digest(split),
        }
    return {
        "schema_version": "d1-cache-v1",
        "status": "completed",
        "task_id": task.task_id,
        "include_test": False,
        "selection_seed": SELECTION_SEED,
        "train_sample_rule": "sorted(np.random.default_rng(42).choice(N_train, min(2000,N_train), replace=False))",
        "class_schema": schema,
        "source_files": {path: sha256_file(Path(path)) for path in task.source_files},
        "tokenizer": {
            "files": dict(tokenizer_hashes),
            "digest": _tokenizer_digest(tokenizer_hashes),
            "max_length": MAX_LENGTH,
            "padding": "dynamic_batch",
            "truncation": True,
        },
        "splits": splits,
    }


def _extract_task_model(
    task: TaskDataset,
    prepared: Mapping[str, PreparedSplit],
    *,
    model_id: str,
    tokenizer: Any,
    tokenizer_hashes: Mapping[str, str],
    device: torch.device,
    cache_root: Path,
    force: bool,
) -> dict[str, Any]:
    model_dir = MODEL_DIRS[model_id]
    output_dir = cache_root / task.task_id / model_id
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not force:
        raise FileExistsError(
            f"Completed or stale cache already exists at {output_dir}; use --force only for an intentional rerun"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model, loading_info = _load_encoder(model_id, device)
    batch_size = TEACHER_BATCH_SIZE if model_id == "teacher" else TRAIN_BATCH_SIZE
    arrays: dict[str, np.ndarray] = {}
    empty_rows: dict[str, int] = {}
    try:
        for split_name in ("train", "validation"):
            print(f"[{task.task_id}/{model_id}] extracting {split_name} ({len(prepared[split_name].ids)} rows)", flush=True)
            cls, mean, empty = _extract_split(
                model,
                prepared[split_name],
                tokenizer=tokenizer,
                device=device,
                batch_size=batch_size,
            )
            arrays[f"{split_name}_cls"] = cls
            arrays[f"{split_name}_mean"] = mean
            empty_rows[split_name] = empty
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    array_metadata: dict[str, Any] = {}
    for name, array in arrays.items():
        path = output_dir / f"{name}.npy"
        _save_npy_atomic(path, array)
        array_metadata[name] = {
            "path": str(path.relative_to(ROOT)),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": sha256_file(path),
            "finite": bool(np.isfinite(array).all()),
        }
    model_hashes = _directory_hashes(model_dir)
    source_files = {
        "extractor": sha256_file(Path(__file__)),
        "data_loader": sha256_file(ROOT / "src" / "goibert" / "data.py"),
        "contract": sha256_file(ROOT / "docs" / "D1_CACHE_CONTRACT.md"),
    }
    manifest = {
        "schema_version": "d1-cache-v1",
        "status": "completed",
        "task_id": task.task_id,
        "model_id": model_id,
        "model_path": str(model_dir.relative_to(ROOT)),
        "model_lineage": MODEL_LINEAGE[model_id],
        "checkpoint_files": model_hashes,
        "checkpoint_digest": _directory_digest(model_hashes),
        "tokenizer_files": dict(tokenizer_hashes),
        "tokenizer_digest": _tokenizer_digest(tokenizer_hashes),
        "source_hashes": source_files,
        "input_contract": {
            "max_length": MAX_LENGTH,
            "selection_seed": SELECTION_SEED,
            "include_test": False,
            "train_count": len(prepared["train"].ids),
            "validation_count": len(prepared["validation"].ids),
            "train_input_sha256": _input_digest(prepared["train"]),
            "validation_input_sha256": _input_digest(prepared["validation"]),
        },
        "runtime": {
            "device": str(device),
            "precision": PRECISION,
            "attention_implementation": ATTENTION_IMPLEMENTATION,
            "batch_size": batch_size,
            "torch_version": torch.__version__,
            "transformers_version": __import__("transformers").__version__,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "elapsed_seconds": time.time() - started,
            "loading_info": loading_info,
        },
        "representations": {
            "layer_count": int(next(iter(arrays.values())).shape[0]),
            "hidden_size": int(next(iter(arrays.values())).shape[2]),
            "layer_0": "embedding_output",
            "cls": "first_token",
            "mean": "attention_valid_non_special_tokens; CLS fallback for empty rows",
            "empty_mean_rows": empty_rows,
        },
        "arrays": array_metadata,
    }
    # The manifest is the completion marker and is written only after every
    # array has been atomically replaced and hashed.
    _write_json_atomic(manifest_path, manifest)
    print(
        f"[{task.task_id}/{model_id}] completed in {time.time() - started:.1f}s "
        f"layers={manifest['representations']['layer_count']} hidden={manifest['representations']['hidden_size']}",
        flush=True,
    )
    return manifest


def _prepare_task(task_id: str, tokenizer: Any) -> tuple[TaskDataset, dict[str, PreparedSplit]]:
    task = load_task(task_id, ROOT, include_test=False)
    if task.has_test:
        raise RuntimeError("D1 must never load test")
    indices = _select_train_indices(len(task.train))
    prepared = {
        "train": _prepare_split(task, "train", indices, tokenizer),
        "validation": _prepare_split(task, "validation", range(len(task.validation)), tokenizer),
    }
    if prepared["train"].source_indices != tuple(int(value) for value in indices):
        raise RuntimeError("Train source index selection was not preserved")
    return task, prepared


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, action="append", help="Extract only this task (repeatable)")
    parser.add_argument("--model", choices=MODEL_IDS, action="append", help="Extract only this model (repeatable)")
    parser.add_argument("--device", default="cuda:2", help="CUDA device; default is the approved GPU 2")
    parser.add_argument("--cache-root", type=Path, default=ROOT / "artifacts" / "d1" / "cache")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing cache intentionally")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("D1 extraction requires CUDA; sandbox CPU fallback is not an accepted run")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("D1 extraction must run on CUDA")
    if device.index is None:
        raise ValueError("Use an explicit CUDA device such as cuda:2")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"BF16 is not supported on {device}")
    torch.cuda.set_device(device)
    tasks = tuple(args.task) if args.task else TASKS
    models = tuple(args.model) if args.model else MODEL_IDS
    tokenizer_dirs = {model_id: MODEL_DIRS[model_id] for model_id in MODEL_IDS}
    tokenizer_hashes = {model_id: _tokenizer_hashes(path) for model_id, path in tokenizer_dirs.items()}
    if _tokenizer_digest(tokenizer_hashes["student"]) != _tokenizer_digest(tokenizer_hashes["teacher"]):
        raise RuntimeError("Student and teacher tokenizer/checkpoint directory hashes differ")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIRS["student"], local_files_only=True, use_fast=True
    )
    if tokenizer.pad_token_id is None or not tokenizer.all_special_ids:
        raise RuntimeError("Tokenizer is missing pad or special token IDs")
    cache_root = args.cache_root.resolve()
    prepared_by_task: dict[str, tuple[TaskDataset, dict[str, PreparedSplit]]] = {}
    for task_id in tasks:
        task, prepared = _prepare_task(task_id, tokenizer)
        prepared_by_task[task_id] = (task, prepared)
        metadata = _dataset_metadata(task, prepared, tokenizer_hashes=tokenizer_hashes["student"])
        _write_json_atomic(cache_root / task_id / "dataset.json", metadata)
        print(
            f"[{task_id}] prepared train={len(prepared['train'].ids)} validation={len(prepared['validation'].ids)} "
            f"train_input={metadata['splits']['train']['input_sha256'][:12]}",
            flush=True,
        )

    for task_id in tasks:
        task, prepared = prepared_by_task[task_id]
        for model_id in models:
            _extract_task_model(
                task,
                prepared,
                model_id=model_id,
                tokenizer=tokenizer,
                tokenizer_hashes=tokenizer_hashes[model_id],
                device=device,
                cache_root=cache_root,
                force=args.force,
            )
    print("D1 extraction completed for " + ", ".join(f"{task}/{model}" for task in tasks for model in models))


if __name__ == "__main__":
    main()
