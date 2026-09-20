#!/usr/bin/env python3
"""Extract D2 representations from the selected post-fine-tuning checkpoints.

This is a deliberately small adapter around the frozen D1 extractor.  It
reuses D1's train cohort, validation cohort, tokenizer, padding, masks and
hidden-state views, but loads the encoder from the selected E1/E2 task
checkpoint.  The task head and optimizer are intentionally discarded.

The script never opens the test split.  A cache manifest is the completion
marker and is written only after all four arrays and the weight-drift audit
are complete.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from transformers import AutoTokenizer, RobertaModel

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Import the frozen D1 helpers rather than duplicating their data and feature
# extraction semantics.  This module is loaded by path because scripts/ is
# intentionally not a Python package.
_D1_SPEC = importlib.util.spec_from_file_location(
    "goibert_d1_extractor", ROOT / "scripts" / "d1_extract_features.py"
)
if _D1_SPEC is None or _D1_SPEC.loader is None:
    raise RuntimeError("Could not locate the frozen D1 extractor")
d1 = importlib.util.module_from_spec(_D1_SPEC)
sys.modules[_D1_SPEC.name] = d1
_D1_SPEC.loader.exec_module(d1)


TASKS = tuple(d1.TASKS)
MODEL_IDS = tuple(d1.MODEL_IDS)
ATTENTION_IMPLEMENTATION = d1.ATTENTION_IMPLEMENTATION
PRECISION = d1.PRECISION
MAX_LENGTH = d1.MAX_LENGTH
SELECTION_SEED = d1.SELECTION_SEED
TRAIN_SAMPLE_SIZE = d1.TRAIN_SAMPLE_SIZE
TRAIN_BATCH_SIZE = d1.TRAIN_BATCH_SIZE
TEACHER_BATCH_SIZE = d1.TEACHER_BATCH_SIZE

TEACHER_GAP_PATH = ROOT / "artifacts" / "diagnostics" / "teacher_gap.json"
E1_SCREENING_PATH = ROOT / "artifacts" / "e1" / ".orchestrator" / "screening_approved.json"
E2_TEACHER_SELECTION_PATH = (
    ROOT / "artifacts" / "e2" / ".orchestrator" / "stage0_teacher_selection.json"
)
D1_CACHE_ROOT = ROOT / "artifacts" / "d1" / "cache"
D2_CACHE_ROOT = ROOT / "artifacts" / "d2" / "cache"


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


def _canonical_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _hash_file(path: Path) -> str:
    return d1.sha256_file(path)


def _directory_hashes(root: Path) -> dict[str, str]:
    return d1._directory_hashes(root)


def _directory_digest(hashes: Mapping[str, str]) -> str:
    return d1._directory_digest(hashes)


def _resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _load_selection_inputs() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return locked E1 seed-42 and E2 teacher selections by task."""

    if not TEACHER_GAP_PATH.is_file():
        raise FileNotFoundError(TEACHER_GAP_PATH)
    gap = _canonical_json(TEACHER_GAP_PATH)
    prediction_files = gap.get("input_prediction_files", {})
    e1: dict[str, dict[str, Any]] = {}
    teacher: dict[str, dict[str, Any]] = {}
    for task_id in TASKS:
        e1_key = f"{task_id}.e1.seed42"
        teacher_key = f"{task_id}.teacher"
        for key in (e1_key, teacher_key):
            entry = prediction_files.get(key)
            if not isinstance(entry, Mapping) or not entry.get("path"):
                raise RuntimeError(f"teacher_gap.json has no usable {key}")
            prediction_path = _resolve(str(entry["path"]))
            if not prediction_path.is_file():
                raise FileNotFoundError(prediction_path)
            artifact_dir = prediction_path.parent
            metadata_path = artifact_dir / "metadata.json"
            checkpoint_path = artifact_dir / "best_checkpoint.pt"
            if not metadata_path.is_file() or not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Selected {key} lacks metadata/checkpoint in {artifact_dir}"
                )
            payload = {
                "task_id": task_id,
                "prediction_path": prediction_path,
                "metadata_path": metadata_path,
                "checkpoint_path": checkpoint_path,
                "prediction_sha256": str(entry.get("sha256", _hash_file(prediction_path))),
                "metadata": _canonical_json(metadata_path),
            }
            if key.endswith(".e1.seed42"):
                e1[task_id] = payload
            else:
                teacher[task_id] = payload

    # E1 screening_approved is the durable record for seed 42.  The diagnostic
    # used the same selected artifact paths; enforce that it is a completed
    # CE E1 run on the original student and not a later E3/E4/E5/E6 artifact.
    if not E1_SCREENING_PATH.is_file():
        raise FileNotFoundError(E1_SCREENING_PATH)
    approved = _canonical_json(E1_SCREENING_PATH).get("approved", [])
    approved_by_path = {_resolve(item["artifact_dir"]): item for item in approved if item.get("artifact_dir")}
    for task_id, item in e1.items():
        metadata = item["metadata"]
        artifact_meta = approved_by_path.get(item["metadata_path"].parent)
        if artifact_meta is None:
            raise RuntimeError(f"E1 seed-42 artifact is not in screening approval: {task_id}")
        if int(metadata.get("seed", -1)) != 42 or int(artifact_meta.get("seed", -1)) != 42:
            raise RuntimeError(f"E1 seed mismatch for {task_id}")
        if metadata.get("stage") != "screening" or metadata.get("loss") != "cross_entropy":
            raise RuntimeError(f"E1 selection is not the CE screening run for {task_id}")
        if metadata.get("include_test") is not False:
            raise RuntimeError(f"E1 selection unexpectedly opened test for {task_id}")
        if metadata.get("model", {}).get("backbone") != "goiturn-s-gd":
            raise RuntimeError(f"Unexpected E1 backbone for {task_id}")
        if artifact_meta.get("learning_rate") != metadata.get("learning_rate"):
            raise RuntimeError(f"E1 approval/metadata learning-rate mismatch for {task_id}")

    # E2 stage0_teacher_selection is the locked selection record.  It also
    # prevents accidentally extracting an E2 KD/NC student checkpoint.
    if not E2_TEACHER_SELECTION_PATH.is_file():
        raise FileNotFoundError(E2_TEACHER_SELECTION_PATH)
    e2_selection = _canonical_json(E2_TEACHER_SELECTION_PATH)
    selected = e2_selection.get("selected", {})
    selected_by_path = {
        _resolve(item["artifact_dir"]): item
        for item in selected.values()
        if isinstance(item, Mapping) and item.get("artifact_dir")
    }
    for task_id, item in teacher.items():
        metadata = item["metadata"]
        selection = selected_by_path.get(item["metadata_path"].parent)
        if selection is None:
            raise RuntimeError(f"E2 teacher artifact is not in locked stage0 selection: {task_id}")
        if int(metadata.get("seed", -1)) != 42 or int(selection.get("seed", -1)) != 42:
            raise RuntimeError(f"E2 teacher seed mismatch for {task_id}")
        if metadata.get("stage") != "stage0_teacher" or metadata.get("method") != "teacher":
            raise RuntimeError(f"Selected E2 artifact is not stage0 teacher for {task_id}")
        if metadata.get("include_test") is not False:
            raise RuntimeError(f"E2 teacher selection unexpectedly opened test for {task_id}")
        if metadata.get("model", {}).get("backbone") != "goibert-b-dapt":
            raise RuntimeError(f"Unexpected E2 teacher backbone for {task_id}")
        if selection.get("learning_rate") != metadata.get("learning_rate"):
            raise RuntimeError(f"E2 approval/metadata learning-rate mismatch for {task_id}")
    return e1, teacher


def _validate_d1_cohort(task_id: str, tokenizer: Any) -> tuple[Any, dict[str, Any], Path]:
    """Rebuild D1's cohort and require byte-equivalent metadata."""

    task, prepared = d1._prepare_task(task_id, tokenizer)
    generated = d1._dataset_metadata(
        task,
        prepared,
        tokenizer_hashes=d1._tokenizer_hashes(d1.MODEL_DIRS["student"]),
    )
    d1_dataset_path = D1_CACHE_ROOT / task_id / "dataset.json"
    if not d1_dataset_path.is_file():
        raise FileNotFoundError(d1_dataset_path)
    existing = _canonical_json(d1_dataset_path)
    # The original D1 dataset files predate a harmless ``slot_label_ids``
    # metadata addition in the helper.  Compare every cohort/tokenization
    # field explicitly while allowing that schema-only addition.
    for key in ("schema_version", "status", "task_id", "include_test", "selection_seed", "train_sample_rule", "source_files", "tokenizer"):
        if generated.get(key) != existing.get(key):
            raise RuntimeError(f"D1 metadata differs at {task_id}/{key}; refusing D2 extraction")
    generated_schema = generated.get("class_schema", {})
    existing_schema = existing.get("class_schema", {})
    for key in ("task_id", "label_names", "aspect_names", "slot_label_names", "pair_scored_aspects", "label_policy"):
        if generated_schema.get(key) != existing_schema.get(key):
            raise RuntimeError(f"D1 metadata differs at {task_id}/class_schema/{key}; refusing D2 extraction")
    for split_name in ("train", "validation"):
        generated_split = generated["splits"][split_name]
        existing_split = existing["splits"].get(split_name)
        if existing_split is None:
            raise RuntimeError(f"D1 metadata is missing {task_id}/{split_name}")
        for key in (
            "count",
            "source_indices",
            "ids",
            "text_sha256",
            "raw_gold_labels",
            "model_gold_labels",
            "support_counts_raw",
            "support_counts_model",
            "sequence_lengths",
            "input_sha256",
        ):
            if generated_split.get(key) != existing_split.get(key):
                raise RuntimeError(f"D1 cohort/tokenization differs at {task_id}/{split_name}/{key}")
    return task, prepared, d1_dataset_path


def _check_base_hashes(model_id: str, metadata: Mapping[str, Any]) -> tuple[Path, dict[str, str]]:
    base_dir = d1.MODEL_DIRS[model_id]
    actual = _directory_hashes(base_dir)
    expected = metadata.get("backbone_hashes")
    if not isinstance(expected, Mapping):
        raise RuntimeError(f"Selected {model_id} metadata has no backbone_hashes")
    for name, value in expected.items():
        if actual.get(name) != value:
            raise RuntimeError(
                f"Base {model_id} file hash changed for {name}: {actual.get(name)} != {value}"
            )
    return base_dir, actual


def _load_after_encoder(
    model_id: str,
    selected: Mapping[str, Any],
    device: torch.device,
) -> tuple[RobertaModel, dict[str, Any], dict[str, Any]]:
    """Load base config, then strictly replace it with checkpoint encoder weights."""

    metadata = selected["metadata"]
    base_dir, base_hashes = _check_base_hashes(model_id, metadata)
    checkpoint_path = selected["checkpoint_path"]
    checkpoint_sha256 = _hash_file(checkpoint_path)
    checkpoint_metadata = checkpoint_path.parent / "metadata.json"
    # Checkpoint metadata is the source of the selected best epoch and LR; the
    # explicit seed/stage checks happen before this function.
    load_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "add_pooling_layer": False,
        "attn_implementation": ATTENTION_IMPLEMENTATION,
        "output_loading_info": True,
    }
    loaded = RobertaModel.from_pretrained(base_dir, **load_kwargs)
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise RuntimeError(f"Could not obtain base loading info for {model_id}")
    model, loading_info = loaded
    if not isinstance(model, RobertaModel) or not isinstance(loading_info, Mapping):
        raise RuntimeError(f"Unexpected base model load for {model_id}")
    missing = [key for key in loading_info.get("missing_keys", []) if not key.startswith("pooler.")]
    unexpected = [key for key in loading_info.get("unexpected_keys", []) if not key.startswith("lm_head.")]
    mismatched = list(loading_info.get("mismatched_keys", []))
    if missing or unexpected or mismatched:
        raise RuntimeError(
            f"Unsafe base model load for {model_id}: missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
        )

    # Keep the original CPU tensors for the drift audit.  The checkpoint is
    # loaded with mmap when supported, avoiding an unnecessary second copy.
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False, mmap=True
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping) or "model" not in checkpoint:
        raise RuntimeError(f"Checkpoint {checkpoint_path} has no model state")
    full_state = checkpoint["model"]
    if not isinstance(full_state, Mapping):
        raise RuntimeError(f"Checkpoint {checkpoint_path} model state is not a mapping")
    non_encoder = [key for key in full_state if not (key.startswith("encoder.") or key.startswith("head."))]
    if non_encoder:
        raise RuntimeError(f"Checkpoint {checkpoint_path} has unexpected top-level state keys: {non_encoder[:5]}")
    head_keys = [key for key in full_state if key.startswith("head.")]
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in full_state.items()
        if key.startswith("encoder.")
    }
    # ``state_dict()`` normally returns views into the module.  Clone to CPU
    # before strict loading so the before/after drift audit cannot be erased
    # by ``load_state_dict`` mutating the module in place.
    base_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    if set(encoder_state) != set(base_state):
        missing_keys = sorted(set(base_state) - set(encoder_state))
        unexpected_keys = sorted(set(encoder_state) - set(base_state))
        raise RuntimeError(
            f"Checkpoint encoder keys do not match {model_id} RobertaModel: "
            f"missing={missing_keys[:8]}, unexpected={unexpected_keys[:8]}"
        )
    # Strict load is an explicit contract: no randomly initialized encoder
    # parameter may pass into the representation extraction.
    load_result = model.load_state_dict(encoder_state, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            f"Strict encoder load reported missing/unexpected keys: {load_result}"
        )

    drift = _weight_drift(base_state, encoder_state)
    changed_keys = [key for key, value in drift["parameters"].items() if value["abs_l2"] > 0.0]
    if not changed_keys:
        raise RuntimeError(f"Fine-tuned {model_id} checkpoint is identical to original encoder")
    model.to(device)
    model.eval()
    source = {
        "selected_checkpoint_path": str(checkpoint_path),
        "selected_checkpoint_sha256": checkpoint_sha256,
        "selected_metadata_path": str(checkpoint_metadata),
        "selected_metadata_sha256": _hash_file(checkpoint_metadata),
        "selected_seed": int(metadata.get("seed", -1)),
        "selected_stage": metadata.get("stage"),
        "selected_method": metadata.get("method", "supervised_ce"),
        "selected_learning_rate": metadata.get("learning_rate"),
        "selected_best_epoch": metadata.get("best_epoch"),
        "selected_best_validation_f1": metadata.get("best_selection_f1"),
        "selected_include_test": metadata.get("include_test"),
        "base_model_path": str(base_dir),
        "base_model_hashes": base_hashes,
        "base_model_digest": _directory_digest(base_hashes),
        "base_loading_info": {
            "missing_keys": list(loading_info.get("missing_keys", [])),
            "unexpected_keys": list(loading_info.get("unexpected_keys", [])),
            "mismatched_keys": list(loading_info.get("mismatched_keys", [])),
        },
        "checkpoint_state_key_counts": {
            "total": len(full_state),
            "encoder": len(encoder_state),
            "head": len(head_keys),
        },
        "strict_encoder_load": True,
        "changed_encoder_parameter_count": len(changed_keys),
        "encoder_parameter_count": len(encoder_state),
    }
    # Release the optimizer/RNG/head entries before the next model.  The model
    # has its own loaded tensors and no optimizer is needed for inference.
    del checkpoint
    return model, source, drift


def _empty_group() -> dict[str, float | int]:
    return {"parameter_count": 0, "abs_l2": 0.0, "before_l2": 0.0, "relative_l2": 0.0}


def _accumulate_group(group: dict[str, float | int], before: torch.Tensor, after: torch.Tensor) -> None:
    before_f = before.detach().to(dtype=torch.float32, device="cpu")
    after_f = after.detach().to(dtype=torch.float32, device="cpu")
    difference = after_f - before_f
    group["parameter_count"] = int(group["parameter_count"]) + int(before_f.numel())
    group["abs_l2"] = float(group["abs_l2"]) + float(torch.sum(difference * difference).item())
    group["before_l2"] = float(group["before_l2"]) + float(torch.sum(before_f * before_f).item())


def _finish_group(group: dict[str, float | int]) -> dict[str, float | int]:
    abs_l2 = float(np.sqrt(max(float(group["abs_l2"]), 0.0)))
    before_l2 = float(np.sqrt(max(float(group["before_l2"]), 0.0)))
    group["abs_l2"] = abs_l2
    group["before_l2"] = before_l2
    group["relative_l2"] = abs_l2 / before_l2 if before_l2 > 0.0 else None
    return group


def _drift_category(remainder: str) -> str:
    """Map one transformer parameter to the intended update group.

    Attention output projections also contain ``output.dense`` in their
    parameter name, so the attention prefix must be checked before the FFN
    prefix.  LayerNorm is checked first for the same explicitness.
    """

    if ".LayerNorm." in remainder:
        return "layernorm"
    if remainder.startswith("attention."):
        return "attention"
    if remainder.startswith("intermediate.") or remainder.startswith("output.dense."):
        return "ffn"
    return "other"


def _weight_drift(base_state: Mapping[str, torch.Tensor], after_state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Compute per-parameter and grouped encoder L2 changes on CPU."""

    parameters: dict[str, Any] = {}
    groups: dict[str, dict[str, float | int]] = {"embeddings": _empty_group()}
    layer_groups: dict[str, dict[str, dict[str, float | int]]] = {}
    all_group = _empty_group()
    layer_re = re.compile(r"^encoder\.layer\.(\d+)\.(.*)$")
    for key in sorted(base_state):
        before = base_state[key].detach().cpu()
        after = after_state[key].detach().cpu()
        if tuple(before.shape) != tuple(after.shape):
            raise RuntimeError(f"Weight shape mismatch in drift audit for {key}")
        before_f = before.float()
        after_f = after.float()
        difference = after_f - before_f
        before_sq = float(torch.sum(before_f * before_f).item())
        abs_sq = float(torch.sum(difference * difference).item())
        before_l2 = float(np.sqrt(max(before_sq, 0.0)))
        abs_l2 = float(np.sqrt(max(abs_sq, 0.0)))
        parameters[key] = {
            "shape": list(before.shape),
            "parameter_count": int(before.numel()),
            "abs_l2": abs_l2,
            "before_l2": before_l2,
            "relative_l2": abs_l2 / before_l2 if before_l2 > 0.0 else None,
            "max_abs": float(torch.max(torch.abs(difference)).item()) if difference.numel() else 0.0,
        }
        _accumulate_group(all_group, before, after)
        match = layer_re.match(key)
        if key.startswith("embeddings."):
            group_name = "embeddings"
            groups.setdefault(group_name, _empty_group())
            _accumulate_group(groups[group_name], before, after)
        elif match:
            layer_index = int(match.group(1))
            remainder = match.group(2)
            category = _drift_category(remainder)
            layer_name = f"layer_{layer_index}"
            layer_groups.setdefault(
                layer_name,
                {name: _empty_group() for name in ("attention", "ffn", "layernorm", "other")},
            )
            _accumulate_group(layer_groups[layer_name][category], before, after)
        else:
            groups.setdefault("other", _empty_group())
            _accumulate_group(groups["other"], before, after)
    return {
        "schema_version": "d2-weight-drift-v1",
        "dtype_for_norms": "float32_cpu",
        "parameters": parameters,
        "groups": {name: _finish_group(value) for name, value in groups.items()},
        "layers": {
            layer: {name: _finish_group(value) for name, value in categories.items()}
            for layer, categories in sorted(layer_groups.items())
        },
        "all_encoder": _finish_group(all_group),
    }


def _copy_or_verify_dataset(task_id: str, d1_dataset_path: Path, d2_task_dir: Path) -> str:
    d2_dataset_path = d2_task_dir / "dataset.json"
    source_hash = _hash_file(d1_dataset_path)
    if d2_dataset_path.exists():
        if _hash_file(d2_dataset_path) != source_hash:
            raise RuntimeError(f"Existing D2 dataset metadata differs for {task_id}")
    else:
        d2_task_dir.mkdir(parents=True, exist_ok=True)
        temporary = d2_dataset_path.with_suffix(".json.tmp")
        shutil.copyfile(d1_dataset_path, temporary)
        temporary.replace(d2_dataset_path)
    return source_hash


def _extract_one(
    task_id: str,
    model_id: str,
    selected: Mapping[str, Any],
    task: Any,
    prepared: Mapping[str, Any],
    tokenizer: Any,
    tokenizer_hashes: Mapping[str, str],
    device: torch.device,
    d2_task_dir: Path,
    d1_dataset_path: Path,
    d1_dataset_hash: str,
    force: bool,
) -> dict[str, Any]:
    output_dir = d2_task_dir / model_id
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not force:
        raise FileExistsError(f"Completed D2 cache exists at {output_dir}; use --force to rerun")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model, source, drift = _load_after_encoder(model_id, selected, device)
    batch_size = TEACHER_BATCH_SIZE if model_id == "teacher" else TRAIN_BATCH_SIZE
    arrays: dict[str, np.ndarray] = {}
    empty_rows: dict[str, int] = {}
    try:
        for split_name in ("train", "validation"):
            print(
                f"[D2 {task_id}/{model_id}] extracting {split_name} ({len(prepared[split_name].ids)} rows)",
                flush=True,
            )
            cls, mean, empty = d1._extract_split(
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
            "sha256": _hash_file(path),
            "finite": bool(np.isfinite(array).all()),
        }
    drift_path = output_dir / "weight_drift.json"
    _write_json_atomic(drift_path, drift)
    source_files = {
        "extractor": _hash_file(Path(__file__)),
        "d1_extractor": _hash_file(ROOT / "scripts" / "d1_extract_features.py"),
        "data_loader": _hash_file(ROOT / "src" / "goibert" / "data.py"),
        "d1_contract": _hash_file(ROOT / "docs" / "D1_CACHE_CONTRACT.md"),
        "d1_dataset": d1_dataset_hash,
        "teacher_gap": _hash_file(TEACHER_GAP_PATH),
        "e1_screening_approval": _hash_file(E1_SCREENING_PATH),
        "e2_teacher_selection": _hash_file(E2_TEACHER_SELECTION_PATH),
    }
    first_array = next(iter(arrays.values()))
    manifest: dict[str, Any] = {
        # Keep the D1 schema marker so D1 analysis readers can consume these
        # arrays.  d2_phase/source_checkpoint carry the additional identity.
        "schema_version": "d1-cache-v1",
        "status": "completed",
        "diagnostic_phase": "d2_after_task_finetune",
        "task_id": task_id,
        "model_id": model_id,
        "model_path": source["base_model_path"],
        "model_lineage": (
            "GoiBERT student encoder after selected E1 CE task fine-tune"
            if model_id == "student"
            else "DAPT teacher encoder after selected E2 stage0 task fine-tune"
        ),
        "checkpoint_files": source["base_model_hashes"],
        "checkpoint_digest": source["base_model_digest"],
        "tokenizer_files": dict(tokenizer_hashes),
        "tokenizer_digest": d1._tokenizer_digest(tokenizer_hashes),
        "source_hashes": source_files,
        "source_checkpoint": source,
        "input_contract": {
            "max_length": MAX_LENGTH,
            "selection_seed": SELECTION_SEED,
            "include_test": False,
            "train_count": len(prepared["train"].ids),
            "validation_count": len(prepared["validation"].ids),
            "train_input_sha256": d1._input_digest(prepared["train"]),
            "validation_input_sha256": d1._input_digest(prepared["validation"]),
            "d1_dataset_manifest": str(d1_dataset_path.relative_to(ROOT)),
            "d1_dataset_manifest_sha256": d1_dataset_hash,
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
        },
        "representations": {
            "layer_count": int(first_array.shape[0]),
            "hidden_size": int(first_array.shape[2]),
            "layer_0": "embedding_output",
            "cls": "first_token",
            "mean": "attention_valid_non_special_tokens; CLS fallback for empty rows",
            "empty_mean_rows": empty_rows,
        },
        "weight_drift": {
            "path": str(drift_path.relative_to(ROOT)),
            "sha256": _hash_file(drift_path),
            "all_encoder_relative_l2": drift["all_encoder"]["relative_l2"],
        },
        "arrays": array_metadata,
    }
    _write_json_atomic(manifest_path, manifest)
    print(
        f"[D2 {task_id}/{model_id}] completed in {time.time() - started:.1f}s "
        f"layers={manifest['representations']['layer_count']} hidden={manifest['representations']['hidden_size']} "
        f"weight_relative_l2={drift['all_encoder']['relative_l2']:.6g}",
        flush=True,
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, action="append", help="Extract only this task (repeatable)")
    parser.add_argument("--model", choices=MODEL_IDS, action="append", help="Extract only this model (repeatable)")
    parser.add_argument("--device", default="cuda:2", help="Explicit CUDA device; default is cuda:2")
    parser.add_argument("--cache-root", type=Path, default=D2_CACHE_ROOT)
    parser.add_argument("--force", action="store_true", help="Rerun into existing D2 directories")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("D2 extraction requires CUDA; CPU fallback is not an accepted run")
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("Use an explicit CUDA device such as cuda:2")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"BF16 is not supported on {device}")
    torch.cuda.set_device(device)
    tasks = tuple(args.task) if args.task else TASKS
    models = tuple(args.model) if args.model else MODEL_IDS
    selected_e1, selected_teacher = _load_selection_inputs()
    tokenizer_dirs = {model_id: d1.MODEL_DIRS[model_id] for model_id in MODEL_IDS}
    tokenizer_hashes_by_model = {
        model_id: d1._tokenizer_hashes(path) for model_id, path in tokenizer_dirs.items()
    }
    if d1._tokenizer_digest(tokenizer_hashes_by_model["student"]) != d1._tokenizer_digest(
        tokenizer_hashes_by_model["teacher"]
    ):
        raise RuntimeError("Student and teacher tokenizer assets differ")
    tokenizer = AutoTokenizer.from_pretrained(
        d1.MODEL_DIRS["student"], local_files_only=True, use_fast=True
    )
    if tokenizer.pad_token_id is None or not tokenizer.all_special_ids:
        raise RuntimeError("Tokenizer is missing pad or special token IDs")
    cache_root = _resolve(args.cache_root)
    prepared_by_task: dict[str, tuple[Any, dict[str, Any], Path, str]] = {}
    for task_id in tasks:
        task, prepared, d1_dataset_path = _validate_d1_cohort(task_id, tokenizer)
        d2_task_dir = cache_root / task_id
        dataset_hash = _copy_or_verify_dataset(task_id, d1_dataset_path, d2_task_dir)
        prepared_by_task[task_id] = (task, prepared, d1_dataset_path, dataset_hash)
        print(
            f"[D2 {task_id}] verified D1 cohort train={len(prepared['train'].ids)} "
            f"validation={len(prepared['validation'].ids)} dataset_sha256={dataset_hash[:12]}",
            flush=True,
        )

    manifests: list[dict[str, Any]] = []
    for task_id in tasks:
        task, prepared, d1_dataset_path, dataset_hash = prepared_by_task[task_id]
        selections = {"student": selected_e1[task_id], "teacher": selected_teacher[task_id]}
        for model_id in models:
            manifest = _extract_one(
                task_id,
                model_id,
                selections[model_id],
                task,
                prepared,
                tokenizer,
                tokenizer_hashes_by_model[model_id],
                device,
                cache_root / task_id,
                d1_dataset_path,
                dataset_hash,
                args.force,
            )
            manifests.append(manifest)
    _write_json_atomic(
        cache_root / "d2_extraction_summary.json",
        {
            "schema_version": "d2-after-extraction-v1",
            "status": "completed",
            "tasks": list(tasks),
            "models": list(models),
            "count": len(manifests),
            "cache_manifests": [
                str(cache_root / item["task_id"] / item["model_id"] / "manifest.json")
                for item in manifests
            ],
            "test_opened": False,
            "created_at_unix": time.time(),
        },
    )
    print("D2 after-feature extraction completed", flush=True)


if __name__ == "__main__":
    main()
