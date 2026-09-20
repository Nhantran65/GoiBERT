#!/usr/bin/env python3
"""D1 layerwise linear-CKA diagnostic for the GoiBERT student and teacher.

This script consumes the immutable feature cache produced by the D1 extraction
worker.  It deliberately has no model construction, tokenisation, test-split
loading, or encoder inference path.  The cache contract is:

    artifacts/d1/cache/<task>/<student|teacher>/
        train_cls.npy, validation_cls.npy,
        train_mean.npy, validation_mean.npy, manifest.json

Arrays are ``[layers, rows, hidden_width]``.  CLS is the primary view; the
content mean is retained as a secondary diagnostic only.  Linear CKA is
computed from column-centred representations in float64 and centred Gram
matrices.  Gram matrices make the within-model layer sweep inexpensive while
supporting different student and teacher widths.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TASKS = (
    "vsfc_sentiment",
    "vsfc_topic",
    "visfd",
    "absa_restaurant",
    "absa_hotel",
)
MODELS = ("student", "teacher")
VIEWS = ("cls", "mean")
SPLITS = ("train", "validation")
EXPECTED_LAYERS = {"student": 11, "teacher": 13}
GD_MAPPING = (0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12)
SAMPLE_SEED = 123
TRAIN_SAMPLE_SIZE = 1024
PERMUTATION_SEED = 314159


class ContractError(RuntimeError):
    """Raised when the extraction cache does not satisfy the D1 contract."""


@dataclass
class CacheView:
    task: str
    model: str
    arrays: dict[str, np.ndarray]
    manifest_paths: list[Path]
    metadata: dict[str, Any]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Cannot serialise {type(value)!r}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - message is the useful part
        raise ContractError(f"Could not parse manifest {path}: {exc}") from exc


def _json_candidates(cache_root: Path, task: str, model: str) -> list[Path]:
    model_dir = cache_root / task / model
    task_dir = cache_root / task
    candidates = [
        model_dir / "manifest.json",
        model_dir / "features_manifest.json",
        model_dir / "cache_manifest.json",
        task_dir / f"{model}_manifest.json",
        task_dir / "manifest.json",
        cache_root / "manifest.json",
        cache_root / "cache_manifest.json",
    ]
    for directory in (model_dir, task_dir):
        if directory.is_dir():
            candidates.extend(sorted(directory.glob("*.json")))
    return list(dict.fromkeys(path for path in candidates if path.is_file()))


def _contains_completion_marker(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in {"status", "state", "phase", "result"}:
                if isinstance(child, str) and child.lower() in {"completed", "complete", "success", "ready"}:
                    return True
                if child is True:
                    return True
            if _contains_completion_marker(child):
                return True
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_completion_marker(child) for child in value)
    return False


def _find_manifests(cache_root: Path, task: str, model: str) -> tuple[list[Path], dict[str, Any]]:
    paths = _json_candidates(cache_root, task, model)
    loaded: list[tuple[Path, Any]] = []
    for path in paths:
        try:
            loaded.append((path, _load_json(path)))
        except ContractError:
            continue
    if not loaded:
        raise ContractError(f"No JSON manifest found for {task}/{model} under {cache_root}")

    # Prefer the most local manifest and one mentioning this task/model.  A
    # root manifest is still retained in provenance and can carry row IDs.
    def score(item: tuple[Path, Any]) -> tuple[int, int]:
        path, payload = item
        text = json.dumps(payload, ensure_ascii=False, default=_json_default).lower()
        local = int(path.parent == cache_root / task / model)
        mentions = int(task.lower() in text) + int(model.lower() in text)
        return (local * 10 + mentions, -len(str(path)))

    chosen_path, chosen = max(loaded, key=score)
    if not isinstance(chosen, Mapping):
        chosen = {"manifest_value": chosen}
    metadata = dict(chosen)
    metadata["selected_manifest"] = str(chosen_path.resolve())
    metadata["completion_marker_found"] = any(_contains_completion_marker(payload) for _, payload in loaded)
    return [path for path, _ in loaded], metadata


def _normalise_ids(value: Any, split: str) -> list[str] | None:
    """Find explicit split row IDs without mistaking shape/count arrays for IDs."""
    if isinstance(value, Mapping):
        explicit = (
            f"{split}_ids",
            f"{split}_source_ids",
            f"{split}_row_ids",
            "source_ids",
            "row_ids",
            "ids",
        )
        for key in explicit:
            if key not in value:
                continue
            child = value[key]
            if isinstance(child, Mapping):
                for split_key in (split, split.lower(), split.capitalize()):
                    if split_key in child:
                        result = _normalise_ids(child[split_key], split)
                        if result is not None:
                            return result
            elif isinstance(child, Sequence) and not isinstance(child, (str, bytes)):
                if child and all(isinstance(item, (str, int)) for item in child):
                    return [str(item) for item in child]
        for key in ("splits", "ids_by_split", "source_ids_by_split", "rows_by_split"):
            if key in value:
                result = _normalise_ids(value[key], split)
                if result is not None:
                    return result
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if value and all(isinstance(item, (str, int)) for item in value):
            return [str(item) for item in value]
    return None


def _input_hash(value: Any, split: str) -> str | None:
    if not isinstance(value, Mapping):
        return None
    candidates = (
        f"{split}_input_sha256",
        f"{split}_text_sha256",
        f"{split}_source_sha256",
        f"{split}_hash",
    )
    for key in candidates:
        child = value.get(key)
        if isinstance(child, str) and child:
            return child
    for key in ("input_contract", "source", "inputs", "splits"):
        child = value.get(key)
        result = _input_hash(child, split)
        if result is not None:
            return result
    for child in value.values():
        if isinstance(child, Mapping):
            result = _input_hash(child, split)
            if result is not None:
                return result
    return None

def _collect_ids(manifest_paths: Iterable[Path], split: str) -> tuple[list[str] | None, str | None, str | None]:
    input_hash: str | None = None
    for path in manifest_paths:
        try:
            payload = _load_json(path)
        except ContractError:
            continue
        if input_hash is None:
            input_hash = _input_hash(payload, split)
        ids = _normalise_ids(payload, split)
        if ids is not None:
            encoded = json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            return ids, hashlib.sha256(encoded).hexdigest(), input_hash
    return None, None, input_hash

def _load_cache_view(cache_root: Path, task: str, model: str) -> CacheView:
    model_dir = cache_root / task / model
    if not model_dir.is_dir():
        raise ContractError(f"Missing feature-cache directory: {model_dir}")
    manifest_paths, metadata = _find_manifests(cache_root, task, model)
    if not metadata.get("completion_marker_found", False):
        raise ContractError(f"No completed/ready marker found in manifest for {task}/{model}")

    arrays: dict[str, np.ndarray] = {}
    for split in SPLITS:
        for view in VIEWS:
            key = f"{split}_{view}"
            path = model_dir / f"{key}.npy"
            if not path.is_file():
                raise ContractError(f"Missing required feature array: {path}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.ndim != 3:
                raise ContractError(f"{path} must have shape [layers, rows, hidden], got {array.shape}")
            if array.shape[0] != EXPECTED_LAYERS[model]:
                raise ContractError(
                    f"{path} has {array.shape[0]} layers; expected {EXPECTED_LAYERS[model]} for {model}"
                )
            if not np.isfinite(np.asarray(array[0, : min(8, array.shape[1])], dtype=np.float64)).all():
                raise ContractError(f"{path} has non-finite feature values")
            arrays[key] = array

    for split in SPLITS:
        ids, ids_hash, input_hash = _collect_ids(manifest_paths, split)
        if ids is not None and len(ids) != arrays[f"{split}_cls"].shape[1]:
            raise ContractError(
                f"{task}/{model} {split} manifest has {len(ids)} IDs, arrays have {arrays[f'{split}_cls'].shape[1]} rows"
            )
        metadata[f"{split}_ids"] = ids
        metadata[f"{split}_ids_hash"] = ids_hash
        metadata[f"{split}_input_hash"] = input_hash
    metadata["array_files"] = {
        key: str((model_dir / f"{key}.npy").resolve()) for key in arrays
    }
    metadata["array_sha256"] = {
        key: _sha256_file(model_dir / f"{key}.npy") for key in arrays
    }
    return CacheView(task, model, arrays, manifest_paths, metadata)


def _validate_pair(student: CacheView, teacher: CacheView, task: str) -> dict[str, Any]:
    checks: dict[str, Any] = {"task": task, "splits": {}}
    for split in SPLITS:
        sx = student.arrays[f"{split}_cls"]
        tx = teacher.arrays[f"{split}_cls"]
        if sx.shape[1] != tx.shape[1]:
            raise ContractError(f"{task}/{split} student/teacher row counts differ: {sx.shape} vs {tx.shape}")
        student_ids = student.metadata.get(f"{split}_ids")
        teacher_ids = teacher.metadata.get(f"{split}_ids")
        if student_ids is not None and teacher_ids is not None:
            if student_ids != teacher_ids:
                raise ContractError(f"{task}/{split} student and teacher IDs differ")
            alignment_method = "explicit_ids"
        else:
            student_hash = student.metadata.get(f"{split}_input_hash")
            teacher_hash = teacher.metadata.get(f"{split}_input_hash")
            if not student_hash or not teacher_hash or student_hash != teacher_hash:
                raise ContractError(f"{task}/{split} lacks matching explicit IDs or input hashes")
            alignment_method = "input_contract_hash"
        for view in VIEWS:
            sa = student.arrays[f"{split}_{view}"]
            ta = teacher.arrays[f"{split}_{view}"]
            if sa.shape[1] != ta.shape[1]:
                raise ContractError(f"{task}/{split}/{view} row counts differ")
            if not np.isfinite(np.asarray(sa[:, : min(8, sa.shape[1])], dtype=np.float64)).all():
                raise ContractError(f"{task}/{split}/{view} student sample contains non-finite values")
            if not np.isfinite(np.asarray(ta[:, : min(8, ta.shape[1])], dtype=np.float64)).all():
                raise ContractError(f"{task}/{split}/{view} teacher sample contains non-finite values")
        checks["splits"][split] = {
            "rows": sx.shape[1],
            "ids_match": student_ids is not None and teacher_ids is not None,
            "alignment_method": alignment_method,
            "student_ids_sha256": student.metadata[f"{split}_ids_hash"],
            "teacher_ids_sha256": teacher.metadata[f"{split}_ids_hash"],
            "student_input_hash": student.metadata.get(f"{split}_input_hash"),
            "teacher_input_hash": teacher.metadata.get(f"{split}_input_hash"),
            "student_hidden_width": sx.shape[2],
            "teacher_hidden_width": tx.shape[2],
        }
    return checks


def _sample_indices(n_rows: int, *, size: int, seed: int) -> np.ndarray:
    count = min(int(size), int(n_rows))
    if count <= 0:
        raise ContractError("Cannot sample from an empty feature cache")
    rng = np.random.default_rng(seed)
    # Sort to make row order stable and easy to audit while preserving the
    # exact same selected rows for student and teacher.
    return np.sort(rng.choice(n_rows, size=count, replace=False).astype(np.int64))


def _center_features(array: np.ndarray, indices: np.ndarray) -> np.ndarray:
    # The explicit float64 conversion is part of the numerical contract.  It
    # also prevents BF16 storage from causing a silently all-zero covariance.
    values = np.asarray(array[:, indices, :], dtype=np.float64)
    values -= values.mean(axis=1, keepdims=True)
    return values


def _gram_stack(array: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    centered = _center_features(array, indices)
    layers, rows, _ = centered.shape
    grams = np.empty((layers, rows, rows), dtype=np.float64)
    centered_norms = np.empty(layers, dtype=np.float64)
    for layer in range(layers):
        gram = centered[layer] @ centered[layer].T
        grams[layer] = gram
        centered_norms[layer] = math.sqrt(float(np.sum(gram * gram, dtype=np.float64)))
    tolerance = max(1e-12, np.finfo(np.float64).eps * max(1.0, rows))
    zero_layers = [int(i) for i, value in enumerate(centered_norms) if value <= tolerance]
    return grams, {
        "rows": int(rows),
        "hidden_width": int(array.shape[2]),
        "centered_frobenius_gram_norms": centered_norms.tolist(),
        "zero_denominator_layers": zero_layers,
        "zero_norm_tolerance": tolerance,
    }


def _cka_from_grams(left: np.ndarray, right: np.ndarray, left_norms: np.ndarray, right_norms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.full((left.shape[0], right.shape[0]), np.nan, dtype=np.float64)
    undefined = np.zeros(values.shape, dtype=bool)
    denominator_tolerance = max(1e-24, np.finfo(np.float64).eps)
    for i in range(left.shape[0]):
        for j in range(right.shape[0]):
            denominator = float(left_norms[i] * right_norms[j])
            if denominator <= denominator_tolerance:
                undefined[i, j] = True
                continue
            numerator = float(np.sum(left[i] * right[j], dtype=np.float64))
            # Small floating overshoots can put an exact self-similarity just
            # outside [0, 1].  Clip only finite, defined values.
            values[i, j] = float(np.clip(numerator / denominator, -1.0, 1.0))
    return values, undefined


def _linear_cka_matrix(array: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    grams, info = _gram_stack(array, indices)
    norms = np.asarray(info["centered_frobenius_gram_norms"], dtype=np.float64)
    values, undefined = _cka_from_grams(grams, grams, norms, norms)
    info["undefined_pairs"] = [[int(i), int(j)] for i, j in zip(*np.where(undefined))]
    return values, info


def _cross_cka(
    student: np.ndarray,
    teacher: np.ndarray,
    student_indices: np.ndarray,
    teacher_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if teacher_indices is None:
        teacher_indices = student_indices
    if len(student_indices) != len(teacher_indices):
        raise ContractError("Student and teacher CKA row selections must have the same length")
    student_grams, student_info = _gram_stack(student, student_indices)
    teacher_grams, teacher_info = _gram_stack(teacher, teacher_indices)
    student_norms = np.asarray(student_info["centered_frobenius_gram_norms"], dtype=np.float64)
    teacher_norms = np.asarray(teacher_info["centered_frobenius_gram_norms"], dtype=np.float64)
    values, undefined = _cka_from_grams(student_grams, teacher_grams, student_norms, teacher_norms)
    info = {
        "rows": int(len(student_indices)),
        "student": student_info,
        "teacher": teacher_info,
        "undefined_pairs": [[int(i), int(j)] for i, j in zip(*np.where(undefined))],
    }
    return values, info


def _pair_rows(matrix: np.ndarray, kind: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for student_layer, teacher_layer in enumerate(GD_MAPPING):
        value = float(matrix[student_layer, teacher_layer])
        rows.append(
            {
                "student_layer": student_layer,
                "teacher_layer": teacher_layer,
                "value": value if math.isfinite(value) else None,
                "kind": kind,
            }
        )
    return rows


def _row_best_teacher_layer_gap(matrix: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for student_layer, declared_teacher_layer in enumerate(GD_MAPPING):
        values = matrix[student_layer]
        finite = np.isfinite(values)
        declared = float(values[declared_teacher_layer])
        if not finite.any():
            rows.append(
                {
                    "student_layer": student_layer,
                    "declared_teacher_layer": declared_teacher_layer,
                    "best_teacher_layer": None,
                    "best_value": None,
                    "declared_value": None,
                    "best_minus_declared": None,
                }
            )
            continue
        best_teacher_layer = int(np.nanargmax(values))
        best_value = float(values[best_teacher_layer])
        rows.append(
            {
                "student_layer": student_layer,
                "declared_teacher_layer": declared_teacher_layer,
                "best_teacher_layer": best_teacher_layer,
                "best_value": best_value,
                "declared_value": declared if math.isfinite(declared) else None,
                "best_minus_declared": (best_value - declared) if math.isfinite(declared) else None,
            }
        )
    return rows

def _finite_summary(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"finite_count": 0, "mean": None, "min": None, "max": None}
    return {
        "finite_count": int(finite.size),
        "mean": float(np.mean(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _sanity_checks() -> dict[str, Any]:
    rng = np.random.default_rng(7)
    base = rng.normal(size=(64, 9)).astype(np.float64)

    def scalar_cka(left: np.ndarray, right: np.ndarray) -> float | None:
        left_c = left.astype(np.float64) - left.astype(np.float64).mean(axis=0, keepdims=True)
        right_c = right.astype(np.float64) - right.astype(np.float64).mean(axis=0, keepdims=True)
        left_g = left_c @ left_c.T
        right_g = right_c @ right_c.T
        denom = math.sqrt(float(np.sum(left_g * left_g)) * float(np.sum(right_g * right_g)))
        if denom <= 1e-24:
            return None
        return float(np.sum(left_g * right_g) / denom)

    rotation, _ = np.linalg.qr(rng.normal(size=(9, 9)))
    checks = {
        "identity": scalar_cka(base, base),
        "orthogonal_rotation": scalar_cka(base, base @ rotation),
        "scaling": scalar_cka(base, 11.0 * base),
        "constant_undefined": scalar_cka(np.ones((64, 9)), base),
    }
    checks["pass"] = bool(
        checks["identity"] is not None
        and abs(float(checks["identity"]) - 1.0) < 1e-12
        and checks["orthogonal_rotation"] is not None
        and abs(float(checks["orthogonal_rotation"]) - 1.0) < 1e-12
        and checks["scaling"] is not None
        and abs(float(checks["scaling"]) - 1.0) < 1e-12
        and checks["constant_undefined"] is None
    )
    return checks


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    import struct
    import zlib

    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def _write_png(path: Path, pixels: np.ndarray) -> None:
    import struct
    import zlib

    pixels = np.asarray(pixels, dtype=np.uint8)
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError(f"Expected RGB pixels, got {pixels.shape}")
    raw = b"".join(b"\x00" + pixels[row].tobytes() for row in range(pixels.shape[0]))
    payload = b"\x89PNG\r\n\x1a\n"
    payload += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", pixels.shape[1], pixels.shape[0], 8, 2, 0, 0, 0))
    payload += _png_chunk(b"IDAT", zlib.compress(raw, 6))
    payload += _png_chunk(b"IEND", b"")
    path.write_bytes(payload)


def _viridis(value: float | None) -> tuple[int, int, int]:
    if value is None or not math.isfinite(value):
        return (190, 190, 190)
    anchors = np.asarray(
        [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]],
        dtype=np.float64,
    )
    position = float(np.clip(value, 0.0, 1.0)) * (len(anchors) - 1)
    left = int(math.floor(position))
    right = min(left + 1, len(anchors) - 1)
    fraction = position - left
    rgb = (1.0 - fraction) * anchors[left] + fraction * anchors[right]
    return tuple(int(round(item)) for item in rgb)


def _draw_rect(pixels: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    h, w, _ = pixels.shape
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x0 < x1 and y0 < y1:
        pixels[y0:y1, x0:x1, :] = np.asarray(color, dtype=np.uint8)


def _draw_line(pixels: np.ndarray, points: Sequence[tuple[int, int]], color: tuple[int, int, int], width: int = 2) -> None:
    if len(points) < 2:
        return
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for step in range(steps + 1):
            x = int(round(x0 + (x1 - x0) * step / steps))
            y = int(round(y0 + (y1 - y0) * step / steps))
            _draw_rect(pixels, x - width // 2, y - width // 2, x + width // 2 + 1, y + width // 2 + 1, color)


def _svg_escape(value: Any) -> str:
    from xml.sax.saxutils import escape

    return escape(str(value))


def _plot_heatmap(matrix: np.ndarray, path_png: Path, path_svg: Path, title: str, x_label: str, y_label: str, x_ticks: Sequence[str], y_ticks: Sequence[str]) -> None:
    rows, cols = matrix.shape
    cell_w, cell_h = 42, 32
    left, top, right, bottom = 86, 54, 18, 74
    width, height = left + cols * cell_w + right, top + rows * cell_h + bottom
    pixels = np.full((height, width, 3), 255, dtype=np.uint8)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{width/2:.1f}" y="24" text-anchor="middle" font-family="sans-serif" font-size="15">{_svg_escape(title)}</text>']
    for i in range(rows):
        for j in range(cols):
            value = float(matrix[i, j])
            color = _viridis(value if math.isfinite(value) else None)
            x0, y0 = left + j * cell_w, top + i * cell_h
            _draw_rect(pixels, x0, y0, x0 + cell_w - 1, y0 + cell_h - 1, color)
            fill = "#%02x%02x%02x" % color
            label = "NA" if not math.isfinite(value) else f"{value:.2f}"
            text_color = "black" if math.isfinite(value) and value > 0.55 else "white"
            svg.append(f'<rect x="{x0}" y="{y0}" width="{cell_w-1}" height="{cell_h-1}" fill="{fill}"/>')
            svg.append(f'<text x="{x0+cell_w/2:.1f}" y="{y0+cell_h/2+4:.1f}" text-anchor="middle" font-family="sans-serif" font-size="9" fill="{text_color}">{label}</text>')
    for j, label in enumerate(x_ticks):
        x = left + j * cell_w + cell_w / 2
        svg.append(f'<text x="{x:.1f}" y="{top+rows*cell_h+16}" text-anchor="end" transform="rotate(-45 {x:.1f},{top+rows*cell_h+16})" font-family="sans-serif" font-size="10">{_svg_escape(label)}</text>')
    for i, label in enumerate(y_ticks):
        y = top + i * cell_h + cell_h / 2 + 4
        svg.append(f'<text x="{left-8}" y="{y:.1f}" text-anchor="end" font-family="sans-serif" font-size="10">{_svg_escape(label)}</text>')
    svg.extend([f'<text x="{left+cols*cell_w/2:.1f}" y="{height-10}" text-anchor="middle" font-family="sans-serif" font-size="11">{_svg_escape(x_label)}</text>', f'<text x="16" y="{top+rows*cell_h/2:.1f}" text-anchor="middle" transform="rotate(-90 16,{top+rows*cell_h/2:.1f})" font-family="sans-serif" font-size="11">{_svg_escape(y_label)}</text>', '</svg>'])
    path_png.parent.mkdir(parents=True, exist_ok=True)
    _write_png(path_png, pixels)
    path_svg.write_text("\n".join(svg) + "\n", encoding="utf-8")


def _plot_pairs(rows_matched: list[dict[str, Any]], rows_permuted: list[dict[str, Any]], path_png: Path, path_svg: Path, task: str) -> None:
    width, height = 900, 390
    left, top, right, bottom = 62, 42, 20, 74
    plot_w, plot_h = width - left - right, height - top - bottom
    pixels = np.full((height, width, 3), 255, dtype=np.uint8)
    x_values = np.linspace(left, left + plot_w, len(GD_MAPPING))
    def y_value(value: float | None) -> int | None:
        return None if value is None or not math.isfinite(value) else int(round(top + (1.0 - value) * plot_h))
    series = [(rows_matched, (31, 119, 180), "matched rows"), (rows_permuted, (214, 39, 40), "same rows, teacher permuted")]
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{width/2}" y="22" text-anchor="middle" font-family="sans-serif" font-size="15">{_svg_escape(task)}: CKA on declared GD layer pairs (train sample)</text>']
    for tick in (0.0, 0.5, 1.0):
        y = int(round(top + (1.0 - tick) * plot_h))
        _draw_line(pixels, [(left, y), (left+plot_w, y)], (220, 220, 220), 1)
        svg.append(f'<line x1="{left}" y1="{y}" x2="{left+plot_w}" y2="{y}" stroke="#dddddd"/>')
        svg.append(f'<text x="{left-8}" y="{y+4}" text-anchor="end" font-family="sans-serif" font-size="10">{tick:.1f}</text>')
    for rows, color, label in series:
        points = [(int(round(x)), y) for x, row in zip(x_values, rows) if (y := y_value(row["value"])) is not None]
        _draw_line(pixels, points, color, 2)
        if points:
            svg.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in points)}" fill="none" stroke="rgb{color}" stroke-width="2"/>')
        for x, row in zip(x_values, rows):
            y = y_value(row["value"])
            if y is not None:
                _draw_rect(pixels, int(round(x))-3, y-3, int(round(x))+4, y+4, color)
                svg.append(f'<circle cx="{x:.1f}" cy="{y}" r="3" fill="rgb{color}"/>')
    for i, layer in enumerate(GD_MAPPING):
        x = x_values[i]
        svg.append(f'<text x="{x:.1f}" y="{top+plot_h+20}" text-anchor="end" transform="rotate(-45 {x:.1f},{top+plot_h+20})" font-family="sans-serif" font-size="10">S{i}/T{layer}</text>')
    svg.extend([f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="black"/>', f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="black"/>', '</svg>'])
    path_png.parent.mkdir(parents=True, exist_ok=True)
    _write_png(path_png, pixels)
    path_svg.write_text("\n".join(svg) + "\n", encoding="utf-8")


def _plot_norms(info: Mapping[str, Any], path_png: Path, path_svg: Path, title: str) -> None:
    width, height = 900, 330
    left, top, right, bottom = 62, 42, 20, 52
    plot_w, plot_h = width - left - right, height - top - bottom
    pixels = np.full((height, width, 3), 255, dtype=np.uint8)
    series = [("student", np.asarray(info["student"]["centered_frobenius_gram_norms"], dtype=np.float64), (31,119,180)), ("teacher", np.asarray(info["teacher"]["centered_frobenius_gram_norms"], dtype=np.float64), (214,39,40))]
    all_values = np.concatenate([values for _, values, _ in series])
    transformed = np.sign(all_values) * np.log10(1.0 + np.abs(all_values))
    max_value = max(float(np.max(np.abs(transformed))), 1e-12)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{width/2}" y="22" text-anchor="middle" font-family="sans-serif" font-size="15">{_svg_escape(title)}</text>']
    for label, values, color in series:
        vals = np.sign(values) * np.log10(1.0 + np.abs(values))
        xs = np.linspace(left, left + plot_w, len(values))
        points = [(int(round(x)), int(round(top + (1.0 - (v / max_value + 1.0) / 2.0) * plot_h))) for x, v in zip(xs, vals)]
        _draw_line(pixels, points, color, 2)
        svg.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in points)}" fill="none" stroke="rgb{color}" stroke-width="2"/>')
        for x, y in points:
            _draw_rect(pixels, x-3, y-3, x+4, y+4, color)
            svg.append(f'<circle cx="{x}" cy="{y}" r="3" fill="rgb{color}"/>')
        svg.append(f'<text x="{left+plot_w-100}" y="{top+18+18*len(svg)}" font-family="sans-serif" font-size="10" fill="rgb{color}">{label}</text>')
    svg.extend([f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="black"/>', f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="black"/>', '</svg>'])
    path_png.parent.mkdir(parents=True, exist_ok=True)
    _write_png(path_png, pixels)
    path_svg.write_text("\n".join(svg) + "\n", encoding="utf-8")

def _analyse_task(task: str, cache_root: Path, output_root: Path, sample_size: int, seed: int) -> dict[str, Any]:
    student = _load_cache_view(cache_root, task, "student")
    teacher = _load_cache_view(cache_root, task, "teacher")
    contract = _validate_pair(student, teacher, task)
    train_rows = student.arrays["train_cls"].shape[1]
    train_indices = _sample_indices(train_rows, size=sample_size, seed=seed)
    permutation = np.random.default_rng(PERMUTATION_SEED).permutation(len(train_indices)).astype(np.int64)
    permuted_teacher_indices = train_indices[permutation]
    task_out = output_root / task
    task_out.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "task": task,
        "cache_root": str(cache_root.resolve()),
        "analysis_split": "train",
        "sample": {
            "seed": seed,
            "requested_train_rows": sample_size,
            "train_indices": train_indices.tolist(),
            "train_rows_used": int(len(train_indices)),
            "row_permutation_seed": PERMUTATION_SEED,
            "row_permutation": permutation.tolist(),
            "permuted_teacher_indices": permuted_teacher_indices.tolist(),
        },
        "contract": contract,
        "models": {"student": student.metadata, "teacher": teacher.metadata},
        "views": {},
    }

    for view in VIEWS:
        train_cross, train_cross_info = _cross_cka(
            student.arrays[f"train_{view}"], teacher.arrays[f"train_{view}"], train_indices
        )
        train_permuted, train_permuted_info = _cross_cka(
            student.arrays[f"train_{view}"],
            teacher.arrays[f"train_{view}"],
            train_indices,
            permuted_teacher_indices,
        )
        train_student_within, train_student_info = _linear_cka_matrix(
            student.arrays[f"train_{view}"], train_indices
        )
        train_teacher_within, train_teacher_info = _linear_cka_matrix(
            teacher.arrays[f"train_{view}"], train_indices
        )
        train_pairs = _pair_rows(train_cross, "matched")
        permuted_pairs = _pair_rows(train_permuted, "row_permuted")
        matched_values = np.asarray([row["value"] for row in train_pairs if row["value"] is not None], dtype=np.float64)
        permuted_values = np.asarray([row["value"] for row in permuted_pairs if row["value"] is not None], dtype=np.float64)
        if matched_values.size == 0 or permuted_values.size == 0:
            raise ContractError(f"{task}/{view} row-permutation control has no finite mapped pairs")
        control_delta = float(np.mean(matched_values) - np.mean(permuted_values))
        if not math.isfinite(control_delta) or abs(control_delta) <= 1e-3:
            raise ContractError(
                f"{task}/{view} row-permutation control is not materially different: delta={control_delta}"
            )
        view_result: dict[str, Any] = {
            "primary": view == "cls",
            "train_cross": train_cross.tolist(),
            "train_row_permuted_cross": train_permuted.tolist(),
            "train_student_within": train_student_within.tolist(),
            "train_teacher_within": train_teacher_within.tolist(),
            "train_cross_info": train_cross_info,
            "train_row_permuted_cross_info": train_permuted_info,
            "train_student_within_info": train_student_info,
            "train_teacher_within_info": train_teacher_info,
            "gd_pair_rows_train": train_pairs,
            "gd_pair_rows_row_permuted": permuted_pairs,
            "row_best_teacher_layer_gap": _row_best_teacher_layer_gap(train_cross),
            "train_cross_summary": _finite_summary(train_cross),
            "train_row_permuted_summary": _finite_summary(train_permuted),
            "row_permutation_control": {
                "matched_mapped_mean": float(np.mean(matched_values)),
                "row_permuted_mapped_mean": float(np.mean(permuted_values)),
                "matched_minus_row_permuted": control_delta,
                "materially_different": True,
            },
        }
        _write_json(task_out / f"{view}_cka.json", view_result)

        labels_s = [f"S{i}" for i in range(EXPECTED_LAYERS["student"])]
        labels_t = [f"T{i}" for i in range(EXPECTED_LAYERS["teacher"])]
        _plot_heatmap(
            train_cross,
            task_out / f"{view}_cross_train.png",
            task_out / f"{view}_cross_train.svg",
            f"{task}: {view} student–teacher CKA (train sample)",
            "teacher layer",
            "student layer",
            labels_t,
            labels_s,
        )
        _plot_heatmap(
            train_student_within,
            task_out / f"{view}_within_student_train.png",
            task_out / f"{view}_within_student_train.svg",
            f"{task}: {view} within-student CKA (train sample)",
            "student layer",
            "student layer",
            labels_s,
            labels_s,
        )
        _plot_heatmap(
            train_teacher_within,
            task_out / f"{view}_within_teacher_train.png",
            task_out / f"{view}_within_teacher_train.svg",
            f"{task}: {view} within-teacher CKA (train sample)",
            "teacher layer",
            "teacher layer",
            labels_t,
            labels_t,
        )
        _plot_pairs(
            train_pairs,
            permuted_pairs,
            task_out / f"{view}_gd_pairs.png",
            task_out / f"{view}_gd_pairs.svg",
            task,
        )
        _plot_norms(
            train_cross_info,
            task_out / f"{view}_norms_train.png",
            task_out / f"{view}_norms_train.svg",
            f"{task}: {view} centered representation norms",
        )
        result["views"][view] = view_result
        del train_cross, train_permuted, train_student_within, train_teacher_within
        del train_cross_info, train_permuted_info, train_student_info, train_teacher_info
        gc.collect()
    _write_json(task_out / "result.json", result)
    return result

def _markdown_report(results: list[dict[str, Any]], out_path: Path, cache_root: Path, sample_size: int, seed: int, sanity: Mapping[str, Any]) -> None:
    lines = [
        "# D1 CKA layer diagnostic",
        "",
        "Ngày chạy: " + datetime.now(timezone.utc).isoformat(),
        "",
        "Phân tích này chỉ đọc feature cache; không load test, không khởi tạo model và không chạy encoder inference.",
        "",
        "## Contract và phương pháp",
        "",
        f"- Cache: `{cache_root.resolve()}`; mỗi mảng có dạng `[layer, row, hidden]`.",
        "- CKA chính dùng CLS; mean pooling là view phụ để kiểm tra độ nhạy của kết luận.",
        f"- CKA và within-model matrices chỉ dùng tối đa {sample_size} dòng train, chọn cố định bằng seed {seed}.",
        "- Validation IDs vẫn được kiểm tra để xác nhận cùng hàng giữa student và teacher, nhưng không dùng để tính CKA trong pass này.",
        "- Student có 11 layer (embedding + 10 transformer), teacher có 13 layer (embedding + 12 transformer).",
        f"- Mapping GD được khai báo trước: `{list(GD_MAPPING)}`. Các cặp này chỉ được báo cáo, không được xem là mapping tối ưu do CKA.",
        f"- Reference control: cùng các dòng nhưng teacher bị permute bằng seed {PERMUTATION_SEED}; đây là reference hữu ích cho hiệu ứng finite-sample, không phải p-value hay permutation-test grid.",
        "- CKA là similarity bất biến với phép biến đổi trực giao và scale; similarity thấp không tự chứng minh bottleneck có thể sửa.",
        "",
        "## Sanity checks",
        "",
        f"`{json.dumps(dict(sanity), ensure_ascii=False)}`",
        "",
        "Identity, orthogonal rotation và scaling phải cho CKA gần 1; biểu diễn hằng phải cho undefined/NaN do mẫu số bằng 0. NaN ở layer embedding CLS vì vậy là cờ zero-denominator, không bị đổi thành 0.",
        "",
        "## CKA theo cặp GD (CLS, primary; train sample)",
        "",
        "| Task | Matched mean | Row-permuted mean | Matched range | Matched minus permuted | Undefined pairs |\n|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        view = result["views"]["cls"]
        matched = [row["value"] for row in view["gd_pair_rows_train"] if row["value"] is not None]
        permuted = [row["value"] for row in view["gd_pair_rows_row_permuted"] if row["value"] is not None]
        undefined = len(view["train_cross_info"]["undefined_pairs"])
        matched_mean = float(np.mean(matched)) if matched else float("nan")
        permuted_mean = float(np.mean(permuted)) if permuted else float("nan")
        lines.append(
            f"| {result['task']} | {matched_mean:.4f} | {permuted_mean:.4f} | "
            f"{min(matched) if matched else float('nan'):.4f}–{max(matched) if matched else float('nan'):.4f} | "
            f"{matched_mean - permuted_mean:.4f} | {undefined} |"
        )
    lines.extend(
        [
            "",
            "## Reading the evidence",
            "",
            "The CLS cross-CKA matrices are primary. Mean-pooling matrices are a secondary coverage view and must not be used to select a pooling strategy.",
            "",
            "Within-model CKA is reported only on the fixed train sample. It describes redundancy/similarity among layer representations, not task quality. The row-wise best teacher layer and gap are retained in each JSON artifact for audit, but this report does not turn them into a recommended GD mapping.",
            "",
            "Finite-sample caveat: student and teacher widths differ (384 vs 768), and CKA is computed on the same rows but does not directly measure functional equivalence or task accuracy. Any intervention should require agreement with the layerwise probe curves before being proposed.",
            "",
            "## Per-task artifact index",
            "",
        ]
    )
    for result in results:
        lines.append(f"- `{result['task']}`: `artifacts/d1/cka/{result['task']}/result.json`, train-only CLS/mean cross and within-model heatmaps (`.png` and `.svg`).")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--train-sample-size", type=int, default=TRAIN_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    cache_root = (args.cache_root or repo_root / "artifacts" / "d1" / "cache").resolve()
    output_root = (args.output_root or repo_root / "artifacts" / "d1" / "cka").resolve()
    report_path = (args.report or repo_root / "docs" / "D1_CKA.md").resolve()
    if args.train_sample_size <= 0:
        raise SystemExit("--train-sample-size must be positive")
    output_root.mkdir(parents=True, exist_ok=True)
    sanity = _sanity_checks()
    if not sanity["pass"]:
        raise SystemExit(f"CKA analytic sanity checks failed: {sanity}")

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for task in args.tasks:
        try:
            print(f"[D1 CKA] analysing {task}", flush=True)
            results.append(_analyse_task(task, cache_root, output_root, args.train_sample_size, args.seed))
        except Exception as exc:
            failures.append({"task": task, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[D1 CKA] FAILED {task}: {exc}", file=sys.stderr, flush=True)
            if len(args.tasks) == 1:
                raise

    summary = {
        "schema": "goibert-d1-cka-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cache_root": str(cache_root),
        "output_root": str(output_root),
        "report": str(report_path),
        "analysis_script_sha256": _sha256_file(Path(__file__).resolve()),
        "tasks_requested": list(args.tasks),
        "tasks_completed": [result["task"] for result in results],
        "failures": failures,
        "train_sample_size_requested": args.train_sample_size,
        "sample_seed": args.seed,
        "row_permutation_seed": PERMUTATION_SEED,
        "analysis_split": "train",
        "gd_mapping": list(GD_MAPPING),
        "expected_layers": EXPECTED_LAYERS,
        "sanity_checks": sanity,
        "no_test_split_loaded": True,
        "no_encoder_inference": True,
    }
    _write_json(output_root / "summary.json", summary)
    if results:
        _markdown_report(results, report_path, cache_root, args.train_sample_size, args.seed, sanity)
    if failures:
        raise SystemExit(f"D1 CKA completed with failures: {failures}")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
