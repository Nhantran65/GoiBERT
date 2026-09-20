"""Immutable train-only teacher-logit caches for E2.

The cache is deliberately a small, self-describing ``torch.save`` payload.
Only ordered source IDs and fp32 logits are stored.  Labels, validation rows
and test rows never enter the payload.  A cache is identified by the complete
input contract (task, IDs/text hash, tokenizer, max length, teacher
checkpoint and slot schema); an existing path is never silently overwritten.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from goibert.train import capture_rng_state, make_autocast, _move_batch


CACHE_SCHEMA = "goibert-e2-train-logits-v1"


def _stable_hash(values: Sequence[Any]) -> str:
    encoded = json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _text_hash(texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = str(text).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def text_hash(texts: Sequence[str]) -> str:
    """Stable hash of the ordered train text used to bind a cache."""

    return _text_hash(texts)


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class TeacherLogitCache:
    """Validated cache view; logits are always CPU fp32."""

    path: Path
    metadata: dict[str, Any]
    logits: Tensor

    @property
    def ids(self) -> tuple[Any, ...]:
        return tuple(self.metadata["source_ids"])

    @property
    def task_id(self) -> str:
        return str(self.metadata["task_id"])

    def lookup(self, ids: Sequence[Any], *, device: torch.device | None = None) -> Tensor:
        positions = self.metadata.get("id_positions")
        if not isinstance(positions, Mapping):
            positions = {str(value): index for index, value in enumerate(self.ids)}
        try:
            indices = [int(positions[str(value)]) for value in ids]
        except KeyError as error:
            raise ValueError(f"Teacher-logit cache is missing source ID {error.args[0]!r}") from error
        values = self.logits[indices]
        return values.to(device) if device is not None else values


def _validate_payload(payload: Any, path: Path) -> TeacherLogitCache:
    if not isinstance(payload, Mapping):
        raise ValueError(f"Teacher cache {path} is not a mapping payload")
    if payload.get("schema") != CACHE_SCHEMA:
        raise ValueError(f"Teacher cache {path} has an unsupported schema")
    metadata = dict(payload.get("metadata", {}))
    required = {
        "task_id",
        "split",
        "source_ids",
        "source_ids_hash",
        "source_text_hash",
        "tokenizer_hashes",
        "max_length",
        "teacher_hash",
        "aspect_class_counts",
        "num_aspects",
        "num_labels",
        "data_provenance",
        "inference_precision",
        "e1_dependency_source_hash",
        "source_hashes",
    }
    missing = sorted(required.difference(metadata))
    if missing:
        raise ValueError(f"Teacher cache {path} is missing metadata fields: {missing}")
    if metadata.get("split") != "train":
        raise ValueError("Teacher-logit caches may only contain the train split")
    for forbidden in ("labels", "gold_labels", "test", "validation", "valid"):
        if forbidden in payload or forbidden in metadata:
            raise ValueError(f"Teacher cache {path} contains forbidden field {forbidden!r}")
    ids = list(metadata["source_ids"])
    if metadata["source_ids_hash"] != _stable_hash(ids):
        raise ValueError(f"Teacher cache {path} source ID hash does not match IDs")
    logits = payload.get("logits")
    if not isinstance(logits, Tensor):
        raise ValueError(f"Teacher cache {path} logits are not a tensor")
    if logits.ndim < 2 or logits.shape[0] != len(ids):
        raise ValueError(f"Teacher cache {path} logits and source IDs are misaligned")
    logits = logits.detach().cpu().float().contiguous()
    if not torch.isfinite(logits).all():
        raise ValueError(f"Teacher cache {path} contains non-finite logits")
    metadata["id_positions"] = {str(value): index for index, value in enumerate(ids)}
    return TeacherLogitCache(path=path, metadata=metadata, logits=logits)


def load_teacher_logit_cache(path: Path) -> TeacherLogitCache:
    """Load and validate an immutable cache without touching any dataset split."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Teacher-logit cache does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return _validate_payload(payload, path)


def _cache_identity(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _jsonable(metadata[key])
        for key in (
            "task_id",
            "split",
            "source_ids",
            "source_ids_hash",
            "source_text_hash",
            "tokenizer_hashes",
            "max_length",
            "teacher_hash",
            "aspect_class_counts",
            "num_aspects",
            "num_labels",
            "data_provenance",
            "inference_precision",
            "e1_dependency_source_hash",
            "source_hashes",
        )
    }


def _assert_identity(existing: TeacherLogitCache, requested: Mapping[str, Any]) -> None:
    old = _cache_identity(existing.metadata)
    new = _cache_identity(requested)
    if old != new:
        raise FileExistsError(
            "Refusing to overwrite an existing teacher-logit cache with a different contract"
        )


def build_teacher_logit_cache(
    path: Path,
    *,
    teacher: nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    precision: str,
    task_id: str,
    source_ids: Sequence[Any],
    source_texts: Sequence[str],
    tokenizer_hashes: Mapping[str, str],
    max_length: int,
    teacher_hash: str,
    aspect_class_counts: Sequence[int],
    num_aspects: int,
    num_labels: int,
    e1_dependency_source_hash: str | None = None,
    data_provenance: Mapping[str, Any] | None = None,
    inference_precision: str | None = None,
    source_hashes: Mapping[str, str] | None = None,
) -> TeacherLogitCache:
    """Materialize ordered train logits and atomically create an immutable cache."""

    path = Path(path)
    ids = list(source_ids)
    texts = list(source_texts)
    if len(ids) != len(texts):
        raise ValueError("source_ids and source_texts must have equal lengths")
    if not ids:
        raise ValueError("Cannot create a teacher-logit cache for an empty train split")
    if len(set(str(value) for value in ids)) != len(ids):
        raise ValueError("source_ids must be unique for cache alignment")
    metadata: dict[str, Any] = {
        "task_id": str(task_id),
        "split": "train",
        "source_ids": ids,
        "source_ids_hash": _stable_hash(ids),
        "source_text_hash": _text_hash(texts),
        "tokenizer_hashes": dict(tokenizer_hashes),
        "max_length": int(max_length),
        "teacher_hash": str(teacher_hash),
        "aspect_class_counts": [int(value) for value in aspect_class_counts],
        "num_aspects": int(num_aspects),
        "num_labels": int(num_labels),
        "data_provenance": dict(data_provenance or {}),
        "inference_precision": str(inference_precision or precision),
        "e1_dependency_source_hash": str(e1_dependency_source_hash or ""),
        "source_hashes": dict(source_hashes or {}),
    }
    if path.exists():
        return _assert_existing_cache(path, metadata)

    before = capture_rng_state()
    try:
        teacher.eval()
        values: list[Tensor] = []
        observed_ids: list[Any] = []
        with torch.inference_mode():
            for batch in loader:
                moved = _move_batch(batch, device)
                with make_autocast(precision, device):
                    logits = teacher(
                        moved["input_ids"],
                        moved["attention_mask"],
                        token_type_ids=moved.get("token_type_ids"),
                    )
                logits = logits.detach().float().cpu()
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Teacher produced non-finite train logits")
                values.append(logits)
                observed_ids.extend(batch["ids"])
        if observed_ids != ids:
            raise ValueError("Ordered teacher cache loader IDs differ from source_ids")
        logits = torch.cat(values, dim=0)
        if logits.shape[0] != len(ids) or logits.shape[1] != int(num_labels):
            # Uniform ABSA heads return [N, aspects, classes], so compare the
            # flattened class product rather than requiring a rank-2 tensor.
            if logits.shape[0] != len(ids) or int(torch.tensor(logits.shape[1:]).prod()) != int(num_labels):
                raise ValueError(
                    f"Teacher logits shape {tuple(logits.shape)} does not match source schema"
                )
        payload = {
            "schema": CACHE_SCHEMA,
            "metadata": metadata,
            "logits": logits,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        if temporary.exists():
            raise FileExistsError(f"Temporary teacher cache path already exists: {temporary}")
        torch.save(payload, temporary)
        # ``link`` is an exclusive commit on the same filesystem: unlike
        # ``replace``, it can never overwrite a cache created concurrently.
        try:
            # The destination is created with an exclusive hard-link.  If a
            # process already committed the identity, this raises without
            # touching its bytes.
            path.hardlink_to(temporary)
            temporary.unlink(missing_ok=True)
        except FileExistsError:
            temporary.unlink(missing_ok=True)
            return _assert_existing_cache(path, metadata)
        return _validate_payload(payload, path)
    finally:
        _restore_rng_state(before)


def _assert_existing_cache(path: Path, requested: Mapping[str, Any]) -> TeacherLogitCache:
    existing = load_teacher_logit_cache(path)
    _assert_identity(existing, requested)
    return existing


# Readable compatibility alias for callers that prefer a verb phrase.
create_teacher_logit_cache = build_teacher_logit_cache
