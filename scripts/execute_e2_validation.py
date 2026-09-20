"""Durable, validation-only orchestration for the E2 training ablation.

The E2 runner is intentionally a separate process from this scheduler.  This
module owns the queue, stage gates, selection manifests, provenance hashes and
recovery rules; ``goibert_e2.train`` owns one training or cache operation.

The default command is a dry run.  A caller must pass ``--execute`` to spawn a
runner.  There is no test stage and this script never requests or evaluates a
test split.

The runner contract is documented in ``docs/E2_ORCHESTRATION.md``.  Keeping
the contract here in one place lets the training implementation evolve without
silently changing the experiment queue.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


PROTOCOL = "E2-training-ablation-v1"
REVISION = "e2-20260908-r1"
RUNNER_MODULE = "goibert_e2.train"
E1_DEPENDENCY_HASH = "456e3020bd405ec39c7b4cbab3255d0bb3cb2a8a4bda3e37cc0620d51b559873"

TASK_IDS = (
    "vsfc_sentiment",
    "vsfc_topic",
    "visfd",
    "absa_restaurant",
    "absa_hotel",
)
CATEGORICAL_TASKS = frozenset(("vsfc_sentiment", "vsfc_topic"))
ABSA_TASKS = frozenset(set(TASK_IDS) - set(CATEGORICAL_TASKS))
TEACHER_LEARNING_RATES = (1e-5, 2e-5, 5e-5)
STUDENT_LEARNING_RATE = {
    "vsfc_sentiment": 1e-5,
    "vsfc_topic": 1e-5,
    "visfd": 1e-4,
    "absa_restaurant": 1e-4,
    "absa_hotel": 1e-4,
}
KD_ALPHAS = (0.25, 0.5, 0.75)
NC_LAMBDAS = (0.1, 0.5)
STAGE0_SEED = 42
STAGEA_SEED = 42
STAGEB_SEEDS = (43, 44)
ALL_STAGEB_SEEDS = (42, 43, 44)
TEMPERATURE = 2.0
EPOCHS = 30
EFFECTIVE_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 32
TEACHER_MICRO_BATCH_SIZE = 8
PRECISION = "bf16"
ATTENTION_IMPLEMENTATION = "sdpa"
STUDENT_MIN_FREE_MEMORY_GIB = 5.0
TEACHER_MIN_FREE_MEMORY_GIB = 5.0
NC_MIN_FREE_MEMORY_GIB = 5.0
DEFAULT_DEVICES = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
ACTIVE_STUDENT_MEMORY_GUARD_GIB = STUDENT_MIN_FREE_MEMORY_GIB
ACTIVE_TEACHER_MEMORY_GUARD_GIB = TEACHER_MIN_FREE_MEMORY_GIB
ACTIVE_NC_MEMORY_GUARD_GIB = NC_MIN_FREE_MEMORY_GIB


class OrchestrationBlocked(RuntimeError):
    """The scheduler stopped safely and needs a review or a new invocation."""


@dataclass(frozen=True)
class TrialSpec:
    """One immutable identity in an E2 queue."""

    stage: str
    task_id: str
    method: str
    learning_rate: float
    seed: int
    coefficient: float | None
    expected_dir: str

    @property
    def key(self) -> tuple[str, str, str, float, int, float | None]:
        return (
            self.stage,
            self.task_id,
            self.method,
            float(self.learning_rate),
            int(self.seed),
            None if self.coefficient is None else float(self.coefficient),
        )

    def slug(self) -> str:
        return (
            f"{self.stage}__{self.task_id}__{self.method}__"
            f"{lr_slug(self.learning_rate)}__{coefficient_slug(self.coefficient)}"
            f"__seed{self.seed}"
        )


def lr_slug(value: float) -> str:
    return f"{float(value):.0e}"


def coefficient_slug(value: float | None) -> str:
    if value is None:
        return "none"
    # Stable, human-readable names that do not depend on locale or repr.
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return "coef_" + text.replace("-", "m").replace(".", "p")


def _stage_dir(root: Path, stage: str, task_id: str, method: str, coefficient: float | None) -> Path:
    if stage == "stage0_teacher":
        return root / "stage0_teacher" / task_id / lr_slug(float(coefficient))
    if stage == "stage0_cache":
        return root / "stage0_cache" / task_id / f"seed_{STAGE0_SEED}"
    if stage == "stagea_kd":
        return root / "stageA" / "kd" / task_id / coefficient_slug(coefficient)
    if stage == "stagea_nc":
        return root / "stageA" / "nc" / task_id / coefficient_slug(coefficient)
    if stage == "stagea_ce_replay":
        return root / "stageA" / "ce_replay" / task_id
    if stage == "stageb":
        return root / "stageB" / method / task_id / coefficient_slug(coefficient)
    raise ValueError(f"Unknown E2 stage {stage!r}")


def make_spec(
    *,
    output_root: Path,
    stage: str,
    task_id: str,
    method: str,
    learning_rate: float,
    seed: int,
    coefficient: float | None = None,
) -> TrialSpec:
    if task_id not in TASK_IDS:
        raise ValueError(f"Unknown task_id {task_id!r}")
    base = _stage_dir(output_root, stage, task_id, method, coefficient)
    expected = base if stage == "stage0_cache" else base / f"seed_{seed}"
    if stage == "stage0_cache":
        expected = expected / "teacher_logits.pt"
    return TrialSpec(
        stage=stage,
        task_id=task_id,
        method=method,
        learning_rate=float(learning_rate),
        seed=int(seed),
        coefficient=None if coefficient is None else float(coefficient),
        expected_dir=str(expected),
    )


def teacher_specs(output_root: Path, tasks: Sequence[str] = TASK_IDS) -> list[TrialSpec]:
    return [
        make_spec(
            output_root=output_root,
            stage="stage0_teacher",
            task_id=task,
            method="teacher",
            learning_rate=lr,
            seed=STAGE0_SEED,
            coefficient=lr,
        )
        for task in tasks
        for lr in TEACHER_LEARNING_RATES
    ]


def cache_specs(
    output_root: Path,
    selected_teachers: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[TrialSpec]:
    """Build one train-logit cache identity per selected teacher task."""

    if selected_teachers is None:
        return [
            make_spec(
                output_root=output_root,
                stage="stage0_cache",
                task_id=task,
                method="cache_train_logits",
                learning_rate=0.0,
                seed=STAGE0_SEED,
            )
            for task in TASK_IDS
        ]
    specs: list[TrialSpec] = []
    for task in TASK_IDS:
        row = selected_teachers.get(task)
        if not isinstance(row, Mapping):
            raise OrchestrationBlocked(f"Selected-teacher manifest has no row for {task}")
        specs.append(
            make_spec(
                output_root=output_root,
                stage="stage0_cache",
                task_id=task,
                method="cache_train_logits",
                learning_rate=float(row["learning_rate"]),
                seed=STAGE0_SEED,
            )
        )
    return specs


def stagea_kd_specs(output_root: Path, tasks: Sequence[str] = TASK_IDS) -> list[TrialSpec]:
    return [
        make_spec(
            output_root=output_root,
            stage="stagea_kd",
            task_id=task,
            method="kd",
            learning_rate=STUDENT_LEARNING_RATE[task],
            seed=STAGEA_SEED,
            coefficient=alpha,
        )
        for task in tasks
        for alpha in KD_ALPHAS
    ]


def stagea_nc_specs(output_root: Path, tasks: Sequence[str] = TASK_IDS) -> list[TrialSpec]:
    return [
        make_spec(
            output_root=output_root,
            stage="stagea_nc",
            task_id=task,
            method="nc",
            learning_rate=STUDENT_LEARNING_RATE[task],
            seed=STAGEA_SEED,
            coefficient=lam,
        )
        for task in tasks
        for lam in NC_LAMBDAS
    ]


def ce_replay_specs(output_root: Path) -> list[TrialSpec]:
    """One E1-compatible student replay used as the E2 engineering gate."""

    return [
        make_spec(
            output_root=output_root,
            stage="stagea_ce_replay",
            task_id="vsfc_sentiment",
            method="ce_replay",
            learning_rate=STUDENT_LEARNING_RATE["vsfc_sentiment"],
            seed=STAGEA_SEED,
        )
    ]


def stageb_specs(
    output_root: Path,
    selected_kd: Mapping[str, float],
    selected_nc: Mapping[str, float],
    tasks: Sequence[str] = TASK_IDS,
) -> list[TrialSpec]:
    specs: list[TrialSpec] = []
    for task in tasks:
        if task not in selected_kd or task not in selected_nc:
            raise OrchestrationBlocked(f"Stage B selection is incomplete for {task}")
        for method, coefficient in (("kd", selected_kd[task]), ("nc", selected_nc[task])):
            for seed in STAGEB_SEEDS:
                specs.append(
                    make_spec(
                        output_root=output_root,
                        stage="stageb",
                        task_id=task,
                        method=method,
                        learning_rate=STUDENT_LEARNING_RATE[task],
                        seed=seed,
                        coefficient=coefficient,
                    )
                )
    return specs


def _float_equal(left: Any, right: float | None) -> bool:
    if right is None:
        return left is None
    return isinstance(left, (int, float)) and math.isclose(float(left), float(right), rel_tol=0, abs_tol=1e-12)


def metadata_identity_matches(metadata: Mapping[str, Any], spec: TrialSpec) -> bool:
    if metadata.get("task_id") != spec.task_id or metadata.get("seed") != spec.seed:
        return False
    if metadata.get("stage") != spec.stage or metadata.get("method") != spec.method:
        return False
    if metadata.get("revision") != REVISION:
        return False
    if not _float_equal(metadata.get("coefficient"), spec.coefficient):
        return False
    expected_mode = {
        "teacher": "teacher",
        "ce_replay": "ce",
        "kd": "kd",
        "nc": "nc",
    }.get(spec.method)
    if spec.stage != "stage0_cache" and metadata.get("mode") != expected_mode:
        return False
    if not _float_equal(metadata.get("learning_rate"), spec.learning_rate):
        return False
    if spec.method == "kd":
        hyper = metadata.get("hyperparameters")
        return isinstance(hyper, Mapping) and _float_equal(hyper.get("alpha"), spec.coefficient)
    if spec.method == "nc":
        hyper = metadata.get("hyperparameters")
        return isinstance(hyper, Mapping) and _float_equal(hyper.get("consistency_lambda"), spec.coefficient)
    return True


def _required_train_artifacts(artifact_dir: Path) -> bool:
    return all(
        (artifact_dir / name).is_file()
        for name in (
            "metadata.json",
            "resolved_config.json",
            "epoch_curves.json",
            "best_checkpoint.pt",
            "best_validation_metrics.json",
            "validation_predictions.json",
        )
    )


def _load_cache_summary(cache_path: Path) -> dict[str, Any] | None:
    """Read the sibling-owned immutable cache contract without reimplementing it."""

    if not cache_path.is_file():
        return None
    try:
        from goibert_e2.cache import load_teacher_logit_cache

        cache = load_teacher_logit_cache(cache_path)
    except Exception:
        return None
    metadata = dict(cache.metadata)
    metadata.update(
        {
            "cache_path": str(cache_path.resolve()),
            "cache_sha256": sha256_file(cache_path),
            "cache_schema": "goibert-e2-train-logits-v1",
            "status": "completed",
        }
    )
    return metadata


def _cache_sidecar_path(cache_path: Path) -> Path:
    return cache_path.with_name("scheduler_manifest.json")


def _cache_sidecar_matches(cache_path: Path, spec: TrialSpec, hashes: Mapping[str, str]) -> bool:
    try:
        payload = read_json(_cache_sidecar_path(cache_path))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("protocol") == PROTOCOL
        and payload.get("revision") == REVISION
        and payload.get("stage") == spec.stage
        and payload.get("task_id") == spec.task_id
        and payload.get("method") == spec.method
        and _float_equal(payload.get("learning_rate"), spec.learning_rate)
        and payload.get("seed") == spec.seed
        and payload.get("cache_sha256") == sha256_file(cache_path)
        and payload.get("e1_dependency_hash") == E1_DEPENDENCY_HASH
        and payload.get("e2_code_hash") == hashes.get("e2_code_hash")
        and payload.get("e2_config_hash") == hashes.get("e2_config_hash")
        and payload.get("orchestrator_hash") == hashes.get("orchestrator_hash")
    )


def _write_cache_sidecar(cache_path: Path, spec: TrialSpec, hashes: Mapping[str, str], summary: Mapping[str, Any]) -> None:
    write_json(
        _cache_sidecar_path(cache_path),
        {
            "protocol": PROTOCOL,
            "revision": REVISION,
            "stage": spec.stage,
            "task_id": spec.task_id,
            "method": spec.method,
            "learning_rate": spec.learning_rate,
            "seed": spec.seed,
            "cache_sha256": sha256_file(cache_path),
            "teacher_hash": summary.get("teacher_hash"),
            **dict(hashes),
        },
    )


def _runtime_precision(metadata: Mapping[str, Any]) -> str | None:
    runtime = metadata.get("runtime")
    if isinstance(runtime, Mapping):
        value = runtime.get("precision")
        if isinstance(value, str):
            return value
    value = metadata.get("precision")
    return value if isinstance(value, str) else None


def _has_test_artifact(artifact_dir: Path) -> bool:
    return any(
        (artifact_dir / name).exists()
        for name in (
            "test_metrics.json",
            "test_predictions.json",
            "test_evaluation_metadata.json",
            "test_logits.pt",
        )
    )


def metadata_matches(
    metadata: Mapping[str, Any],
    artifact_dir: Path,
    spec: TrialSpec,
    *,
    e2_code_hash: str,
    e2_config_hash: str,
    orchestrator_hash: str,
) -> bool:
    """Strict completed-artifact contract; smoke and test artifacts never pass."""

    if spec.stage == "stage0_cache":
        cache_metadata = _load_cache_summary(artifact_dir)
        if cache_metadata is None:
            return False
        if cache_metadata.get("task_id") != spec.task_id or cache_metadata.get("split") != "train":
            return False
        if cache_metadata.get("source_ids_hash") != _stable_cache_id_hash(cache_metadata.get("source_ids", [])):
            return False
        if cache_metadata.get("e1_dependency_source_hash") != E1_DEPENDENCY_HASH:
            return False
        cache_hashes = cache_metadata.get("source_hashes")
        if not isinstance(cache_hashes, Mapping):
            return False
        if cache_hashes.get("e1_dependency") != E1_DEPENDENCY_HASH:
            return False
        if cache_hashes.get("e2") != e2_code_hash:
            return False
        if cache_hashes.get("e2_config") != e2_config_hash:
            return False
        if cache_hashes.get("orchestrator") != orchestrator_hash:
            return False
        if _has_test_artifact(artifact_dir):
            return False
        return True

    if not metadata_identity_matches(metadata, spec):
        return False
    if metadata.get("status") != "completed" or metadata.get("smoke") is True:
        return False
    source_hashes = metadata.get("source_hashes")
    if not isinstance(source_hashes, Mapping):
        return False
    if source_hashes.get("e2") != e2_code_hash:
        return False
    if source_hashes.get("e1_dependency") != E1_DEPENDENCY_HASH:
        return False
    if source_hashes.get("e2_config") != e2_config_hash:
        return False
    if source_hashes.get("orchestrator") != orchestrator_hash:
        return False
    if metadata.get("protocol") != PROTOCOL:
        return False
    if metadata.get("include_test") is not False:
        return False
    if metadata.get("epochs") != EPOCHS:
        return False
    if metadata.get("effective_batch_size") != EFFECTIVE_BATCH_SIZE:
        return False
    expected_micro = TEACHER_MICRO_BATCH_SIZE if spec.stage in {"stage0_teacher"} else MICRO_BATCH_SIZE
    if metadata.get("micro_batch_size") != expected_micro:
        return False
    if _runtime_precision(metadata) != PRECISION:
        return False
    if metadata.get("attention_implementation") != ATTENTION_IMPLEMENTATION:
        return False
    if _has_test_artifact(artifact_dir):
        return False

    if not isinstance(metadata.get("best_epoch"), int) or metadata["best_epoch"] < 1:
        return False
    value = metadata.get("best_selection_f1")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return False
    if not _required_train_artifacts(artifact_dir):
        return False
    return True


def read_json(path: Path) -> dict[str, Any] | list[Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_cache_id_hash(values: Sequence[Any]) -> str:
    payload = json.dumps(list(values), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_files(repo_root: Path, files: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted({path.resolve() for path in files if path.is_file()}):
        try:
            relative = str(path.relative_to(repo_root.resolve()))
        except ValueError:
            relative = str(path)
        digest.update(relative.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def e2_code_hash(repo_root: Path) -> str:
    roots = [repo_root / "src" / "goibert_e2", repo_root / "src" / "goibert" / "e2_losses.py"]
    files: list[Path] = []
    for root in roots:
        if root.is_dir():
            files.extend(path for path in root.rglob("*") if path.suffix in {".py", ".json"})
        elif root.is_file():
            files.append(root)
    # A missing implementation is still represented by a stable digest; the
    # runner will fail clearly when --execute is eventually requested.
    return _hash_files(repo_root, files)


def e2_config_hash(repo_root: Path) -> str:
    root = repo_root / "configs" / "e2"
    files = [path for path in root.rglob("*") if path.is_file()] if root.is_dir() else []
    return _hash_files(repo_root, files)


def orchestrator_hash(repo_root: Path) -> str:
    return sha256_file(repo_root / "scripts" / "execute_e2_validation.py")


def source_hashes(repo_root: Path) -> dict[str, str]:
    try:
        from goibert.train import source_hash as e1_source_hash

        dependency = e1_source_hash(repo_root)
    except Exception as exc:  # pragma: no cover - the real runner reports details
        raise OrchestrationBlocked(f"Unable to compute frozen E1 dependency hash: {exc}") from exc
    if dependency != E1_DEPENDENCY_HASH:
        raise OrchestrationBlocked(
            f"E1 dependency hash mismatch: expected {E1_DEPENDENCY_HASH}, found {dependency}"
        )
    return {
        "e1_dependency_hash": dependency,
        "e2_code_hash": e2_code_hash(repo_root),
        "e2_config_hash": e2_config_hash(repo_root),
        "orchestrator_hash": orchestrator_hash(repo_root),
    }


def _read_metadata(path: Path) -> dict[str, Any] | None:
    try:
        value = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def score(record: Mapping[str, Any]) -> float:
    value = record.get("best_selection_f1")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise OrchestrationBlocked(f"Record lacks finite best_selection_f1: {record}")
    return float(value)


def candidate_records(
    output_root: Path,
    spec: TrialSpec,
    *,
    e2_code_hash_value: str,
    e2_config_hash_value: str,
    orchestrator_hash_value: str,
) -> list[dict[str, Any]]:
    """Return exact complete records; ambiguous or occupied identities block."""

    if spec.stage == "stage0_cache":
        cache_path = Path(spec.expected_dir)
        summary = _load_cache_summary(cache_path)
        if summary is None or not _cache_sidecar_matches(
            cache_path,
            spec,
            {
                "e2_code_hash": e2_code_hash_value,
                "e2_config_hash": e2_config_hash_value,
                "orchestrator_hash": orchestrator_hash_value,
            },
        ):
            return []
        if not metadata_matches(
            summary,
            cache_path,
            spec,
            e2_code_hash=e2_code_hash_value,
            e2_config_hash=e2_config_hash_value,
            orchestrator_hash=orchestrator_hash_value,
        ):
            raise OrchestrationBlocked(f"Existing cache fails the frozen E2 cache contract: {cache_path}")
        return [
            {
                "artifact_dir": str(cache_path.parent.resolve()),
                "cache_path": str(cache_path.resolve()),
                "stage": spec.stage,
                "task_id": spec.task_id,
                "method": spec.method,
                "learning_rate": spec.learning_rate,
                "seed": spec.seed,
                "coefficient": spec.coefficient,
                **summary,
            }
        ]

    # Stage identity is encoded in the canonical path because the sibling
    # runner intentionally records its own mode/hyperparameters rather than
    # scheduler-only stage labels.  Never scan rerun directories and never
    # choose the best of multiple attempts.
    artifact_dir = Path(spec.expected_dir)
    metadata_path = artifact_dir / "metadata.json"
    if not metadata_path.is_file():
        return []
    metadata = _read_metadata(metadata_path)
    if metadata is None or not metadata_identity_matches(metadata, spec):
        raise OrchestrationBlocked(f"Canonical E2 artifact has the wrong identity: {artifact_dir}")
    if not metadata_matches(
        metadata,
        artifact_dir,
        spec,
        e2_code_hash=e2_code_hash_value,
        e2_config_hash=e2_config_hash_value,
        orchestrator_hash=orchestrator_hash_value,
    ):
        raise OrchestrationBlocked(f"Canonical E2 artifact fails the frozen contract: {artifact_dir}")
    return [{"artifact_dir": str(artifact_dir.resolve()), "stage": spec.stage, "method": spec.method, "coefficient": spec.coefficient, **metadata}]


def _record_identity(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("stage"),
        record.get("task_id"),
        record.get("method"),
        float(record.get("learning_rate", 0.0)),
        int(record.get("seed", -1)),
        None if record.get("coefficient") is None else float(record.get("coefficient")),
    )


def _append_unique(rows: list[dict[str, Any]], record: dict[str, Any]) -> None:
    identity = _record_identity(record)
    for existing in rows:
        if _record_identity(existing) == identity:
            if existing.get("artifact_dir") != record.get("artifact_dir"):
                raise OrchestrationBlocked(f"Approved manifest has ambiguous identity: {identity}")
            return
    rows.append(record)


class StateStore:
    """Atomic state/event store shared by one locked scheduler process."""

    def __init__(
        self,
        root: Path,
        hashes: Mapping[str, str],
        memory_guards: Mapping[str, float] | None = None,
    ) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.hashes = dict(hashes)
        self.memory_guards = dict(
            memory_guards
            or {
                "student": ACTIVE_STUDENT_MEMORY_GUARD_GIB,
                "teacher": ACTIVE_TEACHER_MEMORY_GUARD_GIB,
                "nc": ACTIVE_NC_MEMORY_GUARD_GIB,
            }
        )
        self.state_path = root / "state.json"
        if self.state_path.is_file():
            loaded = read_json(self.state_path)
            if not isinstance(loaded, dict):
                raise OrchestrationBlocked("Malformed E2 orchestration state")
            if loaded.get("protocol") != PROTOCOL or loaded.get("revision") != REVISION:
                raise OrchestrationBlocked("Existing E2 state uses another protocol/revision")
            for key, value in self.hashes.items():
                if loaded.get(key) != value:
                    raise OrchestrationBlocked(f"Existing E2 state has a different {key}")
            if loaded.get("memory_guards") != self.memory_guards:
                raise OrchestrationBlocked("Existing E2 state has different memory guards")
            self.state = loaded
        else:
            self.state = {
                "protocol": PROTOCOL,
                "revision": REVISION,
                **self.hashes,
                "memory_guards": self.memory_guards,
                "created_at": _now(),
                "updated_at": _now(),
                "jobs": [],
                "approved": {},
                "reused": [],
                "failed": [],
                "stages": {},
            }
            self.save()

    def save(self) -> None:
        self.state["updated_at"] = _now()
        write_json(self.state_path, self.state)

    def event(self, event: str, **payload: Any) -> None:
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": _now(), "event": event, **payload}, ensure_ascii=False) + "\n")
        self.state.setdefault("events", []).append({"time": _now(), "event": event, **payload})
        self.save()

    def approved_manifest(self) -> None:
        write_json(
            self.root / "approved_records.json",
            {
                "protocol": PROTOCOL,
                "revision": REVISION,
                **self.hashes,
                "memory_guards": self.memory_guards,
                "approved": self.state.get("approved", {}),
                "reused": self.state.get("reused", []),
                "failed": self.state.get("failed", []),
                "created_at": self.state.get("created_at"),
                "updated_at": _now(),
            },
        )


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _lock_orchestration(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    handle = (root / "orchestrator.lock").open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise OrchestrationBlocked("Another E2 orchestrator is already running") from exc
    return handle


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _gpu_index(device: str) -> str:
    match = re.search(r"(\d+)$", device)
    if not match:
        raise ValueError(f"Expected CUDA device such as cuda:0, got {device!r}")
    return match.group(1)


def free_memory_gib(device: str) -> float:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if query.returncode != 0:
        raise OrchestrationBlocked(f"nvidia-smi preflight failed: {query.stderr.strip()}")
    target = _gpu_index(device)
    for line in query.stdout.splitlines():
        pieces = [piece.strip() for piece in line.split(",")]
        if len(pieces) == 2 and pieces[0] == target:
            return float(pieces[1]) / 1024.0
    raise OrchestrationBlocked(f"nvidia-smi did not report approved device {device}")


def memory_guard_gib(spec: TrialSpec) -> float:
    if spec.stage in {"stage0_teacher", "stage0_cache"}:
        return ACTIVE_TEACHER_MEMORY_GUARD_GIB
    if spec.method == "nc":
        return ACTIVE_NC_MEMORY_GUARD_GIB
    return ACTIVE_STUDENT_MEMORY_GUARD_GIB


def _selected_teacher_path(selected: Mapping[str, Any], task: str) -> Path:
    row = selected.get(task)
    if not isinstance(row, Mapping):
        raise OrchestrationBlocked(f"Teacher selection has no row for {task}")
    checkpoint = row.get("checkpoint")
    if not isinstance(checkpoint, str) or not Path(checkpoint).is_file():
        raise OrchestrationBlocked(f"Selected teacher checkpoint is missing for {task}: {checkpoint}")
    return Path(checkpoint)


def _selected_cache_path(output_root: Path, task: str) -> Path:
    return output_root / "stage0_cache" / task / f"seed_{STAGE0_SEED}" / "teacher_logits.pt"


def _validate_teacher_binding(
    records: Sequence[Mapping[str, Any]],
    selected_teachers: Mapping[str, Mapping[str, Any]],
    *,
    output_root: Path,
    require_cache: bool,
) -> None:
    """Reject a student artifact trained against a different teacher/cache."""

    for record in records:
        task = str(record.get("task_id"))
        selected = selected_teachers.get(task)
        if not isinstance(selected, Mapping):
            raise OrchestrationBlocked(f"No selected teacher for student task {task}")
        expected_hash = selected.get("checkpoint_sha256")
        if record.get("stage") == "stage0_cache":
            if record.get("teacher_hash") != expected_hash:
                raise OrchestrationBlocked(f"Teacher cache checkpoint hash mismatch for {task}")
        elif record.get("method") == "kd":
            # KD consumes the immutable cache; its metadata binds the cache
            # path/hash, and the cache itself binds teacher_hash.
            cache_summary = _load_cache_summary(_selected_cache_path(output_root, task))
            if cache_summary is None or cache_summary.get("teacher_hash") != expected_hash:
                raise OrchestrationBlocked(f"KD cache teacher hash mismatch for {task}")
        if require_cache and record.get("method") == "kd":
            expected_cache = str(_selected_cache_path(output_root, task).resolve())
            hyper = record.get("hyperparameters")
            if not isinstance(hyper, Mapping):
                raise OrchestrationBlocked(f"KD metadata has no hyperparameters for {task}")
            if str(Path(str(hyper.get("teacher_cache", ""))).resolve()) != expected_cache:
                raise OrchestrationBlocked(f"Student cache binding mismatch for {task}")
            if hyper.get("teacher_cache_hash") != sha256_file(Path(expected_cache)):
                raise OrchestrationBlocked(f"Student cache hash mismatch for {task}")


def build_command(
    spec: TrialSpec,
    *,
    repo_root: Path,
    output_root: Path | None = None,
    device: str,
    hashes: Mapping[str, str],
    selected_teachers: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[str]:
    """Build the stable CLI contract for the separate E2 runner."""

    is_cache = spec.stage == "stage0_cache"
    mode = {
        "teacher": "teacher",
        "ce_replay": "ce",
        "kd": "kd",
        "nc": "nc",
    }.get(spec.method)
    if not is_cache and mode is None:
        raise ValueError(f"No runner mode for method {spec.method!r}")
    command = [sys.executable, "-m", RUNNER_MODULE]
    command.extend([
        "--mode",
        "cache" if is_cache else mode,
        "--repo-root",
        str(repo_root),
        "--task-id",
        spec.task_id,
        "--learning-rate",
        repr(spec.learning_rate),
        "--seed",
        str(spec.seed),
        "--output-dir",
        str(Path(spec.expected_dir).parent if is_cache else Path(spec.expected_dir)),
        "--device",
        device,
        "--precision",
        PRECISION,
        "--attention-implementation",
        ATTENTION_IMPLEMENTATION,
        "--epochs",
        str(EPOCHS),
        "--effective-batch-size",
        str(EFFECTIVE_BATCH_SIZE),
        "--micro-batch-size",
        str(TEACHER_MICRO_BATCH_SIZE if spec.stage in {"stage0_teacher", "stage0_cache"} else MICRO_BATCH_SIZE),
        "--stage",
        spec.stage,
        "--method",
        spec.method,
        "--protocol",
        PROTOCOL,
        "--revision",
        REVISION,
        "--expected-e2-code-hash",
        hashes["e2_code_hash"],
        "--expected-e2-config-hash",
        hashes["e2_config_hash"],
        "--expected-orchestrator-hash",
        hashes["orchestrator_hash"],
        "--include-test",
        "false",
    ])
    if spec.coefficient is not None:
        command.extend(["--coefficient", repr(float(spec.coefficient))])
    if is_cache:
        teacher = _selected_teacher_path(selected_teachers or {}, spec.task_id)
        command.extend(
            [
                "--teacher-checkpoint",
                str(teacher),
                "--cache-path",
                str(Path(spec.expected_dir)),
            ]
        )
        return command

    if spec.method == "kd":
        cache_root = output_root or Path(spec.expected_dir).parents[4]
        cache = _selected_cache_path(cache_root, spec.task_id)
        command.extend(
            [
                "--cache-path",
                str(cache),
                "--alpha",
                repr(float(spec.coefficient)),
            ]
        )
    elif spec.method == "nc":
        command.extend(["--consistency-lambda", repr(float(spec.coefficient))])
    return command


def _job_failed(store: StateStore, spec: TrialSpec) -> bool:
    return any(_record_identity(row.get("spec", {})) == spec.key for row in store.state.get("failed", []))


def _attempt_path_occupied(spec: TrialSpec) -> bool:
    path = Path(spec.expected_dir)
    return path.exists() and not path.is_dir() or path.is_dir() and any(path.iterdir())


def _validate_completed_artifact(
    path: Path,
    spec: TrialSpec,
    *,
    hashes: Mapping[str, str],
) -> dict[str, Any]:
    if spec.stage == "stage0_cache":
        summary = _load_cache_summary(path)
        if summary is None or not metadata_matches(summary, path, spec, **hashes):
            raise OrchestrationBlocked(f"Completed E2 cache is invalid: {path}")
        return {
            "artifact_dir": str(path.parent.resolve()),
            "cache_path": str(path.resolve()),
            "stage": spec.stage,
            "task_id": spec.task_id,
            "method": spec.method,
            "learning_rate": spec.learning_rate,
            "seed": spec.seed,
            "coefficient": spec.coefficient,
            **summary,
        }
    metadata = _read_metadata(path / "metadata.json")
    if metadata is None or not metadata_matches(metadata, path, spec, **hashes):
        raise OrchestrationBlocked(f"Completed E2 process produced an invalid artifact: {path}")
    return {
        "artifact_dir": str(path.resolve()),
        "stage": spec.stage,
        "method": spec.method,
        "coefficient": spec.coefficient,
        **metadata,
    }


def _append_approved(store: StateStore, record: dict[str, Any]) -> None:
    rows = store.state.setdefault("approved", {}).setdefault(record.get("stage", "unknown"), [])
    _append_unique(rows, record)


def _mark_job(store: StateStore, job: dict[str, Any], status: str, **extra: Any) -> None:
    job.update({"status": status, **extra, "updated_at": _now()})
    store.save()


def _recover_running_jobs(store: StateStore, *, hashes: Mapping[str, str]) -> None:
    for job in store.state.get("jobs", []):
        if job.get("status") != "running":
            continue
        pid = int(job.get("pid", -1))
        if _pid_alive(pid):
            raise OrchestrationBlocked(
                f"Existing E2 job is still running; refusing duplicate: pid={pid}, spec={job.get('spec')}"
            )
        spec = TrialSpec(**(job.get("spec") or {}))
        artifact = Path(str(job.get("output_dir", spec.expected_dir)))
        if spec.stage == "stage0_cache":
            summary = _load_cache_summary(artifact)
            if summary is not None and metadata_matches(summary, artifact, spec, **hashes):
                record = _validate_completed_artifact(artifact, spec, hashes=hashes)
                _write_cache_sidecar(artifact, spec, hashes, record)
                _append_approved(store, record)
                _mark_job(store, job, "completed", artifact_dir=record["artifact_dir"], recovered=True)
                continue
        elif artifact.is_dir():
            metadata = _read_metadata(artifact / "metadata.json")
            if metadata is not None and metadata_matches(metadata, artifact, spec, **hashes):
                record = _validate_completed_artifact(artifact, spec, hashes=hashes)
                _append_approved(store, record)
                _mark_job(store, job, "completed", artifact_dir=record["artifact_dir"], recovered=True)
                continue
        failure = {"spec": asdict(spec), "output_dir": str(artifact), "reason": "process ended without exact artifact"}
        store.state.setdefault("failed", []).append(failure)
        _mark_job(store, job, "failed", reason=failure["reason"])
        raise OrchestrationBlocked(f"Recovered E2 job failed contract: {spec.key}")


def _launch_job(
    store: StateStore,
    spec: TrialSpec,
    *,
    output_root: Path,
    repo_root: Path,
    device: str,
    hashes: Mapping[str, str],
    selected_teachers: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], subprocess.Popen[str], Any]:
    free = free_memory_gib(device)
    needed = memory_guard_gib(spec)
    if free < needed:
        raise OrchestrationBlocked(f"{device} has {free:.2f} GiB free; need {needed:.2f} GiB")
    output_dir = Path(spec.expected_dir)
    if output_dir.exists():
        raise OrchestrationBlocked(f"Expected E2 path is occupied; refusing overwrite: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    log_dir = store.root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{spec.slug()}.log"
    command = build_command(
        spec,
        repo_root=repo_root,
        output_root=output_root,
        device=device,
        hashes=hashes,
        selected_teachers=selected_teachers,
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    log_handle = log_path.open("w", encoding="utf-8")
    log_handle.write("$ " + " ".join(command) + "\n")
    log_handle.flush()
    process = subprocess.Popen(
        command,
        cwd=repo_root,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    job = {
        "spec": asdict(spec),
        "output_dir": str(output_dir),
        "device": device,
        "pid": process.pid,
        "command": command,
        "log_path": str(log_path),
        "status": "running",
        "started_at": _now(),
        "free_memory_gib_at_launch": free,
        "memory_guard_gib": needed,
    }
    store.state.setdefault("jobs", []).append(job)
    store.event("job_started", spec=asdict(spec), pid=process.pid, device=device, log_path=str(log_path))
    return job, process, log_handle


def _finish_process(
    store: StateStore,
    job: dict[str, Any],
    process: subprocess.Popen[str],
    spec: TrialSpec,
    log_handle: Any,
    *,
    hashes: Mapping[str, str],
) -> dict[str, Any]:
    return_code = process.returncode
    log_handle.close()
    output_dir = Path(str(job["output_dir"]))
    if return_code != 0:
        failure = {
            "spec": asdict(spec),
            "output_dir": str(output_dir),
            "returncode": return_code,
            "log_path": job.get("log_path"),
            "status": "failed",
        }
        store.state.setdefault("failed", []).append(failure)
        _mark_job(store, job, "failed", returncode=return_code)
        raise OrchestrationBlocked(f"E2 trial failed: {spec.key}; see {job.get('log_path')}")
    record = _validate_completed_artifact(output_dir, spec, hashes=hashes)
    if spec.stage == "stage0_cache":
        _write_cache_sidecar(Path(spec.expected_dir), spec, hashes, record)
    _append_approved(store, record)
    _mark_job(store, job, "completed", artifact_dir=record["artifact_dir"], returncode=0)
    store.event("job_completed", spec=asdict(spec), artifact_dir=record["artifact_dir"])
    return record


def run_specs(
    store: StateStore,
    specs: Sequence[TrialSpec],
    *,
    output_root: Path,
    repo_root: Path,
    devices: Sequence[str],
    poll_seconds: float,
    hashes: Mapping[str, str],
    selected_teachers: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Run missing identities once; matching records are reused exactly."""

    approved: list[dict[str, Any]] = []
    pending: list[TrialSpec] = []
    for spec in specs:
        if _job_failed(store, spec):
            raise OrchestrationBlocked(f"A prior attempt failed; refusing automatic rerun: {spec.key}")
        exact = candidate_records(
            output_root,
            spec,
            e2_code_hash_value=hashes["e2_code_hash"],
            e2_config_hash_value=hashes["e2_config_hash"],
            orchestrator_hash_value=hashes["orchestrator_hash"],
        )
        if exact:
            approved.extend(exact)
            for record in exact:
                _append_approved(store, record)
                store.state.setdefault("reused", []).append({"spec": asdict(spec), **record})
                store.event("record_reused", spec=asdict(spec), artifact_dir=record["artifact_dir"])
            continue
        if _attempt_path_occupied(spec):
            raise OrchestrationBlocked(
                f"Expected E2 path is occupied without an exact completed record: {spec.expected_dir}"
            )
        pending.append(spec)
    store.approved_manifest()

    active: dict[str, tuple[dict[str, Any], subprocess.Popen[str], TrialSpec, Any]] = {}
    while pending or active:
        launched = False
        for device in devices:
            if device in active or not pending:
                continue
            spec = pending[0]
            free = free_memory_gib(device)
            if free < memory_guard_gib(spec):
                continue
            pending.pop(0)
            job, process, log_handle = _launch_job(
                store,
                spec,
                output_root=output_root,
                repo_root=repo_root,
                device=device,
                hashes=hashes,
                selected_teachers=selected_teachers,
            )
            active[device] = (job, process, spec, log_handle)
            launched = True
        completed_devices: list[str] = []
        for device, (job, process, spec, log_handle) in list(active.items()):
            if process.poll() is None:
                continue
            approved.append(_finish_process(store, job, process, spec, log_handle, hashes=hashes))
            completed_devices.append(device)
        for device in completed_devices:
            del active[device]
        store.approved_manifest()
        if pending or active:
            if not launched and not completed_devices:
                time.sleep(poll_seconds)
    return approved


def _records_for_stage(store: StateStore, stage: str) -> list[dict[str, Any]]:
    rows = list(store.state.get("approved", {}).get(stage, []))
    identities = [_record_identity(row) for row in rows]
    if len(identities) != len(set(identities)):
        raise OrchestrationBlocked(f"Approved {stage} manifest contains duplicate identities")
    return rows


def select_teachers(
    records: Sequence[Mapping[str, Any]],
    *,
    e1_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Choose one teacher/task by score, lower LR, then earlier epoch."""

    selected: dict[str, dict[str, Any]] = {}
    for task in TASK_IDS:
        rows = [dict(row) for row in records if row.get("task_id") == task and row.get("stage") == "stage0_teacher"]
        if len(rows) != len(TEACHER_LEARNING_RATES):
            raise OrchestrationBlocked(f"Teacher grid is incomplete for {task}")
        if {float(row.get("learning_rate")) for row in rows} != set(TEACHER_LEARNING_RATES):
            raise OrchestrationBlocked(f"Teacher grid has wrong learning rates for {task}")
        if any(int(row.get("seed", -1)) != STAGE0_SEED for row in rows):
            raise OrchestrationBlocked(f"Teacher seed mismatch for {task}")
        winner = sorted(rows, key=lambda row: (-score(row), float(row["learning_rate"]), int(row["best_epoch"])))[0]
        checkpoint = Path(str(winner.get("artifact_dir", ""))) / "best_checkpoint.pt"
        if not checkpoint.is_file():
            raise OrchestrationBlocked(f"Selected teacher checkpoint missing: {checkpoint}")
        winner["checkpoint"] = str(checkpoint.resolve())
        winner["checkpoint_sha256"] = sha256_file(checkpoint)
        reference = e1_reference_score(e1_root, task, float(winner["learning_rate"])) if e1_root else None
        winner["e1_reference_selection_f1"] = reference
        winner["teacher_strength_val"] = (
            "stronger_than_e1_reference" if reference is not None and score(winner) > reference
            else "not_stronger_or_equal_to_e1_reference" if reference is not None
            else "e1_reference_unavailable"
        )
        winner["teacher_strength_is_gate"] = False
        selected[task] = winner
    return selected


def e1_reference_score(e1_root: Path | None, task: str, learning_rate: float) -> float | None:
    if e1_root is None or not e1_root.is_dir():
        return None
    rows: list[dict[str, Any]] = []
    for path in sorted(e1_root.rglob("metadata.json")):
        metadata = _read_metadata(path)
        if not metadata or metadata.get("smoke") is True:
            continue
        if (
            metadata.get("task_id") == task
            and metadata.get("seed") == STAGE0_SEED
            and isinstance(metadata.get("best_selection_f1"), (int, float))
            and _float_equal(metadata.get("learning_rate"), learning_rate)
            and metadata.get("status") == "completed"
            and metadata.get("include_test") is False
        ):
            rows.append(metadata)
    if len(rows) > 1:
        raise OrchestrationBlocked(f"Ambiguous E1 reference for {task} LR {learning_rate:g}")
    return score(rows[0]) if rows else None


def _select_coefficients(
    records: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    method: str,
    candidates: Sequence[float],
) -> dict[str, float]:
    selected: dict[str, float] = {}
    for task in TASK_IDS:
        rows = [dict(row) for row in records if row.get("stage") == stage and row.get("task_id") == task and row.get("method") == method]
        if len(rows) != len(candidates):
            raise OrchestrationBlocked(f"{stage}/{method} grid is incomplete for {task}")
        if {float(row.get("coefficient")) for row in rows} != set(float(value) for value in candidates):
            raise OrchestrationBlocked(f"{stage}/{method} has wrong coefficients for {task}")
        if any(int(row.get("seed", -1)) != STAGEA_SEED for row in rows):
            raise OrchestrationBlocked(f"{stage}/{method} seed mismatch for {task}")
        winner = sorted(rows, key=lambda row: (-score(row), float(row["coefficient"])))[0]
        selected[task] = float(winner["coefficient"])
    return selected


def paired_e1_comparison(
    e2_records: Sequence[Mapping[str, Any]],
    *,
    e1_root: Path,
    task: str,
    method: str,
    coefficient: float,
) -> dict[str, Any]:
    """Compare Stage B seeds 42..44 with E1 at the fixed sentinel LR."""

    e2_by_seed: dict[int, float] = {}
    for row in e2_records:
        if row.get("task_id") != task or row.get("method") != method:
            continue
        stage_allowed = row.get("stage") == "stageb" or (
            row.get("stage") == ("stagea_kd" if method == "kd" else "stagea_nc")
            and int(row.get("seed", -1)) == STAGEA_SEED
        )
        if stage_allowed and _float_equal(row.get("coefficient"), coefficient):
            seed = int(row["seed"])
            if seed in e2_by_seed:
                raise OrchestrationBlocked(f"Ambiguous E2 paired row for {task}/{method} seed {seed}")
            e2_by_seed[seed] = score(row)
    e1_by_seed: dict[int, float] = {}
    for path in sorted(e1_root.rglob("metadata.json")):
        metadata = _read_metadata(path)
        if not metadata or metadata.get("smoke") is True:
            continue
        if (
            metadata.get("task_id") == task
            and _float_equal(metadata.get("learning_rate"), STUDENT_LEARNING_RATE[task])
            and metadata.get("include_test") is False
            and metadata.get("status") == "completed"
            and isinstance(metadata.get("seed"), int)
            and metadata["seed"] in ALL_STAGEB_SEEDS
        ):
            if metadata["seed"] in e1_by_seed:
                raise OrchestrationBlocked(f"Ambiguous E1 paired reference for {task} seed {metadata['seed']}")
            e1_by_seed[int(metadata["seed"])] = score(metadata)
    seeds = [seed for seed in ALL_STAGEB_SEEDS if seed in e2_by_seed and seed in e1_by_seed]
    e2_values = [e2_by_seed[seed] for seed in seeds]
    e1_values = [e1_by_seed[seed] for seed in seeds]
    return {
        "task_id": task,
        "method": method,
        "coefficient": coefficient,
        "sentinel_learning_rate": STUDENT_LEARNING_RATE[task],
        "paired_seeds": seeds,
        "e2_scores": {str(seed): e2_by_seed[seed] for seed in seeds},
        "e1_scores": {str(seed): e1_by_seed[seed] for seed in seeds},
        "e2_mean": sum(e2_values) / len(e2_values) if e2_values else None,
        "e1_mean": sum(e1_values) / len(e1_values) if e1_values else None,
        "mean_delta_e2_minus_e1": (sum(e2_values) - sum(e1_values)) / len(e2_values) if e2_values else None,
        "status": "complete" if len(seeds) == len(ALL_STAGEB_SEEDS) else "incomplete_reference",
    }


def write_selection_manifest(
    path: Path,
    *,
    stage: str,
    hashes: Mapping[str, str],
    payload: Mapping[str, Any],
) -> None:
    write_json(
        path,
        {
            "protocol": PROTOCOL,
            "revision": REVISION,
            "stage": stage,
            **dict(hashes),
            "locked": True,
            "created_at": _now(),
            **dict(payload),
        },
    )


def smoke_gate_ok(path: Path) -> bool:
    try:
        payload = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("status") == "passed" and payload.get("test_access") is False


def plan_payload(
    output_root: Path,
    hashes: Mapping[str, str],
    *,
    student_guard: float = STUDENT_MIN_FREE_MEMORY_GIB,
    teacher_guard: float = TEACHER_MIN_FREE_MEMORY_GIB,
    nc_guard: float = NC_MIN_FREE_MEMORY_GIB,
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "revision": REVISION,
        **dict(hashes),
        "runner_module": RUNNER_MODULE,
        "precision": PRECISION,
        "attention_implementation": ATTENTION_IMPLEMENTATION,
        "micro_batch_size": {"student": MICRO_BATCH_SIZE, "teacher_and_cache": TEACHER_MICRO_BATCH_SIZE},
        "devices": list(DEFAULT_DEVICES),
        "memory_guard_gib": {"student": student_guard, "teacher": teacher_guard, "nc": nc_guard},
        "test_access": False,
        "stages": {
            "stage0_teacher": {"trial_count": 15, "grid": "5 tasks x 3 LR x seed 42"},
            "stage0_cache": {"trial_count": 5, "grid": "selected teacher x task; train IDs/logits only"},
            "stagea_ce_replay": {"trial_count": 1, "grid": "VSFC sentiment x LR 1e-5 x seed 42"},
            "stagea_kd": {"trial_count": 15, "grid": "5 tasks x alpha .25/.5/.75 x seed 42"},
            "stagea_nc": {"trial_count": 10, "grid": "5 tasks x lambda .1/.5 x seed 42"},
            "stageb": {"trial_count": 20, "grid": "selected KD + NC x seeds 43/44"},
            "combined_kd_nc": {"trial_count": 0, "status": "manual gate only; no autopromotion"},
        },
        "full_training_trial_count": 61,
        "cache_pass_count": 5,
        "output_root": str(output_root),
        "reuse_policy": "exact completed identity only; smoke/failed/rerun artifacts never selected",
        "launch_policy": "dry-run unless --execute is supplied by root GO",
    }


def _write_stage_status(store: StateStore, stage: str, expected: int, records: Sequence[Mapping[str, Any]]) -> None:
    if len(records) != expected:
        raise OrchestrationBlocked(f"{stage} expected {expected} records, found {len(records)}")
    payload = {
        "protocol": PROTOCOL,
        "revision": REVISION,
        **store.hashes,
        "stage": stage,
        "expected_count": expected,
        "approved_count": len(records),
        "approved": list(records),
        "created_at": _now(),
    }
    write_json(store.root / f"{stage}_approved.json", payload)
    store.state.setdefault("stages", {})[stage] = {
        "status": "completed",
        "expected_count": expected,
        "approved_count": len(records),
    }
    store.approved_manifest()


def _load_teacher_selection(path: Path, hashes: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, dict) or payload.get("locked") is not True:
        raise OrchestrationBlocked(f"Teacher selection is not locked: {path}")
    if payload.get("protocol") != PROTOCOL or payload.get("revision") != REVISION:
        raise OrchestrationBlocked("Teacher selection uses another protocol/revision")
    for key, value in hashes.items():
        if payload.get(key) != value:
            raise OrchestrationBlocked(f"Teacher selection has a different {key}")
    selected = payload.get("selected")
    if not isinstance(selected, dict):
        raise OrchestrationBlocked("Teacher selection has no selected mapping")
    for task in TASK_IDS:
        _selected_teacher_path(selected, task)
    return {str(key): dict(value) for key, value in selected.items() if isinstance(value, Mapping)}


def _run_stage(
    store: StateStore,
    *,
    stage: str,
    output_root: Path,
    repo_root: Path,
    devices: Sequence[str],
    poll_seconds: float,
    hashes: Mapping[str, str],
    selected_teachers: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if stage == "stage0_teacher":
        specs = teacher_specs(output_root)
        expected = 15
    elif stage == "stage0_cache":
        specs = cache_specs(output_root, selected_teachers)
        expected = 5
    elif stage == "stagea_kd":
        specs = stagea_kd_specs(output_root)
        expected = 15
    elif stage == "stagea_nc":
        specs = stagea_nc_specs(output_root)
        expected = 10
    elif stage == "stagea_ce_replay":
        specs = ce_replay_specs(output_root)
        expected = 1
    else:
        raise ValueError(f"Unsupported direct stage {stage!r}")
    records = run_specs(
        store,
        specs,
        output_root=output_root,
        repo_root=repo_root,
        devices=devices,
        poll_seconds=poll_seconds,
        hashes=hashes,
        selected_teachers=selected_teachers,
    )
    # Stage cache identities can be returned in arbitrary completion order.
    records = sorted(records, key=_record_identity)
    _write_stage_status(store, stage, expected, records)
    return records


def run_orchestration(args: argparse.Namespace) -> int:
    global ACTIVE_STUDENT_MEMORY_GUARD_GIB, ACTIVE_TEACHER_MEMORY_GUARD_GIB, ACTIVE_NC_MEMORY_GUARD_GIB
    ACTIVE_STUDENT_MEMORY_GUARD_GIB = float(
        getattr(args, "student_memory_guard_gib", STUDENT_MIN_FREE_MEMORY_GIB)
    )
    ACTIVE_TEACHER_MEMORY_GUARD_GIB = float(
        getattr(args, "teacher_memory_guard_gib", TEACHER_MIN_FREE_MEMORY_GIB)
    )
    ACTIVE_NC_MEMORY_GUARD_GIB = float(
        getattr(args, "nc_memory_guard_gib", NC_MIN_FREE_MEMORY_GIB)
    )
    if min(ACTIVE_STUDENT_MEMORY_GUARD_GIB, ACTIVE_TEACHER_MEMORY_GUARD_GIB, ACTIVE_NC_MEMORY_GUARD_GIB) <= 0:
        raise OrchestrationBlocked("Memory guards must be positive")
    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    hashes = source_hashes(repo_root)
    orchestrator_root = output_root / ".orchestrator"
    if not args.execute:
        plan = plan_payload(
            output_root,
            hashes,
            student_guard=ACTIVE_STUDENT_MEMORY_GUARD_GIB,
            teacher_guard=ACTIVE_TEACHER_MEMORY_GUARD_GIB,
            nc_guard=ACTIVE_NC_MEMORY_GUARD_GIB,
        )
        write_json(orchestrator_root / "plan.json", plan)
        print(json.dumps({"plan": str(orchestrator_root / "plan.json"), **plan}, indent=2))
        return 0
    if args.stage == "combined_kd_nc":
        raise OrchestrationBlocked("Combined KD+NC is manual-gate only; no automatic promotion")
    if args.root_smoke_gate is not None and not smoke_gate_ok(args.root_smoke_gate):
        raise OrchestrationBlocked(f"Root smoke gate is not passed: {args.root_smoke_gate}")
    if args.stage in {"stagea_nc", "stagea_ce_replay"} and args.root_smoke_gate is None:
        raise OrchestrationBlocked(f"{args.stage} requires --root-smoke-gate before launch")

    lock = _lock_orchestration(orchestrator_root)
    try:
        store = StateStore(
            orchestrator_root,
            hashes,
            memory_guards={
                "student": ACTIVE_STUDENT_MEMORY_GUARD_GIB,
                "teacher": ACTIVE_TEACHER_MEMORY_GUARD_GIB,
                "nc": ACTIVE_NC_MEMORY_GUARD_GIB,
            },
        )
        _recover_running_jobs(store, hashes=hashes)
        if args.stage == "stage0_teacher":
            records = _run_stage(
                store,
                stage="stage0_teacher",
                output_root=output_root,
                repo_root=repo_root,
                devices=args.devices,
                poll_seconds=args.poll_seconds,
                hashes=hashes,
            )
            selected = select_teachers(records, e1_root=repo_root / "artifacts" / "e1")
            write_selection_manifest(
                orchestrator_root / "stage0_teacher_selection.json",
                stage="stage0_teacher_selection",
                hashes=hashes,
                payload={"selected": selected, "selection_rule": "score desc, LR asc, best epoch asc", "teacher_strength_is_gate": False},
            )
            return 0
        selected_teachers: dict[str, dict[str, Any]] | None = None
        if args.stage in {"stage0_cache", "stagea_kd", "stageb"}:
            selection_path = orchestrator_root / "stage0_teacher_selection.json"
            selected_teachers = _load_teacher_selection(selection_path, hashes)
        if args.stage == "stagea_ce_replay":
            _run_stage(
                store,
                stage="stagea_ce_replay",
                output_root=output_root,
                repo_root=repo_root,
                devices=args.devices,
                poll_seconds=args.poll_seconds,
                hashes=hashes,
            )
            return 0
        if args.stage == "stage0_cache":
            cache_records = _run_stage(
                store,
                stage="stage0_cache",
                output_root=output_root,
                repo_root=repo_root,
                devices=args.devices,
                poll_seconds=args.poll_seconds,
                hashes=hashes,
                selected_teachers=selected_teachers,
            )
            assert selected_teachers is not None
            _validate_teacher_binding(
                cache_records,
                selected_teachers,
                output_root=output_root,
                require_cache=False,
            )
            return 0
        if args.stage == "stagea_kd":
            if not (orchestrator_root / "stage0_cache_approved.json").is_file():
                raise OrchestrationBlocked("StageA KD requires the completed stage0 cache gate")
            if not (orchestrator_root / "stagea_ce_replay_approved.json").is_file():
                raise OrchestrationBlocked("StageA KD requires the completed CE replay gate")
            kd_records = _run_stage(
                store,
                stage="stagea_kd",
                output_root=output_root,
                repo_root=repo_root,
                devices=args.devices,
                poll_seconds=args.poll_seconds,
                hashes=hashes,
                selected_teachers=selected_teachers,
            )
            assert selected_teachers is not None
            _validate_teacher_binding(
                kd_records,
                selected_teachers,
                output_root=output_root,
                require_cache=True,
            )
            return 0
        if args.stage == "stagea_nc":
            if not (orchestrator_root / "stagea_ce_replay_approved.json").is_file():
                raise OrchestrationBlocked("StageA NC requires the completed CE replay gate")
            _run_stage(
                store,
                stage="stagea_nc",
                output_root=output_root,
                repo_root=repo_root,
                devices=args.devices,
                poll_seconds=args.poll_seconds,
                hashes=hashes,
            )
            return 0
        if args.stage == "stageb":
            if not (orchestrator_root / "stagea_ce_replay_approved.json").is_file():
                raise OrchestrationBlocked("Stage B requires the completed CE replay gate")
            kd_records = _records_for_stage(store, "stagea_kd")
            nc_records = _records_for_stage(store, "stagea_nc")
            selected_kd = _select_coefficients(kd_records, stage="stagea_kd", method="kd", candidates=KD_ALPHAS)
            selected_nc = _select_coefficients(nc_records, stage="stagea_nc", method="nc", candidates=NC_LAMBDAS)
            write_selection_manifest(
                orchestrator_root / "stageA_selection.json",
                stage="stageA_selection",
                hashes=hashes,
                payload={
                    "selected_kd_alpha": selected_kd,
                    "selected_nc_lambda": selected_nc,
                    "selection_rule": "validation score desc, coefficient asc",
                    "test_access": False,
                },
            )
            specs = stageb_specs(output_root, selected_kd, selected_nc)
            records = run_specs(
                store,
                specs,
                output_root=output_root,
                repo_root=repo_root,
                devices=args.devices,
                poll_seconds=args.poll_seconds,
                hashes=hashes,
                selected_teachers=selected_teachers,
            )
            _write_stage_status(store, "stageb", 20, sorted(records, key=_record_identity))
            assert selected_teachers is not None
            _validate_teacher_binding(
                [row for row in records if row.get("method") == "kd"],
                selected_teachers,
                output_root=output_root,
                require_cache=True,
            )
            _validate_teacher_binding(
                [row for row in records if row.get("method") == "nc"],
                selected_teachers,
                output_root=output_root,
                require_cache=False,
            )
            comparison_records = kd_records + nc_records + records
            comparisons = [
                paired_e1_comparison(
                    comparison_records,
                    e1_root=repo_root / "artifacts" / "e1",
                    task=task,
                    method=method,
                    coefficient=coefficient,
                )
                for task in TASK_IDS
                for method, coefficient in (("kd", selected_kd[task]), ("nc", selected_nc[task]))
            ]
            write_json(orchestrator_root / "stageB_paired_e1_comparison.json", {"comparisons": comparisons, "test_access": False})
            return 0
        if args.stage == "all":
            teacher_records = _run_stage(
                store,
                stage="stage0_teacher",
                output_root=output_root,
                repo_root=repo_root,
                devices=args.devices,
                poll_seconds=args.poll_seconds,
                hashes=hashes,
            )
            selected_teachers = select_teachers(teacher_records, e1_root=repo_root / "artifacts" / "e1")
            write_selection_manifest(
                orchestrator_root / "stage0_teacher_selection.json",
                stage="stage0_teacher_selection",
                hashes=hashes,
                payload={"selected": selected_teachers, "selection_rule": "score desc, LR asc, best epoch asc", "teacher_strength_is_gate": False},
            )
            _run_stage(
                store,
                stage="stage0_cache",
                output_root=output_root,
                repo_root=repo_root,
                devices=args.devices,
                poll_seconds=args.poll_seconds,
                hashes=hashes,
                selected_teachers=selected_teachers,
            )
            if not (orchestrator_root / "stagea_ce_replay_approved.json").is_file():
                raise OrchestrationBlocked(
                    "Teacher/cache stages are complete; run the CE replay gate before KD/NC"
                )
            # NC is deliberately queued after the root smoke gate and is
            # independent of teacher selection.  It can be invoked separately
            # to overlap with stage0 on a different orchestrator root.
            kd_records = _run_stage(
                store,
                stage="stagea_kd",
                output_root=output_root,
                repo_root=repo_root,
                devices=args.devices,
                poll_seconds=args.poll_seconds,
                hashes=hashes,
                selected_teachers=selected_teachers,
            )
            assert selected_teachers is not None
            _validate_teacher_binding(kd_records, selected_teachers, output_root=output_root, require_cache=True)
            nc_records = _run_stage(
                store,
                stage="stagea_nc",
                output_root=output_root,
                repo_root=repo_root,
                devices=args.devices,
                poll_seconds=args.poll_seconds,
                hashes=hashes,
            )
            selected_kd = _select_coefficients(kd_records, stage="stagea_kd", method="kd", candidates=KD_ALPHAS)
            selected_nc = _select_coefficients(nc_records, stage="stagea_nc", method="nc", candidates=NC_LAMBDAS)
            write_selection_manifest(
                orchestrator_root / "stageA_selection.json",
                stage="stageA_selection",
                hashes=hashes,
                payload={
                    "selected_kd_alpha": selected_kd,
                    "selected_nc_lambda": selected_nc,
                    "selection_rule": "validation score desc, coefficient asc",
                    "test_access": False,
                },
            )
            confirmation_records = run_specs(
                store,
                stageb_specs(output_root, selected_kd, selected_nc),
                output_root=output_root,
                repo_root=repo_root,
                devices=args.devices,
                poll_seconds=args.poll_seconds,
                hashes=hashes,
                selected_teachers=selected_teachers,
            )
            _write_stage_status(store, "stageb", 20, sorted(confirmation_records, key=_record_identity))
            _validate_teacher_binding(
                [row for row in confirmation_records if row.get("method") == "kd"],
                selected_teachers,
                output_root=output_root,
                require_cache=True,
            )
            _validate_teacher_binding(
                [row for row in confirmation_records if row.get("method") == "nc"],
                selected_teachers,
                output_root=output_root,
                require_cache=False,
            )
            comparisons = [
                paired_e1_comparison(
                    kd_records + nc_records + confirmation_records,
                    e1_root=repo_root / "artifacts" / "e1",
                    task=task,
                    method=method,
                    coefficient=coefficient,
                )
                for task in TASK_IDS
                for method, coefficient in (("kd", selected_kd[task]), ("nc", selected_nc[task]))
            ]
            write_json(orchestrator_root / "stageB_paired_e1_comparison.json", {"comparisons": comparisons, "test_access": False})
            return 0
        raise ValueError(f"Unknown stage {args.stage!r}")
    finally:
        lock.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "stage0_teacher", "stage0_cache", "stagea_kd", "stagea_nc", "stageb", "combined_kd_nc"), default="all")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "artifacts" / "e2")
    parser.add_argument("--devices", nargs="+", default=list(DEFAULT_DEVICES))
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--student-memory-guard-gib", type=float, default=STUDENT_MIN_FREE_MEMORY_GIB)
    parser.add_argument("--teacher-memory-guard-gib", type=float, default=TEACHER_MIN_FREE_MEMORY_GIB)
    parser.add_argument("--nc-memory-guard-gib", type=float, default=NC_MIN_FREE_MEMORY_GIB)
    parser.add_argument("--root-smoke-gate", type=Path)
    parser.add_argument("--execute", action="store_true", help="Required before any runner subprocess can launch")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_orchestration(args)
    except OrchestrationBlocked as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
