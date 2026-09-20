"""Durable validation-only E1 orchestration.

This script is the operator-facing wrapper around ``goibert.train``.  It is
deliberately separate from the frozen runner so that recovery, process
bookkeeping and approval of completed records cannot change the experiment
recipe.  The default mode only plans; ``--execute`` is required to launch
training subprocesses.  No test split is ever requested here.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from goibert.train import (  # noqa: E402
    LEARNING_RATES,
    SCREENING_SEED,
    SHORTLIST_SEEDS,
    TASK_IDS,
    FINAL_SEEDS,
    read_json,
    source_hash,
    write_json,
)


PROTOCOL = "E1-supervised-v1-durable-validation"
FROZEN_SOURCE_HASH = "456e3020bd405ec39c7b4cbab3255d0bb3cb2a8a4bda3e37cc0620d51b559873"
PRECISION = "bf16"
ATTENTION_IMPLEMENTATION = "sdpa"
EPOCHS = 30
EFFECTIVE_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 32
MIN_FREE_MEMORY_GIB = 8.0
DEFAULT_DEVICES = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")


class OrchestrationBlocked(RuntimeError):
    """The script stopped safely and requires an operator decision/retry."""


@dataclass(frozen=True)
class TrialSpec:
    task_id: str
    stage: str
    learning_rate: float
    seed: int
    expected_dir: str

    @property
    def key(self) -> tuple[str, str, float, int]:
        return (self.task_id, self.stage, self.learning_rate, self.seed)

    def slug(self) -> str:
        lr = f"{self.learning_rate:.0e}"
        return f"{self.task_id}__{self.stage}__{lr}__seed{self.seed}"


def lr_slug(value: float) -> str:
    return f"{value:.0e}"


def make_spec(
    task_id: str,
    stage: str,
    learning_rate: float,
    seed: int,
    output_root: Path,
) -> TrialSpec:
    expected = output_root / task_id / stage / lr_slug(learning_rate) / f"seed_{seed}"
    return TrialSpec(task_id, stage, float(learning_rate), int(seed), str(expected))


def screening_specs(output_root: Path, tasks: Sequence[str] = TASK_IDS) -> list[TrialSpec]:
    return [
        make_spec(task, "screening", lr, SCREENING_SEED, output_root)
        for task in tasks
        for lr in LEARNING_RATES
    ]


def shortlist_specs(
    output_root: Path,
    selected_lrs: dict[str, Sequence[float]],
) -> list[TrialSpec]:
    return [
        make_spec(task, "shortlist", lr, seed, output_root)
        for task in TASK_IDS
        for lr in selected_lrs[task]
        for seed in (43, 44)
    ]


def final_specs(output_root: Path, selected_lrs: dict[str, float]) -> list[TrialSpec]:
    return [
        make_spec(task, "final", selected_lrs[task], seed, output_root)
        for task in TASK_IDS
        for seed in (45, 46)
    ]


def _float_equal(left: Any, right: float) -> bool:
    return isinstance(left, (int, float)) and math.isclose(float(left), right, rel_tol=0, abs_tol=1e-12)


def _required_artifacts(artifact_dir: Path) -> bool:
    return all(
        (artifact_dir / filename).is_file()
        for filename in (
            "metadata.json",
            "resolved_config.json",
            "epoch_curves.json",
            "best_checkpoint.pt",
            "best_validation_metrics.json",
            "validation_predictions.json",
        )
    )


def metadata_identity_matches(metadata: dict[str, Any], spec: TrialSpec) -> bool:
    return (
        metadata.get("task_id") == spec.task_id
        and metadata.get("stage") == spec.stage
        and _float_equal(metadata.get("learning_rate"), spec.learning_rate)
        and metadata.get("seed") == spec.seed
    )


def metadata_matches(
    metadata: dict[str, Any],
    artifact_dir: Path,
    spec: TrialSpec,
    *,
    expected_hash: str = FROZEN_SOURCE_HASH,
) -> bool:
    """Return true only for a complete, exact, non-smoke validation trial."""

    runtime = metadata.get("runtime")
    if not isinstance(runtime, dict):
        return False
    if not metadata_identity_matches(metadata, spec):
        return False
    if metadata.get("status") != "completed":
        return False
    if metadata.get("source_hash") != expected_hash:
        return False
    if metadata.get("epochs") != EPOCHS:
        return False
    if metadata.get("effective_batch_size") != EFFECTIVE_BATCH_SIZE:
        return False
    if metadata.get("micro_batch_size") != MICRO_BATCH_SIZE:
        return False
    if metadata.get("precision") not in (None, PRECISION) and runtime.get("precision") != PRECISION:
        return False
    if runtime.get("precision") != PRECISION:
        return False
    if metadata.get("attention_implementation") != ATTENTION_IMPLEMENTATION:
        return False
    if metadata.get("include_test") is not False:
        return False
    if metadata.get("smoke") is True:
        return False
    if not isinstance(metadata.get("best_epoch"), int) or metadata["best_epoch"] < 1:
        return False
    if not isinstance(metadata.get("best_selection_f1"), (int, float)):
        return False
    if any((artifact_dir / filename).exists() for filename in (
        "test_metrics.json",
        "test_predictions.json",
        "test_evaluation_metadata.json",
    )):
        return False
    return _required_artifacts(artifact_dir)


def _read_metadata(path: Path) -> dict[str, Any] | None:
    try:
        value = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return value


def candidate_records(output_root: Path, spec: TrialSpec) -> list[dict[str, Any]]:
    """Find exact completed artifacts and reject identity collisions."""

    matches: list[dict[str, Any]] = []
    identity_collisions: list[str] = []
    if not output_root.exists():
        return matches
    for metadata_path in sorted(output_root.rglob("metadata.json")):
        artifact_dir = metadata_path.parent
        if ".orchestrator" in artifact_dir.parts:
            continue
        # Task-level smoke artifacts intentionally use the same task/stage/LR/
        # seed identity as a real screening trial, but are capped to a few
        # rows/epochs and must never become identity collisions in the durable
        # validation queue.  Keep them under an explicit ``smoke`` directory
        # and exclude them before applying the strict completed-trial contract.
        if "smoke" in artifact_dir.parts:
            continue
        metadata = _read_metadata(metadata_path)
        if metadata is None:
            continue
        if metadata_identity_matches(metadata, spec):
            if metadata_matches(metadata, artifact_dir, spec):
                matches.append({"artifact_dir": str(artifact_dir.resolve()), **metadata})
            else:
                identity_collisions.append(str(artifact_dir))
    if identity_collisions:
        raise OrchestrationBlocked(
            "Found identity-matching artifacts that fail the frozen validation contract: "
            + ", ".join(identity_collisions)
        )
    return matches


def score(record: dict[str, Any]) -> float:
    value = record.get("best_selection_f1")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise OrchestrationBlocked(f"Approved record lacks finite best_selection_f1: {record}")
    return float(value)


def select_top_two_lrs(records: Sequence[dict[str, Any]]) -> dict[str, list[float]]:
    """Select exactly two LRs/task from the complete seed-42 screening set."""

    selected: dict[str, list[float]] = {}
    for task in TASK_IDS:
        rows = [row for row in records if row.get("task_id") == task]
        by_lr: dict[float, list[dict[str, Any]]] = {}
        for row in rows:
            by_lr.setdefault(float(row["learning_rate"]), []).append(row)
        if set(by_lr) != set(LEARNING_RATES):
            raise OrchestrationBlocked(f"Screening is incomplete for {task}: {sorted(by_lr)}")
        if any(len(values) != 1 for values in by_lr.values()):
            raise OrchestrationBlocked(f"Screening has duplicate LR records for {task}")
        if any(int(values[0].get("seed", -1)) != SCREENING_SEED for values in by_lr.values()):
            raise OrchestrationBlocked(f"Screening seed mismatch for {task}")
        ordered = sorted(
            ((score(values[0]), lr) for lr, values in by_lr.items()),
            key=lambda item: (-item[0], item[1]),
        )
        selected[task] = [lr for _, lr in ordered[:2]]
    return selected


def select_final_lrs(
    screening_records: Sequence[dict[str, Any]],
    shortlist_records: Sequence[dict[str, Any]],
    top_two: dict[str, Sequence[float]],
) -> dict[str, float]:
    selected: dict[str, float] = {}
    for task in TASK_IDS:
        rows = [row for row in screening_records + shortlist_records if row.get("task_id") == task]
        candidate_scores: list[tuple[float, float]] = []
        for lr in top_two[task]:
            lr_rows = [row for row in rows if _float_equal(row.get("learning_rate"), lr)]
            by_seed: dict[int, list[dict[str, Any]]] = {}
            for row in lr_rows:
                by_seed.setdefault(int(row.get("seed", -1)), []).append(row)
            if set(by_seed) != set(SHORTLIST_SEEDS) or any(len(v) != 1 for v in by_seed.values()):
                raise OrchestrationBlocked(f"Shortlist is incomplete or duplicated for {task} LR {lr:g}")
            mean = sum(score(values[0]) for values in by_seed.values()) / len(SHORTLIST_SEEDS)
            candidate_scores.append((mean, lr))
        candidate_scores.sort(key=lambda item: (-item[0], item[1]))
        selected[task] = candidate_scores[0][1]
    return selected


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


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class StateStore:
    def __init__(self, root: Path, source_hash_value: str) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = root / "state.json"
        if self.state_path.is_file():
            self.state = read_json(self.state_path)
            if self.state.get("protocol") != PROTOCOL:
                raise OrchestrationBlocked("Existing orchestration state uses another protocol")
            if self.state.get("source_hash") != source_hash_value:
                raise OrchestrationBlocked("Existing orchestration state has a different source hash")
        else:
            self.state = {
                "protocol": PROTOCOL,
                "source_hash": source_hash_value,
                "created_at": _now(),
                "updated_at": _now(),
                "jobs": [],
                "approved": {"screening": [], "shortlist": [], "final": []},
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
                "source_hash": self.state["source_hash"],
                "approved": self.state.get("approved", {}),
                "reused": self.state.get("reused", []),
                "created_at": self.state.get("created_at"),
                "updated_at": _now(),
            },
        )


def _lock_orchestration(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    handle = (root / "orchestrator.lock").open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise OrchestrationBlocked("Another validation orchestrator is already running") from exc
    return handle


def _exact_record_for_spec(store: StateStore, spec: TrialSpec) -> dict[str, Any] | None:
    matches = candidate_records(Path(spec.expected_dir).parents[3], spec)
    if len(matches) > 1:
        raise OrchestrationBlocked(f"Ambiguous exact completed artifacts for {spec.key}")
    return matches[0] if matches else None


def _candidate_record(output_root: Path, spec: TrialSpec) -> dict[str, Any] | None:
    matches = candidate_records(output_root, spec)
    if len(matches) > 1:
        raise OrchestrationBlocked(f"Ambiguous exact completed artifacts for {spec.key}")
    return matches[0] if matches else None


def _attempt_dir(spec: TrialSpec, attempt_root: Path) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    base = attempt_root / f"{spec.slug()}__attempt_{stamp}"
    counter = 1
    candidate = base
    while candidate.exists():
        candidate = attempt_root / f"{base.name}_{counter}"
        counter += 1
    return candidate


def _command(spec: TrialSpec, attempt_dir: Path, repo_root: Path, device: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "goibert.train",
        "--task-id",
        spec.task_id,
        "--learning-rate",
        repr(spec.learning_rate),
        "--seed",
        str(spec.seed),
        "--repo-root",
        str(repo_root),
        "--output-dir",
        str(attempt_dir),
        "--device",
        device,
        "--precision",
        PRECISION,
        "--attention-implementation",
        ATTENTION_IMPLEMENTATION,
        "--stage",
        spec.stage,
        "--expected-source-hash",
        FROZEN_SOURCE_HASH,
    ]


def _validate_completed_artifact(path: Path, spec: TrialSpec) -> dict[str, Any]:
    metadata_path = path / "metadata.json"
    metadata = _read_metadata(metadata_path)
    if metadata is None or not metadata_matches(metadata, path, spec):
        raise OrchestrationBlocked(f"Completed process produced invalid artifact for {spec.key}: {path}")
    return {"artifact_dir": str(path.resolve()), **metadata}


def _mark_job(store: StateStore, job: dict[str, Any], status: str, **extra: Any) -> None:
    job.update({"status": status, **extra, "updated_at": _now()})
    store.save()


def _append_approved(store: StateStore, stage: str, record: dict[str, Any]) -> None:
    rows = store.state.setdefault("approved", {}).setdefault(stage, [])
    key = (
        record.get("task_id"),
        record.get("stage"),
        float(record.get("learning_rate")),
        int(record.get("seed")),
    )
    for existing in rows:
        existing_key = (
            existing.get("task_id"),
            existing.get("stage"),
            float(existing.get("learning_rate")),
            int(existing.get("seed")),
        )
        if existing_key == key:
            if existing.get("artifact_dir") != record.get("artifact_dir"):
                raise OrchestrationBlocked(f"Approved manifest has ambiguous record: {key}")
            return
    rows.append(record)


def _recover_running_jobs(store: StateStore, output_root: Path) -> None:
    for job in store.state.get("jobs", []):
        if job.get("status") != "running":
            continue
        pid = int(job.get("pid", -1))
        if _pid_alive(pid):
            raise OrchestrationBlocked(
                f"Existing validation job is still running; refusing duplicate: pid={pid}, "
                f"spec={job.get('spec')}"
            )
        spec_data = job.get("spec") or {}
        spec = TrialSpec(**spec_data)
        artifact = Path(str(job.get("attempt_dir", "")))
        if artifact.is_dir():
            metadata = _read_metadata(artifact / "metadata.json")
            if metadata is not None and metadata_matches(metadata, artifact, spec):
                record = {"artifact_dir": str(artifact.resolve()), **metadata}
                _append_approved(store, spec.stage, record)
                _mark_job(store, job, "completed", artifact_dir=str(artifact.resolve()))
                continue
        store.state.setdefault("failed", []).append({"spec": asdict(spec), "attempt_dir": str(artifact), "reason": "process ended without exact artifact"})
        _mark_job(store, job, "failed", reason="process ended without exact artifact")
        raise OrchestrationBlocked(f"Recovered job failed contract: {spec.key}")


def _launch_job(
    store: StateStore,
    spec: TrialSpec,
    output_root: Path,
    repo_root: Path,
    device: str,
    attempt_root: Path,
) -> tuple[dict[str, Any], subprocess.Popen[str], Any]:
    free = free_memory_gib(device)
    if free < MIN_FREE_MEMORY_GIB:
        raise OrchestrationBlocked(
            f"{device} has only {free:.2f} GiB free; need {MIN_FREE_MEMORY_GIB:.2f} GiB"
        )
    attempt_dir = _attempt_dir(spec, attempt_root)
    log_dir = store.root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{attempt_dir.name}.log"
    command = _command(spec, attempt_dir, repo_root, device)
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
        "attempt_dir": str(attempt_dir),
        "device": device,
        "pid": process.pid,
        "command": command,
        "log_path": str(log_path),
        "status": "running",
        "started_at": _now(),
        "free_memory_gib_at_launch": free,
    }
    store.state.setdefault("jobs", []).append(job)
    store.event("job_started", spec=asdict(spec), pid=process.pid, device=device, log_path=str(log_path))
    # Keep the handle outside the JSON state; only the path is persisted.
    return job, process, log_handle


def _finish_process(
    store: StateStore,
    job: dict[str, Any],
    process: subprocess.Popen[str],
    spec: TrialSpec,
    log_handle: Any,
) -> dict[str, Any]:
    return_code = process.returncode
    log_handle.close()
    attempt_dir = Path(str(job["attempt_dir"]))
    if return_code != 0:
        failure = {
            "spec": asdict(spec),
            "attempt_dir": str(attempt_dir),
            "returncode": return_code,
            "log_path": job.get("log_path"),
            "status": "failed",
        }
        store.state.setdefault("failed", []).append(failure)
        _mark_job(store, job, "failed", returncode=return_code)
        raise OrchestrationBlocked(f"Validation trial failed: {spec.key}; see {job.get('log_path')}")
    record = _validate_completed_artifact(attempt_dir, spec)
    _append_approved(store, spec.stage, record)
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
) -> list[dict[str, Any]]:
    """Run only missing specs, with one active subprocess per device."""

    approved: list[dict[str, Any]] = []
    pending: list[TrialSpec] = []
    for spec in specs:
        exact = _candidate_record(output_root, spec)
        if exact is not None:
            approved.append(exact)
            _append_approved(store, spec.stage, exact)
            store.state.setdefault("reused", []).append({"spec": asdict(spec), **exact})
            store.event("record_reused", spec=asdict(spec), artifact_dir=exact["artifact_dir"])
            continue
        expected_dir = Path(spec.expected_dir)
        if expected_dir.exists():
            raise OrchestrationBlocked(
                f"Expected artifact path is occupied without an exact completed record; "
                f"refusing duplicate: {expected_dir}"
            )
        pending.append(spec)
    store.approved_manifest()
    active: dict[str, tuple[dict[str, Any], subprocess.Popen[str], TrialSpec, Any]] = {}
    attempt_root = store.root / "attempts"
    attempt_root.mkdir(parents=True, exist_ok=True)
    while pending or active:
        launched = False
        for device in devices:
            if device in active or not pending:
                continue
            # Memory is checked immediately before each spawn.  A busy device
            # is left for the next polling round rather than being occupied.
            try:
                free = free_memory_gib(device)
            except OrchestrationBlocked:
                raise
            if free < MIN_FREE_MEMORY_GIB:
                continue
            spec = pending.pop(0)
            job, process, log_handle = _launch_job(
                store, spec, output_root, repo_root, device, attempt_root
            )
            active[device] = (job, process, spec, log_handle)
            launched = True
        completed_devices: list[str] = []
        for device, (job, process, spec, log_handle) in list(active.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            approved.append(_finish_process(store, job, process, spec, log_handle))
            completed_devices.append(device)
        for device in completed_devices:
            del active[device]
        store.approved_manifest()
        if pending or active:
            if not launched and not completed_devices:
                time.sleep(poll_seconds)
    return approved


def _stage_records(store: StateStore, stage: str) -> list[dict[str, Any]]:
    rows = list(store.state.setdefault("approved", {}).get(stage, []))
    keys = [(row.get("task_id"), row.get("stage"), float(row.get("learning_rate")), int(row.get("seed"))) for row in rows]
    if len(keys) != len(set(keys)):
        raise OrchestrationBlocked(f"Approved {stage} manifest contains duplicate records")
    return rows


def _write_stage_manifest(store: StateStore, stage: str, expected: int, records: Sequence[dict[str, Any]], reused: Sequence[dict[str, Any]] = ()) -> None:
    if len(records) != expected:
        raise OrchestrationBlocked(f"{stage} expected {expected} approved records, found {len(records)}")
    store.state.setdefault("approved", {})[stage] = list(records)
    payload = {
        "protocol": PROTOCOL,
        "source_hash": FROZEN_SOURCE_HASH,
        "stage": stage,
        "expected_count": expected,
        "approved_count": len(records),
        "approved": list(records),
        "reused": list(reused),
        "created_at": _now(),
    }
    write_json(store.root / f"{stage}_approved.json", payload)
    store.state.setdefault("stages", {})[stage] = {
        "status": "completed",
        "expected_count": expected,
        "approved_count": len(records),
        "reused_count": len(reused),
    }
    store.approved_manifest()


def _wait_for_pilot(
    output_root: Path,
    pilot_path: Path,
    poll_seconds: float,
    pilot_pid: int | None = None,
) -> None:
    """Wait for an externally launched pilot, without starting a duplicate."""

    while True:
        metadata = _read_metadata(pilot_path / "metadata.json") if pilot_path.is_dir() else None
        pilot_spec = TrialSpec("vsfc_sentiment", "screening", 2e-5, 42, str(pilot_path))
        if metadata is not None and metadata_matches(metadata, pilot_path, pilot_spec):
            return
        if metadata is not None and metadata.get("status") == "completed":
            raise OrchestrationBlocked(
                f"External pilot completed but does not match frozen recipe/hash: {pilot_path}"
            )
        if pilot_pid is not None and not _pid_alive(pilot_pid):
            raise OrchestrationBlocked(
                f"External pilot PID {pilot_pid} ended without a completed metadata artifact: {pilot_path}"
            )
        if pilot_path.exists():
            print(f"WAIT: external pilot path exists but is incomplete: {pilot_path}", flush=True)
        else:
            print(f"WAIT: waiting for external pilot path: {pilot_path}", flush=True)
        time.sleep(poll_seconds)


def plan_payload(output_root: Path) -> dict[str, Any]:
    top = 25
    shortlist = 20
    final = 10
    return {
        "protocol": PROTOCOL,
        "source_hash": FROZEN_SOURCE_HASH,
        "precision": PRECISION,
        "attention_implementation": ATTENTION_IMPLEMENTATION,
        "devices": list(DEFAULT_DEVICES),
        "min_free_memory_gib": MIN_FREE_MEMORY_GIB,
        "stages": {
            "screening": {"new_trials_max": top, "expected_approved": top},
            "shortlist": {"new_trials_max": shortlist, "expected_approved": shortlist},
            "final": {"new_trials_max": final, "expected_approved": final},
        },
        "output_root": str(output_root),
    }


def run_orchestration(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    orchestrator_root = output_root / ".orchestrator"
    actual_hash = source_hash(repo_root)
    if actual_hash != FROZEN_SOURCE_HASH:
        raise OrchestrationBlocked(
            f"Frozen runner hash mismatch: expected {FROZEN_SOURCE_HASH}, found {actual_hash}"
        )
    if not args.execute:
        plan_path = orchestrator_root / "plan.json"
        write_json(plan_path, plan_payload(output_root))
        print(json.dumps({"plan": str(plan_path), **plan_payload(output_root)}, indent=2))
        return 0
    if args.stage == "test":
        raise OrchestrationBlocked("This validation orchestrator has no test stage")
    lock = _lock_orchestration(orchestrator_root)
    try:
        store = StateStore(orchestrator_root, FROZEN_SOURCE_HASH)
        _recover_running_jobs(store, output_root)
        if store.state.get("failed"):
            raise OrchestrationBlocked(
                "Existing failed validation attempts require operator review; "
                "automatic retry is disabled"
            )
        if args.wait_for_pilot:
            _wait_for_pilot(
                output_root,
                args.pilot_path.resolve(),
                args.poll_seconds,
                args.pilot_pid,
            )
        tasks = tuple(args.tasks)
        if args.stage in {"all", "screening"}:
            expected = screening_specs(output_root, tasks)
            records = run_specs(
                store,
                expected,
                output_root=output_root,
                repo_root=repo_root,
                devices=args.devices,
                poll_seconds=args.poll_seconds,
            )
            if tasks == TASK_IDS:
                _write_stage_manifest(store, "screening", 25, records)
            else:
                _write_stage_manifest(store, "screening", len(expected), records)
            if args.stage == "screening":
                return 0
        screening = _stage_records(store, "screening")
        top_two = select_top_two_lrs(screening)
        if args.stage in {"all", "shortlist"}:
            expected_shortlist = [
                make_spec(task, "shortlist", lr, seed, output_root)
                for task in tasks
                for lr in top_two[task]
                for seed in (43, 44)
            ]
            shortlist_new = run_specs(
                store,
                expected_shortlist,
                output_root=output_root,
                repo_root=repo_root,
                devices=args.devices,
                poll_seconds=args.poll_seconds,
            )
            if tasks == TASK_IDS:
                _write_stage_manifest(store, "shortlist", 20, shortlist_new)
            else:
                _write_stage_manifest(store, "shortlist", len(expected_shortlist), shortlist_new)
            reused = [
                row
                for row in screening
                if row.get("task_id") in tasks
                and float(row.get("learning_rate")) in set(top_two[row["task_id"]])
            ]
            store.state.setdefault("stages", {}).setdefault("shortlist", {})["reused_seed42"] = reused
            store.approved_manifest()
            if args.stage == "shortlist":
                return 0
        shortlist = _stage_records(store, "shortlist")
        selected = select_final_lrs(screening, shortlist, top_two)
        if args.stage in {"all", "final"}:
            expected_final = [
                make_spec(task, "final", selected[task], seed, output_root)
                for task in tasks
                for seed in (45, 46)
            ]
            final_new = run_specs(
                store,
                expected_final,
                output_root=output_root,
                repo_root=repo_root,
                devices=args.devices,
                poll_seconds=args.poll_seconds,
            )
            _write_stage_manifest(store, "final", len(expected_final), final_new)
            store.state.setdefault("stages", {}).setdefault("final", {})["selected_learning_rates"] = selected
            store.approved_manifest()
        return 0
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Durable validation-only E1 orchestrator")
    parser.add_argument("--stage", choices=("all", "screening", "shortlist", "final", "test"), default="all")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "artifacts" / "e1")
    parser.add_argument("--tasks", nargs="+", choices=TASK_IDS, default=list(TASK_IDS))
    parser.add_argument("--devices", nargs="+", default=list(DEFAULT_DEVICES))
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--wait-for-pilot", action="store_true")
    parser.add_argument(
        "--pilot-path",
        type=Path,
        default=REPO_ROOT / "artifacts" / "e1" / "vsfc_sentiment" / "screening" / "2e-05" / "seed_42",
    )
    parser.add_argument("--pilot-pid", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
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
