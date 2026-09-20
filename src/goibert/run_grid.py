"""E1 trial queue and selection stages.

The default command is a dry run.  Launching processes requires ``--execute``
and an explicit device list, so an accidental invocation cannot start the
55-trial budget.  One subprocess is assigned to each device at a time.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from .train import (
    FINAL_SEEDS,
    LEARNING_RATES,
    SCREENING_SEED,
    SHORTLIST_SEEDS,
    TASK_IDS,
    evaluate_locked_checkpoint,
    read_json,
    source_hash,
    write_json,
)


def lr_slug(value: float) -> str:
    return f"{value:.0e}"


@dataclass(frozen=True)
class TrialSpec:
    task_id: str
    stage: str
    learning_rate: float
    seed: int
    output_dir: str

    def command(
        self,
        *,
        repo_root: Path,
        device: str,
        precision: str,
        attention_implementation: str,
        source_hash_value: str,
    ) -> list[str]:
        return [
            sys.executable,
            "-m",
            "goibert.train",
            "--task-id",
            self.task_id,
            "--learning-rate",
            repr(self.learning_rate),
            "--seed",
            str(self.seed),
            "--repo-root",
            str(repo_root),
            "--output-dir",
            self.output_dir,
            "--device",
            device,
            "--precision",
            precision,
            "--attention-implementation",
            attention_implementation,
            "--stage",
            self.stage,
            "--expected-source-hash",
            source_hash_value,
        ]


def make_trial_spec(
    *,
    task_id: str,
    stage: str,
    learning_rate: float,
    seed: int,
    output_root: Path,
) -> TrialSpec:
    path = output_root / task_id / stage / lr_slug(learning_rate) / f"seed_{seed}"
    return TrialSpec(task_id, stage, learning_rate, seed, str(path))


def _latest_metadata(output_root: Path, *, stage: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not output_root.is_dir():
        return records
    for path in sorted(output_root.glob(f"*/{stage}/*/seed_*/metadata.json")):
        try:
            record = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        record["artifact_dir"] = str(path.parent)
        records.append(record)
    return records


def _successful(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if r.get("status") == "completed" and "best_selection_f1" in r]


def _score(record: dict[str, Any]) -> float:
    value = record.get("best_selection_f1")
    if not isinstance(value, (int, float)):
        raise ValueError(f"Trial has no numeric best_selection_f1: {record}")
    return float(value)


def _same_hash(records: Iterable[dict[str, Any]], expected: str) -> None:
    mismatches = [
        (r.get("artifact_dir"), r.get("source_hash"))
        for r in records
        if r.get("source_hash") != expected
    ]
    if mismatches:
        raise RuntimeError(
            "Refusing to aggregate mixed source versions; mismatches: "
            + json.dumps(mismatches, ensure_ascii=False)
        )


def _screening_specs(tasks: Sequence[str], output_root: Path) -> list[TrialSpec]:
    return [
        make_trial_spec(
            task_id=task,
            stage="screening",
            learning_rate=lr,
            seed=SCREENING_SEED,
            output_root=output_root,
        )
        for task in tasks
        for lr in LEARNING_RATES
    ]


def _shortlist_specs(tasks: Sequence[str], output_root: Path) -> list[TrialSpec]:
    screening = _successful(_latest_metadata(output_root, stage="screening"))
    specs: list[TrialSpec] = []
    for task in tasks:
        rows = [r for r in screening if r.get("task_id") == task]
        by_lr: dict[float, list[dict[str, Any]]] = {}
        for row in rows:
            by_lr.setdefault(float(row["learning_rate"]), []).append(row)
        if set(by_lr) != set(LEARNING_RATES):
            raise RuntimeError(f"Shortlist requires all screening LRs for {task}")
        for lr, candidates in by_lr.items():
            if len(candidates) != 1 or int(candidates[0].get("seed", -1)) != SCREENING_SEED:
                raise RuntimeError(
                    f"Expected exactly one seed-42 screening result for {task} LR {lr:g}"
                )
        winners = sorted(rows, key=lambda r: (-_score(r), float(r["learning_rate"])))[:2]
        for winner in winners:
            for seed in SHORTLIST_SEEDS:
                if seed == SCREENING_SEED:
                    continue
                specs.append(
                    make_trial_spec(
                        task_id=task,
                        stage="shortlist",
                        learning_rate=float(winner["learning_rate"]),
                        seed=seed,
                        output_root=output_root,
                    )
                )
    return specs


def selected_learning_rates(output_root: Path, tasks: Sequence[str]) -> dict[str, float]:
    screening = _successful(_latest_metadata(output_root, stage="screening"))
    shortlist = _successful(_latest_metadata(output_root, stage="shortlist"))
    selected: dict[str, float] = {}
    for task in tasks:
        screen_rows = [r for r in screening if r.get("task_id") == task]
        short_rows = [r for r in shortlist if r.get("task_id") == task]
        by_lr_seed: dict[tuple[float, int], list[float]] = {}
        for row in screen_rows + short_rows:
            key = (float(row["learning_rate"]), int(row["seed"]))
            by_lr_seed.setdefault(key, []).append(_score(row))
        by_lr: dict[float, list[float]] = {}
        for (lr, seed), scores in by_lr_seed.items():
            if len(scores) != 1:
                raise RuntimeError(f"Duplicate validation result for {task} LR {lr:g} seed {seed}")
            by_lr.setdefault(lr, []).extend(scores)
        candidates = [
            (sum(scores) / len(scores), lr)
            for lr, scores in by_lr.items()
            if set(seed for (candidate_lr, seed) in by_lr_seed if candidate_lr == lr)
            == set(SHORTLIST_SEEDS)
        ]
        if len(candidates) < 2:
            raise RuntimeError(f"Shortlist requires 3 validation seeds for {task}")
        # Highest mean, then lower LR.  The epoch tie-break already happened
        # within each trial by keeping the earliest best epoch.
        candidates.sort(key=lambda item: (-item[0], item[1]))
        selected[task] = candidates[0][1]
    return selected


def _final_specs(tasks: Sequence[str], output_root: Path) -> list[TrialSpec]:
    selected = selected_learning_rates(output_root, tasks)
    return [
        make_trial_spec(
            task_id=task,
            stage="final",
            learning_rate=selected[task],
            seed=seed,
            output_root=output_root,
        )
        for task in tasks
        for seed in (45, 46)
    ]


def _plan_index(stage: str, specs: Sequence[TrialSpec], launch_hash: str) -> dict[str, Any]:
    return {
        "protocol": "E1-supervised-v1",
        "stage": stage,
        "launch_source_hash": launch_hash,
        "trial_count": len(specs),
        "trials": [asdict(spec) for spec in specs],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _write_index(output_root: Path, stage: str, payload: dict[str, Any]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = output_root / f"{stage}_index_{stamp}.json"
    counter = 1
    while path.exists():
        path = output_root / f"{stage}_index_{stamp}_{counter}.json"
        counter += 1
    write_json(path, payload)
    return path


def _run_one(
    spec: TrialSpec,
    *,
    repo_root: Path,
    device: str,
    precision: str,
    attention_implementation: str,
    launch_hash: str,
) -> dict[str, Any]:
    output_dir = Path(spec.output_dir)
    allocated = output_dir
    suffix = 1
    while allocated.exists():
        allocated = output_dir.with_name(f"{output_dir.name}__rerun{suffix}")
        suffix += 1
    spec = TrialSpec(
        task_id=spec.task_id,
        stage=spec.stage,
        learning_rate=spec.learning_rate,
        seed=spec.seed,
        output_dir=str(allocated),
    )
    output_dir = allocated
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_dir.parent / f"{output_dir.name}.launch.log"
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root / "src") + (os.pathsep + existing if existing else "")
    command = spec.command(
        repo_root=repo_root,
        device=device,
        precision=precision,
        attention_implementation=attention_implementation,
        source_hash_value=launch_hash,
    )
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        failure = {
            "status": "failed",
            "returncode": completed.returncode,
            "command": command,
            "device": device,
            "log_path": str(log_path),
        }
        write_json(output_dir.parent / f"seed_{spec.seed}.failure.json", failure)
        return failure | {"task_id": spec.task_id, "stage": spec.stage, "seed": spec.seed}
    metadata = Path(spec.output_dir) / "metadata.json"
    if not metadata.is_file():
        # A fresh rerun path is used when a previous artifact occupies the
        # planned directory.  Find the completed child without guessing a
        # source version; aggregation still validates its hash.
        candidates = sorted(output_dir.parent.glob(f"seed_{spec.seed}*/metadata.json"))
        if len(candidates) != 1:
            raise RuntimeError(f"Completed process did not write unique metadata: {spec}")
        metadata = candidates[0]
    result = read_json(metadata)
    result["artifact_dir"] = str(metadata.parent)
    result["device"] = device
    result["log_path"] = str(log_path)
    return result


def run_specs(
    specs: Sequence[TrialSpec],
    *,
    repo_root: Path,
    devices: Sequence[str],
    precision: str,
    attention_implementation: str,
    launch_hash: str,
) -> list[dict[str, Any]]:
    if not devices:
        raise ValueError("At least one approved device is required")
    results: list[dict[str, Any]] = []
    # A batch contains at most one process per approved device.  The next
    # batch starts only after every process in the current batch exits.
    for offset in range(0, len(specs), len(devices)):
        batch = list(specs[offset : offset + len(devices)])
        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            futures = {
                pool.submit(
                    _run_one,
                    spec,
                    repo_root=repo_root,
                    device=devices[index],
                    precision=precision,
                    attention_implementation=attention_implementation,
                    launch_hash=launch_hash,
                ): spec
                for index, spec in enumerate(batch)
            }
            for future in as_completed(futures):
                results.append(future.result())
    return results


def _resolve_devices(value: str, *, execute: bool) -> list[str]:
    if value != "auto":
        devices = [part.strip() for part in value.split(",") if part.strip()]
        if not devices:
            raise ValueError("--devices must contain at least one device")
        return devices
    if not execute:
        return ["cuda:0"]
    # Do not inspect or claim free GPUs here; environment_smoke/root must pass
    # the approved allocation explicitly.  ``auto`` is intentionally rejected
    # for execution to avoid colliding with another workload.
    raise ValueError("--devices auto is only valid for --dry-run; pass root-approved IDs")


def _check_gpu_memory(devices: Sequence[str], minimum_gib: float) -> None:
    """Fail before launch when an approved GPU lacks the measured headroom."""

    cuda_devices = [device for device in devices if device.startswith("cuda")]
    if not cuda_devices:
        return
    indices: list[str] = []
    for device in cuda_devices:
        match = re.search(r"(\d+)$", device)
        if match is None:
            raise ValueError(f"Cannot map CUDA device {device!r} to nvidia-smi index")
        indices.append(match.group(1))
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if query.returncode != 0:
        raise RuntimeError(f"nvidia-smi memory preflight failed: {query.stderr.strip()}")
    free_by_index: dict[str, float] = {}
    for line in query.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2:
            free_by_index[parts[0]] = float(parts[1]) / 1024.0
    missing = [index for index in indices if index not in free_by_index]
    if missing:
        raise RuntimeError(f"nvidia-smi did not report approved GPU(s): {missing}")
    below = {
        index: round(free_by_index[index], 2)
        for index in indices
        if free_by_index[index] < minimum_gib
    }
    if below:
        raise RuntimeError(
            f"Insufficient free GPU memory before launch (need >= {minimum_gib:.2f} GiB): {below}"
        )


def _build_stage_specs(stage: str, tasks: Sequence[str], output_root: Path) -> list[TrialSpec]:
    if stage == "screening":
        return _screening_specs(tasks, output_root)
    if stage == "shortlist":
        return _shortlist_specs(tasks, output_root)
    if stage == "final":
        return _final_specs(tasks, output_root)
    raise ValueError(f"Training grid does not support stage {stage!r}")


def _test_records(selection_manifest: Path, *, expected_source_hash: str) -> list[dict[str, Any]]:
    manifest = read_json(selection_manifest)
    if manifest.get("protocol") != "E1-test-gate-v1" or manifest.get("locked") is not True:
        raise RuntimeError("Test stage requires locked E1-test-gate-v1 selection manifest")
    if manifest.get("source_hash") != expected_source_hash:
        raise RuntimeError("Selection manifest source hash differs from current runner")
    rows = manifest.get("selections")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("Selection manifest has no checkpoint selections")
    records = [row for row in rows if isinstance(row, dict)]
    if len(records) != len(rows):
        raise RuntimeError("Selection manifest contains a malformed selection")
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        task = row.get("task_id")
        seed = row.get("seed")
        artifact = row.get("artifact_dir")
        if task not in TASK_IDS or not isinstance(seed, int) or not isinstance(artifact, str):
            raise RuntimeError("Selection manifest row lacks task_id, integer seed, or artifact_dir")
        required_fields = {
            "checkpoint_sha256",
            "metadata_sha256",
            "backbone_hashes",
            "tokenizer_hashes",
            "data_provenance",
            "precision",
            "learning_rate",
            "best_epoch",
            "model",
        }
        missing = sorted(field for field in required_fields if field not in row)
        if missing:
            raise RuntimeError(f"Selection manifest row for {task}/seed{seed} misses {missing}")
        by_task.setdefault(task, []).append(row)
    required_seeds = set(FINAL_SEEDS)
    for task, task_rows in by_task.items():
        seeds = [int(row["seed"]) for row in task_rows]
        if len(seeds) != len(set(seeds)) or set(seeds) != required_seeds:
            raise RuntimeError(
                f"Test selection must contain exactly one checkpoint for seeds {sorted(required_seeds)} "
                f"per task; {task} has {seeds}"
            )
    return records


def _run_test_one(
    row: dict[str, Any],
    *,
    selection_manifest: Path,
    repo_root: Path,
    device: str,
    precision: str,
    attention_implementation: str,
) -> dict[str, Any]:
    return evaluate_locked_checkpoint(
        selection_manifest=selection_manifest,
        artifact_dir=Path(str(row["artifact_dir"])),
        repo_root=repo_root,
        device_name=device,
        precision=precision,
        attention_implementation=attention_implementation,
    )


def run_test_records(
    records: Sequence[dict[str, Any]],
    *,
    selection_manifest: Path,
    repo_root: Path,
    devices: Sequence[str],
    precision: str,
    attention_implementation: str,
) -> list[dict[str, Any]]:
    if not devices:
        raise ValueError("At least one approved device is required")
    results: list[dict[str, Any]] = []
    for offset in range(0, len(records), len(devices)):
        batch = list(records[offset : offset + len(devices)])
        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            futures = {
                pool.submit(
                    _run_test_one,
                    row,
                    selection_manifest=selection_manifest,
                    repo_root=repo_root,
                    device=devices[index],
                    precision=precision,
                    attention_implementation=attention_implementation,
                ): row
                for index, row in enumerate(batch)
            }
            for future in as_completed(futures):
                results.append(future.result())
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan or launch GoiBERT E1 trials")
    parser.add_argument("--stage", choices=("screening", "shortlist", "final", "test"), required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASK_IDS, default=list(TASK_IDS))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/e1"))
    parser.add_argument("--devices", default="auto", help="comma-separated approved IDs, e.g. cuda:0,cuda:1")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--attention-implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--min-free-memory-gib", type=float, default=8.0)
    parser.add_argument("--execute", action="store_true", help="actually launch subprocesses")
    parser.add_argument("--dry-run", action="store_true", help="print/write plan without launching")
    parser.add_argument("--selection-manifest", type=Path, default=None)
    parser.add_argument("--confirm-test-gate", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.execute and args.dry_run:
        raise SystemExit("--execute and --dry-run are mutually exclusive")
    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    launch_hash = source_hash(repo_root)
    if args.stage == "test":
        if args.selection_manifest is None:
            raise SystemExit("--selection-manifest is required for --stage test")
        records = _test_records(args.selection_manifest.resolve(), expected_source_hash=launch_hash)
        payload = {
            "protocol": "E1-test-gate-v1",
            "stage": "test",
            "launch_source_hash": launch_hash,
            "selection_manifest": str(args.selection_manifest.resolve()),
            "trial_count": len(records),
            "selections": records,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        specs: list[TrialSpec] = []
    else:
        # Dependencies must all come from the same source snapshot.  This
        # check prevents a code edit between stages from mixing score tables.
        if args.stage != "screening":
            dependency_records = _successful(
                _latest_metadata(output_root, stage="screening")
                + (_latest_metadata(output_root, stage="shortlist") if args.stage == "final" else [])
            )
            _same_hash(dependency_records, launch_hash)
        specs = _build_stage_specs(args.stage, args.tasks, output_root)
        payload = _plan_index(args.stage, specs, launch_hash)
    index_path = _write_index(output_root, args.stage, payload)
    print(json.dumps({"plan": str(index_path), **payload}, ensure_ascii=False, indent=2))
    if not args.execute:
        return 0
    if args.stage == "test" and not args.confirm_test_gate:
        raise SystemExit("Test stage requires --confirm-test-gate after reviewing the locked manifest")
    devices = _resolve_devices(args.devices, execute=True)
    _check_gpu_memory(devices, args.min_free_memory_gib)
    if args.stage == "test":
        results = run_test_records(
            records,
            selection_manifest=args.selection_manifest.resolve(),
            repo_root=repo_root,
            devices=devices,
            precision=args.precision,
            attention_implementation=args.attention_implementation,
        )
    else:
        results = run_specs(
            specs,
            repo_root=repo_root,
            devices=devices,
            precision=args.precision,
            attention_implementation=args.attention_implementation,
            launch_hash=launch_hash,
        )
        _same_hash(_successful(results), launch_hash)
    completed_payload = payload | {
        "executed": True,
        "devices": list(devices),
        "results": results,
    }
    completed_path = _write_index(output_root, args.stage, completed_payload)
    print(json.dumps({"results": str(completed_path), "count": len(results)}, indent=2))
    return 0 if all(result.get("status") == "completed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
