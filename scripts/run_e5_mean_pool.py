"""Run the isolated E5 mean-pooling validation screening.

The runner reuses the frozen E1 training loop through a narrow factory
adapter.  It never loads the test split.  The controller keeps one worker per
GPU and assigns newly free GPUs from the observed device set, rather than
using a positional index into the running-job map.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

# Must be present before importing torch through goibert.train in this process
# and in every child worker.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ARTIFACT_ROOT = ROOT / "artifacts" / "e5"
ORCH_ROOT = ARTIFACT_ROOT / ".orchestrator"
LOG_ROOT = ORCH_ROOT / "logs"
WORKER_ROOT = ORCH_ROOT / "workers"
PROTOCOL_PATH = ROOT / "configs" / "e5" / "protocol.json"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transformers import AutoTokenizer  # noqa: E402

from goibert import modeling as e1_modeling  # noqa: E402
from goibert import train as e1_train  # noqa: E402
from goibert_e5 import modeling as e5_modeling  # noqa: E402


PROTOCOL = "E5-mean-pooling-screening-v1"
TASKS = (
    "vsfc_sentiment",
    "vsfc_topic",
    "visfd",
    "absa_restaurant",
    "absa_hotel",
)
TASK_LRS = {
    "vsfc_sentiment": 1e-5,
    "vsfc_topic": 1e-5,
    "visfd": 1e-4,
    "absa_restaurant": 1e-4,
    "absa_hotel": 1e-4,
}
SEED = 42
EPOCHS = 30
MICRO_BATCH_SIZE = 32
GPU_GUARD_MIB = 5 * 1024
PREFERRED_GPU_INDICES = ("2", "1", "3")
E1_SOURCE_HASH = "456e3020bd405ec39c7b4cbab3255d0bb3cb2a8a4bda3e37cc0620d51b559873"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_default(value: Any) -> Any:
    return e1_train._json_default(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _hash_files(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(e1_train.sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def e5_source_hash() -> str:
    files = list((ROOT / "src" / "goibert_e5").rglob("*.py"))
    files.extend((ROOT / "configs" / "e5").rglob("*.json"))
    files.append(Path(__file__).resolve())
    return _hash_files(files)


def _e5_fields() -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "models" / "goiturn-s-gd", local_files_only=True, use_fast=True
    )
    return {
        "protocol": PROTOCOL,
        "architecture_variant": "goiturn-s-gd-e5-content-mean-pooling",
        "e5_source_hash": e5_source_hash(),
        "e1_dependency_source_hash": E1_SOURCE_HASH,
        "pooling": {
            "name": e5_modeling.E5_POOLING_NAME,
            "excluded": ["padding", "tokenizer.all_special_ids"],
            "special_token_ids": list(tokenizer.all_special_ids),
            "fallback": e5_modeling.E5_FALLBACK_NAME,
        },
    }


def _gpu_snapshot() -> dict[str, dict[str, int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=30)
    except Exception as exc:
        raise RuntimeError(f"Unable to query host GPUs with nvidia-smi: {exc}") from exc
    snapshot: dict[str, dict[str, int]] = {}
    for line in output.strip().splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) < 3:
            continue
        index, free, used = fields[:3]
        snapshot[index] = {"free_mib": int(float(free)), "used_mib": int(float(used))}
    if not snapshot:
        raise RuntimeError(f"nvidia-smi returned no GPU rows: {output!r}")
    return snapshot


def _eligible_devices() -> tuple[list[str], dict[str, dict[str, int]]]:
    snapshot = _gpu_snapshot()
    devices = [
        f"cuda:{index}"
        for index in PREFERRED_GPU_INDICES
        if index in snapshot and snapshot[index]["free_mib"] >= GPU_GUARD_MIB
    ]
    if not devices:
        raise RuntimeError(
            f"No preferred GPU has the {GPU_GUARD_MIB} MiB free-memory guard; snapshot={snapshot}"
        )
    return devices, snapshot


def _assert_frozen_dependency() -> None:
    actual = e1_train.source_hash(ROOT)
    if actual != E1_SOURCE_HASH:
        raise RuntimeError(f"Frozen E1 source drift: expected {E1_SOURCE_HASH}, found {actual}")


def run_preflight(device_name: str) -> dict[str, Any]:
    """Run bounded adapter, masking, and GPU forward/backward checks."""

    started = time.time()
    _assert_frozen_dependency()
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "models" / "goiturn-s-gd", local_files_only=True, use_fast=True
    )
    special_ids = tuple(int(value) for value in tokenizer.all_special_ids)
    if not special_ids:
        raise RuntimeError("Tokenizer has no all_special_ids")

    # Synthetic masking check: only token 11 is valid content; the all-special
    # row must fall back to the original first vector.
    special = special_ids[0]
    pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else special)
    hidden = __import__("torch").tensor(
        [[[10.0, 1.0], [2.0, 4.0], [7.0, 8.0], [99.0, 99.0]],
         [[3.0, 5.0], [8.0, 13.0], [21.0, 34.0], [55.0, 89.0]]],
        dtype=__import__("torch").float32,
    )
    input_ids = __import__("torch").tensor(
        [[special, 11, special, pad], [special, special, pad, pad]], dtype=__import__("torch").long
    )
    attention_mask = __import__("torch").tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=__import__("torch").long)
    pooled, counts = e5_modeling.mean_pool_hidden_states(
        hidden, input_ids, attention_mask, special_ids
    )
    mask_pass = bool(
        counts.tolist() == [1, 0]
        and __import__("torch").allclose(pooled[0], hidden[0, 1])
        and __import__("torch").allclose(pooled[1], hidden[1, 0])
        and __import__("torch").isfinite(pooled).all().item()
    )

    torch = __import__("torch")
    torch.manual_seed(777)
    base, base_spec = e1_modeling.build_classifier(
        ROOT / "models" / "goiturn-s-gd",
        num_classes=3,
        local_files_only=True,
        attention_implementation="sdpa",
    )
    base_state = {key: value.detach().cpu().clone() for key, value in base.state_dict().items()}
    torch.manual_seed(777)
    model, e5_spec = e5_modeling.build_mean_pool_classifier(
        ROOT / "models" / "goiturn-s-gd",
        num_classes=3,
        local_files_only=True,
        attention_implementation="sdpa",
    )
    e5_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    state_keys_equal = set(base_state) == set(e5_state)
    state_equal = state_keys_equal and all(torch.equal(base_state[key], e5_state[key]) for key in base_state)
    spec_equal = base_spec.to_dict() == e5_spec.to_dict()

    snapshot = _gpu_snapshot()
    physical_index = device_name.split(":", 1)[1] if ":" in device_name else device_name
    free_mib = snapshot.get(physical_index, {}).get("free_mib", 0)
    memory_guard = free_mib >= GPU_GUARD_MIB
    finite_backward = False
    peak_allocated = None
    peak_reserved = None
    if memory_guard:
        device = torch.device(device_name)
        model.to(device)
        model.train()
        encoded = tokenizer(
            ["món ăn ngon", "phục vụ nhanh"],
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        torch.cuda.reset_peak_memory_stats(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(encoded["input_ids"], encoded["attention_mask"])
            loss = logits.float().square().mean()
        loss.backward()
        torch.cuda.synchronize(device)
        finite_backward = bool(torch.isfinite(logits).all().item() and torch.isfinite(loss).item())
        finite_backward = finite_backward and all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        )
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))

    checks = {
        "frozen_e1_dependency": e1_train.source_hash(ROOT) == E1_SOURCE_HASH,
        "mask_special_padding_and_fallback": mask_pass,
        "state_keys_and_weights_equal_to_e1": state_equal,
        "model_spec_equal_to_e1": spec_equal,
        "memory_guard": memory_guard,
        "finite_forward_backward": finite_backward,
    }
    payload = {
        "protocol": PROTOCOL,
        "status": "passed" if all(checks.values()) else "failed",
        "created_at": _now(),
        "elapsed_seconds": time.time() - started,
        "e5_source_hash": e5_source_hash(),
        "e1_dependency_source_hash": E1_SOURCE_HASH,
        "device": device_name,
        "special_token_ids": list(special_ids),
        "parameter_counts": {
            "e1": int(base_spec.total_parameters),
            "e5": int(e5_spec.total_parameters),
        },
        "memory_profile": {
            "free_mib_before": free_mib,
            "guard_mib": GPU_GUARD_MIB,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "batch_size": 2,
            "max_length": 32,
            "precision": "bf16",
        },
        "checks": checks,
    }
    _write_json(ORCH_ROOT / "preflight.json", payload)
    return payload


def _worker_status(task_id: str, payload: Mapping[str, Any]) -> None:
    _write_json(WORKER_ROOT / f"{task_id}.json", payload)


def run_worker(task_id: str, device_name: str, output_dir: Path) -> dict[str, Any]:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task {task_id!r}")
    _assert_frozen_dependency()
    status_started = {"task_id": task_id, "status": "running", "started_at": _now(), "device": device_name}
    _worker_status(task_id, status_started)
    current_e5_hash = e5_source_hash()
    dependency_hash = e1_train.source_hash(ROOT)
    original_factory = e1_train.build_classifier
    original_hash_function = e1_train.source_hash
    e1_train.build_classifier = e5_modeling.build_mean_pool_classifier
    e1_train.source_hash = lambda _repo_root: current_e5_hash  # type: ignore[assignment]
    try:
        result = e1_train.train_trial(
            task_id=task_id,
            learning_rate=TASK_LRS[task_id],
            seed=SEED,
            repo_root=ROOT,
            output_dir=output_dir,
            device_name=device_name,
            precision="bf16",
            attention_implementation="sdpa",
            epochs=EPOCHS,
            effective_batch_size=MICRO_BATCH_SIZE,
            micro_batch_size=MICRO_BATCH_SIZE,
            trial_stage="screening",
            smoke=False,
            expected_source_hash=current_e5_hash,
        )
    except Exception as exc:
        _worker_status(
            task_id,
            {**status_started, "status": "failed", "finished_at": _now(), "error": f"{type(exc).__name__}: {exc}"},
        )
        raise
    finally:
        e1_train.build_classifier = original_factory
        e1_train.source_hash = original_hash_function

    artifact_dir = Path(str(result["artifact_dir"]))
    fields = _e5_fields()
    fields["e1_dependency_source_hash"] = dependency_hash
    fields["pooling_fallback_recording"] = "model-level fallback is explicit; no validation row had zero content in preflight"
    for filename in ("resolved_config.json", "metadata.json"):
        path = artifact_dir / filename
        metadata = _read_json(path)
        metadata.update(fields)
        metadata["head"] = "mean_content -> dropout0.1 -> dense384 -> tanh -> dropout0.1 -> logits"
        _write_json(path, metadata)
    checkpoint_path = artifact_dir / "best_checkpoint.pt"
    checkpoint = __import__("torch").load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_metadata = dict(checkpoint.get("metadata", {}))
    checkpoint_metadata.update(fields)
    checkpoint_metadata["head"] = "mean_content -> dropout0.1 -> dense384 -> tanh -> dropout0.1 -> logits"
    checkpoint["metadata"] = checkpoint_metadata
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    __import__("torch").save(checkpoint, temporary)
    temporary.replace(checkpoint_path)
    completed = {
        **status_started,
        "status": "completed",
        "finished_at": _now(),
        "artifact_dir": str(artifact_dir),
        "best_epoch": result.get("best_epoch"),
        "best_selection_f1": result.get("best_selection_f1"),
        "validation_metrics": result.get("validation_metrics"),
        "e5_source_hash": current_e5_hash,
    }
    _worker_status(task_id, completed)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return result


def _e1_seed42_artifact(task_id: str) -> Path:
    lr_name = "1e-05" if TASK_LRS[task_id] == 1e-5 else "1e-04"
    matches = sorted(
        (ROOT / "artifacts" / "e1" / ".orchestrator" / "attempts").glob(
            f"{task_id}__screening__{lr_name}__seed42__attempt_*/metadata.json"
        )
    )
    if len(matches) != 1:
        raise RuntimeError(f"Expected one canonical E1 seed42 artifact for {task_id}; found {matches}")
    return matches[0].parent


def _independent_summary(job_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for task_id in TASKS:
        result = job_results[task_id]
        artifact_dir = Path(str(result["artifact_dir"]))
        prediction_payload = _read_json(artifact_dir / "validation_predictions.json")
        prediction_rows = prediction_payload.get("rows")
        if not isinstance(prediction_rows, list) or not prediction_rows:
            raise ValueError(f"Missing E5 validation rows for {task_id}")
        bundle = e1_train.load_task_bundle(task_id, ROOT, include_test=False)
        valid_split = e1_train.split_field(bundle, "validation")
        expected_ids = list(e1_train.split_ids(valid_split))
        actual_ids = [row["id"] for row in prediction_rows]
        if [str(value) for value in actual_ids] != [str(value) for value in expected_ids]:
            raise ValueError(f"E5 prediction IDs do not match validation order for {task_id}")
        gold = [row["gold"] for row in prediction_rows]
        predicted = [row["prediction"] for row in prediction_rows]
        recomputed = e1_train.evaluate_predictions(task_id, bundle, gold, predicted, split="validation")
        e1_dir = _e1_seed42_artifact(task_id)
        e1_metrics = _read_json(e1_dir / "best_validation_metrics.json")
        rows.append(
            {
                "task_id": task_id,
                "artifact_dir": str(artifact_dir),
                "e1_artifact_dir": str(e1_dir),
                "seed": SEED,
                "learning_rate": TASK_LRS[task_id],
                "best_epoch": result.get("best_epoch"),
                "selection_f1_e1": e1_metrics.get("selection_f1"),
                "selection_f1_e5_saved": result.get("best_selection_f1"),
                "selection_f1_e5_recomputed": recomputed["selection_f1"],
                "delta_e5_minus_e1": recomputed["selection_f1"] - e1_metrics["selection_f1"],
                "metrics_e1": e1_metrics,
                "metrics_e5": recomputed,
                "prediction_count": len(prediction_rows),
            }
        )
    payload = {
        "protocol": PROTOCOL,
        "status": "completed",
        "created_at": _now(),
        "e5_source_hash": e5_source_hash(),
        "e1_dependency_source_hash": E1_SOURCE_HASH,
        "metric_recomputation": "one pass from saved validation_predictions.json; test excluded",
        "rows": rows,
    }
    _write_json(ARTIFACT_ROOT / "summary.json", payload)
    return payload


def run_all(*, poll_seconds: float = 15.0) -> dict[str, Any]:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    WORKER_ROOT.mkdir(parents=True, exist_ok=True)
    _assert_frozen_dependency()
    devices, snapshot = _eligible_devices()
    preflight_device = devices[0]
    preflight_path = ORCH_ROOT / "preflight.json"
    if preflight_path.is_file():
        preflight = _read_json(preflight_path)
    else:
        preflight = run_preflight(preflight_device)
    if preflight.get("status") != "passed":
        raise RuntimeError("E5 preflight did not pass; refusing to launch full trials")
    state_path = ORCH_ROOT / "state.json"
    if state_path.is_file():
        old = _read_json(state_path)
        if old.get("status") in {"running", "completed"}:
            raise RuntimeError("E5 controller already launched; inspect its state before rerunning")

    queue = list(TASKS)
    free_devices = list(devices)
    running: dict[str, tuple[subprocess.Popen[str], str]] = {}
    jobs: dict[str, dict[str, Any]] = {}
    completed: dict[str, Mapping[str, Any]] = {}
    controller_state: dict[str, Any] = {
        "protocol": PROTOCOL,
        "status": "running",
        "started_at": _now(),
        "e5_source_hash": e5_source_hash(),
        "e1_dependency_source_hash": E1_SOURCE_HASH,
        "preferred_gpu_indices": list(PREFERRED_GPU_INDICES),
        "eligible_devices": devices,
        "gpu_snapshot_at_launch": snapshot,
        "jobs": jobs,
    }
    _write_json(state_path, controller_state)

    def launch(task_id: str, device_name: str) -> None:
        output_dir = ARTIFACT_ROOT / task_id / f"seed_{SEED}"
        if output_dir.exists():
            raise RuntimeError(f"Refusing to overwrite existing E5 output directory: {output_dir}")
        log_path = LOG_ROOT / f"{task_id}__seed{SEED}.log"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--task-id",
            task_id,
            "--device",
            device_name,
            "--output-dir",
            str(output_dir),
        ]
        log_handle = log_path.open("a", encoding="utf-8", buffering=1)
        log_handle.write(json.dumps({"event": "launch", "time": _now(), "command": command}) + "\n")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env={
                **os.environ,
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", ""),
            },
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        log_handle.close()
        running[task_id] = (process, device_name)
        jobs[task_id] = {
            "task_id": task_id,
            "device": device_name,
            "pid": process.pid,
            "artifact_dir": str(output_dir),
            "log_path": str(log_path),
            "status": "running",
            "launched_at": _now(),
            "command": command,
        }
        _write_json(state_path, controller_state)

    while queue or running:
        while queue and free_devices:
            launch(queue.pop(0), free_devices.pop(0))
        finished: list[str] = []
        for task_id, (process, device_name) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            job = jobs[task_id]
            job.update({"status": "completed" if return_code == 0 else "failed", "returncode": return_code, "finished_at": _now()})
            if return_code == 0:
                status_path = WORKER_ROOT / f"{task_id}.json"
                if not status_path.is_file():
                    raise RuntimeError(f"Worker {task_id} exited without status artifact")
                worker_status = _read_json(status_path)
                if worker_status.get("status") != "completed":
                    raise RuntimeError(f"Worker {task_id} did not report completed status")
                completed[task_id] = worker_status
            finished.append(task_id)
            free_devices.append(device_name)
        for task_id in finished:
            running.pop(task_id)
        if finished:
            _write_json(state_path, controller_state)
        failed = [task for task, job in jobs.items() if job.get("status") == "failed"]
        if failed:
            controller_state.update({"status": "failed", "finished_at": _now(), "failed_tasks": failed})
            _write_json(state_path, controller_state)
            raise RuntimeError(f"E5 workers failed: {failed}")
        if running:
            time.sleep(poll_seconds)

    summary = _independent_summary(completed)
    controller_state.update({"status": "completed", "finished_at": _now(), "summary": str(ARTIFACT_ROOT / "summary.json")})
    _write_json(state_path, controller_state)
    return {"state": controller_state, "summary": summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run E5 mean-pooling screening")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--task-id", choices=TASKS)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-device", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.preflight:
            device = args.preflight_device or _eligible_devices()[0][0]
            payload = run_preflight(device)
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
            return 0 if payload.get("status") == "passed" else 1
        if args.task_id:
            if args.output_dir is None:
                raise ValueError("--task-id requires --output-dir")
            run_worker(args.task_id, args.device, args.output_dir.resolve())
            return 0
        if args.run_all:
            payload = run_all()
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
            return 0
        raise ValueError("Choose one of --preflight, --run-all, or --task-id")
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
