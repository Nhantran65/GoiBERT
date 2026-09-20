"""Run the minimal E4 FFN expansion screening.

The worker reuses the frozen E1 training loop in an isolated Python process
and replaces only the model factory.  ``--run-all`` performs one preflight,
then keeps at most one worker on each eligible GPU and queues the remaining
tasks.  No test split is loaded by this script.
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

import torch
from transformers import AutoTokenizer, RobertaModel


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from goibert import modeling as e1_modeling  # noqa: E402
from goibert import train as e1_train  # noqa: E402
from goibert_e4 import modeling as e4_modeling  # noqa: E402


PROTOCOL = "E4-ffn-expansion-screening-v1"
ARTIFACT_ROOT = ROOT / "artifacts" / "e4"
ORCH_ROOT = ARTIFACT_ROOT / ".orchestrator"
LOG_ROOT = ORCH_ROOT / "logs"
WORKER_STATUS_ROOT = ORCH_ROOT / "workers"
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
MAX_LENGTH = 256
GPU_GUARD_GIB = 5.0
PREFLIGHT_ATOL = 1e-4
PREFLIGHT_RTOL = 1e-4


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def e4_source_hash() -> str:
    """Hash the isolated adapter, protocol, and this immutable worker."""

    paths = [
        *sorted((SRC / "goibert_e4").rglob("*.py")),
        ROOT / "configs" / "e4" / "protocol.json",
        ROOT / "scripts" / "run_e4_architecture.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _gpu_free_gib() -> dict[str, float]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    free: dict[str, float] = {}
    for line in result.stdout.splitlines():
        pieces = [piece.strip() for piece in line.split(",")]
        if len(pieces) == 2:
            free[pieces[0]] = float(pieces[1]) / 1024.0
    return free


def _eligible_devices() -> list[str]:
    free = _gpu_free_gib()
    preferred_order = ("2", "1", "3", "0")
    selected = [
        index for index in preferred_order if free.get(index, 0.0) >= GPU_GUARD_GIB
    ]
    if not selected:
        raise RuntimeError(
            f"No GPU has at least {GPU_GUARD_GIB:.1f} GiB free; observed {free}"
        )
    return [f"cuda:{index}" for index in selected]


def _raw_expanded_encoder() -> tuple[RobertaModel, dict[str, Any]]:
    encoder = RobertaModel.from_pretrained(
        ROOT / "models" / "goiturn-s-gd",
        local_files_only=True,
        add_pooling_layer=False,
    )
    with torch.random.fork_rng(devices=[]):
        conversion = e4_modeling.expand_roberta_ffn(encoder)
    return encoder, conversion


def run_preflight(device_name: str = "cuda:2") -> dict[str, Any]:
    """Run the single bounded conversion/function/memory preflight."""

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "models" / "goiturn-s-gd", local_files_only=True, use_fast=True
    )
    bundle = e1_train.load_task_bundle("vsfc_sentiment", ROOT, include_test=False)
    train_split = e1_train.split_field(bundle, "train")
    train_texts = list(e1_train.split_texts(train_split))
    lengths = [
        (len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]), index)
        for index, text in enumerate(train_texts)
    ]
    lengths.sort()
    sample_indices = [lengths[0][1], lengths[-1][1]]
    sample_texts = [train_texts[index] for index in sample_indices]
    inputs = tokenizer(
        sample_texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    baseline_encoder = RobertaModel.from_pretrained(
        ROOT / "models" / "goiturn-s-gd",
        local_files_only=True,
        add_pooling_layer=False,
    ).eval()
    expanded_encoder, conversion = _raw_expanded_encoder()
    expanded_encoder.eval()
    with torch.inference_mode():
        baseline_output = baseline_encoder(**inputs, return_dict=True).last_hidden_state
        expanded_output = expanded_encoder(**inputs, return_dict=True).last_hidden_state
    difference = (baseline_output - expanded_output).abs()
    max_abs_diff = float(difference.max().item())
    max_rel_diff = float((difference / (baseline_output.abs() + 1e-12)).max().item())
    function_equivalent = bool(
        torch.allclose(
            baseline_output,
            expanded_output,
            atol=PREFLIGHT_ATOL,
            rtol=PREFLIGHT_RTOL,
        )
    )

    baseline_state = baseline_encoder.state_dict()
    expanded_state = expanded_encoder.state_dict()
    expected_shape_changes = {
        f"encoder.layer.{index}.intermediate.dense.{suffix}"
        for index in range(10)
        for suffix in ("weight", "bias")
    }
    expected_shape_changes.update(
        f"encoder.layer.{index}.output.dense.weight" for index in range(10)
    )
    shape_contract = True
    unchanged_weights = True
    unexpected_keys: list[str] = []
    for key, source in baseline_state.items():
        target = expanded_state.get(key)
        if target is None:
            unexpected_keys.append(key)
            shape_contract = False
            continue
        if key in expected_shape_changes:
            expected = {
                "intermediate.dense.weight": (3072, 384),
                "intermediate.dense.bias": (3072,),
                "output.dense.weight": (384, 3072),
            }[key.split(".", 4)[-2] + "." + key.split(".", 4)[-1]]
            if tuple(target.shape) != expected:
                shape_contract = False
        elif tuple(source.shape) != tuple(target.shape) or not torch.equal(source, target):
            unchanged_weights = False
            unexpected_keys.append(key)
    extras = sorted(set(expanded_state).difference(baseline_state))
    if extras:
        shape_contract = False
        unexpected_keys.extend(extras)
    for key in expected_shape_changes:
        if key not in baseline_state or key not in expanded_state:
            shape_contract = False
    output_biases_equal = all(
        torch.equal(
            baseline_state[f"encoder.layer.{index}.output.dense.bias"],
            expanded_state[f"encoder.layer.{index}.output.dense.bias"],
        )
        for index in range(10)
    )

    # The expansion must not consume the process RNG.  Build the two task
    # classifiers after resetting the same seed and compare only their heads.
    torch.manual_seed(777)
    baseline_classifier, baseline_spec = e1_modeling.build_classifier(
        ROOT / "models" / "goiturn-s-gd",
        num_classes=3,
        local_files_only=True,
        attention_implementation="sdpa",
    )
    torch.manual_seed(777)
    expanded_classifier, expanded_spec = e4_modeling.build_expanded_classifier(
        ROOT / "models" / "goiturn-s-gd",
        num_classes=3,
        local_files_only=True,
        attention_implementation="sdpa",
    )
    head_equal = all(
        torch.equal(value, expanded_classifier.head.state_dict()[key])
        for key, value in baseline_classifier.head.state_dict().items()
    )
    base_params = int(baseline_spec.total_parameters)
    expanded_params = int(expanded_spec.to_dict()["total_parameters"])
    parameter_delta = expanded_params - base_params

    expanded_encoder.zero_grad(set_to_none=True)
    expanded_encoder.train(False)
    gradient_output = expanded_encoder(**inputs, return_dict=True).last_hidden_state
    gradient_loss = gradient_output[:, 0].float().square().mean()
    gradient_loss.backward()
    gradient = expanded_encoder.encoder.layer[0].intermediate.dense.weight.grad
    gradient_finite = bool(gradient is not None and torch.isfinite(gradient).all().item())
    cloned_gradient_delta = float(
        (gradient[:1536] - gradient[1536:]).abs().max().item()
        if gradient is not None
        else float("nan")
    )
    cloned_gradients_can_diverge = bool(cloned_gradient_delta > 0.0)

    # Profile the exact requested microbatch and maximum sequence length on a
    # real task batch.  This is a bounded forward/backward probe only.
    memory_device = torch.device(device_name)
    if memory_device.type != "cuda":
        raise ValueError("E4 memory preflight requires a CUDA device")
    free_before_gib = _gpu_free_gib().get(str(memory_device.index), 0.0)
    memory_guard = free_before_gib >= GPU_GUARD_GIB
    profile_peak_allocated = None
    profile_peak_reserved = None
    if memory_guard:
        profile_model, _ = e4_modeling.build_expanded_classifier(
            ROOT / "models" / "goiturn-s-gd",
            num_classes=3,
            local_files_only=True,
            attention_implementation="sdpa",
        )
        profile_model.to(memory_device)
        profile_model.train()
        profile_inputs = tokenizer(
            [train_texts[index % len(train_texts)] for index in range(MICRO_BATCH_SIZE)],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        profile_inputs = {key: value.to(memory_device) for key, value in profile_inputs.items()}
        torch.cuda.reset_peak_memory_stats(memory_device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            profile_logits = profile_model(
                profile_inputs["input_ids"],
                profile_inputs["attention_mask"],
                token_type_ids=profile_inputs.get("token_type_ids"),
            )
            profile_loss = profile_logits.float().square().mean()
        profile_loss.backward()
        torch.cuda.synchronize(memory_device)
        profile_peak_allocated = int(torch.cuda.max_memory_allocated(memory_device))
        profile_peak_reserved = int(torch.cuda.max_memory_reserved(memory_device))
        del profile_model, profile_inputs, profile_logits, profile_loss
        torch.cuda.empty_cache()

    checks = {
        "function_equivalent": function_equivalent,
        "shape_contract": shape_contract,
        "unchanged_non_ffn_weights": unchanged_weights,
        "output_biases_identical": output_biases_equal,
        "head_rng_preserved": head_equal,
        "gradient_finite": gradient_finite,
        "cloned_gradients_can_diverge": cloned_gradients_can_diverge,
        "memory_guard": memory_guard,
    }
    passed = all(checks.values())
    payload = {
        "protocol": PROTOCOL,
        "status": "passed" if passed else "failed",
        "created_at": _now(),
        "elapsed_seconds": time.time() - started,
        "e4_source_hash": e4_source_hash(),
        "e1_dependency_source_hash": e1_train.source_hash(ROOT),
        "sample_indices": sample_indices,
        "sample_token_lengths": [item[0] for item in lengths[:1] + lengths[-1:]],
        "tolerances": {"atol": PREFLIGHT_ATOL, "rtol": PREFLIGHT_RTOL},
        "max_abs_diff_fp32": max_abs_diff,
        "max_rel_diff_fp32": max_rel_diff,
        "parameter_counts": {
            "source_classifier": base_params,
            "expanded_classifier": expanded_params,
            "delta": parameter_delta,
            "expected_delta": 11_811_840,
        },
        "expanded_config": {
            "num_hidden_layers": int(expanded_encoder.config.num_hidden_layers),
            "hidden_size": int(expanded_encoder.config.hidden_size),
            "num_attention_heads": int(expanded_encoder.config.num_attention_heads),
            "intermediate_size": int(expanded_encoder.config.intermediate_size),
        },
        "conversion": conversion,
        "unexpected_state_keys": sorted(set(unexpected_keys)),
        "gradient": {
            "finite": gradient_finite,
            "cloned_half_max_abs_difference": cloned_gradient_delta,
        },
        "memory_profile": {
            "device": str(memory_device),
            "free_before_gib": free_before_gib,
            "guard_gib": GPU_GUARD_GIB,
            "peak_allocated_bytes": profile_peak_allocated,
            "peak_reserved_bytes": profile_peak_reserved,
            "batch_size": MICRO_BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "precision": "bf16",
        },
        "checks": checks,
    }
    _write_json(ORCH_ROOT / "preflight.json", payload)
    return payload


def _e4_metadata_fields() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "architecture_variant": e4_modeling.EXPANDED_BACKBONE_NAME,
        "source_architecture": "goiturn-s-gd",
        "e4_source_hash": e4_source_hash(),
        "e1_dependency_source_hash": e1_train.source_hash(ROOT),
        "ffn_expansion": {
            "source_intermediate_size": e4_modeling.ORIGINAL_INTERMEDIATE_SIZE,
            "expanded_intermediate_size": e4_modeling.EXPANDED_INTERMEDIATE_SIZE,
            "conversion_seed": e4_modeling.CONVERSION_SEED,
            "noise_scale_relative_to_w2_std": e4_modeling.CONVERSION_NOISE_SCALE,
            "mapping": "W1'=[W1;W1], b1'=[b1;b1], W2'=[0.5W2+D;0.5W2-D], b2'=b2",
            "generator": "local_cpu_single_generator_advanced_across_layers",
        },
    }


def _worker_status(task_id: str, payload: Mapping[str, Any]) -> None:
    _write_json(WORKER_STATUS_ROOT / f"{task_id}.json", dict(payload))


def run_worker(task_id: str, device_name: str, output_dir: Path) -> dict[str, Any]:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task {task_id!r}")
    status_started = {"task_id": task_id, "status": "running", "started_at": _now(), "device": device_name}
    _worker_status(task_id, status_started)
    dependency_hash = e1_train.source_hash(ROOT)
    current_e4_hash = e4_source_hash()
    original_factory = e1_train.build_classifier
    original_hash_function = e1_train.source_hash
    e1_train.build_classifier = e4_modeling.build_expanded_classifier
    e1_train.source_hash = lambda _repo_root: current_e4_hash  # type: ignore[assignment]
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
            expected_source_hash=current_e4_hash,
        )
    except Exception as exc:
        _worker_status(
            task_id,
            {
                **status_started,
                "status": "failed",
                "finished_at": _now(),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    finally:
        e1_train.build_classifier = original_factory
        e1_train.source_hash = original_hash_function

    artifact_dir = Path(str(result["artifact_dir"]))
    fields = _e4_metadata_fields()
    fields["e1_dependency_source_hash"] = dependency_hash
    for filename in ("resolved_config.json", "metadata.json"):
        path = artifact_dir / filename
        metadata = _read_json(path)
        metadata.update(fields)
        _write_json(path, metadata)
    # Keep the resumable checkpoint's embedded audit metadata in sync with the
    # sidecar metadata.  This is one metadata-only rewrite; model weights are
    # untouched.
    checkpoint_path = artifact_dir / "best_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_metadata = dict(checkpoint.get("metadata", {}))
    checkpoint_metadata.update(fields)
    checkpoint["metadata"] = checkpoint_metadata
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(checkpoint_path)
    completed = {
        **status_started,
        "status": "completed",
        "finished_at": _now(),
        "artifact_dir": str(artifact_dir),
        "best_epoch": result.get("best_epoch"),
        "best_selection_f1": result.get("best_selection_f1"),
        "validation_metrics": result.get("validation_metrics"),
        "e4_source_hash": current_e4_hash,
    }
    _worker_status(task_id, completed)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return result


def _independent_summary(job_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute each E4 validation score once from saved prediction rows."""

    rows_out: list[dict[str, Any]] = []
    for task_id in TASKS:
        result = job_results[task_id]
        artifact_dir = Path(str(result["artifact_dir"]))
        prediction_payload = _read_json(artifact_dir / "validation_predictions.json")
        rows = prediction_payload.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Missing E4 validation prediction rows for {task_id}")
        bundle = e1_train.load_task_bundle(task_id, ROOT, include_test=False)
        expected_ids = list(e1_train.split_ids(e1_train.split_field(bundle, "validation")))
        ids = [str(row["id"]) for row in rows]
        if ids != expected_ids:
            raise ValueError(f"E4 prediction IDs do not match validation order for {task_id}")
        gold = [row["gold"] for row in rows]
        predicted = [row["prediction"] for row in rows]
        recomputed = e1_train.evaluate_predictions(
            task_id,
            bundle,
            gold,
            predicted,
            split="validation",
        )
        rows_out.append(
            {
                "task_id": task_id,
                "artifact_dir": str(artifact_dir),
                "seed": SEED,
                "learning_rate": TASK_LRS[task_id],
                "best_epoch": result.get("best_epoch"),
                "selection_f1_saved": result.get("best_selection_f1"),
                "selection_f1_recomputed": recomputed["selection_f1"],
                "metrics_recomputed": recomputed,
                "prediction_count": len(rows),
            }
        )
    payload = {
        "protocol": PROTOCOL,
        "status": "completed",
        "created_at": _now(),
        "e4_source_hash": e4_source_hash(),
        "metric_recomputation": "one pass from saved validation_predictions.json; test excluded",
        "rows": rows_out,
    }
    _write_json(ARTIFACT_ROOT / "summary.json", payload)
    return payload


def run_all(*, preflight_device: str = "cuda:2", poll_seconds: float = 15.0) -> dict[str, Any]:
    ORCH_ROOT.mkdir(parents=True, exist_ok=True)
    preflight_path = ORCH_ROOT / "preflight.json"
    if preflight_path.is_file():
        preflight = _read_json(preflight_path)
    else:
        preflight = run_preflight(preflight_device)
    if preflight.get("status") != "passed":
        raise RuntimeError("E4 preflight did not pass; refusing to launch full trials")
    state_path = ORCH_ROOT / "state.json"
    if state_path.is_file():
        old_state = _read_json(state_path)
        if old_state.get("status") in {"running", "completed"}:
            raise RuntimeError("E4 controller already launched; inspect its state before rerunning")
    devices = _eligible_devices()
    queue = list(TASKS)
    running: dict[str, tuple[subprocess.Popen[str], str]] = {}
    jobs: dict[str, dict[str, Any]] = {}
    completed_results: dict[str, Mapping[str, Any]] = {}
    controller_state: dict[str, Any] = {
        "protocol": PROTOCOL,
        "status": "running",
        "started_at": _now(),
        "e4_source_hash": e4_source_hash(),
        "preflight": str(preflight_path),
        "eligible_devices": devices,
        "jobs": jobs,
    }
    _write_json(state_path, controller_state)

    def launch(task_id: str, device_name: str) -> None:
        output_dir = ARTIFACT_ROOT / task_id / f"seed_{SEED}"
        log_path = LOG_ROOT / f"{task_id}__seed{SEED}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
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
        while queue and len(running) < len(devices):
            launch(queue.pop(0), devices[len(running)])
        finished: list[str] = []
        for task_id, (process, device_name) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            job = jobs[task_id]
            job.update(
                {
                    "status": "completed" if return_code == 0 else "failed",
                    "returncode": return_code,
                    "finished_at": _now(),
                }
            )
            if return_code == 0:
                worker_status_path = WORKER_STATUS_ROOT / f"{task_id}.json"
                if not worker_status_path.is_file():
                    raise RuntimeError(f"Worker {task_id} exited without status artifact")
                worker_status = _read_json(worker_status_path)
                completed_results[task_id] = worker_status
            finished.append(task_id)
        for task_id in finished:
            running.pop(task_id)
        if finished:
            _write_json(state_path, controller_state)
        if running:
            time.sleep(poll_seconds)

    failed = [task for task, job in jobs.items() if job.get("status") != "completed"]
    if failed:
        controller_state.update({"status": "failed", "finished_at": _now(), "failed_tasks": failed})
        _write_json(state_path, controller_state)
        raise RuntimeError(f"E4 workers failed: {failed}")
    # Worker status contains the returned artifact path and selected epoch.
    summary = _independent_summary(completed_results)
    controller_state.update({"status": "completed", "finished_at": _now(), "summary": str(ARTIFACT_ROOT / "summary.json")})
    _write_json(state_path, controller_state)
    return {"state": controller_state, "summary": summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run E4 FFN expansion screening")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--task-id", choices=TASKS)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-device", default="cuda:2")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.preflight:
            payload = run_preflight(args.preflight_device)
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
            return 0 if payload.get("status") == "passed" else 1
        if args.task_id:
            if args.output_dir is None:
                raise ValueError("--task-id requires --output-dir")
            run_worker(args.task_id, args.device, args.output_dir.resolve())
            return 0
        if args.run_all:
            payload = run_all(preflight_device=args.preflight_device)
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
            return 0
        raise ValueError("Choose one of --preflight, --run-all, or --task-id")
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

