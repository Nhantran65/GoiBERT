"""Run the minimal E6 MLM-backbone supervised screening.

The E1 trainer remains frozen.  Each worker installs a narrow adapter for the
encoder factory and tokenizer loader, so the only model input changed by E6 is
the final exported MLM backbone.  The worker restores the frozen functions
after training and annotates every artifact with the actual E6 directory
hashes before reporting completion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

# This must be set before importing torch through goibert.train.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
E6_BACKBONE = ROOT / "models" / "goiturn-s-gd-mlm-e6"
ARTIFACT_ROOT = ROOT / "artifacts" / "e6" / "finetune"
ORCH_ROOT = ARTIFACT_ROOT / ".orchestrator"
LOG_ROOT = ORCH_ROOT / "logs"
WORKER_ROOT = ORCH_ROOT / "workers"
PROTOCOL_PATH = ROOT / "configs" / "e6" / "protocol.json"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402
from transformers import AutoTokenizer, RobertaModel  # noqa: E402

from goibert import modeling as e1_modeling  # noqa: E402
from goibert import train as e1_train  # noqa: E402
from goibert_e6 import modeling as e6_modeling  # noqa: E402


PROTOCOL = "E6-mlm-backbone-supervised-screening-v1"
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
PRIOR_METHODS = {
    "vsfc_sentiment": ("NC", 0.847722),
    "vsfc_topic": ("KD", 0.807842),
    "visfd": ("NC", 0.810884),
    "absa_restaurant": ("E3_polarity_loss", 0.701299),
    "absa_hotel": ("NC", 0.735776),
}
CONFIG_FIELDS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "intermediate_size",
    "vocab_size",
    "max_position_embeddings",
)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_default(value: Any) -> Any:
    return e1_train._json_default(value)


def _write_json(path: Path, payload: Any) -> None:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_files(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(str(path.resolve().relative_to(ROOT)).encode("utf-8"))
        digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


def e6_source_hash() -> str:
    paths = [
        # Keep the downstream adapter hash separate from the MLM worker's
        # source hash.  The final export manifest carries the latter.
        SRC / "goibert_e6" / "modeling.py",
        PROTOCOL_PATH,
        Path(__file__).resolve(),
    ]
    return _hash_files(paths)


def _directory_hashes(root: Path) -> dict[str, str]:
    return e1_train.directory_hashes(root)


def _manifest_path() -> Path | None:
    for name in (
        "e6_mlm_manifest.json",
        "mlm_manifest.json",
        "export_manifest.json",
        "pretrain_manifest.json",
    ):
        candidate = E6_BACKBONE / name
        if candidate.is_file():
            return candidate
    # The current MLM worker keeps the provenance manifest beside its
    # training artifacts and writes it only after the encoder export.
    artifact_manifest = ARTIFACT_ROOT.parent / "mlm" / "mlm_metadata.json"
    if artifact_manifest.is_file():
        return artifact_manifest
    return None


def validate_e6_export(*, load_model: bool = True) -> dict[str, Any]:
    """Validate the final export before any supervised trial can launch."""

    required = (
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    )
    missing = [name for name in required if not (E6_BACKBONE / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"E6 MLM export is incomplete at {E6_BACKBONE}; missing {missing}"
        )
    source_config = _read_json(ROOT / "models" / "goiturn-s-gd" / "config.json")
    e6_config = _read_json(E6_BACKBONE / "config.json")
    config_match = {
        field: source_config.get(field) == e6_config.get(field) for field in CONFIG_FIELDS
    }
    if not all(config_match.values()):
        raise RuntimeError(f"E6 config is not architecture-compatible: {config_match}")

    source_tokenizer = _directory_hashes(ROOT / "models" / "goiturn-s-gd")
    e6_hashes = _directory_hashes(E6_BACKBONE)
    tokenizer_names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
    tokenizer_hash_match = {
        name: source_tokenizer.get(name) == e6_hashes.get(name) for name in tokenizer_names
    }
    source_tokenizer_obj = AutoTokenizer.from_pretrained(
        ROOT / "models" / "goiturn-s-gd", local_files_only=True, use_fast=True
    )
    e6_tokenizer_obj = AutoTokenizer.from_pretrained(
        E6_BACKBONE, local_files_only=True, use_fast=True
    )
    tokenizer_semantic_match = {
        "vocab": source_tokenizer_obj.get_vocab() == e6_tokenizer_obj.get_vocab(),
        "special_token_ids": source_tokenizer_obj.all_special_ids == e6_tokenizer_obj.all_special_ids,
        "pad_token_id": source_tokenizer_obj.pad_token_id == e6_tokenizer_obj.pad_token_id,
        "mask_token_id": source_tokenizer_obj.mask_token_id == e6_tokenizer_obj.mask_token_id,
    }
    if not all(tokenizer_hash_match.values()):
        raise RuntimeError(
            "E6 export tokenizer bytes differ from the frozen E1 tokenizer; "
            f"hash_comparison={tokenizer_hash_match}, semantic_comparison={tokenizer_semantic_match}"
        )
    tokenizer_match = tokenizer_hash_match

    loaded = False
    hidden_size = None
    parameter_count = None
    if load_model:
        model = RobertaModel.from_pretrained(
            E6_BACKBONE,
            local_files_only=True,
            add_pooling_layer=False,
            attn_implementation="sdpa",
        )
        model.eval()
        hidden_size = int(model.config.hidden_size)
        parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
        with torch.inference_mode():
            tokenizer = AutoTokenizer.from_pretrained(
                E6_BACKBONE, local_files_only=True, use_fast=True
            )
            encoded = tokenizer(
                ["món ăn ngon", "phục vụ nhanh"],
                padding=True,
                truncation=True,
                max_length=32,
                return_tensors="pt",
            )
            output = model(**encoded, return_dict=True).last_hidden_state
        if not bool(torch.isfinite(output).all().item()):
            raise RuntimeError("E6 export produced non-finite CPU hidden states")
        loaded = True
        del model

    manifest = _manifest_path()
    if manifest is None:
        raise FileNotFoundError(
            "E6 MLM export has no recognized provenance manifest; expected "
            "e6_mlm_manifest.json"
        )
    manifest_payload: dict[str, Any] | None = None
    if manifest is not None:
        try:
            manifest_payload = _read_json(manifest)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unable to read E6 MLM export manifest {manifest}: {exc}") from exc
    mlm_source_hash = None
    if manifest_payload is not None:
        manifest_status = manifest_payload.get("status")
        if manifest_status not in {"completed", "complete", "ready"}:
            raise RuntimeError(
                "E6 MLM export manifest is not a final completed export: "
                f"status={manifest_status!r}"
            )
        for key in ("mlm_source_hash", "source_hash", "code_hash", "mlm_code_hash"):
            value = manifest_payload.get(key)
            if isinstance(value, str):
                mlm_source_hash = value
                break
    ready_marker = ARTIFACT_ROOT.parent / "mlm" / "READY_FOR_FINETUNE.json"
    if not ready_marker.is_file():
        raise FileNotFoundError(
            "E6 MLM export has no READY_FOR_FINETUNE.json marker written after export"
        )
    ready_payload = _read_json(ready_marker)
    if ready_payload.get("status") != "ready":
        raise RuntimeError(
            "E6 MLM READY_FOR_FINETUNE marker is not ready: "
            f"status={ready_payload.get('status')!r}"
        )
    if mlm_source_hash is not None and ready_payload.get("source_hash") != mlm_source_hash:
        raise RuntimeError("E6 READY_FOR_FINETUNE source hash does not match MLM metadata")
    payload = {
        "protocol": PROTOCOL,
        "status": "passed",
        "created_at": _now(),
        "path": str(E6_BACKBONE),
        "directory_hashes": e6_hashes,
        "directory_sha256": _hash_files(
            [path for path in sorted(E6_BACKBONE.rglob("*")) if path.is_file()]
        ),
        "manifest_path": str(manifest) if manifest else None,
        "manifest_sha256": _sha256(manifest) if manifest else None,
        "mlm_source_hash": mlm_source_hash,
        "manifest_payload": manifest_payload,
        "ready_marker_path": str(ready_marker),
        "ready_marker_sha256": _sha256(ready_marker),
        "config_match_to_source": config_match,
        "tokenizer_match_to_e1": tokenizer_match,
        "tokenizer_hash_match_to_e1": tokenizer_hash_match,
        "tokenizer_semantic_match_to_e1": tokenizer_semantic_match,
        "model_loaded": loaded,
        "hidden_size": hidden_size,
        "parameter_count": parameter_count,
    }
    return payload


def _gpu_snapshot() -> dict[str, dict[str, int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,memory.used",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    snapshot: dict[str, dict[str, int]] = {}
    for line in result.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) >= 3:
            snapshot[fields[0]] = {
                "free_mib": int(float(fields[1])),
                "used_mib": int(float(fields[2])),
            }
    return snapshot


def _eligible_devices() -> tuple[list[str], dict[str, dict[str, int]]]:
    snapshot = _gpu_snapshot()
    devices = [
        f"cuda:{index}"
        for index in PREFERRED_GPU_INDICES
        if snapshot.get(index, {}).get("free_mib", 0) >= GPU_GUARD_MIB
    ]
    if not devices:
        raise RuntimeError(
            f"No approved GPU has at least {GPU_GUARD_MIB} MiB free; snapshot={snapshot}"
        )
    return devices, snapshot


def _assert_frozen_dependency() -> None:
    actual = e1_train.source_hash(ROOT)
    if actual != E1_SOURCE_HASH:
        raise RuntimeError(f"Frozen E1 source drift: expected {E1_SOURCE_HASH}, found {actual}")


def _patch_artifact_metadata(artifact_dir: Path, export: Mapping[str, Any]) -> None:
    fields = {
        "protocol": PROTOCOL,
        "architecture_variant": e6_modeling.E6_BACKBONE_NAME,
        "backbone_name": e6_modeling.E6_BACKBONE_NAME,
        "backbone_path": str(E6_BACKBONE),
        "mlm_backbone_path": str(E6_BACKBONE),
        "mlm_backbone_hashes": export["directory_hashes"],
        "mlm_backbone_directory_sha256": export["directory_sha256"],
        "mlm_export_manifest_path": export.get("manifest_path"),
        "mlm_export_manifest_sha256": export.get("manifest_sha256"),
        "mlm_ready_marker_path": export.get("ready_marker_path"),
        "mlm_ready_marker_sha256": export.get("ready_marker_sha256"),
        "mlm_source_hash": export.get("mlm_source_hash"),
        "e6_source_hash": e6_source_hash(),
        "e1_dependency_source_hash": E1_SOURCE_HASH,
        "backbone_hashes": export["directory_hashes"],
        "tokenizer_hashes": {
            name: export["directory_hashes"][name]
            for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
        },
        "backbone_provenance": {
            "source": "student GoiTurn-S-GD after E6 supervised-independent MLM continuation",
            "encoder_only_export": True,
            "architecture_compatibility": export["config_match_to_source"],
            "tokenizer_compatibility": export["tokenizer_match_to_e1"],
            "tokenizer_hash_match": export.get("tokenizer_hash_match_to_e1"),
        },
    }
    for filename in ("resolved_config.json", "metadata.json"):
        path = artifact_dir / filename
        metadata = _read_json(path)
        metadata.update(fields)
        metadata["source_hash"] = fields["e6_source_hash"]
        _write_json(path, metadata)

    checkpoint_path = artifact_dir / "best_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_metadata = dict(checkpoint.get("metadata", {}))
    checkpoint_metadata.update(fields)
    checkpoint_metadata["source_hash"] = fields["e6_source_hash"]
    checkpoint["metadata"] = checkpoint_metadata
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(checkpoint_path)


def _worker_status(task_id: str, payload: Mapping[str, Any]) -> None:
    _write_json(WORKER_ROOT / f"{task_id}.json", payload)


def run_preflight(device_name: str) -> dict[str, Any]:
    started = time.time()
    _assert_frozen_dependency()
    export = validate_e6_export(load_model=True)
    torch.manual_seed(777)
    baseline, baseline_spec = e1_modeling.build_classifier(
        ROOT / "models" / "goiturn-s-gd",
        num_classes=3,
        local_files_only=True,
        attention_implementation="sdpa",
    )
    torch.manual_seed(777)
    adapted, adapted_spec = e6_modeling.build_e6_classifier(
        ROOT / "models" / "goiturn-s-gd",
        e6_backbone_dir=E6_BACKBONE,
        num_classes=3,
        local_files_only=True,
        attention_implementation="sdpa",
    )
    head_equal = all(
        torch.equal(value, adapted.head.state_dict()[key])
        for key, value in baseline.head.state_dict().items()
    )
    parameter_count_equal = baseline_spec.total_parameters == adapted_spec.total_parameters
    backbone_architecture_equal = all(
        getattr(baseline.encoder.config, field) == getattr(adapted.encoder.config, field)
        for field in CONFIG_FIELDS
    )

    physical_index = device_name.split(":", 1)[1] if ":" in device_name else device_name
    snapshot = _gpu_snapshot()
    free_mib = snapshot.get(physical_index, {}).get("free_mib", 0)
    memory_guard = free_mib >= GPU_GUARD_MIB
    finite_backward = False
    peak_allocated = None
    peak_reserved = None
    if memory_guard:
        tokenizer = AutoTokenizer.from_pretrained(E6_BACKBONE, local_files_only=True, use_fast=True)
        device = torch.device(device_name)
        adapted.to(device).train()
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
            logits = adapted(encoded["input_ids"], encoded["attention_mask"])
            loss = logits.float().square().mean()
        loss.backward()
        torch.cuda.synchronize(device)
        finite_backward = bool(torch.isfinite(logits).all().item() and torch.isfinite(loss).item())
        finite_backward = finite_backward and all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in adapted.parameters()
        )
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))

    checks = {
        "frozen_e1_dependency": e1_train.source_hash(ROOT) == E1_SOURCE_HASH,
        "export_validation": export.get("status") == "passed",
        "architecture_unchanged": backbone_architecture_equal,
        "head_rng_and_weights_equal": head_equal,
        "parameter_count_equal": parameter_count_equal,
        "memory_guard": memory_guard,
        "finite_forward_backward": finite_backward,
    }
    manifest_path = _manifest_path()
    mlm_source_hash = None
    if manifest_path is not None:
        manifest_payload = _read_json(manifest_path)
        for key in ("mlm_source_hash", "source_hash", "code_hash", "mlm_code_hash"):
            if isinstance(manifest_payload.get(key), str):
                mlm_source_hash = manifest_payload[key]
                break
    payload = {
        "protocol": PROTOCOL,
        "status": "passed" if all(checks.values()) else "failed",
        "created_at": _now(),
        "elapsed_seconds": time.time() - started,
        "e6_source_hash": e6_source_hash(),
        "e1_dependency_source_hash": E1_SOURCE_HASH,
        "device": device_name,
        "export": export,
        "parameter_counts": {
            "e1": int(baseline_spec.total_parameters),
            "e6": int(adapted_spec.total_parameters),
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


def run_worker(task_id: str, device_name: str, output_dir: Path) -> dict[str, Any]:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task {task_id!r}")
    _assert_frozen_dependency()
    preflight_path = ORCH_ROOT / "preflight.json"
    if not preflight_path.is_file():
        raise RuntimeError("E6 worker requires a passed preflight artifact")
    preflight = _read_json(preflight_path)
    if preflight.get("status") != "passed" or not isinstance(preflight.get("export"), dict):
        raise RuntimeError("E6 worker requires a passed preflight/export contract")
    export = preflight["export"]
    status_started = {
        "task_id": task_id,
        "status": "running",
        "started_at": _now(),
        "device": device_name,
        "e6_source_hash": e6_source_hash(),
        "mlm_backbone_directory_sha256": export["directory_sha256"],
    }
    _worker_status(task_id, status_started)

    original_factory = e1_train.build_classifier
    original_hash_function = e1_train.source_hash
    original_tokenizer_class = e1_train.AutoTokenizer
    current_e6_hash = e6_source_hash()

    def factory(_backbone_dir: Path, **kwargs: Any):
        return e6_modeling.build_e6_classifier(
            _backbone_dir,
            e6_backbone_dir=E6_BACKBONE,
            **kwargs,
        )

    class _E6TokenizerProxy:
        @staticmethod
        def from_pretrained(*_args: Any, **kwargs: Any):
            return original_tokenizer_class.from_pretrained(E6_BACKBONE, **kwargs)

    e1_train.build_classifier = factory  # type: ignore[assignment]
    e1_train.source_hash = lambda _repo_root: current_e6_hash  # type: ignore[assignment]
    e1_train.AutoTokenizer = _E6TokenizerProxy  # type: ignore[assignment]
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
            expected_source_hash=current_e6_hash,
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
        e1_train.AutoTokenizer = original_tokenizer_class

    artifact_dir = Path(str(result["artifact_dir"]))
    _patch_artifact_metadata(artifact_dir, export)
    completed = {
        **status_started,
        "status": "completed",
        "finished_at": _now(),
        "artifact_dir": str(artifact_dir),
        "best_epoch": result.get("best_epoch"),
        "best_selection_f1": result.get("best_selection_f1"),
        "validation_metrics": result.get("validation_metrics"),
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
            raise ValueError(f"Missing E6 validation rows for {task_id}")
        bundle = e1_train.load_task_bundle(task_id, ROOT, include_test=False)
        valid_split = e1_train.split_field(bundle, "validation")
        expected_ids = list(e1_train.split_ids(valid_split))
        actual_ids = [row["id"] for row in prediction_rows]
        if [str(value) for value in actual_ids] != [str(value) for value in expected_ids]:
            raise ValueError(f"E6 prediction IDs do not match validation order for {task_id}")
        gold = [row["gold"] for row in prediction_rows]
        predicted = [row["prediction"] for row in prediction_rows]
        recomputed = e1_train.evaluate_predictions(task_id, bundle, gold, predicted, split="validation")
        e1_dir = _e1_seed42_artifact(task_id)
        e1_metrics = _read_json(e1_dir / "best_validation_metrics.json")
        prior_name, prior_score = PRIOR_METHODS[task_id]
        e6_score = float(recomputed["selection_f1"])
        rows.append(
            {
                "task_id": task_id,
                "artifact_dir": str(artifact_dir),
                "e1_artifact_dir": str(e1_dir),
                "seed": SEED,
                "learning_rate": TASK_LRS[task_id],
                "best_epoch": result.get("best_epoch"),
                "selection_f1_e1": e1_metrics.get("selection_f1"),
                "selection_f1_e6_saved": result.get("best_selection_f1"),
                "selection_f1_e6_recomputed": e6_score,
                "delta_e6_minus_e1": e6_score - float(e1_metrics["selection_f1"]),
                "prior_method": prior_name,
                "prior_selection_f1": prior_score,
                "delta_e6_minus_prior": e6_score - prior_score,
                "metrics_e1": e1_metrics,
                "metrics_e6": recomputed,
                "prediction_count": len(prediction_rows),
            }
        )
    summary_manifest = _manifest_path()
    summary_mlm_source_hash = None
    if summary_manifest is not None:
        summary_manifest_payload = _read_json(summary_manifest)
        for key in ("mlm_source_hash", "source_hash", "code_hash", "mlm_code_hash"):
            if isinstance(summary_manifest_payload.get(key), str):
                summary_mlm_source_hash = summary_manifest_payload[key]
                break
    payload = {
        "protocol": PROTOCOL,
        "status": "completed",
        "created_at": _now(),
        "e6_source_hash": e6_source_hash(),
        "e1_dependency_source_hash": E1_SOURCE_HASH,
        "mlm_backbone": str(E6_BACKBONE),
        "mlm_source_hash": summary_mlm_source_hash,
        "metric_recomputation": "one pass from saved validation_predictions.json; test excluded",
        "rows": rows,
    }
    summary_json = ARTIFACT_ROOT / "summary.json"
    _write_json(summary_json, payload)
    with (ARTIFACT_ROOT / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "task_id",
            "seed",
            "learning_rate",
            "selection_f1_e1",
            "selection_f1_e6_recomputed",
            "delta_e6_minus_e1",
            "prior_method",
            "prior_selection_f1",
            "delta_e6_minus_prior",
            "best_epoch",
            "prediction_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    return payload


def run_all(*, poll_seconds: float = 15.0) -> dict[str, Any]:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    WORKER_ROOT.mkdir(parents=True, exist_ok=True)
    _assert_frozen_dependency()
    devices, snapshot = _eligible_devices()
    preflight_path = ORCH_ROOT / "preflight.json"
    preflight = run_preflight(devices[0])
    if preflight.get("status") != "passed":
        raise RuntimeError("E6 preflight did not pass; refusing to launch full trials")
    export = preflight["export"]
    state_path = ORCH_ROOT / "state.json"
    if state_path.is_file():
        old = _read_json(state_path)
        if old.get("status") in {"running", "completed"}:
            raise RuntimeError("E6 controller already launched; inspect its state before rerunning")

    queue = list(TASKS)
    free_devices = list(devices)
    running: dict[str, tuple[subprocess.Popen[str], str]] = {}
    jobs: dict[str, dict[str, Any]] = {}
    completed: dict[str, Mapping[str, Any]] = {}
    controller_state: dict[str, Any] = {
        "protocol": PROTOCOL,
        "status": "running",
        "started_at": _now(),
        "e6_source_hash": e6_source_hash(),
        "e1_dependency_source_hash": E1_SOURCE_HASH,
        "mlm_backbone_path": str(E6_BACKBONE),
        "mlm_backbone_directory_sha256": export["directory_sha256"],
        "mlm_export_manifest_path": export["manifest_path"],
        "mlm_export_manifest_sha256": export["manifest_sha256"],
        "mlm_ready_marker_path": export["ready_marker_path"],
        "mlm_ready_marker_sha256": export["ready_marker_sha256"],
        "mlm_source_hash": export.get("mlm_source_hash"),
        "preferred_gpu_indices": list(PREFERRED_GPU_INDICES),
        "eligible_devices": devices,
        "gpu_snapshot_at_launch": snapshot,
        "jobs": jobs,
    }
    _write_json(state_path, controller_state)

    def launch(task_id: str, device_name: str) -> None:
        output_dir = ARTIFACT_ROOT / task_id / f"seed_{SEED}"
        if output_dir.exists():
            raise RuntimeError(f"Refusing to overwrite existing E6 output directory: {output_dir}")
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
            job.update(
                {
                    "status": "completed" if return_code == 0 else "failed",
                    "returncode": return_code,
                    "finished_at": _now(),
                }
            )
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
            raise RuntimeError(f"E6 workers failed: {failed}")
        if running:
            time.sleep(poll_seconds)

    summary = _independent_summary(completed)
    controller_state.update(
        {
            "status": "completed",
            "finished_at": _now(),
            "summary": str(ARTIFACT_ROOT / "summary.json"),
            "preflight": str(preflight_path),
        }
    )
    _write_json(state_path, controller_state)
    return {"state": controller_state, "summary": summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run E6 MLM-backbone supervised screening")
    parser.add_argument("--validate-export", action="store_true")
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
        if args.validate_export:
            payload = validate_e6_export(load_model=True)
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
            return 0
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
        raise ValueError("Choose one of --validate-export, --preflight, --run-all, or --task-id")
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
