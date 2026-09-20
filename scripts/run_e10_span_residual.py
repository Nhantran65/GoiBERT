"""E10 span-routing versus multi-kernel convolution screening.

This is an isolated execution adapter around the frozen E2 NC loop and E7
optimizer.  It composes two E10 model variants, explicit raw-text alignment
collators, and a durable ten-trial queue without changing E1--E9 sources.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ARTIFACT_ROOT = ROOT / "artifacts" / "e10"
ORCH_ROOT = ARTIFACT_ROOT / ".orchestrator"
LOG_ROOT = ORCH_ROOT / "logs"
WORKER_ROOT = ORCH_ROOT / "workers"
PROTOCOL_PATH = ROOT / "configs" / "e10" / "protocol.json"
MANIFEST_PATH = ARTIFACT_ROOT / "source_manifest.json"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402
from torch import Tensor, nn  # noqa: E402
from transformers import AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

from goibert import train as e1_train  # noqa: E402
from goibert_e2 import train as e2_train  # noqa: E402
from goibert_e2.noise import noisy_batch  # noqa: E402
from goibert_e7 import optim as e7_optim  # noqa: E402
from goibert_e10.modeling import (  # noqa: E402
    CONV_RANK,
    INSERTION_LAYER_INDEX,
    SPAN_RANK,
    SPAN_WIDTHS,
    E10Classifier,
    TokenAlignment,
    build_e10_student,
    build_token_alignment,
    make_e10_loader,
    raw_units,
    _tokenize_with_alignment,
)


PROTOCOL = "E10-span-residual-vs-multikernel-conv-v1"
REVISION = "e10-20260911-r1"
VARIANTS = ("span", "conv")
TASKS = (
    "vsfc_sentiment",
    "vsfc_topic",
    "visfd",
    "absa_restaurant",
    "absa_hotel",
)
TRIALS = tuple((variant, task_id) for variant in VARIANTS for task_id in TASKS)
TASK_LRS = {
    "vsfc_sentiment": 1e-5,
    "vsfc_topic": 1e-5,
    "visfd": 1e-4,
    "absa_restaurant": 1e-4,
    "absa_hotel": 1e-4,
}
NC_LAMBDAS = {
    "vsfc_sentiment": 0.1,
    "vsfc_topic": 0.5,
    "visfd": 0.5,
    "absa_restaurant": 0.1,
    "absa_hotel": 0.1,
}
SEED = 42
EPOCHS = 30
EFFECTIVE_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 32
MAX_LENGTH = 256
GAMMA = 0.8
NUM_LAYERS = 10
WARMUP_FRACTION = 0.1
GPU_GUARD_MIB = 5 * 1024
PREFERRED_GPU_INDICES = ("0", "1", "2", "3")
E1_SOURCE_HASH = "456e3020bd405ec39c7b4cbab3255d0bb3cb2a8a4bda3e37cc0620d51b559873"
E2_CODE_HASH = "efb81ac232de2f0ae4287688ad0125a1d84fb1e689e2b1566ef26d8d6e2b60b5"
E2_CONFIG_HASH = "45e6c36196fe95ae0eb9193e9197224bdf2980515c15ddf01998267b668b61b8"


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
    with path.open(encoding="utf-8") as handle:
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


def e10_source_hash() -> str:
    return _hash_files((Path(__file__).resolve(), PROTOCOL_PATH, SRC / "goibert_e10" / "modeling.py"))


def e7_dependency_hash() -> str:
    paths = [*sorted((SRC / "goibert_e7").rglob("*.py")), ROOT / "configs" / "e7" / "protocol.json"]
    return _hash_files(paths)


def _source_hashes() -> dict[str, str]:
    values = e2_train.source_hashes(ROOT)
    if values["e1_dependency"] != E1_SOURCE_HASH:
        raise RuntimeError(f"Frozen E1 dependency drift: {values['e1_dependency']}")
    if values["e2"] != E2_CODE_HASH or values["e2_config"] != E2_CONFIG_HASH:
        raise RuntimeError(f"Frozen E2 dependency drift: {values}")
    return values


def _gpu_snapshot() -> dict[str, dict[str, int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=30)
    snapshot: dict[str, dict[str, int]] = {}
    for line in output.strip().splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) < 3:
            continue
        snapshot[fields[0]] = {
            "free_mib": int(float(fields[1])),
            "used_mib": int(float(fields[2])),
            "utilization_gpu": int(float(fields[3])) if len(fields) >= 4 else -1,
        }
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
    return devices, snapshot


def _pid_alive(pid: Any) -> bool:
    try:
        value = int(pid)
    except (TypeError, ValueError):
        return False
    if value <= 0:
        return False
    try:
        os.kill(value, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _trial_key(variant: str, task_id: str) -> str:
    return f"{variant}__{task_id}__seed{SEED}"


def _trial_dir(variant: str, task_id: str) -> Path:
    return ARTIFACT_ROOT / variant / task_id / f"seed_{SEED}"


def _worker_path(variant: str, task_id: str) -> Path:
    return WORKER_ROOT / f"{_trial_key(variant, task_id)}.json"


def _completed_artifact(variant: str, task_id: str) -> Path | None:
    artifact = _trial_dir(variant, task_id)
    if not all((artifact / name).is_file() for name in ("status.json", "metadata.json", "best_checkpoint.pt")):
        return None
    try:
        status = _read_json(artifact / "status.json")
        metadata = _read_json(artifact / "metadata.json")
    except Exception:
        return None
    if status.get("status") == "completed" and metadata.get("protocol") == "E2-training-ablation-v1":
        architecture = metadata.get("architecture", {})
        metadata_variant = metadata.get("variant")
        if isinstance(architecture, Mapping):
            metadata_variant = architecture.get("variant", metadata_variant)
        if metadata_variant == variant:
            return artifact
    return None


def _set_worker_status(variant: str, task_id: str, payload: Mapping[str, Any]) -> None:
    _write_json(_worker_path(variant, task_id), dict(payload))


def _preflight_optimizer(model: nn.Module, base_lr: float) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    optimizer = e7_optim.build_layerwise_optimizer(model, base_lr, gamma=GAMMA, num_layers=NUM_LAYERS)
    coverage = e7_optim.validate_optimizer_groups(
        model,
        optimizer,
        base_lr=base_lr,
        gamma=GAMMA,
        num_layers=NUM_LAYERS,
        expected_group_names=[
            f"{region}_{kind}"
            for region in ("embeddings", *[f"block_{i:02d}" for i in range(NUM_LAYERS)], "head")
            for kind in ("decay", "no_decay")
        ],
    )
    if not coverage["all_trainable"] or coverage["duplicate_parameter_count"] != 0:
        raise AssertionError("E10 optimizer does not cover every trainable parameter exactly once")
    branch_rows = [row for row in coverage["groups"] if row["region"] == "block_01"]
    if not branch_rows or any(not any("e10_block" in name for name in row["parameter_names"]) for row in branch_rows):
        raise AssertionError("E10 branch is not visible in the block_01 optimizer groups")
    expected_branch_lr = float(base_lr * GAMMA**8)
    if any(not math.isclose(float(row["lr"]), expected_branch_lr, rel_tol=1e-12) for row in branch_rows):
        raise AssertionError(f"E10 branch LR mismatch: {branch_rows}")
    return optimizer, coverage


def _alignment_preflight(tokenizer: Any) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    checks["unicode_units"] = len(raw_units("Cà phê—ngon, 😄!")) == 4
    checks["punctuation_separator"] = [unit.segment for unit in raw_units("a,b-c")] == [0, 1, 2]
    checks["empty"] = len(raw_units("")) == 0 and not build_token_alignment("", [(0, 0)]).token_to_units[0]
    clean = "Cà phê ngon"
    noisy = "Ca phe ngon"
    clean_encoded = tokenizer(clean, add_special_tokens=True, truncation=True, max_length=32, return_offsets_mapping=True)
    noisy_encoded = tokenizer(noisy, add_special_tokens=True, truncation=True, max_length=32, return_offsets_mapping=True)
    clean_alignment = build_token_alignment(clean, clean_encoded["offset_mapping"])
    noisy_alignment = build_token_alignment(noisy, noisy_encoded["offset_mapping"])
    checks["clean_noisy_recomputed"] = (
        clean_alignment.raw_text == clean
        and noisy_alignment.raw_text == noisy
        and clean_alignment.raw_text != noisy_alignment.raw_text
    )
    long_text = " ".join(["Càphê"] * 100)
    truncated = tokenizer(long_text, add_special_tokens=True, truncation=True, max_length=8, return_offsets_mapping=True)
    trunc_alignment = build_token_alignment(long_text, truncated["offset_mapping"])
    checks["truncation_visible_only"] = len(trunc_alignment.token_to_units) == len(truncated["input_ids"])
    if not all(checks.values()):
        raise AssertionError(f"E10 alignment preflight failed: {checks}")
    return checks


def _gradient_preflight() -> dict[str, Any]:
    e2_train.seed_everything(42)
    base, _ = build_e10_student(
        ROOT / "models" / "goiturn-s-gd",
        variant="span",
        num_classes=3,
        attention_implementation="sdpa",
    )
    branch = base._e10_layer_wrapper.e10_block
    alignment = build_token_alignment("Cà phê ngon", [(0, 0), (0, 2), (3, 5), (6, 10), (0, 0)])
    hidden = torch.randn(1, 5, 384)
    target = torch.randn_like(hidden)
    optimizer = torch.optim.SGD(branch.parameters(), lr=0.1)
    optimizer.zero_grad(set_to_none=True)
    first = branch(hidden, [alignment])
    (first - target).square().mean().backward()
    output_grad = float(branch.output.weight.grad.abs().sum())
    inner_grad_step1 = float(branch.query.weight.grad.abs().sum())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second = branch(hidden, [alignment])
    (second - target).square().mean().backward()
    inner_grad_step2 = float(branch.query.weight.grad.abs().sum())
    if output_grad <= 0 or inner_grad_step1 != 0.0 or inner_grad_step2 <= 0:
        raise AssertionError(
            f"E10 gradient gate failed: output={output_grad}, inner1={inner_grad_step1}, inner2={inner_grad_step2}"
        )
    return {
        "output_projection_grad_step1": output_grad,
        "inner_grad_step1": inner_grad_step1,
        "inner_grad_step2": inner_grad_step2,
    }


def _head_init_preflight() -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "models" / "goiturn-s-gd", local_files_only=True, use_fast=True
    )
    encoded, alignments = _tokenize_with_alignment(tokenizer, ["Cà phê ngon"], 32)
    e2_train.seed_everything(SEED)
    baseline, _ = e2_train.build_e2_student(
        ROOT / "models" / "goiturn-s-gd", num_classes=3, attention_implementation="sdpa"
    )
    baseline_head = {name: value.detach().clone() for name, value in baseline.head.state_dict().items()}
    e2_train.seed_everything(SEED)
    candidate, _ = build_e10_student(
        ROOT / "models" / "goiturn-s-gd", variant="span", num_classes=3, attention_implementation="sdpa"
    )
    candidate_head = candidate.head.state_dict()
    equal = all(torch.equal(baseline_head[name], candidate_head[name]) for name in baseline_head)
    if not equal:
        raise AssertionError("E10 branch construction changed the E1/E8 CLS-head initialization")
    baseline.eval()
    candidate.eval()
    with torch.inference_mode():
        baseline_logits = baseline(
            encoded["input_ids"], encoded["attention_mask"]
        )
        candidate_logits = candidate(
            encoded["input_ids"], encoded["attention_mask"], alignment=alignments
        )
    logits_equal = torch.equal(baseline_logits, candidate_logits)
    if not logits_equal:
        max_diff = float((baseline_logits - candidate_logits).abs().max())
        raise AssertionError(f"E10 zero-output initialization changed baseline logits; max_diff={max_diff}")
    return {
        "head_logits_init_identity": True,
        "head_parameter_count": len(baseline_head),
        "baseline_candidate_logits_exact": True,
    }


def _strict_roundtrip_preflight() -> dict[str, Any]:
    e2_train.seed_everything(SEED)
    model, spec = build_e10_student(
        ROOT / "models" / "goiturn-s-gd", variant="conv", num_classes=3, attention_implementation="sdpa"
    )
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    e2_train.seed_everything(SEED)
    restored, restored_spec = build_e10_student(
        ROOT / "models" / "goiturn-s-gd", variant="conv", num_classes=3, attention_implementation="sdpa"
    )
    missing, unexpected = restored.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise AssertionError(f"E10 strict checkpoint roundtrip failed: {missing}, {unexpected}")
    if spec.to_dict()["variant"] != restored_spec.to_dict()["variant"]:
        raise AssertionError("E10 constructor metadata lost variant identity")
    return {"strict_load": True, "state_key_count": len(state), "variant": "conv"}


def run_preflight(device_name: str | None = None) -> dict[str, Any]:
    source_values = _source_hashes()
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "models" / "goiturn-s-gd", local_files_only=True, use_fast=True
    )
    alignment_checks = _alignment_preflight(tokenizer)
    head_checks = _head_init_preflight()
    gradient_checks = _gradient_preflight()
    roundtrip_checks = _strict_roundtrip_preflight()
    shape_rows: list[dict[str, Any]] = []
    optimizer_coverage: dict[str, Any] | None = None
    for task_id in TASKS:
        contract = e2_train._task_contract(task_id, ROOT, smoke=False)
        e2_train.seed_everything(SEED)
        model, spec = build_e10_student(
            ROOT / "models" / "goiturn-s-gd",
            variant="span",
            num_classes=contract["num_classes"],
            num_aspects=contract["num_aspects"],
            aspect_class_counts=contract["aspect_class_counts"] if contract["num_aspects"] else None,
            attention_implementation="sdpa",
        )
        if optimizer_coverage is None:
            _, optimizer_coverage = _preflight_optimizer(model, TASK_LRS[task_id])
        shape_rows.append(
            {
                "task_id": task_id,
                "num_labels": contract["num_classes"],
                "num_aspects": contract["num_aspects"],
                "aspect_class_counts": list(contract["aspect_class_counts"]),
                "model_spec": spec.to_dict(),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            }
        )
        del model
    devices: list[str] = []
    snapshot: dict[str, Any] = {}
    gpu_error = None
    try:
        devices, snapshot = _eligible_devices()
    except Exception as exc:
        gpu_error = f"{type(exc).__name__}: {exc}"
    payload = {
        "protocol": PROTOCOL,
        "revision": REVISION,
        "status": "passed",
        "created_at": _now(),
        "e10_source_hash": e10_source_hash(),
        "e7_dependency_source_hash": e7_dependency_hash(),
        "e2_source_hashes": source_values,
        "recipe": {
            "seed": SEED,
            "epochs": EPOCHS,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "precision": "bf16",
            "attention_implementation": "sdpa",
            "gamma": GAMMA,
            "nc_lambdas": NC_LAMBDAS,
        },
        "tasks": list(TASKS),
        "variants": {
            "span": {"rank": SPAN_RANK, "route": "content_softmax_plus_null"},
            "conv": {"rank": CONV_RANK, "route": "uniform_valid_kernel_1_to_4"},
        },
        "alignment_checks": alignment_checks,
        "head_checks": head_checks,
        "gradient_checks": gradient_checks,
        "strict_roundtrip_checks": roundtrip_checks,
        "shape_rows": shape_rows,
        "optimizer_coverage": optimizer_coverage,
        "eligible_devices": devices,
        "gpu_snapshot": snapshot,
        "gpu_query_error": gpu_error,
        "full_training_launched": False,
    }
    _write_json(ORCH_ROOT / "preflight.json", payload)
    return payload


def _e10_move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensors while carrying alignment tied to this exact raw batch."""

    if "alignment" not in batch or "raw_texts" not in batch:
        raise ValueError("E10 batch must carry explicit raw_texts and alignment")
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _make_noisy_batch_e10(
    tokenizer: Any,
    ids: Sequence[Any],
    text_by_id: Mapping[str, str],
    *,
    max_length: int,
    seed: int,
    epoch: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    clean_texts = [text_by_id[str(example_id)] for example_id in ids]
    noisy_result = noisy_batch(clean_texts, ids, seed=seed, epoch=epoch)
    batch, alignments = _tokenize_with_alignment(tokenizer, list(noisy_result.texts), max_length)
    return (
        {
            **{key: value.to(device) for key, value in batch.items() if isinstance(value, Tensor)},
            "alignment": alignments,
            "raw_texts": list(noisy_result.texts),
        },
        noisy_result.to_dict(),
    )


def _run_validation_e10(
    model: E10Classifier,
    loader: Any,
    device: torch.device,
    precision: str,
    task_id: str,
    bundle: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    counts = e1_train.task_aspect_class_counts(bundle)
    gold: list[Any] = []
    predicted: list[Any] = []
    ids: list[Any] = []
    with torch.inference_mode():
        for batch in loader:
            moved = _e10_move_batch(batch, device)
            with e1_train.make_autocast(precision, device):
                logits = model(
                    moved["input_ids"],
                    moved["attention_mask"],
                    token_type_ids=moved.get("token_type_ids"),
                    alignment=moved["alignment"],
                )
            predicted.extend(e1_train._predictions(logits, aspect_class_counts=counts))
            gold.extend(batch["gold_labels"].tolist())
            ids.extend(batch["ids"])
    metrics = e1_train.evaluate_predictions(task_id, bundle, gold, predicted, split="validation")
    rows = [
        {"id": example_id, "gold": target, "prediction": prediction}
        for example_id, target, prediction in zip(ids, gold, predicted)
    ]
    return metrics, rows


def _train_loop_e10(
    *,
    model: E10Classifier,
    train_loader: Any,
    valid_loader: Any,
    bundle: Any,
    task_id: str,
    device: torch.device,
    precision: str,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epochs: int,
    effective_batch_size: int,
    micro_batch_size: int,
    mode: str,
    alpha: float | None,
    consistency_lambda: float | None,
    cache: Any,
    tokenizer: Any,
    train_split: Any,
    max_length: int,
    seed: int,
    artifact_dir: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if mode != "nc" or cache is not None or alpha is not None:
        raise ValueError("E10 runner only permits the frozen E8 NC objective")
    if consistency_lambda is None:
        raise ValueError("E10 NC lambda is required")
    metadata.update(model.architecture_metadata())
    counts = e2_train.task_aspect_class_counts(bundle)
    num_aspects = e2_train.task_num_aspects(bundle)
    accumulation_steps = effective_batch_size // micro_batch_size
    text_by_id = {
        str(example_id): text
        for example_id, text in zip(e2_train.split_ids(train_split), e2_train.split_texts(train_split))
    }
    _, nc_fn = e2_train._load_loss_api()
    best_score = -math.inf
    best_epoch = 0
    best_metrics: dict[str, Any] = {}
    best_predictions: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_examples = 0
        loss_totals = {"total": 0.0, "ce": 0.0, "consistency": 0.0}
        noise_stats: dict[str, Any] = {"count": 0, "changed_count": 0, "operation_counts": {}}
        epoch_started = time.time()
        for batch_index, batch in enumerate(train_loader):
            moved = _e10_move_batch(batch, device)
            with e1_train.make_autocast(precision, device):
                logits = model(
                    moved["input_ids"],
                    moved["attention_mask"],
                    token_type_ids=moved.get("token_type_ids"),
                    alignment=moved["alignment"],
                )
                ce = e2_train.compute_loss(
                    logits,
                    moved["labels"],
                    num_aspects=num_aspects,
                    aspect_class_counts=counts,
                )
                noisy, batch_noise = _make_noisy_batch_e10(
                    tokenizer,
                    batch["ids"],
                    text_by_id,
                    max_length=max_length,
                    seed=seed,
                    epoch=epoch,
                    device=device,
                )
                noisy_logits = model(
                    noisy["input_ids"],
                    noisy["attention_mask"],
                    token_type_ids=noisy.get("token_type_ids"),
                    alignment=noisy["alignment"],
                )
                consistency = nc_fn(
                    logits,
                    noisy_logits,
                    temperature=1.0,
                    slot_class_counts=counts or None,
                )
                total = ce + float(consistency_lambda) * consistency
                window_start = (batch_index // accumulation_steps) * accumulation_steps
                window_end = min(window_start + accumulation_steps, len(train_loader))
                window_examples = sum(
                    min(micro_batch_size, len(train_loader.dataset) - index * micro_batch_size)
                    for index in range(window_start, window_end)
                )
                scaled = total * (len(batch["labels"]) / max(window_examples, 1))
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite E10 loss at epoch={epoch}, batch={batch_index}")
            scaled.backward()
            count = len(batch["labels"])
            total_examples += count
            loss_totals["total"] += float(total.detach().cpu()) * count
            loss_totals["ce"] += float(ce.detach().cpu()) * count
            loss_totals["consistency"] += float(consistency.detach().cpu()) * count
            noise_stats["count"] += int(batch_noise["count"])
            noise_stats["changed_count"] += int(batch_noise["changed_count"])
            for operation, operation_count in batch_noise["operation_counts"].items():
                noise_stats["operation_counts"][operation] = (
                    noise_stats["operation_counts"].get(operation, 0) + int(operation_count)
                )
            boundary = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(train_loader)
            if boundary:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
        val_metrics, val_predictions = _run_validation_e10(
            model, valid_loader, device, precision, task_id, bundle
        )
        score = e1_train.selection_score(val_metrics)
        if not math.isfinite(score):
            raise RuntimeError(f"Non-finite E10 validation selection_f1 at epoch {epoch}")
        noise_stats["change_rate"] = noise_stats["changed_count"] / max(noise_stats["count"], 1)
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_losses": {key: value / max(total_examples, 1) for key, value in loss_totals.items()},
            "validation": val_metrics,
            "selection_f1": score,
            "learning_rate": scheduler.get_last_lr()[0],
            "epoch_seconds": time.time() - epoch_started,
            "noise": noise_stats,
        }
        curves.append(row)
        _write_json(artifact_dir / "epoch_curves.json", {"epochs": curves})
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = val_metrics
            best_predictions = val_predictions
            e2_train._save_model_checkpoint(
                artifact_dir / "best_checkpoint.pt",
                model,
                optimizer,
                scheduler,
                epoch=epoch,
                global_step=global_step,
                metadata=metadata,
            )
            _write_json(artifact_dir / "best_validation_metrics.json", best_metrics)
            _write_json(artifact_dir / "validation_predictions.json", {"rows": best_predictions})
    metadata.update(
        {
            "best_epoch": best_epoch,
            "best_selection_f1": best_score,
            "best_validation_metrics": best_metrics,
            "elapsed_seconds": time.time() - metadata["started_at_unix"],
            "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "peak_gpu_memory_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
            "status": "completed",
            "noise_statistics": [row["noise"] for row in curves],
        }
    )
    checkpoint_path = artifact_dir / "best_checkpoint.pt"
    if checkpoint_path.is_file():
        metadata["checkpoint_sha256"] = e1_train.sha256_file(checkpoint_path)
    _write_json(artifact_dir / "metadata.json", metadata)
    e2_train._set_status(artifact_dir, "completed", best_epoch=best_epoch, best_selection_f1=best_score)
    return {
        "task_id": task_id,
        "mode": mode,
        "seed": metadata["seed"],
        "learning_rate": metadata["learning_rate"],
        "artifact_dir": str(artifact_dir),
        "best_epoch": best_epoch,
        "best_selection_f1": best_score,
        "validation_metrics": best_metrics,
        "status": "completed",
    }


def _install_runtime_adapters(variant: str) -> dict[str, Any]:
    original_builder = e2_train.build_e2_student
    original_loader_set = e2_train._loader_set
    original_train_loop = e2_train._train_loop

    def builder(backbone_dir: Path, *, num_classes: int, num_aspects: int = 0, aspect_class_counts: Sequence[int] | None = None, attention_implementation: str = "sdpa"):
        return build_e10_student(
            backbone_dir,
            variant=variant,
            num_classes=num_classes,
            num_aspects=num_aspects,
            aspect_class_counts=aspect_class_counts,
            attention_implementation=attention_implementation,
        )

    def loader_set(tokenizer: Any, train_split: Any, valid_split: Any, bundle: Any, *, seed: int, micro_batch_size: int, max_length: int):
        train_loader = make_e10_loader(
            tokenizer, train_split,
            texts=e2_train.split_texts(train_split),
            ids=e2_train.split_ids(train_split),
            labels=e2_train.model_labels(bundle, "train", train_split),
            gold_labels=e2_train.split_labels(train_split),
            max_length=max_length,
            batch_size=micro_batch_size,
            shuffle=True,
            seed=seed,
        )
        ordered_loader = make_e10_loader(
            tokenizer, train_split,
            texts=e2_train.split_texts(train_split),
            ids=e2_train.split_ids(train_split),
            labels=e2_train.model_labels(bundle, "train", train_split),
            gold_labels=e2_train.split_labels(train_split),
            max_length=max_length,
            batch_size=micro_batch_size,
            shuffle=False,
            seed=seed,
        )
        valid_loader = make_e10_loader(
            tokenizer, valid_split,
            texts=e2_train.split_texts(valid_split),
            ids=e2_train.split_ids(valid_split),
            labels=e2_train.model_labels(bundle, "validation", valid_split),
            gold_labels=e2_train.split_labels(valid_split),
            max_length=max_length,
            batch_size=micro_batch_size,
            shuffle=False,
            seed=seed,
        )
        return train_loader, ordered_loader, valid_loader

    e2_train.build_e2_student = builder  # type: ignore[assignment]
    e2_train._loader_set = loader_set  # type: ignore[assignment]
    e2_train._train_loop = _train_loop_e10  # type: ignore[assignment]
    return {
        "original_builder": original_builder,
        "original_loader_set": original_loader_set,
        "original_train_loop": original_train_loop,
    }


def _restore_runtime_adapters(originals: Mapping[str, Any]) -> None:
    e2_train.build_e2_student = originals["original_builder"]  # type: ignore[assignment]
    e2_train._loader_set = originals["original_loader_set"]  # type: ignore[assignment]
    e2_train._train_loop = originals["original_train_loop"]  # type: ignore[assignment]


def _annotate_artifact(artifact_dir: Path, *, variant: str, device_name: str, source_values: Mapping[str, str], coverage: Mapping[str, Any], started_at: str) -> None:
    fields = {
        "e10_protocol": PROTOCOL,
        "e10_revision": REVISION,
        "e10_source_hash": e10_source_hash(),
        "e7_dependency_source_hash": e7_dependency_hash(),
        "e2_source_hashes": dict(source_values),
        "variant": variant,
        "device": device_name,
        "worker_started_at": started_at,
        "optimizer_group_contract": coverage["groups"],
        "branch_lr_contract": {"region": "block_01", "exponent": 8, "scale": GAMMA**8},
        "alignment_contract": "explicit_raw_text_offsets_clean_and_noisy",
    }
    for name in ("metadata.json", "resolved_config.json"):
        path = artifact_dir / name
        content = _read_json(path)
        content.update(fields)
        _write_json(path, content)
    checkpoint_path = artifact_dir / "best_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_metadata = dict(checkpoint.get("metadata", {}))
    checkpoint_metadata.update(fields)
    checkpoint["metadata"] = checkpoint_metadata
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(checkpoint_path)


def run_worker(variant: str, task_id: str, device_name: str, output_dir: Path) -> dict[str, Any]:
    if variant not in VARIANTS or task_id not in TASKS:
        raise ValueError(f"Unknown E10 trial {variant}, {task_id}")
    source_values = _source_hashes()
    source_hash = e10_source_hash()
    if MANIFEST_PATH.is_file():
        manifest = _read_json(MANIFEST_PATH)
        if manifest.get("e10_source_hash") != source_hash:
            raise RuntimeError("E10 source changed after provenance freeze; refusing worker launch")
    started_at = _now()
    _set_worker_status(
        variant,
        task_id,
        {
            "protocol": PROTOCOL,
            "variant": variant,
            "task_id": task_id,
            "seed": SEED,
            "status": "running",
            "pid": os.getpid(),
            "device": device_name,
            "e10_source_hash": source_hash,
            "e2_source_hashes": source_values,
            "output_dir": str(output_dir),
            "started_at": started_at,
        },
    )
    captured: dict[str, Any] = {}
    originals = _install_runtime_adapters(variant)
    original_optimizer_factory = e2_train.make_optimizer

    def e10_make_optimizer(model: nn.Module, learning_rate: float) -> torch.optim.Optimizer:
        optimizer = e7_optim.build_layerwise_optimizer(model, learning_rate, gamma=GAMMA, num_layers=NUM_LAYERS)
        captured["optimizer"] = optimizer
        captured["coverage"] = e7_optim.validate_optimizer_groups(
            model, optimizer, base_lr=learning_rate, gamma=GAMMA, num_layers=NUM_LAYERS
        )
        return optimizer

    e2_train.make_optimizer = e10_make_optimizer  # type: ignore[assignment]
    try:
        result = e2_train.train_student_trial(
            task_id=task_id,
            mode="nc",
            repo_root=ROOT,
            output_dir=output_dir,
            seed=SEED,
            consistency_lambda=NC_LAMBDAS[task_id],
            learning_rate=TASK_LRS[task_id],
            device_name=device_name,
            precision="bf16",
            attention_implementation="sdpa",
            epochs=EPOCHS,
            effective_batch_size=EFFECTIVE_BATCH_SIZE,
            micro_batch_size=MICRO_BATCH_SIZE,
            stage="stageE10_span_residual" if variant == "span" else "stageE10_multikernel_conv",
            method=f"e10_{variant}_plus_e8_nc_llrd",
            coefficient=NC_LAMBDAS[task_id],
            revision=REVISION,
            expected_e2_code_hash=E2_CODE_HASH,
            expected_e2_config_hash=E2_CONFIG_HASH,
            expected_orchestrator_hash=source_values["orchestrator"],
        )
    except Exception as exc:
        _set_worker_status(
            variant,
            task_id,
            {
                "protocol": PROTOCOL,
                "variant": variant,
                "task_id": task_id,
                "status": "failed",
                "pid": os.getpid(),
                "device": device_name,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        e2_train.make_optimizer = original_optimizer_factory  # type: ignore[assignment]
        _restore_runtime_adapters(originals)
    optimizer = captured.get("optimizer")
    coverage = captured.get("coverage")
    if optimizer is None or coverage is None:
        raise RuntimeError("E10 optimizer adapter did not capture optimizer/coverage")
    artifact_dir = Path(str(result["artifact_dir"]))
    _annotate_artifact(
        artifact_dir,
        variant=variant,
        device_name=device_name,
        source_values=source_values,
        coverage=coverage,
        started_at=started_at,
    )
    _set_worker_status(
        variant,
        task_id,
        {
            "protocol": PROTOCOL,
            "variant": variant,
            "task_id": task_id,
            "seed": SEED,
            "status": "completed",
            "pid": os.getpid(),
            "device": device_name,
            "artifact_dir": str(artifact_dir),
            "best_epoch": result.get("best_epoch"),
            "best_selection_f1": result.get("best_selection_f1"),
            "validation_metrics": result.get("validation_metrics"),
            "finished_at": _now(),
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return result


def _recompute_validation(variant: str, task_id: str, artifact: Path) -> dict[str, Any]:
    predictions = _read_json(artifact / "validation_predictions.json").get("rows")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError(f"Missing validation predictions for {variant}/{task_id}")
    bundle = e1_train.load_task_bundle(task_id, ROOT, include_test=False)
    valid_split = e1_train.split_field(bundle, "validation")
    expected_ids = [str(value) for value in e1_train.split_ids(valid_split)]
    actual_ids = [str(row["id"]) for row in predictions]
    if actual_ids != expected_ids:
        raise ValueError(f"{variant}/{task_id} validation IDs do not match frozen split")
    return e1_train.evaluate_predictions(
        task_id,
        bundle,
        [row["gold"] for row in predictions],
        [row["prediction"] for row in predictions],
        split="validation",
    )


def _baseline_e8(task_id: str) -> dict[str, Any]:
    return _read_json(ROOT / "artifacts" / "e8" / task_id / "seed_42" / "best_validation_metrics.json")


def _summary_from_artifacts(artifacts: Mapping[tuple[str, str], Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for variant, task_id in TRIALS:
        artifact = artifacts[(variant, task_id)]
        recomputed = _recompute_validation(variant, task_id, artifact)
        baseline = _baseline_e8(task_id)
        metadata = _read_json(artifact / "metadata.json")
        candidate_f1 = float(recomputed["selection_f1"])
        baseline_f1 = float(baseline["selection_f1"])
        rows.append(
            {
                "variant": variant,
                "task_id": task_id,
                "seed": SEED,
                "artifact_dir": str(artifact),
                "best_epoch": metadata.get("best_epoch"),
                "selection_f1": candidate_f1,
                "e8_selection_f1": baseline_f1,
                "candidate_minus_e8_pp": 100.0 * (candidate_f1 - baseline_f1),
                "candidate_f1_percent": 100.0 * candidate_f1,
                "elapsed_seconds": metadata.get("elapsed_seconds"),
                "peak_gpu_memory_allocated_bytes": metadata.get("peak_gpu_memory_allocated_bytes"),
                "peak_gpu_memory_reserved_bytes": metadata.get("peak_gpu_memory_reserved_bytes"),
                "parameter_count": metadata.get("model", {}).get("total_parameters"),
                "metrics": recomputed,
            }
        )
    by_variant_task = {(row["variant"], row["task_id"]): row for row in rows}
    gate_rows: list[dict[str, Any]] = []
    for task_id in TASKS:
        span = by_variant_task[("span", task_id)]
        conv = by_variant_task[("conv", task_id)]
        gate_rows.append(
            {
                "task_id": task_id,
                "span_minus_e8_pp": span["candidate_minus_e8_pp"],
                "conv_minus_e8_pp": conv["candidate_minus_e8_pp"],
                "span_minus_conv_pp": span["candidate_f1_percent"] - conv["candidate_f1_percent"],
            }
        )
    span_e8_gains = [row["span_minus_e8_pp"] for row in gate_rows]
    span_conv_gains = [row["span_minus_conv_pp"] for row in gate_rows]
    gate = {
        "span_vs_e8": {
            "tasks_gain": sum(value > 0 for value in span_e8_gains),
            "tasks_gain_at_least_0p3pp": sum(value >= 0.3 for value in span_e8_gains),
            "no_drop_worse_than_0p3pp": all(value >= -0.3 for value in span_e8_gains),
        },
        "span_vs_conv": {
            "tasks_win": sum(value > 0 for value in span_conv_gains),
            "no_loss_worse_than_0p3pp": all(value >= -0.3 for value in span_conv_gains),
        },
    }
    gate["passes_screening_heuristic"] = bool(
        gate["span_vs_e8"]["tasks_gain"] >= 4
        and gate["span_vs_e8"]["tasks_gain_at_least_0p3pp"] >= 2
        and gate["span_vs_e8"]["no_drop_worse_than_0p3pp"]
        and gate["span_vs_conv"]["tasks_win"] >= 3
        and gate["span_vs_conv"]["no_loss_worse_than_0p3pp"]
    )
    payload = {
        "protocol": PROTOCOL,
        "revision": REVISION,
        "status": "completed",
        "created_at": _now(),
        "e10_source_hash": e10_source_hash(),
        "e7_dependency_source_hash": e7_dependency_hash(),
        "e2_source_hashes": _source_hashes(),
        "baseline_units": "E8 best_validation_metrics.json selection_f1 is 0..1; display and deltas use *100",
        "rows": rows,
        "gate_rows": gate_rows,
        "gate": gate,
    }
    _write_json(ARTIFACT_ROOT / "summary.json", payload)
    fieldnames = [
        "variant", "task_id", "seed", "best_epoch", "selection_f1", "candidate_f1_percent",
        "e8_selection_f1", "candidate_minus_e8_pp", "elapsed_seconds",
        "peak_gpu_memory_allocated_bytes", "peak_gpu_memory_reserved_bytes", "parameter_count",
    ]
    with (ARTIFACT_ROOT / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fieldnames} for row in rows)
    return payload


def _write_source_manifest(source_values: Mapping[str, str], snapshot: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "protocol": PROTOCOL,
        "revision": REVISION,
        "frozen_at": _now(),
        "e10_source_hash": e10_source_hash(),
        "e7_dependency_source_hash": e7_dependency_hash(),
        "e2_source_hashes": dict(source_values),
        "gpu_snapshot_at_freeze": dict(snapshot),
        "tasks": list(TASKS),
        "variants": list(VARIANTS),
        "seed": SEED,
        "epochs": EPOCHS,
    }
    if MANIFEST_PATH.is_file():
        existing = _read_json(MANIFEST_PATH)
        if existing.get("e10_source_hash") != payload["e10_source_hash"]:
            raise RuntimeError("Existing E10 source manifest does not match current code")
        return existing
    _write_json(MANIFEST_PATH, payload)
    return payload


def _adopt_or_requeue() -> tuple[dict[tuple[str, str], Path], dict[tuple[str, str], dict[str, Any]]]:
    completed: dict[tuple[str, str], Path] = {}
    active: dict[tuple[str, str], dict[str, Any]] = {}
    for variant, task_id in TRIALS:
        artifact = _completed_artifact(variant, task_id)
        if artifact is not None:
            completed[(variant, task_id)] = artifact
            continue
        worker_path = _worker_path(variant, task_id)
        if worker_path.is_file():
            worker = _read_json(worker_path)
            if worker.get("status") == "running" and _pid_alive(worker.get("pid")):
                active[(variant, task_id)] = worker
    return completed, active


def run_all(*, poll_seconds: float = 15.0) -> dict[str, Any]:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    ORCH_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    WORKER_ROOT.mkdir(parents=True, exist_ok=True)
    source_values = _source_hashes()
    devices, snapshot = _eligible_devices()
    if not devices:
        raise RuntimeError(f"No GPU meets {GPU_GUARD_MIB} MiB guard; snapshot={snapshot}")
    preflight_path = ORCH_ROOT / "preflight.json"
    if not preflight_path.is_file() or _read_json(preflight_path).get("status") != "passed":
        raise RuntimeError("Run --preflight and obtain root review before --run-all")
    preflight = _read_json(preflight_path)
    if preflight.get("e10_source_hash") != e10_source_hash():
        raise RuntimeError("E10 source changed after preflight; rerun preflight before --run-all")
    _write_source_manifest(source_values, snapshot)
    completed, adopted = _adopt_or_requeue()
    state_path = ORCH_ROOT / "state.json"
    if state_path.is_file():
        state = _read_json(state_path)
        old_pid = state.get("controller_pid")
        if state.get("status") == "running" and old_pid != os.getpid() and _pid_alive(old_pid):
            raise RuntimeError(f"E10 controller already active with PID {old_pid}")
    if len(completed) == len(TRIALS):
        return _summary_from_artifacts(completed)
    jobs: dict[str, dict[str, Any]] = {
        _trial_key(*key): {"variant": key[0], "task_id": key[1], "status": "completed", "artifact_dir": str(path)}
        for key, path in completed.items()
    }
    running_adopted = dict(adopted)
    for key, worker in adopted.items():
        jobs[_trial_key(*key)] = {**worker, "status": "adopted_running", "adopted": True}
    queue = [key for key in TRIALS if key not in completed and key not in adopted]
    controller_state: dict[str, Any] = {
        "protocol": PROTOCOL,
        "status": "running",
        "controller_pid": os.getpid(),
        "started_at": _now(),
        "e10_source_hash": e10_source_hash(),
        "e2_source_hashes": source_values,
        "gpu_snapshot_at_launch": snapshot,
        "eligible_devices": devices,
        "trials": [list(key) for key in TRIALS],
        "jobs": jobs,
        "adopted_active": [list(key) for key in adopted],
    }
    _write_json(state_path, controller_state)
    controller_log = (LOG_ROOT / "controller.log").open("a", encoding="utf-8", buffering=1)
    running_children: dict[tuple[str, str], tuple[subprocess.Popen[str], str]] = {}
    assigned_devices = {str(worker.get("device")) for worker in adopted.values()}

    def launch(key: tuple[str, str], device_name: str) -> None:
        variant, task_id = key
        output_dir = _trial_dir(variant, task_id)
        log_path = LOG_ROOT / f"{_trial_key(variant, task_id)}.log"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--variant", variant,
            "--task-id", task_id,
            "--device", device_name,
            "--output-dir", str(output_dir),
        ]
        log_handle = log_path.open("a", encoding="utf-8", buffering=1)
        log_handle.write(json.dumps({"event": "launch", "time": _now(), "command": command}) + "\n")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", "")},
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        log_handle.close()
        running_children[key] = (process, device_name)
        jobs[_trial_key(variant, task_id)] = {
            "variant": variant,
            "task_id": task_id,
            "device": device_name,
            "pid": process.pid,
            "artifact_dir": str(output_dir),
            "log_path": str(log_path),
            "status": "running",
            "launched_at": _now(),
            "command": command,
        }
        controller_log.write(json.dumps({"event": "launch", "trial": list(key), "pid": process.pid, "device": device_name}) + "\n")
        _write_json(state_path, controller_state)

    try:
        while queue or running_children or running_adopted:
            snapshot_now = _gpu_snapshot()
            free_devices = [
                f"cuda:{index}"
                for index in PREFERRED_GPU_INDICES
                if index in snapshot_now and snapshot_now[index]["free_mib"] >= GPU_GUARD_MIB
                and f"cuda:{index}" not in assigned_devices
            ]
            while queue and free_devices:
                key = queue.pop(0)
                device_name = free_devices.pop(0)
                assigned_devices.add(device_name)
                launch(key, device_name)
            finished: list[tuple[str, str]] = []
            for key, (process, device_name) in list(running_children.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                trial_key = _trial_key(*key)
                jobs[trial_key].update({"status": "completed" if return_code == 0 else "failed", "returncode": return_code, "finished_at": _now()})
                assigned_devices.discard(device_name)
                if return_code == 0:
                    artifact = _completed_artifact(*key)
                    if artifact is None:
                        raise RuntimeError(f"Worker {trial_key} exited successfully without a completed artifact")
                    completed[key] = artifact
                else:
                    controller_state.update({"status": "failed", "failed_trial": list(key), "finished_at": _now()})
                    _write_json(state_path, controller_state)
                    raise RuntimeError(f"E10 worker failed: {trial_key}; inspect {jobs[trial_key].get('log_path')}")
                finished.append(key)
                controller_log.write(json.dumps({"event": "finish", "trial": list(key), "returncode": return_code}) + "\n")
            for key in finished:
                running_children.pop(key)
            for key in list(running_adopted):
                worker = _read_json(_worker_path(*key)) if _worker_path(*key).is_file() else {}
                if worker.get("status") == "completed":
                    artifact = _completed_artifact(*key)
                    if artifact is None:
                        raise RuntimeError(f"Adopted worker {key} reported completed without artifact")
                    completed[key] = artifact
                    running_adopted.pop(key)
                    assigned_devices.discard(str(worker.get("device")))
                elif worker.get("status") == "failed" or not _pid_alive(worker.get("pid")):
                    assigned_devices.discard(str(worker.get("device")))
                    running_adopted.pop(key)
                    queue.insert(0, key)
                    jobs[_trial_key(*key)]["status"] = "requeued_after_worker_exit"
            controller_state["jobs"] = jobs
            controller_state["completed"] = [list(key) for key in completed]
            controller_state["queue"] = [list(key) for key in queue]
            _write_json(state_path, controller_state)
            if queue or running_children or running_adopted:
                time.sleep(poll_seconds)
    finally:
        controller_log.close()
    summary = _summary_from_artifacts(completed)
    controller_state.update({"status": "completed", "finished_at": _now(), "summary": str(ARTIFACT_ROOT / "summary.json")})
    _write_json(state_path, controller_state)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run E10 span versus convolution screening")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--task-id", choices=TASKS)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-device", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.preflight:
            payload = run_preflight(args.preflight_device)
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
            return 0 if payload.get("status") == "passed" else 1
        if args.variant or args.task_id:
            if not args.variant or not args.task_id or args.output_dir is None:
                raise ValueError("worker mode requires --variant, --task-id and --output-dir")
            run_worker(args.variant, args.task_id, args.device, args.output_dir.resolve())
            return 0
        if args.run_all:
            payload = run_all()
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
            return 0
        raise ValueError("Choose one of --preflight, --run-all, or worker arguments")
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
