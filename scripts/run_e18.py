#!/usr/bin/env python3
"""E18 preflight, profile, worker, and run-all controller.

Screening 4B (segment-fork-consistency training on the E2/E7 student; see
``docs/E18_PLAN.md``).  The E2 data/loss loop and E7 layer-wise optimizer
remain imported dependencies, unmodified.  This runner only:

* wraps a fresh ``build_e2_student`` in ``goibert_e18.modeling.E18Classifier``
  (fork-augmented forward, no new parameters);
* monkeypatches ``goibert_e2.train.build_e2_student`` /
  ``make_optimizer`` / ``compute_loss`` / ``run_validation`` inside one
  worker process for the duration of one trial, exactly as
  ``scripts/run_e17.py`` does;
* writes E18-isolated artifacts under ``artifacts/e18/`` and refuses full
  6-trial dispatch until the caller explicitly passes ``--run-all
  --dispatch-approved`` (root review gate).

Never edit ``src/goibert``, ``src/goibert_e2``, ``src/goibert_e7``,
``src/goibert_e17``, other ``scripts/run_e*.py``, ``models/``, or
``artifacts/e1``..``artifacts/e17``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ARTIFACT_ROOT = ROOT / "artifacts" / "e18"
CONFIG_PATH = ROOT / "configs" / "e18" / "protocol.json"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402
from torch import Tensor, nn  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from goibert import train as e1_train  # noqa: E402
from goibert_e2 import train as e2_train  # noqa: E402
from goibert_e2.modeling import build_e2_student  # noqa: E402
from goibert_e7 import optim as e7_optim  # noqa: E402
from goibert_e18.modeling import (  # noqa: E402
    ARMS,
    CONNECTOR_IDS,
    E18Classifier,
    LAMBDA_FORK,
    S_MAX,
    build_e18_student,
)
from goibert_e18.losses import ForkTelemetry, compute_fork_ce, lambda_fork_ramp, make_patched_compute_loss  # noqa: E402


PROTOCOL = "E18-segment-fork-consistency-screening-4B-v1"
LAMBDA_FORK_RAMP_EPOCHS = 3  # linear ramp of lambda_fork over the first 3 epochs (pre-launch amendment, fix 2)
REVISION = "e18-20260913-r1"
TASKS = ("absa_restaurant", "absa_hotel")
ARMS_ORDER = ARMS  # ("b0_baseline", "b1_fork", "b2_random_mask", "b3_fork_detach")
BATCH1_ARMS = ("b0_baseline", "b1_fork", "b2_random_mask")
SEED = 42
EPOCHS = 30
EFFECTIVE_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 16
ACCUMULATION_STEPS = 2
MAX_LENGTH = 256
PRECISION = "bf16"
ATTENTION_IMPLEMENTATION = "sdpa"
GAMMA = 0.8
NUM_LAYERS = 10
WEIGHT_DECAY = 0.01
TASK_LRS = {"absa_restaurant": 1e-4, "absa_hotel": 1e-4}
NC_LAMBDAS = {"absa_restaurant": 0.1, "absa_hotel": 0.1}
GUARD_USED_MIB = 5 * 1024
GUARD_UTIL_PCT = 5.0
RESERVE_HEADROOM_MIB = 4 * 1024
# GPU1 was confirmed idle (20 MiB used, 0% util) at plan time; GPU0/2/3
# belonged to other users' jobs.  Preference order only; eligibility is
# re-checked against the live snapshot every time.
PREFERRED_GPU_INDICES = ("1", "0", "2", "3")
PROFILE_STEP_RATIO_GATE = 2.0  # docs/E18_PLAN.md preflight point 6


# ---------------------------------------------------------------------------
# Generic helpers (ported from scripts/run_e17.py)
# ---------------------------------------------------------------------------
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
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".json", ".safetensors", ".txt", ".model", ".vocab"}
    }


def _tree_py_hashes(root: Path) -> dict[str, str]:
    if not root.is_dir():
        return {}
    return {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in sorted(root.rglob("*.py"))
        if path.is_file()
    }


def _gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=30)
    except Exception as exc:  # noqa: BLE001 - report, do not crash preflight
        return {"available": False, "error": repr(exc), "rows": {}}
    rows: dict[str, dict[str, float]] = {}
    for line in output.strip().splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) >= 4:
            rows[fields[0]] = {
                "free_mib": int(float(fields[1])),
                "used_mib": int(float(fields[2])),
                "utilization_pct": float(fields[3]),
            }
    return {"available": bool(rows), "rows": rows}


def _eligible_devices(snapshot: Mapping[str, Any]) -> list[str]:
    rows = snapshot.get("rows", {})
    eligible = [
        index
        for index, row in rows.items()
        if float(row["used_mib"]) < GUARD_USED_MIB and float(row["utilization_pct"]) <= GUARD_UTIL_PCT
    ]
    ordered = [index for index in PREFERRED_GPU_INDICES if index in eligible]
    ordered += [index for index in eligible if index not in ordered]
    return [f"cuda:{index}" for index in ordered]


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


def _source_manifest() -> dict[str, Any]:
    source_files = [
        Path(__file__).resolve(),
        ROOT / "src" / "goibert_e18" / "__init__.py",
        ROOT / "src" / "goibert_e18" / "modeling.py",
        ROOT / "src" / "goibert_e18" / "losses.py",
        CONFIG_PATH,
        ROOT / "src" / "goibert_e2" / "train.py",
        ROOT / "src" / "goibert_e2" / "modeling.py",
        ROOT / "src" / "goibert_e2" / "losses.py",
        ROOT / "src" / "goibert_e2" / "noise.py",
        ROOT / "src" / "goibert_e2" / "cache.py",
        ROOT / "src" / "goibert_e7" / "optim.py",
    ]
    source_hashes = {str(path.relative_to(ROOT)): _sha256(path) for path in source_files if path.is_file()}
    return {
        "protocol": PROTOCOL,
        "revision": REVISION,
        "created_at": _now(),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "source_hashes": source_hashes,
        "backbone": {
            "path": str((ROOT / "models" / "goiturn-s-gd").resolve()),
            "files": _directory_hashes(ROOT / "models" / "goiturn-s-gd"),
        },
        "frozen_e2_source_hashes": e2_train.source_hashes(ROOT),
        "frozen_e7_files": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in sorted((ROOT / "src" / "goibert_e7").rglob("*.py"))
        },
        "e1_e17_source_py_hashes": {
            name: _tree_py_hashes(ROOT / "src" / name)
            for name in (
                "goibert",
                "goibert_e2",
                "goibert_e3",
                "goibert_e4",
                "goibert_e5",
                "goibert_e6",
                "goibert_e7",
                "goibert_e8",
                "goibert_e9",
                "goibert_e10",
                "goibert_e11",
                "goibert_e12",
                "goibert_e13",
                "goibert_e14",
                "goibert_e15",
                "goibert_e16",
                "goibert_e17",
            )
        },
        "recipe": {
            "tasks": list(TASKS),
            "arms": list(ARMS_ORDER),
            "seed": SEED,
            "epochs": EPOCHS,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "accumulation_steps": ACCUMULATION_STEPS,
            "max_length": MAX_LENGTH,
            "precision": PRECISION,
            "attention_implementation": ATTENTION_IMPLEMENTATION,
            "gamma": GAMMA,
            "weight_decay": WEIGHT_DECAY,
            "lambda_fork": LAMBDA_FORK,
            "s_max": S_MAX,
        },
    }


def _stable_source_binding(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol": manifest["protocol"],
        "revision": manifest["revision"],
        "runner_sha256": manifest["runner_sha256"],
        "source_hashes": manifest["source_hashes"],
        "backbone": manifest["backbone"],
        "frozen_e2_source_hashes": manifest["frozen_e2_source_hashes"],
        "frozen_e7_files": manifest["frozen_e7_files"],
        "e1_e17_source_py_hashes": manifest["e1_e17_source_py_hashes"],
        "recipe": manifest["recipe"],
    }


def _dispatch_vram_gate(device: str, snapshot: Mapping[str, Any]) -> dict[str, Any]:
    profile_path = ARTIFACT_ROOT / "profile.json"
    if not profile_path.is_file():
        raise RuntimeError("E18 profile.json is required before dispatch")
    profile = _read_json(profile_path)
    if profile.get("status") != "passed":
        raise RuntimeError(f"E18 profile gate is not passed: {profile.get('status')!r}")
    max_reserved = max(int(row["peak_reserved_bytes"]) for row in profile.get("rows", []))
    index = device.split(":", 1)[1]
    row = snapshot.get("rows", {}).get(index, {})
    free_mib = int(row.get("free_mib", 0))
    util = float(row.get("utilization_pct", 100.0))
    required_mib = math.ceil(max_reserved / (1024 * 1024)) + RESERVE_HEADROOM_MIB
    gate = {
        "device": device,
        "free_mib": free_mib,
        "utilization_pct": util,
        "profile_peak_reserved_bytes": max_reserved,
        "required_free_mib": required_mib,
        "pass": free_mib >= required_mib and util <= GUARD_UTIL_PCT,
    }
    if not gate["pass"]:
        raise RuntimeError(f"E18 per-launch VRAM/utilization gate failed: {gate}")
    return gate


# ---------------------------------------------------------------------------
# Model/data construction helpers
# ---------------------------------------------------------------------------
def _task_contract(task_id: str) -> dict[str, Any]:
    bundle = e2_train.load_task_bundle(task_id, ROOT, include_test=False, smoke=False)
    train_split = e2_train.split_field(bundle, "train")
    valid_split = e2_train.split_field(bundle, "validation")
    return {
        "bundle": bundle,
        "train": train_split,
        "validation": valid_split,
        "num_aspects": e2_train.task_num_aspects(bundle),
        "aspect_class_counts": e2_train.task_aspect_class_counts(bundle),
        "num_classes": e2_train.task_num_classes(bundle),
    }


def _build(arm: str, contract: Mapping[str, Any]) -> tuple[E18Classifier, Any]:
    return build_e18_student(
        ROOT / "models" / "goiturn-s-gd",
        num_classes=int(contract["num_classes"]),
        num_aspects=int(contract["num_aspects"]),
        aspect_class_counts=contract["aspect_class_counts"] if contract["num_aspects"] else None,
        arm=arm,
        attention_implementation=ATTENTION_IMPLEMENTATION,
        s_max=S_MAX,
        lambda_fork=LAMBDA_FORK,
        seed=SEED,
    )


def _tokenizer() -> Any:
    return AutoTokenizer.from_pretrained(ROOT / "models" / "goiturn-s-gd", local_files_only=True, use_fast=True)


@contextmanager
def _zero_dropout(model: nn.Module):
    """Temporarily zero every dropout probability for exact-parity checks.

    Covers ``nn.Dropout`` submodules (head, embeddings, self-output, FFN
    output) and the SDPA self-attention modules' plain-float
    ``dropout_prob`` attribute (RoBERTa's SDPA path never wraps attention
    dropout in an ``nn.Dropout`` submodule; it passes the float straight to
    ``scaled_dot_product_attention``).
    """

    saved_p: list[tuple[nn.Module, float]] = []
    saved_attr: list[tuple[Any, float]] = []
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            saved_p.append((module, module.p))
            module.p = 0.0
        prob = getattr(module, "dropout_prob", None)
        if isinstance(prob, float):
            saved_attr.append((module, prob))
            module.dropout_prob = 0.0
    try:
        yield
    finally:
        for module, p in saved_p:
            module.p = p
        for module, p in saved_attr:
            module.dropout_prob = p


def _fixed_profile_batch(task_id: str) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    contract = _task_contract(task_id)
    tokenizer = _tokenizer()
    train_split = contract["train"]
    valid_split = contract["validation"]
    train_loader, _, _ = e2_train._loader_set(
        tokenizer,
        train_split,
        valid_split,
        contract["bundle"],
        seed=SEED,
        micro_batch_size=MICRO_BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    batch = next(iter(train_loader))
    text_by_id = {
        str(example_id): text
        for example_id, text in zip(e2_train.split_ids(train_split), e2_train.split_texts(train_split))
    }
    noisy, noise_stats = e2_train._make_noisy_batch(
        tokenizer,
        batch["ids"],
        text_by_id,
        max_length=MAX_LENGTH,
        seed=SEED,
        epoch=1,
        device=torch.device("cpu"),
    )
    return tokenizer, batch, contract, {"noisy": noisy, "noise_stats": noise_stats}


def _multi_aspect_rows(task_id: str, *, split_name: str = "train", limit: int = 256) -> dict[str, Any]:
    """First ``limit`` rows (in split order) with >=2 gold aspects among all aspects."""

    contract = _task_contract(task_id)
    split = e2_train.split_field(contract["bundle"], split_name)
    texts = list(e2_train.split_texts(split))
    labels = e2_train.model_labels(contract["bundle"], split_name, split)
    rows_ids = list(e2_train.split_ids(split))
    selected_texts: list[str] = []
    selected_labels: list[Any] = []
    selected_ids: list[Any] = []
    for example_id, text, label in zip(rows_ids, texts, labels):
        if sum(1 for value in label if int(value) > 0) >= 2:
            selected_texts.append(text)
            selected_labels.append(list(label))
            selected_ids.append(example_id)
        if len(selected_texts) >= limit:
            break
    return {
        "contract": contract,
        "ids": selected_ids,
        "texts": selected_texts,
        "labels": selected_labels,
    }


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
def _parity_check(device: torch.device) -> dict[str, Any]:
    contract = _task_contract("absa_restaurant")
    tokenizer = _tokenizer()
    valid_split = contract["validation"]
    texts = list(e2_train.split_texts(valid_split))[:64]
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    model, spec = _build("b1_fork", contract)
    model.to(device).train()

    with _zero_dropout(model):
        model.forks_armed = True
        fork_cls_fp32 = model(input_ids, attention_mask)
        forks_fp32 = model.last_forks
        plain_fp32 = model._plain_forward(input_ids, attention_mask)
        cls_delta_fp32 = float((fork_cls_fp32.float() - plain_fp32.float()).abs().max().item())
        row0_fp32 = forks_fp32["logits"][:, 0, :]
        plain_flat_fp32 = plain_fp32.reshape(plain_fp32.shape[0], -1)
        row0_delta_fp32 = float((row0_fp32.float() - plain_flat_fp32.float()).abs().max().item())

        model.forks_armed = True
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            fork_cls_bf16 = model(input_ids, attention_mask)
            forks_bf16 = model.last_forks
            plain_bf16 = model._plain_forward(input_ids, attention_mask)
        cls_delta_bf16 = float((fork_cls_bf16.float() - plain_bf16.float()).abs().max().item())
        row0_bf16 = forks_bf16["logits"][:, 0, :]
        plain_flat_bf16 = plain_bf16.reshape(plain_bf16.shape[0], -1)
        row0_delta_bf16 = float((row0_bf16.float() - plain_flat_bf16.float()).abs().max().item())

    # Alternation invariant, exercised over several simulated nc-style steps:
    # clean (fork) -> noisy (plain, re-arm) -> clean (fork) -> ...
    model.forks_armed = True
    alternation: list[dict[str, Any]] = []
    for step in range(4):
        logits = model(input_ids, attention_mask)
        alternation.append(
            {
                "step": step,
                "computed_forks": model.last_forks is not None,
                "forks_armed_after": bool(model.forks_armed),
            }
        )
    model.eval()
    with torch.no_grad():
        _ = model(input_ids, attention_mask)
    eval_never_computes_forks = model.last_forks is None
    model.train()

    return {
        "n_rows": len(texts),
        "cls_delta_fp32": cls_delta_fp32,
        "cls_delta_bf16": cls_delta_bf16,
        "fork_row0_vs_cls_delta_fp32": row0_delta_fp32,
        "fork_row0_vs_cls_delta_bf16": row0_delta_bf16,
        "alternation_rows": alternation,
        "alternation_pattern_matches_clean_noisy": [row["computed_forks"] for row in alternation]
        == [True, False, True, False],
        "eval_never_computes_forks": eval_never_computes_forks,
        "model_spec": spec.to_dict(),
        "checks": {
            "cls_fp32_le_1e4": cls_delta_fp32 <= 1e-4,
            "cls_bf16_le_0_1": cls_delta_bf16 <= 0.1,
            "row0_fp32_le_1e4": row0_delta_fp32 <= 1e-4,
        },
    }


def _b0_lambda_zero_check(device: torch.device) -> dict[str, Any]:
    task_id = "absa_restaurant"
    tokenizer, batch, contract, noisy_payload = _fixed_profile_batch(task_id)
    counts = contract["aspect_class_counts"]
    num_aspects = contract["num_aspects"]
    clean = {key: value.to(device) for key, value in batch.items() if isinstance(value, Tensor)}
    noisy = {key: value.to(device) for key, value in noisy_payload["noisy"].items()}
    nc_fn = e2_train._load_loss_api()[1]

    e1_train.seed_everything(SEED)
    ref_model, _ = build_e2_student(
        ROOT / "models" / "goiturn-s-gd",
        num_classes=int(contract["num_classes"]),
        num_aspects=num_aspects,
        aspect_class_counts=counts if num_aspects else None,
        attention_implementation=ATTENTION_IMPLEMENTATION,
    )
    ref_model.to(device).train()

    e1_train.seed_everything(SEED)
    b0_model, _ = _build("b0_baseline", contract)
    b0_model.to(device).train()

    telemetry = ForkTelemetry()
    patched = make_patched_compute_loss(
        b0_model, original_compute_loss=e1_train.compute_loss, lambda_fork=0.0, telemetry=telemetry
    )

    with _zero_dropout(ref_model), _zero_dropout(b0_model):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            ref_clean = ref_model(clean["input_ids"], clean["attention_mask"])
            ref_noisy = ref_model(noisy["input_ids"], noisy["attention_mask"])
            b0_clean = b0_model(clean["input_ids"], clean["attention_mask"])
            b0_noisy = b0_model(noisy["input_ids"], noisy["attention_mask"])
            ref_ce = e1_train.compute_loss(ref_clean, clean["labels"], num_aspects=num_aspects, aspect_class_counts=counts)
            b0_ce = patched(b0_clean, clean["labels"], num_aspects=num_aspects, aspect_class_counts=counts)
            ref_consistency = nc_fn(ref_clean, ref_noisy, temperature=1.0, slot_class_counts=counts or None)
            b0_consistency = nc_fn(b0_clean, b0_noisy, temperature=1.0, slot_class_counts=counts or None)
            ref_total = ref_ce + NC_LAMBDAS[task_id] * ref_consistency
            b0_total = b0_ce + NC_LAMBDAS[task_id] * b0_consistency

    return {
        "task_id": task_id,
        "clean_logits_delta": float((ref_clean.float() - b0_clean.float()).abs().max().item()),
        "noisy_logits_delta": float((ref_noisy.float() - b0_noisy.float()).abs().max().item()),
        "ce_delta": float((ref_ce.float() - b0_ce.float()).abs().item()),
        "consistency_delta": float((ref_consistency.float() - b0_consistency.float()).abs().item()),
        "total_loss_delta": float((ref_total.float() - b0_total.float()).abs().item()),
        "ref_total_loss": float(ref_total.detach().cpu().item()),
        "b0_total_loss": float(b0_total.detach().cpu().item()),
        "b0_fork_telemetry": telemetry.summary(lambda_fork=0.0),
        "checks": {
            "logits_delta_le_1e4": max(
                float((ref_clean.float() - b0_clean.float()).abs().max().item()),
                float((ref_noisy.float() - b0_noisy.float()).abs().max().item()),
            )
            <= 1e-4,
            "total_loss_delta_le_1e4": float((ref_total.float() - b0_total.float()).abs().item()) <= 1e-4,
            "b0_never_stashes_forks": telemetry.summary(lambda_fork=0.0)["considered"] == 0,
        },
    }


def _gradient_checks(device: torch.device) -> dict[str, Any]:
    pool = _multi_aspect_rows("absa_restaurant", split_name="train", limit=64)
    contract = pool["contract"]
    tokenizer = _tokenizer()
    texts = pool["texts"][:16] if len(pool["texts"]) >= 16 else pool["texts"]
    labels = pool["labels"][: len(texts)]
    if not texts:
        return {"error": "no >=2-gold-aspect rows found for gradient check"}
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)

    results: dict[str, Any] = {}
    for arm in ("b1_fork", "b3_fork_detach"):
        model, _ = _build(arm, contract)
        model.to(device).train()
        model.zero_grad(set_to_none=True)
        model.forks_armed = True
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            _ = model(input_ids, attention_mask)
        forks = model.last_forks
        fork_ce, stats = compute_fork_ce(
            forks, labels_tensor, num_aspects=contract["num_aspects"], aspect_class_counts=contract["aspect_class_counts"]
        )
        row: dict[str, Any] = {"stats": stats}
        if fork_ce is None:
            row["fork_ce"] = None
            row["error"] = "no accepted split in this fixed batch"
        else:
            fork_ce.backward()
            enc_params_with_grad = [p for p in model.encoder.parameters() if p.grad is not None]
            head_params_with_grad = [p for p in model.head.parameters() if p.grad is not None]
            enc_norm = (
                float(sum(p.grad.float().pow(2).sum() for p in enc_params_with_grad).sqrt().item())
                if enc_params_with_grad
                else 0.0
            )
            head_norm = (
                float(sum(p.grad.float().pow(2).sum() for p in head_params_with_grad).sqrt().item())
                if head_params_with_grad
                else 0.0
            )
            row.update(
                {
                    "fork_ce": float(fork_ce.detach().cpu().item()),
                    "encoder_grad_norm": enc_norm,
                    "encoder_params_with_grad": len(enc_params_with_grad),
                    "head_grad_norm": head_norm,
                    "head_params_with_grad": len(head_params_with_grad),
                    "encoder_grad_finite": bool(math.isfinite(enc_norm)),
                    "head_grad_finite": bool(math.isfinite(head_norm)),
                }
            )
        results[arm] = row

    checks = {
        "b1_encoder_grad_finite_nonzero": bool(
            results.get("b1_fork", {}).get("encoder_grad_finite")
            and results.get("b1_fork", {}).get("encoder_grad_norm", 0.0) > 0.0
        ),
        "b1_head_grad_nonzero": bool(results.get("b1_fork", {}).get("head_grad_norm", 0.0) > 0.0),
        "b3_encoder_grad_zero": bool(results.get("b3_fork_detach", {}).get("encoder_params_with_grad", -1) == 0),
        "b3_head_grad_nonzero": bool(results.get("b3_fork_detach", {}).get("head_grad_norm", 0.0) > 0.0),
    }
    return {"batch_size": len(texts), "results": results, "checks": checks}


def _split_acceptance_check(device: torch.device) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for task_id in TASKS:
        pool = _multi_aspect_rows(task_id, split_name="train", limit=256)
        contract = pool["contract"]
        tokenizer = _tokenizer()
        model, _ = _build("b1_fork", contract)
        model.to(device).train()

        considered = accepted = connector_hits = 0
        histogram_at_connector = 0
        histogram_evenly_spaced = 0
        batch_size = 32
        texts, labels = pool["texts"], pool["labels"]
        for start in range(0, len(texts), batch_size):
            chunk_texts = texts[start : start + batch_size]
            chunk_labels = labels[start : start + batch_size]
            if not chunk_texts:
                continue
            encoded = tokenizer(chunk_texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            labels_tensor = torch.tensor(chunk_labels, dtype=torch.long, device=device)
            model.forks_armed = True
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                _ = model(input_ids, attention_mask)
            forks = model.last_forks
            fork_ce, stats = compute_fork_ce(
                forks, labels_tensor, num_aspects=contract["num_aspects"], aspect_class_counts=contract["aspect_class_counts"]
            )
            considered += stats["considered"]
            accepted += stats["accepted"]
            connector_hits += stats["connector_hits"]
            for row_splits, row_flags in zip(forks["splits"], forks["connector_flags"]):
                if row_splits:
                    if any(row_flags):
                        histogram_at_connector += 1
                    else:
                        histogram_evenly_spaced += 1
        rows[task_id] = {
            "n_pool_rows": len(texts),
            "considered": considered,
            "accepted": accepted,
            "acceptance_rate": (accepted / considered) if considered else None,
            "connector_hits": connector_hits,
            "connector_fraction_of_accepted": (connector_hits / accepted) if accepted else None,
            "rows_with_any_connector_split": histogram_at_connector,
            "rows_with_evenly_spaced_splits_only": histogram_evenly_spaced,
        }
    return rows


def _shape_loader_checks(device: torch.device) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for task_id in TASKS:
        contract = _task_contract(task_id)
        tokenizer = _tokenizer()
        texts = list(e2_train.split_texts(contract["validation"]))[:4]
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        model, spec = _build("b1_fork", contract)
        model.to(device)

        model.eval()
        with torch.no_grad():
            eval_logits = model(input_ids, attention_mask)
        eval_forks_none = model.last_forks is None

        model.train()
        model.forks_armed = True
        with torch.no_grad():
            train_logits = model(input_ids, attention_mask)
        forks = model.last_forks

        expected_shape = [len(texts), contract["num_aspects"], contract["aspect_class_counts"][0]]
        rows.append(
            {
                "task_id": task_id,
                "num_aspects": contract["num_aspects"],
                "aspect_class_counts": list(contract["aspect_class_counts"]),
                "eval_logits_shape": list(eval_logits.shape),
                "eval_forks_none": eval_forks_none,
                "train_logits_shape": list(train_logits.shape),
                "fork_logits_shape": list(forks["logits"].shape) if forks is not None else None,
                "expected_cls_shape": expected_shape,
                "eval_finite": bool(torch.isfinite(eval_logits).all().item()),
                "train_finite": bool(torch.isfinite(train_logits).all().item()),
                "shapes_match": list(eval_logits.shape) == expected_shape and list(train_logits.shape) == expected_shape,
                "model_spec": spec.to_dict(),
            }
        )
        del model
    return {"rows": rows, "all_pass": all(row["shapes_match"] and row["eval_finite"] and row["train_finite"] and row["eval_forks_none"] for row in rows)}


def _strict_reload_check(device: torch.device) -> dict[str, Any]:
    contract = {"num_classes": 48, "num_aspects": 12, "aspect_class_counts": (4,) * 12}
    model, spec = _build("b1_fork", contract)
    model.to(device)
    path = ARTIFACT_ROOT / "preflight" / "strict_reload.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "spec": spec.to_dict()}, path)

    reloaded_e2, _ = build_e2_student(
        ROOT / "models" / "goiturn-s-gd",
        num_classes=int(contract["num_classes"]),
        num_aspects=int(contract["num_aspects"]),
        aspect_class_counts=contract["aspect_class_counts"],
        attention_implementation=ATTENTION_IMPLEMENTATION,
    )
    state = torch.load(path, map_location=device, weights_only=False)
    result = reloaded_e2.load_state_dict(state["model"], strict=True)

    reloaded_e18, _ = _build("b0_baseline", contract)
    result_e18 = reloaded_e18.load_state_dict(state["model"], strict=True)

    return {
        "path": str(path),
        "into_plain_e2_student": {
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
            "strict_pass": not result.missing_keys and not result.unexpected_keys,
        },
        "into_fresh_e18_student": {
            "missing_keys": list(result_e18.missing_keys),
            "unexpected_keys": list(result_e18.unexpected_keys),
            "strict_pass": not result_e18.missing_keys and not result_e18.unexpected_keys,
        },
    }


def _optimizer_coverage_check(device: torch.device) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for task_id in TASKS:
        contract = _task_contract(task_id)
        model, _ = _build("b1_fork", contract)
        model.to(device)
        lr = TASK_LRS[task_id]
        optimizer = e7_optim.build_layerwise_optimizer(model, lr, gamma=GAMMA, num_layers=NUM_LAYERS, weight_decay=WEIGHT_DECAY)
        coverage = e7_optim.validate_optimizer_groups(
            model,
            optimizer,
            base_lr=lr,
            gamma=GAMMA,
            num_layers=NUM_LAYERS,
            weight_decay=WEIGHT_DECAY,
            expected_group_names=[
                f"{region}_{kind}"
                for region in ("embeddings", *[f"block_{i:02d}" for i in range(NUM_LAYERS)], "head")
                for kind in ("decay", "no_decay")
            ],
        )
        rows[task_id] = {
            "all_trainable": coverage["all_trainable"],
            "duplicate_parameter_count": coverage["duplicate_parameter_count"],
            "parameter_count": coverage["parameter_count"],
            "group_count": coverage["group_count"],
            "endpoint_scales": coverage["endpoint_scales"],
        }
        del model, optimizer
    return rows


def run_preflight(device_name: str | None = None) -> dict[str, Any]:
    started = time.time()
    snapshot = _gpu_snapshot()
    if device_name is None:
        eligible = _eligible_devices(snapshot)
        device_name = eligible[0] if eligible else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device_name}, but CUDA is unavailable in this process")

    manifest = _source_manifest()
    before_backbone = manifest["backbone"]["files"]
    before_frozen_src = manifest["e1_e17_source_py_hashes"]

    functional = {
        "parity": _parity_check(device),
        "b0_lambda_zero": _b0_lambda_zero_check(device),
        "gradient_checks": _gradient_checks(device),
        "split_acceptance": _split_acceptance_check(device),
        "shape_loader": _shape_loader_checks(device),
        "strict_reload": _strict_reload_check(device),
        "optimizer_coverage": _optimizer_coverage_check(device),
    }

    after_manifest = _source_manifest()
    after_backbone = after_manifest["backbone"]["files"]
    after_frozen_src = after_manifest["e1_e17_source_py_hashes"]

    checks = {
        "parity_cls_fp32": functional["parity"]["checks"]["cls_fp32_le_1e4"],
        "parity_cls_bf16": functional["parity"]["checks"]["cls_bf16_le_0_1"],
        "parity_fork_row0_fp32": functional["parity"]["checks"]["row0_fp32_le_1e4"],
        "parity_alternation_pattern": functional["parity"]["alternation_pattern_matches_clean_noisy"],
        "parity_eval_never_forks": functional["parity"]["eval_never_computes_forks"],
        "b0_logits_delta": functional["b0_lambda_zero"]["checks"]["logits_delta_le_1e4"],
        "b0_total_loss_delta": functional["b0_lambda_zero"]["checks"]["total_loss_delta_le_1e4"],
        "b0_never_stashes_forks": functional["b0_lambda_zero"]["checks"]["b0_never_stashes_forks"],
        "b1_encoder_grad_finite_nonzero": functional["gradient_checks"]["checks"]["b1_encoder_grad_finite_nonzero"],
        "b1_head_grad_nonzero": functional["gradient_checks"]["checks"]["b1_head_grad_nonzero"],
        "b3_encoder_grad_zero": functional["gradient_checks"]["checks"]["b3_encoder_grad_zero"],
        "b3_head_grad_nonzero": functional["gradient_checks"]["checks"]["b3_head_grad_nonzero"],
        "shape_loader_all_pass": functional["shape_loader"]["all_pass"],
        "strict_reload_into_plain_e2": functional["strict_reload"]["into_plain_e2_student"]["strict_pass"],
        "strict_reload_into_fresh_e18": functional["strict_reload"]["into_fresh_e18_student"]["strict_pass"],
        "optimizer_coverage_all_tasks": all(
            row["all_trainable"] and row["duplicate_parameter_count"] == 0
            for row in functional["optimizer_coverage"].values()
        ),
        "backbone_no_mutation": before_backbone == after_backbone,
        "e1_e17_source_no_mutation": before_frozen_src == after_frozen_src,
    }
    gpu_ready = device.type == "cuda" and snapshot.get("available", False)
    payload = {
        "protocol": PROTOCOL,
        "revision": REVISION,
        "status": "passed" if all(checks.values()) and gpu_ready else ("blocked_no_cuda" if not gpu_ready else "failed"),
        "created_at": _now(),
        "elapsed_seconds": time.time() - started,
        "device": str(device),
        "gpu_snapshot": snapshot,
        "eligible_devices": _eligible_devices(snapshot),
        "gpu_preflight_ready": gpu_ready,
        "checks": checks,
        "functional": functional,
        "source_manifest": manifest,
        "backbone_hash_unchanged": before_backbone == after_backbone,
        "e1_e17_source_hash_unchanged": before_frozen_src == after_frozen_src,
        "full_training_dispatch": "not launched; root review required",
    }
    _write_json(ARTIFACT_ROOT / "preflight.json", payload)
    _write_json(ARTIFACT_ROOT / "source_manifest.json", manifest)
    return payload


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------
def _profile_one(
    arm: str,
    device: torch.device,
    task_id: str,
    shape_name: str,
    clean: dict[str, Tensor],
    noisy: dict[str, Tensor],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    model, spec = _build(arm, contract)
    model.to(device).train()
    optimizer = e7_optim.build_layerwise_optimizer(
        model, TASK_LRS[task_id], gamma=GAMMA, num_layers=NUM_LAYERS, weight_decay=WEIGHT_DECAY
    )
    clean = {key: value.to(device) for key, value in clean.items()}
    noisy = {key: value.to(device) for key, value in noisy.items()}
    counts = contract["aspect_class_counts"]
    num_aspects = contract["num_aspects"]
    nc_fn = e2_train._load_loss_api()[1]
    telemetry = ForkTelemetry()
    patched_loss = make_patched_compute_loss(
        model, original_compute_loss=e1_train.compute_loss, lambda_fork=LAMBDA_FORK, telemetry=telemetry
    )

    def one_step() -> tuple[float, float]:
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        last_total = None
        for _ in range(ACCUMULATION_STEPS):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                clean_logits = model(clean["input_ids"], clean["attention_mask"])
                ce = patched_loss(clean_logits, clean["labels"], num_aspects=num_aspects, aspect_class_counts=counts)
                noisy_logits = model(noisy["input_ids"], noisy["attention_mask"])
                consistency = nc_fn(clean_logits, noisy_logits, temperature=1.0, slot_class_counts=counts or None)
                total = ce + NC_LAMBDAS[task_id] * consistency
            if not torch.isfinite(total):
                raise FloatingPointError(f"E18 profile loss non-finite: {arm}/{shape_name}")
            (total / ACCUMULATION_STEPS).backward()
            last_total = total
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize(device)
        assert last_total is not None
        return time.perf_counter() - started, float(last_total.detach().cpu().item())

    for _ in range(2):
        one_step()
    torch.cuda.reset_peak_memory_stats(device)
    telemetry.reset()
    measured_steps: list[float] = []
    loss_value = 0.0
    for _ in range(3):
        step_seconds, loss_value = one_step()
        measured_steps.append(step_seconds)
    step_seconds = float(torch.tensor(measured_steps).median().item())
    result = {
        "arm": arm,
        "task_id": task_id,
        "shape_name": shape_name,
        "input_shape": list(clean["input_ids"].shape),
        "step_seconds": step_seconds,
        "measured_step_seconds": measured_steps,
        "loss": loss_value,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "fork_telemetry": telemetry.summary(lambda_fork=LAMBDA_FORK),
        "model_spec": spec.to_dict(),
        "finite": bool(math.isfinite(loss_value)),
    }
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def run_profile(device_name: str | None = None, task_id: str = "absa_restaurant") -> dict[str, Any]:
    snapshot = _gpu_snapshot()
    manifest = _source_manifest()
    if device_name is None:
        eligible = _eligible_devices(snapshot)
        device_name = eligible[0] if eligible else "cuda:0"
    if not torch.cuda.is_available():
        payload = {
            "protocol": PROTOCOL,
            "status": "blocked_no_cuda",
            "gpu_snapshot": snapshot,
            "requested_device": device_name,
            "source_manifest": manifest,
            "reason": "CUDA unavailable in this process; no profile launched",
        }
        _write_json(ARTIFACT_ROOT / "profile.json", payload)
        return payload
    device = torch.device(device_name)
    tokenizer, actual_batch, contract, noisy_payload = _fixed_profile_batch(task_id)
    clean_actual = {key: value for key, value in actual_batch.items() if isinstance(value, torch.Tensor)}
    noisy_actual = noisy_payload["noisy"]
    synthetic_ids = torch.randint(0, 20480, (MICRO_BATCH_SIZE, MAX_LENGTH), dtype=torch.long)
    synthetic_mask = torch.ones_like(synthetic_ids)
    num_aspects = contract["num_aspects"]
    counts = contract["aspect_class_counts"]
    if num_aspects:
        synthetic_labels = torch.stack(
            [torch.arange(MICRO_BATCH_SIZE, dtype=torch.long) % counts[a] for a in range(num_aspects)], dim=1
        )
    else:
        synthetic_labels = torch.arange(MICRO_BATCH_SIZE, dtype=torch.long) % int(contract["num_classes"])
    synthetic = {"input_ids": synthetic_ids, "attention_mask": synthetic_mask, "labels": synthetic_labels}

    profile_rows = []
    for shape_name, clean, noisy in (
        ("fixed_train_batch", clean_actual, noisy_actual),
        ("synthetic_16x256", synthetic, {"input_ids": synthetic_ids.clone(), "attention_mask": synthetic_mask.clone(), "labels": synthetic_labels.clone()}),
    ):
        for arm in BATCH1_ARMS:
            profile_rows.append(_profile_one(arm, device, task_id, shape_name, clean, noisy, contract))
    by_shape = {shape: {row["arm"]: row for row in profile_rows if row["shape_name"] == shape} for shape in ("fixed_train_batch", "synthetic_16x256")}
    gates = {}
    for shape, rows in by_shape.items():
        baseline = rows["b0_baseline"]["step_seconds"]
        b1 = rows["b1_fork"]["step_seconds"]
        b2 = rows["b2_random_mask"]["step_seconds"]
        free_mib = snapshot.get("rows", {}).get(device_name.split(":")[-1], {}).get("free_mib")
        gates[shape] = {
            "b1_to_b0_step_ratio": b1 / max(baseline, 1e-12),
            "b2_to_b0_step_ratio": b2 / max(baseline, 1e-12),
            "under_gate_ratio": b1 <= PROFILE_STEP_RATIO_GATE * baseline and b2 <= PROFILE_STEP_RATIO_GATE * baseline,
            "reserve_under_free_minus4GiB": free_mib is None
            or max(row["peak_reserved_bytes"] for row in rows.values()) <= (free_mib - RESERVE_HEADROOM_MIB) * 1024 * 1024,
            "free_mib_before": free_mib,
        }
    payload = {
        "protocol": PROTOCOL,
        "revision": REVISION,
        "status": "passed" if all(g["under_gate_ratio"] and g["reserve_under_free_minus4GiB"] for g in gates.values()) else "failed",
        "device": device_name,
        "gpu_snapshot": snapshot,
        "task_id": task_id,
        "rows": profile_rows,
        "budget_gates": gates,
        "source_manifest": manifest,
        "note": (
            "Profile only; full 6-trial (batch 1: b0/b1/b2 x 2 tasks) dispatch remains "
            "root-review gated behind --run-all --dispatch-approved."
        ),
    }
    _write_json(ARTIFACT_ROOT / "profile.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def run_worker(arm: str, task_id: str, device_name: str, output_dir: Path) -> dict[str, Any]:
    if arm not in ARMS_ORDER or task_id not in TASKS:
        raise ValueError(f"E18 arm/task not locked: {arm}, {task_id}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing duplicate E18 output: {output_dir}")

    source_manifest = _source_manifest()
    source_binding = _stable_source_binding(source_manifest)

    telemetry = ForkTelemetry()
    epoch_telemetry_rows: list[dict[str, Any]] = []
    holder: dict[str, Any] = {}

    original_builder = e2_train.build_e2_student
    original_optimizer = e2_train.make_optimizer
    original_compute_loss = e2_train.compute_loss
    original_run_validation = e2_train.run_validation

    def e18_builder(
        backbone_dir: Path,
        *,
        num_classes: int,
        num_aspects: int = 0,
        aspect_class_counts: Any = None,
        attention_implementation: str = "sdpa",
    ):
        model, spec = build_e18_student(
            backbone_dir,
            num_classes=num_classes,
            num_aspects=num_aspects,
            aspect_class_counts=aspect_class_counts,
            arm=arm,
            attention_implementation=attention_implementation,
            s_max=S_MAX,
            lambda_fork=LAMBDA_FORK,
            seed=SEED,
        )
        holder["model"] = model
        return model, spec

    def e18_optimizer(model: torch.nn.Module, learning_rate: float):
        optimizer = e7_optim.build_layerwise_optimizer(
            model, learning_rate, gamma=GAMMA, num_layers=NUM_LAYERS, weight_decay=WEIGHT_DECAY
        )
        coverage = e7_optim.validate_optimizer_groups(
            model, optimizer, base_lr=learning_rate, gamma=GAMMA, num_layers=NUM_LAYERS, weight_decay=WEIGHT_DECAY
        )
        holder["optimizer_coverage"] = coverage
        return optimizer

    def current_epoch() -> int:
        # 1-based training epoch; run_validation fires once at the end of
        # each epoch, so "epochs already validated" is exactly "epochs
        # already trained", and the batches being trained right now belong
        # to the next (not-yet-validated) epoch.
        return len(epoch_telemetry_rows) + 1

    def e18_compute_loss(logits: Tensor, labels: Tensor, *, num_aspects: int, aspect_class_counts: Any = None) -> Tensor:
        model = holder["model"]
        forks = model.last_forks
        model.last_forks = None
        ce = original_compute_loss(logits, labels, num_aspects=num_aspects, aspect_class_counts=aspect_class_counts)
        fork_ce, stats = compute_fork_ce(forks, labels, num_aspects=num_aspects, aspect_class_counts=aspect_class_counts)
        lambda_eff = lambda_fork_ramp(LAMBDA_FORK, current_epoch(), LAMBDA_FORK_RAMP_EPOCHS)
        telemetry.record(
            considered=stats["considered"],
            accepted=stats["accepted"],
            connector_hits=stats["connector_hits"],
            fork_ce_value=None if fork_ce is None else float(fork_ce.detach().cpu().item()),
        )
        if fork_ce is None:
            return ce
        return ce + lambda_eff * fork_ce

    def e18_run_validation(model: Any, loader: Any, device: torch.device, precision: str, validation_task_id: str, bundle: Any, *, split_name: str = "validation"):
        metrics, predictions = original_run_validation(model, loader, device, precision, validation_task_id, bundle, split_name=split_name)
        # The epoch that was just trained (and is now being validated) is
        # current_epoch() as it stood during that epoch's compute_loss
        # calls, i.e. before this row is appended below.
        epoch_just_trained = current_epoch()
        lambda_eff = lambda_fork_ramp(LAMBDA_FORK, epoch_just_trained, LAMBDA_FORK_RAMP_EPOCHS)
        summary = telemetry.summary(lambda_fork=lambda_eff)
        summary.update(
            {
                "epoch": epoch_just_trained,
                "task_id": validation_task_id,
                "arm": arm,
                "split_name": split_name,
                "lambda_fork_base": LAMBDA_FORK,
                "lambda_fork_ramp_epochs": LAMBDA_FORK_RAMP_EPOCHS,
                "lambda_fork_effective": lambda_eff,
            }
        )
        epoch_telemetry_rows.append(summary)
        telemetry.reset()
        _write_json(
            output_dir / "epoch_telemetry.json",
            {
                "protocol": PROTOCOL,
                "arm": arm,
                "task_id": task_id,
                "s_max": S_MAX,
                "lambda_fork": LAMBDA_FORK,
                "lambda_fork_ramp_epochs": LAMBDA_FORK_RAMP_EPOCHS,
                "rows": epoch_telemetry_rows,
            },
        )
        return metrics, predictions

    e2_train.build_e2_student = e18_builder  # type: ignore[assignment]
    e2_train.make_optimizer = e18_optimizer  # type: ignore[assignment]
    e2_train.compute_loss = e18_compute_loss  # type: ignore[assignment]
    e2_train.run_validation = e18_run_validation  # type: ignore[assignment]
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
            precision=PRECISION,
            attention_implementation=ATTENTION_IMPLEMENTATION,
            epochs=EPOCHS,
            effective_batch_size=EFFECTIVE_BATCH_SIZE,
            micro_batch_size=MICRO_BATCH_SIZE,
            stage="stageE18_segment_fork_consistency",
            method=arm,
            coefficient=LAMBDA_FORK,
            revision=REVISION,
        )
    finally:
        e2_train.build_e2_student = original_builder  # type: ignore[assignment]
        e2_train.make_optimizer = original_optimizer  # type: ignore[assignment]
        e2_train.compute_loss = original_compute_loss  # type: ignore[assignment]
        e2_train.run_validation = original_run_validation  # type: ignore[assignment]

    artifact = Path(result["artifact_dir"])
    metadata = _read_json(artifact / "metadata.json")
    telemetry_path = artifact / "epoch_telemetry.json"
    _write_json(
        telemetry_path,
        {"protocol": PROTOCOL, "arm": arm, "task_id": task_id, "s_max": S_MAX, "lambda_fork": LAMBDA_FORK, "rows": epoch_telemetry_rows},
    )
    metadata.update(
        {
            "e18_protocol": PROTOCOL,
            "e18_revision": REVISION,
            "e18_arm": arm,
            "e18_s_max": S_MAX,
            "e18_lambda_fork": LAMBDA_FORK,
            "e18_optimizer_coverage": holder.get("optimizer_coverage"),
            "e18_source_binding": source_binding,
            "e18_epoch_telemetry_path": str(telemetry_path),
        }
    )
    _write_json(artifact / "metadata.json", metadata)
    checkpoint = artifact / "best_checkpoint.pt"
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state.setdefault("metadata", {}).update(
            {
                "e18_protocol": PROTOCOL,
                "e18_arm": arm,
                "e18_s_max": S_MAX,
                "e18_lambda_fork": LAMBDA_FORK,
                "e18_source_binding": source_binding,
                "e18_epoch_telemetry_path": str(telemetry_path),
            }
        )
        torch.save(state, checkpoint)
    return result


# ---------------------------------------------------------------------------
# run-all controller
# ---------------------------------------------------------------------------
def run_all(
    device_names: list[str],
    *,
    dispatch_approved: bool,
    arms: Sequence[str] = BATCH1_ARMS,
    tasks: Sequence[str] = TASKS,
) -> dict[str, Any]:
    if not dispatch_approved:
        raise RuntimeError("Full E18 dispatch requires explicit --dispatch-approved after root review")
    if not device_names:
        raise RuntimeError("No GPU devices available for E18 dispatch")
    for arm in arms:
        if arm not in ARMS_ORDER:
            raise ValueError(f"Unknown E18 arm requested for dispatch: {arm}")
    for task in tasks:
        if task not in TASKS:
            raise ValueError(f"Unknown E18 task requested for dispatch: {task}")

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    state_path = ARTIFACT_ROOT / "controller_state.json"
    source_manifest = _source_manifest()
    source_binding = _stable_source_binding(source_manifest)
    source_manifest_hash = hashlib.sha256(json.dumps(source_manifest, sort_keys=True).encode("utf-8")).hexdigest()

    previous: dict[str, Any] = {}
    if state_path.is_file():
        previous = _read_json(state_path)
        if previous.get("source_binding") is not None and previous.get("source_binding") != source_binding:
            raise RuntimeError("Existing E18 controller source binding differs; refusing resume")
        if previous.get("status") == "running":
            active = [job for job in previous.get("jobs", []) if _pid_alive(job.get("pid")) and job.get("status") == "running"]
            if active:
                raise RuntimeError(f"E18 controller already has active jobs; adopt/refuse duplicate: {active}")

    _write_json(
        ARTIFACT_ROOT / "source_manifest_dispatch.json",
        {"manifest": source_manifest, "manifest_sha256": source_manifest_hash, "source_binding": source_binding, "dispatch_approved": True, "approved_at": _now()},
    )

    queue = [(task, arm) for task in tasks for arm in arms]
    completed_keys = {
        (job.get("task_id"), job.get("arm")) for job in previous.get("jobs", []) if job.get("status") == "completed"
    }
    if previous and previous.get("source_binding") == source_binding:
        for task, arm in queue:
            artifact = ARTIFACT_ROOT / arm / task / "seed_42"
            status_path = artifact / "status.json"
            if status_path.is_file():
                status = _read_json(status_path)
                if status.get("status") == "completed":
                    completed_keys.add((task, arm))
    queue = [(task, arm) for task, arm in queue if (task, arm) not in completed_keys]

    controller = {
        "protocol": PROTOCOL,
        "revision": REVISION,
        "status": "running",
        "created_at": _now(),
        "source_manifest_sha256": source_manifest_hash,
        "source_binding": source_binding,
        "dispatch_approved": True,
        "arms": list(arms),
        "tasks": list(tasks),
        "jobs": [job for job in previous.get("jobs", []) if (job.get("task_id"), job.get("arm")) in completed_keys],
        "queue_remaining": [{"task_id": task, "arm": arm} for task, arm in queue],
    }
    _write_json(state_path, controller)

    active: list[tuple[subprocess.Popen[str], dict[str, Any], Any]] = []
    completed: list[dict[str, Any]] = list(controller["jobs"])

    def wait_job(process: subprocess.Popen[str], job: dict[str, Any], log_handle: Any) -> int:
        while True:
            code = process.poll()
            if code is not None:
                log_handle.close()
                return int(code)
            controller["heartbeat_at"] = _now()
            controller["active_pids"] = [item[1].get("pid") for item in active]
            _write_json(state_path, controller)
            time.sleep(10)

    for index, (task, arm) in enumerate(queue):
        device = device_names[index % len(device_names)]
        snapshot_now = _gpu_snapshot()
        vram_gate = _dispatch_vram_gate(device, snapshot_now)
        output = ARTIFACT_ROOT / arm / task / "seed_42"
        log_path = ARTIFACT_ROOT / "logs" / f"{arm}__{task}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("w", encoding="utf-8")
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--arm", arm, "--task-id", task, "--device", device, "--output-dir", str(output)]
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True, env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONUNBUFFERED": "1"})
        job = {"arm": arm, "task_id": task, "device": device, "pid": process.pid, "output_dir": str(output), "log": str(log_path), "command": command, "status": "running", "started_at": _now(), "vram_gate": vram_gate}
        controller["jobs"].append(job)
        controller["queue_remaining"] = [{"task_id": t, "arm": a} for t, a in queue[index + 1 :]]
        _write_json(state_path, controller)
        active.append((process, job, log))
        if len(active) >= len(device_names):
            process0, meta0, log0 = active[0]
            code = wait_job(process0, meta0, log0)
            active.pop(0)
            meta0["returncode"] = code
            meta0["status"] = "completed" if code == 0 else "failed"
            meta0["finished_at"] = _now()
            completed.append(meta0)
            _write_json(state_path, controller)
            if code != 0:
                raise RuntimeError(f"E18 worker failed: {meta0}")
    for process, meta, log in active:
        code = wait_job(process, meta, log)
        meta["returncode"] = code
        meta["status"] = "completed" if code == 0 else "failed"
        meta["finished_at"] = _now()
        completed.append(meta)
        _write_json(state_path, controller)
        if code != 0:
            raise RuntimeError(f"E18 worker failed: {meta}")
    controller.update({"status": "completed", "finished_at": _now(), "jobs": completed, "queue_remaining": []})
    _write_json(state_path, controller)
    return controller


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E18 preflight/profile/worker/run-all runner")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--dispatch-approved", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--arm", choices=ARMS_ORDER)
    parser.add_argument("--task-id", choices=TASKS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--arms", nargs="+", choices=ARMS_ORDER, default=list(BATCH1_ARMS))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not any((args.preflight, args.profile, args.run_all, args.worker)):
        raise SystemExit("Choose --preflight, --profile, --run-all, or --worker")
    if args.worker:
        if not args.arm or not args.task_id or not args.device or args.output_dir is None:
            raise SystemExit("--worker requires --arm, --task-id, --device, and --output-dir")
        result = run_worker(args.arm, args.task_id, args.device, args.output_dir)
    elif args.run_all:
        snapshot = _gpu_snapshot()
        devices = _eligible_devices(snapshot)
        # Workers use <3 GiB each; allow up to --workers-per-gpu concurrent workers on each eligible GPU.
        devices = [d for d in devices for _ in range(max(1, int(args.workers_per_gpu)))]
        result = run_all(devices, dispatch_approved=args.dispatch_approved, arms=args.arms, tasks=args.tasks)
    elif args.profile:
        result = run_profile(args.device)
    else:
        result = run_preflight(args.device)
    print(json.dumps(result, default=_json_default, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
