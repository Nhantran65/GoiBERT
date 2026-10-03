"""Frozen contract for CafeBERT cross-head pilot on three screening tasks."""

from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.cafe_cross_head.modeling import ARMS, retrofit
from scripts.sealion_benchmark import common as base

ROOT = base.ROOT
ARTIFACT_ROOT = ROOT / "artifacts" / "cafe_cross_head"
PROTOCOL = "cafe-cross-head-v1"
SEED = 42
TASKS = ("absa_restaurant", "visfd", "vsfc_topic")
LEARNING_RATES = (2e-5,)
EPOCHS = 30
EFFECTIVE_BATCH_SIZE = 32
MAX_LENGTH = 256
PRECISION = "bf16"
ATTENTION_IMPLEMENTATION = "matched-eager-top4-sdpa-bottom20"
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1
GRADIENT_CLIP_NORM = 1.0
HEAD_HIDDEN_SIZE = 1024

e1_train = base.e1_train
write_json = base.write_json
read_json = base.read_json
now = base.now
runtime = base.runtime
gpu_snapshot = base.gpu_snapshot
load_model_checkpoint = base.load_model_checkpoint
sha256_file = base.sha256_file
task_contract = base.task_contract
make_loaders = base.make_loaders
load_cafe_tokenizer = base.load_cafe_tokenizer


def source_hash() -> str:
    files = [*(ROOT / "scripts" / "cafe_cross_head" / name for name in ("modeling.py", "common.py", "train.py")), ROOT / "src/goibert/train.py", ROOT / "src/goibert/data.py", ROOT / "src/goibert/metrics.py", ROOT / "src/goibert_e14/modeling.py"]
    digest = hashlib.sha256()
    for file in files:
        digest.update(str(file.relative_to(ROOT)).encode())
        digest.update(base.sha256_file(file).encode())
    return digest.hexdigest()


def build_model(arm: str, contract: dict):
    if arm not in ARMS:
        raise ValueError(arm)
    model, spec = base.build_model("cafebert", contract)
    spec = dict(spec)
    spec.update(retrofit(model.encoder, arm))
    spec["total_parameters"] = sum(p.numel() for p in model.parameters())
    return model, spec


def run_dir(arm: str, task: str, learning_rate: float) -> Path:
    return ARTIFACT_ROOT / arm / task / f"lr_{learning_rate:.0e}" / f"seed_{SEED}"
