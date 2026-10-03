#!/usr/bin/env python3
"""Numerical, mask, gradient and short runtime checks before the GPU grid."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from scripts.cafe_cross_head import common  # noqa: E402
from scripts.sealion_benchmark import common as base  # noqa: E402


def hash_head(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, param in model.head.state_dict().items():
        digest.update(name.encode())
        digest.update(param.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def tensor_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.detach().float().cpu() - b.detach().float().cpu()).abs().max().item()


def build_inputs(tokenizer, device):
    texts = [
        "Đồ ăn ngon nhưng phục vụ khá chậm.",
        "Quán có món ăn hợp khẩu vị, nhân viên nhiệt tình và không gian sạch sẽ.",
        "Giá hợp lý.",
    ]
    batch = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    return {key: value.to(device) for key, value in batch.items() if key in ("input_ids", "attention_mask")}


def check_arm(arm: str, contract: dict, inputs: dict, reference_logits: torch.Tensor, device: torch.device, baseline_head_hash: str) -> tuple[dict, torch.nn.Module]:
    common.e1_train.seed_everything(42)
    model, spec = common.build_model(arm, contract)
    head_hash = hash_head(model)
    if head_hash != baseline_head_hash:
        raise AssertionError(f"{arm}: head initialization differs from MHA")
    model.to(device).eval()
    with torch.inference_mode():
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        modified_ids = inputs["input_ids"].clone()
        padded = inputs["attention_mask"] == 0
        if not padded.any():
            raise AssertionError("preflight inputs have no padding")
        modified_ids[padded] = 100
        changed_pad_logits = model(modified_ids, inputs["attention_mask"])
    init_delta = tensor_diff(logits, reference_logits)
    pad_delta = tensor_diff(logits, changed_pad_logits)
    if init_delta > 1e-3 or pad_delta > 1e-4:
        raise AssertionError(f"{arm}: init_delta={init_delta:g}, pad_delta={pad_delta:g}")
    grads = {}
    if arm != "mha":
        model.train()
        scores = model(inputs["input_ids"], inputs["attention_mask"])
        loss = scores.float().square().mean()
        loss.backward()
        for name, param in model.named_parameters():
            if name.endswith(("logit_mix", "weight_mix")):
                if param.grad is None or not torch.isfinite(param.grad).all():
                    raise AssertionError(f"{arm}: invalid gradient for {name}")
                grads[name] = float(param.grad.norm().item())
        if len(grads) != 8 or any(value <= 0 for value in grads.values()):
            raise AssertionError(f"{arm}: zero or missing gradient: {grads}")
        model.zero_grad(set_to_none=True)
    return {"model_spec": spec, "head_hash": head_hash, "identity_logit_max_delta_fp32": init_delta, "pad_token_mutation_max_delta_fp32": pad_delta, "mix_gradient_norms": grads}, model


def profile(model: torch.nn.Module, arm: str, contract: dict, tokenizer, device: torch.device, steps: int) -> dict:
    model.train()
    loader, valid_loader = base.make_loaders(tokenizer, contract, micro_batch_size=32)
    iterator = iter(loader)
    optimizer = common.e1_train.make_optimizer(model, 2e-5)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        moved = common.e1_train._move_batch(batch, device)
        with common.e1_train.make_autocast("bf16", device):
            logits = model(moved["input_ids"], moved["attention_mask"])
            loss = common.e1_train.compute_loss(logits, moved["labels"], num_aspects=contract["num_aspects"], aspect_class_counts=contract["aspect_class_counts"])
        if not torch.isfinite(loss.detach()):
            raise FloatingPointError(f"non-finite profile loss {arm} at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    train_elapsed = time.perf_counter() - started
    started = time.perf_counter()
    with torch.inference_mode():
        common.e1_train.run_validation(model, valid_loader, device, "bf16", "absa_restaurant", contract["bundle"])
    torch.cuda.synchronize(device)
    valid_elapsed = time.perf_counter() - started
    result = {"steps": steps, "train_seconds": train_elapsed, "seconds_per_step": train_elapsed / steps, "validation_seconds": valid_elapsed, "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3, "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3}
    del optimizer
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--profile-steps", type=int, default=100)
    args = ap.parse_args()
    if args.profile_steps < 1:
        raise ValueError("profile-steps must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for pilot preflight")
    contract = base.task_contract("absa_restaurant")
    tokenizer, tokenizer_manifest = base.load_cafe_tokenizer()
    inputs = build_inputs(tokenizer, device)
    common.e1_train.seed_everything(42)
    original, _ = base.build_model("cafebert", contract)
    original.to(device).eval()
    with torch.inference_mode():
        original_logits = original(inputs["input_ids"], inputs["attention_mask"]).detach().float().cpu()
    del original
    gc.collect()
    torch.cuda.empty_cache()

    result = {"protocol": common.PROTOCOL, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "tokenizer": tokenizer_manifest, "arms": {}}
    mha_logits = None
    baseline_head_hash = None
    for arm in ("mha", "diagonal", "full"):
        common.e1_train.seed_everything(42)
        model, spec = common.build_model(arm, contract)
        head_hash = hash_head(model)
        model.to(device).eval()
        with torch.inference_mode():
            logits = model(inputs["input_ids"], inputs["attention_mask"]).detach().float().cpu()
        if arm == "mha":
            baseline_head_hash = head_hash
            mha_logits = logits
            reference_delta = tensor_diff(logits, original_logits)
            if reference_delta > 1e-3:
                raise AssertionError(f"matched MHA vs official Cafe SDPA differs by {reference_delta:g}")
            result["official_sdpa_vs_matched_mha_max_delta_fp32"] = reference_delta
        del model
        gc.collect()
        torch.cuda.empty_cache()
        arm_checks, model = check_arm(arm, contract, inputs, mha_logits, device, baseline_head_hash)
        arm_checks["profile"] = profile(model, arm, contract, tokenizer, device, args.profile_steps)
        result["arms"][arm] = arm_checks
        del model
        gc.collect()
        torch.cuda.empty_cache()
    result["status"] = "passed"
    common.write_json(common.ARTIFACT_ROOT / "preflight.json", result)
    print(json.dumps({"status": result["status"], "official_delta": result["official_sdpa_vs_matched_mha_max_delta_fp32"], "arms": {arm: {"delta": checks["identity_logit_max_delta_fp32"], "pad_delta": checks["pad_token_mutation_max_delta_fp32"], "seconds_per_step": checks["profile"]["seconds_per_step"], "peak_gib": checks["profile"]["peak_allocated_gib"]} for arm, checks in result["arms"].items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
