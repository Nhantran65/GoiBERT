#!/usr/bin/env python3
"""Bounded convergence sensitivity for the D1 linear probes.

This is a separate diagnostic from the locked 200-step readout.  It reruns
only the final state and the best non-embedding state selected by the fixed
200-step probe, using 1,000 full-batch AdamW updates with the same features,
normalization, learning rate, weight decay, and zero head initialization.
It never loads an encoder or opens the test split.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

try:
    from .d1_linear_probes import (
        MODELS,
        TASKS,
        _normalize_features,
        _raw_label_to_model,
        _task_widths,
        evaluate,
        load_dataset_contract,
        load_features,
        prediction_counts,
        prediction_hash,
        read_json,
        sha256_file,
        write_json,
    )
except ImportError:
    from d1_linear_probes import (
        MODELS,
        TASKS,
        _normalize_features,
        _raw_label_to_model,
        _task_widths,
        evaluate,
        load_dataset_contract,
        load_features,
        prediction_counts,
        prediction_hash,
        read_json,
        sha256_file,
        write_json,
    )


STEPS_200 = 200
STEPS_1000 = 1000
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.01
SEED = 42


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _fit_vectorized(
    train: np.ndarray,
    valid: np.ndarray,
    train_labels: Tensor,
    valid_labels: Tensor,
    widths: Sequence[int],
    *,
    device: torch.device,
    steps: int,
) -> tuple[dict[str, Any], list[Any]]:
    """Fit equivalent per-slot linear heads as one flat linear operation.

    Concatenating the slot heads keeps the same parameterization and zero
    initialization while avoiding thousands of tiny optimizer calls for Hotel.
    The loss remains the unweighted mean of per-slot cross-entropies.
    """
    set_seed(SEED)
    train_x, valid_x, norm_stats = _normalize_features(train, valid, device=device)
    head = nn.Linear(train_x.shape[1], int(sum(widths))).to(device)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_target = train_labels.to(device)
    valid_target = valid_labels.to(device)
    losses: list[float] = []
    started = time.time()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = head(train_x)
        offset = 0
        losses_for_slots: list[Tensor] = []
        for slot, width in enumerate(widths):
            target = train_target if len(widths) == 1 and train_target.ndim == 1 else train_target[:, slot]
            losses_for_slots.append(nn.functional.cross_entropy(logits[:, offset : offset + width], target))
            offset += width
        loss = torch.stack(losses_for_slots).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite sensitivity probe loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    with torch.inference_mode():
        train_logits = head(train_x)
        valid_logits = head(valid_x)
        offset = 0
        train_losses: list[Tensor] = []
        valid_losses: list[Tensor] = []
        predictions_slots: list[Tensor] = []
        for slot, width in enumerate(widths):
            train_target_slot = train_target if len(widths) == 1 and train_target.ndim == 1 else train_target[:, slot]
            valid_target_slot = valid_target if len(widths) == 1 and valid_target.ndim == 1 else valid_target[:, slot]
            train_slot = train_logits[:, offset : offset + width]
            valid_slot = valid_logits[:, offset : offset + width]
            train_losses.append(nn.functional.cross_entropy(train_slot, train_target_slot))
            valid_losses.append(nn.functional.cross_entropy(valid_slot, valid_target_slot))
            predictions_slots.append(valid_slot.argmax(dim=-1))
            offset += width
        train_loss = torch.stack(train_losses).mean()
        valid_loss = torch.stack(valid_losses).mean()
        if len(widths) == 1:
            predictions = predictions_slots[0].detach().cpu().tolist()
        else:
            predictions = torch.stack(predictions_slots, dim=1).detach().cpu().tolist()
    tail_start_index = max(0, steps - 20)
    tail_start_step = max(1, steps - 19)
    checkpoint_step = max(1, steps - 20)
    checkpoint_loss = losses[checkpoint_step - 1]
    tail_start = losses[tail_start_index]
    tail_end = losses[-1]
    relative_improvement = (tail_start - tail_end) / max(abs(tail_start), 1e-12)
    convergence = {
        "steps": steps,
        "initial_train_ce": losses[0],
        f"loss_at_step_{checkpoint_step}": checkpoint_loss,
        "final_step_train_ce": tail_end,
        "post_update_train_ce": float(train_loss.detach().cpu()),
        "last20_mean_train_ce": float(np.mean(losses[-20:])),
        "last20_step_improvement": float(tail_start - tail_end),
        "last20_relative_improvement": float(relative_improvement),
        "last20_start_step": tail_start_step,
        "last20_end_step": steps,
        "learning_failed_flag": bool(losses[0] - np.mean(losses[-20:]) < 1e-3),
        "underconverged_flag": bool(relative_improvement > 0.01),
    }
    return {
        "train_ce": float(train_loss.detach().cpu()),
        "validation_ce": float(valid_loss.detach().cpu()),
        "convergence": convergence,
        "normalization": norm_stats,
        "elapsed_seconds": time.time() - started,
        "probe_parameter_count": int(sum(head_parameter.numel() for head_parameter in head.parameters())),
    }, predictions


def run_task(task: str, cache_root: Path, source_root: Path, output_root: Path, device: torch.device) -> dict[str, Any]:
    task_cache = cache_root / task
    task_200 = source_root / task / "probe_results.json"
    if not task_200.is_file():
        raise FileNotFoundError(f"Missing completed 200-step result: {task_200}")
    previous = read_json(task_200)
    dataset = load_dataset_contract(task_cache, task=task)
    raw_train = np.asarray(dataset["train"]["golds"], dtype=np.int64)
    raw_valid = np.asarray(dataset["validation"]["golds"], dtype=np.int64)
    train_labels_np = _raw_label_to_model(task, raw_train)
    valid_labels_np = _raw_label_to_model(task, raw_valid)
    train_labels = torch.from_numpy(train_labels_np)
    valid_labels = torch.from_numpy(valid_labels_np)
    widths = _task_widths(task, train_labels_np)
    output_task = output_root / task
    output_task.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "protocol": "D1-layerwise-linear-probe-convergence-sensitivity-v1",
        "task_id": task,
        "source_200_step": str(task_200.resolve()),
        "source_200_step_sha256": sha256_file(task_200),
        "probe": {
            "steps_200": STEPS_200,
            "steps_1000": STEPS_1000,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "same_features_normalization_and_zero_init": True,
            "selection": "final state and best nonzero state from fixed 200-step results; no 1000-step validation reselection",
        },
        "models": {},
    }
    for model in MODELS:
        model_dir = task_cache / model
        train_cache, valid_cache, cache_hashes = load_features(
            model_dir, task=task, model=model, view="cls", dataset=dataset
        )
        previous_rows = previous["models"][model]["layers"]
        final_layer = int(previous_rows[-1]["layer_index"])
        best_nonzero_row = max(
            (row for row in previous_rows if int(row["layer_index"]) > 0),
            key=lambda row: (float(row["selection_f1"]), -int(row["layer_index"])),
        )
        best_nonzero_layer = int(best_nonzero_row["layer_index"])
        selected_layers = sorted({final_layer, best_nonzero_layer})
        model_rows: list[dict[str, Any]] = []
        for layer in selected_layers:
            row_200 = next(row for row in previous_rows if int(row["layer_index"]) == layer)
            fit_1000, predictions_1000 = _fit_vectorized(
                train_cache[layer],
                valid_cache[layer],
                train_labels,
                valid_labels,
                widths,
                device=device,
                steps=STEPS_1000,
            )
            metrics_1000 = evaluate(task, dataset["validation"]["golds"], predictions_1000)
            f1_200 = float(row_200["selection_f1"])
            f1_1000 = float(metrics_1000["selection_f1"])
            model_rows.append(
                {
                    "layer_index": layer,
                    "role_200": [
                        role
                        for role, selected in (("final", final_layer), ("best_nonzero", best_nonzero_layer))
                        if layer == selected
                    ],
                    "relative_depth": float(row_200["relative_depth"]),
                    "hidden_size": int(train_cache.shape[2]),
                    "metrics_200": row_200["validation_metrics"],
                    "selection_f1_200": f1_200,
                    "fit_200": row_200["fit"],
                    "metrics_1000": metrics_1000,
                    "selection_f1_1000": f1_1000,
                    "delta_selection_f1": f1_1000 - f1_200,
                    "delta_selection_f1_points": 100.0 * (f1_1000 - f1_200),
                    "prediction_counts_1000": prediction_counts(task, predictions_1000),
                    "prediction_sha256_1000": prediction_hash(predictions_1000),
                    "fit_1000": fit_1000,
                }
            )
        order_200 = sorted(previous_rows, key=lambda row: float(row["selection_f1"]), reverse=True)
        rank_200 = {int(row["layer_index"]): index + 1 for index, row in enumerate(order_200)}
        order_selected_1000 = sorted(model_rows, key=lambda row: float(row["selection_f1_1000"]), reverse=True)
        rank_1000_selected = {int(row["layer_index"]): index + 1 for index, row in enumerate(order_selected_1000)}
        for row in model_rows:
            row["rank_200_all_layers"] = rank_200[row["layer_index"]]
            row["rank_1000_selected_layers"] = rank_1000_selected[row["layer_index"]]
        result["models"][model] = {
            "hidden_size": int(train_cache.shape[2]),
            "state_count": int(train_cache.shape[0]),
            "cache_hashes": cache_hashes,
            "selected_layers": selected_layers,
            "best_nonzero_layer_200": best_nonzero_layer,
            "final_layer": final_layer,
            "rank_stability": {
                "best_nonzero_remains_best_among_selected": bool(
                    rank_1000_selected[best_nonzero_layer] == 1
                ),
                "rank_200_best_nonzero": rank_200[best_nonzero_layer],
                "rank_200_final": rank_200[final_layer],
                "rank_1000_selected_best_nonzero": rank_1000_selected[best_nonzero_layer],
                "rank_1000_selected_final": rank_1000_selected[final_layer],
                "full_layer_1000_rank_not_estimated": True,
            },
            "layers": model_rows,
        }
    write_json(output_task / "sensitivity_results.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    source_root = (args.source_root or repo_root / "artifacts" / "d1" / "probes").resolve()
    cache_root = (args.cache_root or repo_root / "artifacts" / "d1" / "cache").resolve()
    output_root = (args.output_root or repo_root / "artifacts" / "d1" / "probe_sensitivity").resolve()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.torch_threads)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is unavailable")
    device = torch.device(args.device)
    started = time.time()
    all_results: dict[str, Any] = {}
    for task in args.tasks:
        all_results[task] = run_task(task, cache_root, source_root, output_root, device)
        write_json(output_root / "progress.json", {"status": "running", "completed_tasks": list(all_results)})
    summary = {
        "protocol": "D1-layerwise-linear-probe-convergence-sensitivity-v1",
        "status": "completed",
        "tasks": list(all_results),
        "elapsed_seconds": time.time() - started,
        "device": str(device),
        "models": {
            task: {
                model: all_results[task]["models"][model]["rank_stability"]
                for model in MODELS
            }
            for task in all_results
        },
    }
    write_json(output_root / "summary.json", summary)
    write_json(output_root / "progress.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
