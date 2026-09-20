#!/usr/bin/env python3
"""Render D2 before/after probe and same-layer CKA figures.

This renderer only reads JSON artifacts produced by D1/D2 analysis.  It does
not load torch, fit probes, run an encoder, or calculate CKA.  The default
command requires all five D2 tasks so a partial analysis cannot silently
produce a misleading five-panel figure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np


TASKS = (
    "vsfc_sentiment",
    "vsfc_topic",
    "visfd",
    "absa_restaurant",
    "absa_hotel",
)
TASK_LABELS = {
    "vsfc_sentiment": "VSFC\nSentiment",
    "vsfc_topic": "VSFC\nTopic",
    "visfd": "ViSFD",
    "absa_restaurant": "UIT-ABSA\nRestaurant",
    "absa_hotel": "UIT-ABSA\nHotel",
}
COLORS = {"student": "#2166ac", "teacher": "#b35806"}
LIGHT_COLORS = {"student": "#67a9cf", "teacher": "#f1a340"}
MODEL_LABELS = {"student": "Student", "teacher": "Teacher"}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _probe_curve(result: Mapping[str, Any], model: str) -> tuple[np.ndarray, np.ndarray]:
    layers = result["models"][model]["layers"]
    indices = np.asarray([int(row["layer_index"]) for row in layers], dtype=np.int64)
    scores = np.asarray(
        [100.0 * float(row.get("selection_f1", row["validation_metrics"]["selection_f1"])) for row in layers],
        dtype=np.float64,
    )
    if indices.size == 0 or not np.array_equal(indices, np.arange(indices.size)):
        raise ValueError(f"Layer indices are not contiguous from zero for {model}")
    if not np.isfinite(scores).all():
        raise ValueError(f"Non-finite probe curve for {model}")
    return indices, scores


def _sensitivity_points(result: Mapping[str, Any], task: str, model: str) -> tuple[int, float] | None:
    for row in result.get("rows", []):
        if row.get("task") == task and row.get("model") == model:
            value = row.get("after_selection_f1_1000_points")
            if value is None:
                value = 100.0 * float(row["after_selection_f1_1000"])
            return int(row["layer_index"]), float(value)
    return None


def _d1_sensitivity_point(result: Mapping[str, Any], model: str) -> tuple[int, float] | None:
    model_result = result.get("models", {}).get(model)
    if not isinstance(model_result, Mapping):
        return None
    layers = model_result.get("layers", [])
    if not layers:
        return None
    # D1 sensitivity has one final-layer row per model.  Read the explicit
    # final row rather than choosing a layer from the validation result.
    row = next((item for item in layers if item.get("role_200") and "final" in item["role_200"]), layers[-1])
    value = row.get("selection_f1_1000")
    if value is None:
        value = row.get("metrics_1000", {}).get("selection_f1")
    if value is None:
        return None
    return int(row["layer_index"]), 100.0 * float(value)


def _load_probe_inputs(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    before: dict[str, dict[str, Any]] = {}
    after: dict[str, dict[str, Any]] = {}
    before_sensitivity: dict[str, Any] = {}
    for task in TASKS:
        d1_path = root / "artifacts" / "d1" / "probes" / task / "probe_results.json"
        d2_path = root / "artifacts" / "d2" / "probes" / task / "probe_results.json"
        if not d1_path.is_file():
            raise FileNotFoundError(d1_path)
        if not d2_path.is_file():
            raise FileNotFoundError(
                f"D2 probe output missing for {task}: {d2_path}. "
                "The five-panel figure is intentionally withheld until all five tasks are complete."
            )
        before[task] = read_json(d1_path)
        after[task] = read_json(d2_path)
        if before[task].get("dataset", {}).get("include_test") is not False:
            raise ValueError(f"D1 probe opened test for {task}")
        if after[task].get("dataset", {}).get("include_test") is not False:
            raise ValueError(f"D2 probe opened test for {task}")
        d1_sensitivity_path = root / "artifacts" / "d1" / "probe_sensitivity" / task / "sensitivity_results.json"
        before_sensitivity[task] = read_json(d1_sensitivity_path) if d1_sensitivity_path.is_file() else {}
    d2_sensitivity_path = root / "artifacts" / "d2" / "probe_sensitivity" / "final_layer_sensitivity.json"
    d2_sensitivity = read_json(d2_sensitivity_path) if d2_sensitivity_path.is_file() else {}
    return before, after, before_sensitivity, d2_sensitivity


def render_probe(
    root: Path,
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
    before_sensitivity: Mapping[str, Mapping[str, Any]],
    after_sensitivity: Mapping[str, Any],
    output_stem: Path,
) -> None:
    fig, axes = plt.subplots(
        2,
        len(TASKS),
        figsize=(21, 8.2),
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    for col, task in enumerate(TASKS):
        for row, model in enumerate(("student", "teacher")):
            axis = axes[row, col]
            before_x, before_y = _probe_curve(before[task], model)
            after_x, after_y = _probe_curve(after[task], model)
            axis.plot(
                before_x,
                before_y,
                color=LIGHT_COLORS[model],
                linestyle="--",
                linewidth=1.7,
                marker="o",
                markersize=3.2,
                label="Before fine-tune" if col == 0 else None,
            )
            axis.plot(
                after_x,
                after_y,
                color=COLORS[model],
                linestyle="-",
                linewidth=2.0,
                marker="o",
                markersize=3.2,
                label="After fine-tune" if col == 0 else None,
            )
            d1_point = _d1_sensitivity_point(before_sensitivity[task], model)
            d2_point = _sensitivity_points(after_sensitivity, task, model)
            if d1_point is not None:
                axis.scatter(
                    [d1_point[0]],
                    [d1_point[1]],
                    marker="^",
                    s=45,
                    facecolors="none",
                    edgecolors=LIGHT_COLORS[model],
                    linewidths=1.2,
                    zorder=5,
                    label="1,000-step before" if col == 0 else None,
                )
            if d2_point is not None:
                axis.scatter(
                    [d2_point[0]],
                    [d2_point[1]],
                    marker="X",
                    s=48,
                    color=COLORS[model],
                    edgecolors="#222222",
                    linewidths=0.35,
                    zorder=6,
                    label="1,000-step after" if col == 0 else None,
                )
            axis.set_title(TASK_LABELS[task], fontsize=10.5, pad=7)
            axis.set_xlabel("Layer index", fontsize=8.5)
            axis.set_xticks(np.arange(max(len(before_x), len(after_x))))
            axis.set_ylim(0, 100)
            axis.set_yticks(np.arange(0, 101, 20))
            axis.grid(axis="y", color="#d9d9d9", linewidth=0.65, alpha=0.8)
            axis.set_axisbelow(True)
            axis.tick_params(labelsize=7.5)
            axis.axvspan(-0.5, 0.5, color="#999999", alpha=0.1, zorder=0)
            if col == 0:
                axis.set_ylabel(f"{MODEL_LABELS[model]}\nValidation F1 (%)", fontsize=8.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.015), fontsize=8.5)
    fig.suptitle(
        "D2 layerwise linear-probe F1 before and after task fine-tuning",
        fontsize=14,
        y=1.055,
    )
    fig.text(
        0.5,
        -0.025,
        "Lines: fixed 200-step probe. Markers: final-layer 1,000-step sensitivity when available. "
        "Layer 0 is the embedding state and is constant for CLS.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _same_layer_curve(cka_result: Mapping[str, Any], model: str) -> tuple[np.ndarray, np.ndarray]:
    values = cka_result["models"][model]["cls"]["same_layer_diagonal"]
    y = np.asarray(values, dtype=np.float64)
    x = np.arange(y.size, dtype=np.int64)
    finite = np.isfinite(y)
    if y.size == 0 or not finite[1:].any():
        raise ValueError(f"No finite same-layer CKA values for {model}")
    return x, y


def render_cka(
    cka_results: Mapping[str, Mapping[str, Any]],
    output_stem: Path,
) -> None:
    fig, axes = plt.subplots(1, len(TASKS), figsize=(21, 4.5), sharey=True, constrained_layout=True, squeeze=False)
    for col, task in enumerate(TASKS):
        axis = axes[0, col]
        for model in ("student", "teacher"):
            x, y = _same_layer_curve(cka_results[task], model)
            axis.plot(
                x,
                y,
                color=COLORS[model],
                linewidth=2.0,
                marker="o",
                markersize=3.4,
                label=MODEL_LABELS[model] if col == 0 else None,
            )
        axis.axvspan(-0.5, 0.5, color="#999999", alpha=0.1, zorder=0)
        axis.set_title(TASK_LABELS[task], fontsize=10.5, pad=7)
        axis.set_xlabel("Layer index", fontsize=8.5)
        max_layers = max(len(cka_results[task]["models"][model]["cls"]["same_layer_diagonal"]) for model in ("student", "teacher"))
        axis.set_xticks(np.arange(max_layers))
        axis.set_ylim(0, 1.02)
        axis.set_yticks(np.arange(0, 1.01, 0.2))
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.65, alpha=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=7.5)
        axis.text(
            0.04,
            0.95,
            "layer 0 undefined",
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=7,
            color="#555555",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.3},
        )
    axes[0, 0].set_ylabel("CKA(before, after)\nmatched layer", fontsize=8.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01), fontsize=8.5)
    fig.suptitle("D2 same-layer CKA before versus after task fine-tuning", fontsize=14, y=1.07)
    fig.text(
        0.5,
        -0.035,
        "Each point compares the same encoder layer before and after fine-tuning on the train-only CKA cohort. "
        "The embedding CLS state is constant and therefore undefined.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    before, after, before_sensitivity, after_sensitivity = _load_probe_inputs(root)
    cka_results: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        path = root / "artifacts" / "d2" / "cka" / task / "result.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"D2 CKA output missing for {task}: {path}. "
                "The five-panel CKA figure is intentionally withheld until all five tasks are complete."
            )
        cka_results[task] = read_json(path)
    output_dir = root / "artifacts" / "d2" / "figures"
    probe_stem = output_dir / "layerwise_probe_before_after"
    cka_stem = output_dir / "same_layer_cka_before_after"
    render_probe(root, before, after, before_sensitivity, after_sensitivity, probe_stem)
    render_cka(cka_results, cka_stem)
    source_paths: list[Path] = []
    for task in TASKS:
        source_paths.extend(
            [
                root / "artifacts" / "d1" / "probes" / task / "probe_results.json",
                root / "artifacts" / "d2" / "probes" / task / "probe_results.json",
                root / "artifacts" / "d2" / "cka" / task / "result.json",
            ]
        )
        d1_sensitivity = root / "artifacts" / "d1" / "probe_sensitivity" / task / "sensitivity_results.json"
        if d1_sensitivity.is_file():
            source_paths.append(d1_sensitivity)
    d2_sensitivity = root / "artifacts" / "d2" / "probe_sensitivity" / "final_layer_sensitivity.json"
    if d2_sensitivity.is_file():
        source_paths.append(d2_sensitivity)
    source_paths.append(Path(__file__).resolve())
    manifest = {
        "schema": "goibert-d2-figures-v1",
        "status": "completed",
        "test_opened": False,
        "tasks": list(TASKS),
        "source_files": {str(path.relative_to(root)): sha256_file(path) for path in source_paths},
        "probe_figure": {
            "png": str(probe_stem.with_suffix(".png").relative_to(root)),
            "svg": str(probe_stem.with_suffix(".svg").relative_to(root)),
            "metric": "validation selection F1 (%)",
            "layout": "2 rows x 5 tasks; top student, bottom teacher",
            "line": "fixed 200-step probe",
            "markers": "final-layer 1,000-step sensitivity when available",
            "y_limits": [0, 100],
        },
        "cka_figure": {
            "png": str(cka_stem.with_suffix(".png").relative_to(root)),
            "svg": str(cka_stem.with_suffix(".svg").relative_to(root)),
            "metric": "train-only CLS CKA between matched before/after layers",
            "y_limits": [0, 1.02],
            "undefined_layer": 0,
        },
    }
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest["outputs"] = {
        "probe_png_sha256": sha256_file(probe_stem.with_suffix(".png")),
        "probe_svg_sha256": sha256_file(probe_stem.with_suffix(".svg")),
        "cka_png_sha256": sha256_file(cka_stem.with_suffix(".png")),
        "cka_svg_sha256": sha256_file(cka_stem.with_suffix(".svg")),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Rendered {probe_stem} and {cka_stem}")


if __name__ == "__main__":
    main()
