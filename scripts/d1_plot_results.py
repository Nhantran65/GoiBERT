#!/usr/bin/env python3
"""Render the D1 probe and CLS cross-CKA figures from JSON artifacts only.

This script intentionally does no fitting, encoder inference, or CKA
calculation.  It reads the probe/CKA JSON files produced by the analysis
agents, masks undefined CKA entries, and writes two standard Matplotlib
figures in ``artifacts/d1/figures``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

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
COLORS = {"student": "#1f77b4", "teacher": "#d95f02"}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selection_f1(layer: dict[str, Any]) -> float:
    metrics = layer.get("validation_metrics", {})
    value = metrics.get("selection_f1")
    if value is None:
        value = metrics.get("macro_f1")
    if value is None:
        value = metrics.get("pair_micro_f1")
    if value is None:
        raise KeyError("Layer result has no selection_f1/macro_f1/pair_micro_f1")
    return float(value)


def _probe_curves(probe_results: dict[str, dict[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
    curves: dict[str, dict[str, np.ndarray]] = {}
    for task, result in probe_results.items():
        curves[task] = {}
        for model in ("student", "teacher"):
            layers = result["models"][model]["layers"]
            indices = np.asarray([int(layer["layer_index"]) for layer in layers], dtype=np.int64)
            scores = np.asarray([100.0 * _selection_f1(layer) for layer in layers], dtype=np.float64)
            if not np.array_equal(indices, np.arange(len(indices))):
                raise ValueError(f"Non-contiguous layer indices in {task}/{model}")
            if not np.isfinite(scores).all():
                raise ValueError(f"Non-finite probe score in {task}/{model}")
            curves[task][model] = scores
    return curves


def _sensitivity_points(
    sensitivity_results: dict[str, dict[str, Any]],
    model: str,
) -> tuple[np.ndarray, np.ndarray]:
    rows = sensitivity_results["models"][model]["layers"]
    layers = np.asarray([int(row["layer_index"]) for row in rows], dtype=np.int64)
    values = np.asarray([100.0 * float(row["selection_f1_1000"]) for row in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite 1,000-step sensitivity score")
    return layers, values


def render_probe(
    probe_results: dict[str, dict[str, Any]],
    sensitivity_results: dict[str, dict[str, Any]],
    output: Path,
) -> None:
    curves = _probe_curves(probe_results)
    fig, axes = plt.subplots(1, len(TASKS), figsize=(21, 4.7), sharey=True, constrained_layout=True)
    if len(TASKS) == 1:
        axes = [axes]
    for axis, task in zip(axes, TASKS):
        student = curves[task]["student"]
        teacher = curves[task]["teacher"]
        student_x = np.arange(student.size)
        teacher_x = np.arange(teacher.size)
        axis.axvspan(-0.5, 0.5, color="#9e9e9e", alpha=0.14, zorder=0)
        axis.plot(student_x, student, marker="o", markersize=3.5, linewidth=1.8,
                  color=COLORS["student"], label="GoiBERT student")
        axis.plot(teacher_x, teacher, marker="o", markersize=3.5, linewidth=1.8,
                  color=COLORS["teacher"], label="DAPT teacher")
        for model in ("student", "teacher"):
            layers, values = _sensitivity_points(sensitivity_results[task], model)
            axis.scatter(
                layers,
                values,
                marker="X",
                s=38,
                color=COLORS[model],
                edgecolors="#222222",
                linewidths=0.45,
                zorder=5,
            )
        axis.set_title(TASK_LABELS[task], fontsize=11, pad=8)
        axis.set_xlabel("Layer index", fontsize=9)
        axis.set_xticks(np.arange(max(student.size, teacher.size)))
        axis.set_ylim(0, 100)
        axis.set_yticks(np.arange(0, 101, 20))
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=8)
        axis.text(0.04, 0.965, "0 = embedding\n(constant)", transform=axis.transAxes,
                  va="top", ha="left", fontsize=7.5, color="#555555",
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5})
    axes[0].set_ylabel("Validation selection F1 (%)", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], marker="X", linestyle="none", markersize=7,
                              color="#555555", markeredgecolor="#222222", label="1,000-step sensitivity"))
    labels.append("1,000-step sensitivity")
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.03), fontsize=9)
    fig.suptitle(
        "D1 layerwise linear-probe F1 (CLS representations)",
        fontsize=14,
        y=1.08,
    )
    fig.text(
        0.5,
        -0.045,
        "Layer indices are encoder states: student has 10 transformer layers; teacher has 12. "
        "The embedding state (layer 0) is constant for CLS and is shown for reference.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _cka_matrix(cka: dict[str, Any]) -> np.ndarray:
    matrix = np.asarray(cka["train_cross"], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape != (11, 13):
        raise ValueError(f"Unexpected CLS CKA matrix shape: {matrix.shape}")
    return matrix


def render_cka(
    cka_results: dict[str, dict[str, Any]],
    mapping: list[int],
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, len(TASKS), figsize=(22, 5.0), constrained_layout=True)
    if len(TASKS) == 1:
        axes = [axes]
    image = None
    for axis, task in zip(axes, TASKS):
        matrix = np.ma.masked_invalid(_cka_matrix(cka_results[task]))
        axis.set_facecolor("#e6e6e6")
        image = axis.imshow(
            matrix,
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
            aspect="auto",
            interpolation="nearest",
            origin="upper",
        )
        xs = np.asarray(mapping, dtype=float)
        ys = np.arange(len(mapping), dtype=float)
        axis.plot(xs, ys, linestyle="none", marker="s", markersize=4.0,
                  markerfacecolor="none", markeredgecolor="#e31a1c", markeredgewidth=1.2,
                  label="GD target mapping")
        axis.set_title(TASK_LABELS[task], fontsize=11, pad=8)
        axis.set_xlabel("Teacher layer index", fontsize=9)
        axis.set_xticks(np.arange(13))
        axis.set_yticks(np.arange(11))
        axis.set_yticklabels([str(value) for value in range(11)], fontsize=8)
        axis.tick_params(axis="x", labelsize=8)
        axis.grid(False)
        axis.text(0.04, 0.965, "layer 0 = constant\nmasked / undefined",
                  transform=axis.transAxes, va="top", ha="left", fontsize=7.5,
                  color="#444444", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5})
    axes[0].set_ylabel("Student layer index", fontsize=9)
    if image is None:
        raise RuntimeError("No CKA panels rendered")
    colorbar = fig.colorbar(image, ax=axes, location="right", shrink=0.86, pad=0.02)
    colorbar.set_label("CLS cross-model CKA", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.025), fontsize=9)
    fig.suptitle(
        "D1 student–teacher CLS cross-CKA by layer",
        fontsize=14,
        y=1.08,
    )
    fig.text(
        0.5,
        -0.045,
        "Rows are student states (0 + 10 transformer layers); columns are teacher states (0 + 12 transformer layers). "
        "Red squares show the declared GD mapping, not a CKA-optimized alignment. Gray cells are undefined NaN entries.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    probe_results = {
        task: read_json(root / "artifacts" / "d1" / "probes" / task / "probe_results.json")
        for task in TASKS
    }
    sensitivity_results = {
        task: read_json(root / "artifacts" / "d1" / "probe_sensitivity" / task / "sensitivity_results.json")
        for task in TASKS
    }
    cka_results = {
        task: read_json(root / "artifacts" / "d1" / "cka" / task / "cls_cka.json")
        for task in TASKS
    }
    for task in TASKS:
        if probe_results[task].get("dataset", {}).get("include_test") is not False:
            raise ValueError(f"Probe result for {task} does not certify test exclusion")
    summary = read_json(root / "artifacts" / "d1" / "cka" / "summary.json")
    mapping = [int(value) for value in summary["gd_mapping"]]
    if mapping != [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12]:
        raise ValueError(f"Unexpected declared GD mapping: {mapping}")
    output_dir = root / "artifacts" / "d1" / "figures"
    probe_path = output_dir / "layerwise_probe_f1"
    cka_path = output_dir / "cls_cross_cka_heatmaps"
    render_probe(probe_results, sensitivity_results, probe_path)
    render_cka(cka_results, mapping, cka_path)
    manifest = {
        "schema": "goibert-d1-figures-v1",
        "source_only": True,
        "source_files": {
            **{f"probe/{task}": sha256_file(root / "artifacts" / "d1" / "probes" / task / "probe_results.json") for task in TASKS},
            **{f"probe_sensitivity/{task}": sha256_file(root / "artifacts" / "d1" / "probe_sensitivity" / task / "sensitivity_results.json") for task in TASKS},
            **{f"cka/{task}": sha256_file(root / "artifacts" / "d1" / "cka" / task / "cls_cka.json") for task in TASKS},
            "cka/summary": sha256_file(root / "artifacts" / "d1" / "cka" / "summary.json"),
            "renderer": sha256_file(Path(__file__).resolve()),
        },
        "probe_figure": {
            "png": str(probe_path.with_suffix(".png").relative_to(root)),
            "svg": str(probe_path.with_suffix(".svg").relative_to(root)),
            "metric": "JSON validation_metrics.selection_f1, shown as percent; line=200-step probe, X markers=1,000-step sensitivity",
            "y_limits": [0, 100],
            "layer_axis": "actual student/teacher layer indices",
        },
        "cka_figure": {
            "png": str(cka_path.with_suffix(".png").relative_to(root)),
            "svg": str(cka_path.with_suffix(".svg").relative_to(root)),
            "matrix": "JSON train_cross CLS CKA",
            "v_limits": [0, 1],
            "nan_policy": "masked gray",
            "gd_mapping": mapping,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
