#!/usr/bin/env python3
"""Repair D2 grouped weight-drift summaries from saved per-parameter norms.

The original D2 extraction saved correct per-parameter L2 values but grouped
attention output projections under FFN because both names contain
``output.dense``.  This repair never loads a checkpoint or representation
array.  It backs up the old grouped JSON, re-aggregates the saved parameter
norms, and updates only the drift references in each manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("vsfc_sentiment", "vsfc_topic", "visfd", "absa_restaurant", "absa_hotel")
MODELS = ("student", "teacher")
LAYER_RE = re.compile(r"^encoder\.layer\.(\d+)\.(.*)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def empty_group() -> dict[str, float | int]:
    return {"parameter_count": 0, "abs_sq": 0.0, "before_sq": 0.0, "relative_l2": 0.0}


def add(group: dict[str, float | int], parameter: Mapping[str, Any]) -> None:
    group["parameter_count"] = int(group["parameter_count"]) + int(parameter["parameter_count"])
    group["abs_sq"] = float(group["abs_sq"]) + float(parameter["abs_l2"]) ** 2
    group["before_sq"] = float(group["before_sq"]) + float(parameter["before_l2"]) ** 2


def finish(group: dict[str, float | int]) -> dict[str, float | int | None]:
    abs_l2 = float(max(float(group.pop("abs_sq")), 0.0) ** 0.5)
    before_l2 = float(max(float(group.pop("before_sq")), 0.0) ** 0.5)
    group["abs_l2"] = abs_l2
    group["before_l2"] = before_l2
    group["relative_l2"] = abs_l2 / before_l2 if before_l2 > 0.0 else None
    return group


def category(remainder: str) -> str:
    # Order is intentional: attention.output.dense is an attention projection,
    # while output.dense at the block root is the FFN output projection.
    if ".LayerNorm." in remainder:
        return "layernorm"
    if remainder.startswith("attention."):
        return "attention"
    if remainder.startswith("intermediate.") or remainder.startswith("output.dense."):
        return "ffn"
    return "other"


def regroup(parameters: Mapping[str, Mapping[str, Any]], repair: Mapping[str, Any]) -> dict[str, Any]:
    groups: dict[str, dict[str, float | int]] = {"embeddings": empty_group()}
    layers: dict[str, dict[str, dict[str, float | int]]] = {}
    all_group = empty_group()
    for key, parameter in sorted(parameters.items()):
        add(all_group, parameter)
        if key.startswith("embeddings."):
            add(groups["embeddings"], parameter)
            continue
        match = LAYER_RE.match(key)
        if not match:
            groups.setdefault("other", empty_group())
            add(groups["other"], parameter)
            continue
        layer_name = f"layer_{int(match.group(1))}"
        layer = layers.setdefault(
            layer_name,
            {name: empty_group() for name in ("attention", "ffn", "layernorm", "other")},
        )
        add(layer[category(match.group(2))], parameter)
    return {
        "schema_version": "d2-weight-drift-v2",
        "dtype_for_norms": "float32_cpu",
        "repair": dict(repair),
        "parameters": parameters,
        "groups": {name: finish(value) for name, value in groups.items()},
        "layers": {
            layer: {name: finish(value) for name, value in sorted(categories.items())}
            for layer, categories in sorted(layers.items())
        },
        "all_encoder": finish(all_group),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "artifacts" / "d2" / "cache")
    args = parser.parse_args()
    cache_root = args.cache_root.resolve()
    script_hash = sha256_file(Path(__file__).resolve())
    repaired: list[dict[str, Any]] = []
    for task in TASKS:
        for model in MODELS:
            output_dir = cache_root / task / model
            drift_path = output_dir / "weight_drift.json"
            manifest_path = output_dir / "manifest.json"
            if not drift_path.is_file() or not manifest_path.is_file():
                raise FileNotFoundError(f"Incomplete D2 drift cache: {output_dir}")
            old_hash = sha256_file(drift_path)
            backup_path = output_dir / "weight_drift.json.pre_group_fix_20260910.json"
            if not backup_path.exists():
                shutil.copyfile(drift_path, backup_path)
            elif not backup_path.is_file():
                raise RuntimeError(f"Unexpected drift backup state: {backup_path}")
            old = json.loads(drift_path.read_text(encoding="utf-8"))
            parameters = old.get("parameters")
            if not isinstance(parameters, Mapping) or not parameters:
                raise RuntimeError(f"No per-parameter norms in {drift_path}")
            repair = {
                "reason": "Correct attention.output.dense grouping; reaggregate saved per-parameter norms",
                "previous_grouped_path": str(backup_path.relative_to(ROOT)),
                "previous_grouped_sha256": sha256_file(backup_path),
                "repair_script": str(Path(__file__).resolve().relative_to(ROOT)),
                "repair_script_sha256": script_hash,
                "arrays_reextracted": False,
                "checkpoints_reloaded": False,
            }
            corrected = regroup(parameters, repair)
            if corrected["all_encoder"]["parameter_count"] != old["all_encoder"]["parameter_count"]:
                raise RuntimeError(f"Encoder parameter count changed in {drift_path}")
            write_json_atomic(drift_path, corrected)
            new_hash = sha256_file(drift_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            arrays_before = {
                name: details["sha256"]
                for name, details in manifest.get("arrays", {}).items()
            }
            source_hashes = manifest.setdefault("source_hashes", {})
            source_hashes["weight_drift_repair"] = script_hash
            source_hashes["weight_drift_previous_grouped"] = sha256_file(backup_path)
            manifest["weight_drift"] = {
                "path": str(drift_path.relative_to(ROOT)),
                "sha256": new_hash,
                "schema_version": "d2-weight-drift-v2",
                "all_encoder_relative_l2": corrected["all_encoder"]["relative_l2"],
                "repair": repair,
            }
            manifest["weight_drift_repair"] = repair
            if arrays_before != {
                name: details["sha256"] for name, details in manifest.get("arrays", {}).items()
            }:
                raise RuntimeError(f"Array metadata unexpectedly changed in {manifest_path}")
            write_json_atomic(manifest_path, manifest)
            repaired.append(
                {
                    "task": task,
                    "model": model,
                    "path": str(drift_path.relative_to(ROOT)),
                    "previous_sha256": old_hash,
                    "backup_sha256": sha256_file(backup_path),
                    "corrected_sha256": new_hash,
                    "all_encoder_relative_l2": corrected["all_encoder"]["relative_l2"],
                }
            )

    summary_path = cache_root / "d2_extraction_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["weight_drift_grouping_schema"] = "d2-weight-drift-v2"
    summary["weight_drift_repair"] = {
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": script_hash,
        "arrays_reextracted": False,
        "checkpoints_reloaded": False,
        "entries": repaired,
    }
    write_json_atomic(summary_path, summary)
    print(json.dumps({"status": "completed", "repaired": repaired}, indent=2))


if __name__ == "__main__":
    main()
