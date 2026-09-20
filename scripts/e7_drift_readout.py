"""CPU-only parameter-drift readout for completed E7 trials.

This reads the original GoiTurn-S-GD encoder weights plus the selected E1 and
E7 checkpoints.  It performs no encoder forward pass and never loads test
data.  For each encoder region it reports the aggregate relative drift
``||after - before|| / ||before||`` and the selected epoch.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import RobertaModel


ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    "vsfc_sentiment",
    "vsfc_topic",
    "visfd",
    "absa_restaurant",
    "absa_hotel",
)
TASK_LR_NAMES = {
    "vsfc_sentiment": "1e-05",
    "vsfc_topic": "1e-05",
    "visfd": "1e-04",
    "absa_restaurant": "1e-04",
    "absa_hotel": "1e-04",
}
E1_ATTEMPTS = ROOT / "artifacts" / "e1" / ".orchestrator" / "attempts"
E7_ROOT = ROOT / "artifacts" / "e7"
OUTPUT = E7_ROOT / "drift_readout.json"
E1_SOURCE_HASH = "456e3020bd405ec39c7b4cbab3255d0bb3cb2a8a4bda3e37cc0620d51b559873"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _e1_artifact(task_id: str) -> Path:
    pattern = f"{task_id}__screening__{TASK_LR_NAMES[task_id]}__seed42__attempt_*/metadata.json"
    matches = sorted(E1_ATTEMPTS.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one E1 seed42 artifact for {task_id}, found {matches}")
    return matches[0].parent


def _region(name: str) -> str | None:
    if name.startswith("encoder.embeddings."):
        return "embeddings"
    match = re.match(r"encoder\.encoder\.layer\.(\d+)\.", name)
    if match is None:
        return None
    layer = int(match.group(1))
    if layer <= 2:
        return "early_blocks_0_2"
    if layer <= 6:
        return "mid_blocks_3_6"
    if layer <= 9:
        return "late_blocks_7_9"
    return None


def _relative_drift(
    source_state: dict[str, torch.Tensor],
    checkpoint_state: dict[str, torch.Tensor],
) -> dict[str, dict[str, float | int]]:
    sums: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    for checkpoint_name, after in checkpoint_state.items():
        region = _region(checkpoint_name)
        if region is None:
            continue
        source_name = checkpoint_name.removeprefix("encoder.")
        if source_name not in source_state:
            raise KeyError(f"Original encoder is missing {source_name} for {checkpoint_name}")
        before = source_state[source_name]
        if tuple(before.shape) != tuple(after.shape):
            raise ValueError(f"Shape mismatch for {checkpoint_name}: {before.shape} vs {after.shape}")
        before_f = before.detach().to(dtype=torch.float64)
        after_f = after.detach().to(dtype=torch.float64)
        numerator = float(torch.sum((after_f - before_f) ** 2).item())
        denominator = float(torch.sum(before_f**2).item())
        if denominator <= 0 or not math.isfinite(denominator):
            raise ValueError(f"Invalid original norm for {checkpoint_name}")
        if region not in sums:
            sums[region] = [0.0, 0.0]
            counts[region] = 0
        sums[region][0] += numerator
        sums[region][1] += denominator
        counts[region] += 1
    expected = ("embeddings", "early_blocks_0_2", "mid_blocks_3_6", "late_blocks_7_9")
    missing = [name for name in expected if name not in sums]
    if missing:
        raise ValueError(f"Missing encoder regions in checkpoint: {missing}")
    return {
        name: {
            "relative_l2_drift": math.sqrt(sums[name][0] / sums[name][1]),
            "parameter_tensor_count": counts[name],
            "before_squared_l2": sums[name][1],
            "delta_squared_l2": sums[name][0],
        }
        for name in expected
    }


def main() -> int:
    source_model = RobertaModel.from_pretrained(
        ROOT / "models" / "goiturn-s-gd",
        local_files_only=True,
        add_pooling_layer=False,
        attn_implementation="sdpa",
    )
    source_state = {name: tensor.detach().cpu() for name, tensor in source_model.state_dict().items()}
    rows: list[dict[str, Any]] = []
    for task_id in TASKS:
        e1_dir = _e1_artifact(task_id)
        e7_dir = E7_ROOT / task_id / "seed_42"
        e1_metadata = _read_json(e1_dir / "metadata.json")
        e7_metadata = _read_json(e7_dir / "metadata.json")
        e1_checkpoint = torch.load(e1_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False)
        e7_checkpoint = torch.load(e7_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False)
        e1_model = dict(e1_checkpoint["model"])
        e7_model = dict(e7_checkpoint["model"])
        rows.append(
            {
                "task_id": task_id,
                "e1_artifact_dir": str(e1_dir),
                "e7_artifact_dir": str(e7_dir),
                "e1_best_epoch": e1_metadata.get("best_epoch"),
                "e7_best_epoch": e7_metadata.get("best_epoch"),
                "e1_drift_from_original_gd": _relative_drift(source_state, e1_model),
                "e7_drift_from_original_gd": _relative_drift(source_state, e7_model),
            }
        )
    payload = {
        "protocol": "E7-cpu-parameter-drift-readout-v1",
        "status": "completed",
        "source_backbone": "models/goiturn-s-gd",
        "e1_dependency_source_hash": E1_SOURCE_HASH,
        "definition": "aggregate relative encoder drift = sqrt(sum ||after-before||^2 / sum ||before||^2)",
        "regions": {
            "embeddings": "encoder.embeddings.*",
            "early_blocks_0_2": "encoder.encoder.layer.0-2.*",
            "mid_blocks_3_6": "encoder.encoder.layer.3-6.*",
            "late_blocks_7_9": "encoder.encoder.layer.7-9.*",
        },
        "test_accessed": False,
        "encoder_forward_performed": False,
        "rows": rows,
    }
    _write_json(OUTPUT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
