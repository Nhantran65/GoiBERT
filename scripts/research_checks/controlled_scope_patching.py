#!/usr/bin/env python3
"""FP32 token-state patching diagnostic for two authored Restaurant inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from goibert import train as tr  # noqa: E402
from reader_operating_point import load_checkpoint  # noqa: E402
from shadow_reader import FullCapture  # noqa: E402


OUT = ROOT / "artifacts" / "research_diagnostics" / "controlled_scope_patching"
CONNECTORS = (". ", " và ")
SERVICE = "Nhân viên phục vụ tận tình"
FOOD_POS = "Món ăn ngon"
FOOD_NEG = "Món ăn không ngon"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)) + "\n", encoding="utf-8")
    tmp.replace(path)


def aspect_index(bundle: Any, name: str) -> int:
    names = list(bundle.get("aspect_names", ())) if isinstance(bundle, dict) else list(getattr(bundle, "aspect_names", ()))
    return names.index(name)


@torch.inference_mode()
def capture_text(model: Any, tokenizer: Any, text: str, device: torch.device):
    encoded = tokenizer([text], padding=True, truncation=True, max_length=256, return_tensors="pt", return_offsets_mapping=True)
    offsets = encoded.pop("offset_mapping")
    attention = encoded["attention_mask"]
    capture = FullCapture(model)
    all_outputs: dict[int, torch.Tensor] = {}
    handles = []
    for index, layer in enumerate(model.encoder.encoder.layer):
        handles.append(layer.register_forward_hook(lambda _m, _i, output, idx=index: all_outputs.__setitem__(idx, (output[0] if isinstance(output, tuple) else output).detach())) )
    capture.mask2d = attention
    with torch.autocast(device_type=device.type, enabled=False):
        logits = model(encoded["input_ids"].to(device), attention.to(device)).float().cpu().reshape(1, -1)
    for handle in handles: handle.remove()
    capture.restore()
    return {"encoded": encoded, "offsets": offsets, "attention": attention, "logits": logits, "outputs": all_outputs, "capture": capture}


def service_positions(source: dict[str, Any], base: dict[str, Any], tokenizer: Any) -> list[int]:
    source_ids = source["encoded"]["input_ids"][0].tolist(); base_ids = base["encoded"]["input_ids"][0].tolist()
    service_end = len(SERVICE)
    positions_source = []; positions_base = []
    for offsets, positions in ((source["offsets"][0], positions_source), (base["offsets"][0], positions_base)):
        for index, (start, end) in enumerate(offsets.tolist()):
            if start < end and end <= service_end and index > 0:
                positions.append(index)
    if positions_source != positions_base or [source_ids[i] for i in positions_source] != [base_ids[i] for i in positions_base]:
        raise RuntimeError("service token positions/IDs are not exact across source/base")
    if source_ids[: max(positions_source) + 1] != base_ids[: max(positions_base) + 1]:
        raise RuntimeError("source/base prefix token IDs differ before FOOD content")
    return positions_source


def flat_logits(model: Any, hidden: torch.Tensor) -> torch.Tensor:
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        return model.head(hidden).float().cpu().reshape(1, -1)


@torch.inference_mode()
def continue_from(model: Any, capture: FullCapture, state: torch.Tensor, attention: torch.Tensor, start_layer: int, device: torch.device) -> torch.Tensor:
    capture.mask2d = attention
    hidden = state.to(device)
    additive = (1.0 - attention[:, None, None, :].float().to(device)) * -1.0e4
    with torch.autocast(device_type=device.type, enabled=False):
        for index in range(start_layer + 1, len(model.encoder.encoder.layer)):
            hidden = model.encoder.encoder.layer[index](hidden, attention_mask=additive)[0]
        return flat_logits(model, hidden)


def target_readout(logits: torch.Tensor, food: int, service: int) -> dict[str, Any]:
    flat = logits.reshape(logits.shape[0], -1)[0]
    food_start, service_start = food * 4, service * 4
    food_slot = flat[food_start:food_start + 4]; service_slot = flat[service_start:service_start + 4]
    pred = [int(torch.argmax(food_slot)), int(torch.argmax(service_slot))]
    return {"food_logits": [float(x) for x in food_slot], "service_logits": [float(x) for x in service_slot],
            "food_margin_negative": float(food_slot[2] - food_slot[0]), "service_margin_positive": float(service_slot[1] - service_slot[0]),
            "food_prediction": pred[0], "service_prediction": pred[1], "joint_food_negative_service_positive": pred == [2, 1]}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--device", default="cuda:1"); args = parser.parse_args()
    device = torch.device(args.device); OUT.mkdir(parents=True, exist_ok=True); started = time.time()
    model, tokenizer, bundle, counts, checkpoint, _ = load_checkpoint("student_e8", "absa_restaurant", device)
    food = aspect_index(bundle, "FOOD#QUALITY"); service = aspect_index(bundle, "SERVICE#GENERAL")
    inputs = []
    for connector in CONNECTORS:
        source_text = SERVICE + connector + FOOD_POS; base_text = SERVICE + connector + FOOD_NEG
        inputs.extend([{"id": f"source_{'dot' if connector == '. ' else 'and'}", "role": "source", "connector": connector, "text": source_text, "food_expected": 1, "service_expected": 1},
                       {"id": f"base_{'dot' if connector == '. ' else 'and'}", "role": "base", "connector": connector, "text": base_text, "food_expected": 2, "service_expected": 1}])
    captures = {}
    for item in inputs:
        captures[item["id"]] = capture_text(model, tokenizer, item["text"], device)
    patches = []
    for connector in CONNECTORS:
        suffix = "dot" if connector == ". " else "and"; source = captures[f"source_{suffix}"]; base = captures[f"base_{suffix}"]
        service_pos = service_positions(source, base, tokenizer)
        source_readout = target_readout(source["logits"], food, service); base_readout = target_readout(base["logits"], food, service)
        for item, readout in ((source, source_readout), (base, base_readout)):
            item["readout"] = readout; item["token_ids"] = item["encoded"]["input_ids"][0].tolist(); item["attention_mask"] = item["attention"][0].tolist()
        if source_readout["food_prediction"] != 1 or source_readout["service_prediction"] != 1 or base_readout["food_prediction"] != 2 or base_readout["service_prediction"] == 1:
            report = {"protocol": "controlled-scope-patching-v1", "status": "stopped_fp32_reproduction_not_reproduced", "connector": connector, "source": source_readout, "base": base_readout, "checkpoint_sha256": sha256(checkpoint)}
            write_json(OUT / "summary.json", report); return 0
        continuation_capture = FullCapture(model)
        for layer in range(len(model.encoder.encoder.layer)):
            base_state = base["outputs"][layer].clone(); source_state = source["outputs"][layer]
            patched = base_state.clone(); patched[:, service_pos, :] = source_state[:, service_pos, :]
            patched_logits = continue_from(model, continuation_capture, patched, base["attention"], layer, device)
            replay_logits = continue_from(model, continuation_capture, base_state.clone(), base["attention"], layer, device)
            patches.append({"connector": connector, "layer": layer, "service_positions": service_pos, "state_shape": list(base_state.shape),
                            "patched": target_readout(patched_logits, food, service), "base_replay": target_readout(replay_logits, food, service),
                            "base_replay_max_abs_logit_error": float((replay_logits - base["logits"].reshape(1, -1)).abs().max()),
                            "patched_logits": [float(x) for x in patched_logits[0]]})
        continuation_capture.restore()
    predictions = []
    for item in inputs:
        cap = captures[item["id"]]; predictions.append({"id": item["id"], "role": item["role"], "connector": item["connector"], "text": item["text"], "text_sha256": hashlib.sha256(item["text"].encode()).hexdigest(), "token_ids": cap["encoded"]["input_ids"][0].tolist(), "attention_mask": cap["attention"][0].tolist(), "readout": cap["readout"]})
    report = {"protocol": "controlled-scope-patching-v1", "created_at": datetime.now(timezone.utc).isoformat(), "validation_only": True, "test_access": False, "training": False,
              "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint), "source_hashes": {"script": sha256(Path(__file__)), "shadow_helper": sha256(Path(__file__).resolve().parent / "shadow_reader.py")},
              "food_slot": food, "service_slot": service, "inputs": predictions, "patches": patches, "runtime_seconds": time.time() - started,
              "fp32_reproduction": all(item["readout"]["service_prediction"] != 1 for item in predictions if item["role"] == "base"),
              "same_state_controls_max_abs_error": max(item["base_replay_max_abs_logit_error"] for item in patches)}
    if report["same_state_controls_max_abs_error"] > 1e-4:
        raise RuntimeError(f"same-state replay control failed: {report['same_state_controls_max_abs_error']}")
    final_errors = []
    for connector in CONNECTORS:
        suffix = "dot" if connector == ". " else "and"; base_flat = captures[f"base_{suffix}"]["logits"].reshape(1, -1)[0]
        final_errors.append(float((torch.tensor(next(x for x in reversed(patches) if x["connector"] == connector)["patched_logits"]) - base_flat).abs().max()))
    report["final_layer_patch_max_abs_logit_error_by_connector"] = final_errors
    if max(final_errors) > 1e-4:
        raise RuntimeError(f"final-layer patch changed head unexpectedly: {final_errors}")
    write_json(OUT / "input_manifest.json", {"protocol": report["protocol"], "inputs": inputs, "checkpoint_sha256": report["checkpoint_sha256"]})
    write_json(OUT / "predictions.json", {"source_base": predictions, "patches": patches})
    write_json(OUT / "summary.json", report)
    lines = ["# Controlled scope patching", "", "FP32 validation-only token-state patching; authored Restaurant BA inputs only.", "", "| Connector | Base service prediction | Base joint | First layer rescuing joint | Final-layer joint | Replay max abs error |", "|---|---:|---:|---:|---:|---:|"]
    for connector in CONNECTORS:
        suffix = "dot" if connector == ". " else "and"; base = next(x for x in predictions if x["id"] == f"base_{suffix}"); ps = [x for x in patches if x["connector"] == connector]; rescue = next((x["layer"] for x in ps if x["patched"]["joint_food_negative_service_positive"]), None); final = ps[-1]
        lines.append(f"| {connector.strip()} | {base['readout']['service_prediction']} | {base['readout']['joint_food_negative_service_positive']} | {rescue} | {final['patched']['joint_food_negative_service_positive']} | {max(x['base_replay_max_abs_logit_error'] for x in ps):.3g} |")
    (OUT / "FACTUAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
