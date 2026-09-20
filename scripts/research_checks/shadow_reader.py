#!/usr/bin/env python3
"""Shared-content-stack shadow reader diagnostic for frozen E8 checkpoints.

The full pass uses the verified manual FP32 attention path under BF16
autocast. Shadow readers keep the full pass content K/V fixed and evolve only
their CLS state through the original attention output, residual/LN, and FFN.
This is validation-only inference; no training or test access occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import types
from contextlib import nullcontext
from collections import defaultdict
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
from reader_operating_point import (  # noqa: E402
    OUT as READER_OUT,
    clause_spans,
    delta0,
    error_summary,
    load_checkpoint,
    polarity_argmax_from_flat,
    prediction_from_flat,
    select_c4,
    slot_slices,
    curve,
)


OUT = ROOT / "artifacts" / "research_diagnostics" / "shadow_reader"
TASKS = ("absa_restaurant", "absa_hotel")
MAX_LEN = 256


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)) + "\n", encoding="utf-8")
    tmp.replace(path)


class FullCapture:
    """Manual FP32 attention wrapper plus full-stack state hooks."""

    def __init__(self, model: Any):
        self.model = model
        self.mask2d: torch.Tensor | None = None
        self.hidden_inputs: dict[int, torch.Tensor] = {}
        self.content_k: dict[int, torch.Tensor] = {}
        self.content_v: dict[int, torch.Tensor] = {}
        self.full_outputs: dict[int, torch.Tensor] = {}
        self.originals: dict[int, Any] = {}
        self.handles = []
        for index, layer in enumerate(model.encoder.encoder.layer):
            attention = layer.attention.self
            self.originals[index] = attention.forward
            attention.forward = types.MethodType(self._make_forward(index), attention)
            self.handles.append(layer.register_forward_pre_hook(self._pre_hook(index)))
            self.handles.append(layer.register_forward_hook(self._post_hook(index)))

    def _pre_hook(self, index: int):
        state = self

        def hook(_module, inputs):
            state.hidden_inputs[index] = inputs[0].detach()

        return hook

    def _post_hook(self, index: int):
        state = self

        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            state.full_outputs[index] = value[:, 0].detach()

        return hook

    def clear(self) -> None:
        self.hidden_inputs.clear(); self.content_k.clear(); self.content_v.clear(); self.full_outputs.clear()

    def restore(self) -> None:
        for handle in self.handles:
            handle.remove()
        for index, original in self.originals.items():
            self.model.encoder.encoder.layer[index].attention.self.forward = original
        self.handles.clear()

    def _make_forward(self, index: int):
        state = self

        def forward(sa, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                    encoder_attention_mask=None, past_key_value=None, output_attentions=False):
            batch, length, _ = hidden_states.shape
            heads, dim = sa.num_attention_heads, sa.attention_head_size
            with torch.autocast(device_type=hidden_states.device.type, enabled=False):
                x = hidden_states.float()
                query = sa.query(x).view(batch, length, heads, dim).transpose(1, 2)
                key = sa.key(x).view(batch, length, heads, dim).transpose(1, 2)
                value = sa.value(x).view(batch, length, heads, dim).transpose(1, 2)
                valid = state.mask2d.to(x.device).bool()
                scores = torch.matmul(query, key.transpose(-1, -2)) / (float(dim) ** 0.5)
                scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
                probs = torch.softmax(scores, dim=-1)
                context = torch.matmul(probs, value).transpose(1, 2).reshape(batch, length, heads * dim)
            state.content_k[index] = key.detach()
            state.content_v[index] = value.detach()
            return (context.to(hidden_states.dtype),)

        return forward


def build_reader_batch(tokenizer: Any, texts: list[str]):
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LEN,
                        return_tensors="pt", return_offsets_mapping=True)
    offsets = encoded.pop("offset_mapping")
    attention = encoded["attention_mask"]
    rows: list[int] = []
    keeps: list[torch.Tensor] = []
    metadata: list[list[dict[str, Any]]] = [[] for _ in texts]
    for row, text in enumerate(texts):
        length = int(attention[row].sum())
        full = torch.zeros(encoded["input_ids"].shape[1], dtype=torch.bool); full[:length] = True
        rows.append(row); keeps.append(full)
        metadata[row].append({"reader_index": 0, "kind": "full", "span": [0, len(text)],
                              "keep_token_mask": full.tolist(), "valid_token_count": length})
        spans = clause_spans(text)
        if len(spans) >= 2:
            for start, end in spans:
                keep = torch.zeros_like(full); keep[0] = True; keep[length - 1] = True
                for token in range(1, length - 1):
                    os_, oe = offsets[row, token].tolist()
                    if oe > start and os_ < end:
                        keep[token] = True
                if int(keep.sum()) >= 4:
                    index = len(metadata[row]); rows.append(row); keeps.append(keep)
                    metadata[row].append({"reader_index": index, "kind": "clause", "span": [start, end],
                                          "keep_token_mask": keep.tolist(), "valid_token_count": length})
    return encoded, attention, rows, torch.stack(keeps), metadata


@torch.inference_mode()
def full_pass(model: Any, capture: FullCapture, input_ids: torch.Tensor, attention: torch.Tensor,
              device: torch.device, use_bf16: bool = True):
    capture.clear(); capture.mask2d = attention
    context = tr.make_autocast("bf16", device) if use_bf16 else nullcontext()
    with context:
        logits = model(input_ids.to(device), attention.to(device)).float().cpu()
    return logits, {key: value.detach() for key, value in capture.hidden_inputs.items()}, {key: value.detach() for key, value in capture.full_outputs.items()}, {key: value.detach() for key, value in capture.content_k.items()}, {key: value.detach() for key, value in capture.content_v.items()}


@torch.inference_mode()
def shadow_pass(model: Any, hidden_inputs: dict[int, torch.Tensor], full_outputs: dict[int, torch.Tensor],
                content_k: dict[int, torch.Tensor], content_v: dict[int, torch.Tensor],
                attention: torch.Tensor, reader_rows: list[int], reader_keeps: torch.Tensor,
                device: torch.device, use_bf16: bool = True):
    row_tensor = torch.tensor(reader_rows, dtype=torch.long, device=device)
    shadow = hidden_inputs[0][row_tensor, 0].to(device)
    parity_layers: list[torch.Tensor] = []
    context = tr.make_autocast("bf16", device) if use_bf16 else nullcontext()
    attention_device = attention.to(device)
    with context:
        for index, layer in enumerate(model.encoder.encoder.layer):
            sa = layer.attention.self
            k = content_k[index][row_tensor].to(device).clone()
            v = content_v[index][row_tensor].to(device).clone()
            with torch.autocast(device_type=device.type, enabled=False):
                shadow_float = shadow.float()
                query = sa.query(shadow_float).view(len(reader_rows), sa.num_attention_heads, sa.attention_head_size)
                key0 = sa.key(shadow_float).view(len(reader_rows), sa.num_attention_heads, sa.attention_head_size)
                value0 = sa.value(shadow_float).view(len(reader_rows), sa.num_attention_heads, sa.attention_head_size)
                k[:, :, 0, :] = key0
                v[:, :, 0, :] = value0
                scores = torch.einsum("rhd,rhnd->rhn", query, k) / (float(sa.attention_head_size) ** 0.5)
                valid = attention_device[row_tensor].bool() & reader_keeps.to(device).bool()
                scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
                probs = torch.softmax(scores, dim=-1)
                context = torch.einsum("rhn,rhnd->rhd", probs, v).reshape(len(reader_rows), -1)
            hidden = shadow.unsqueeze(1)
            attention_output = layer.attention.output.dense(context.to(shadow.dtype)).unsqueeze(1)
            hidden = layer.attention.output.LayerNorm(attention_output + hidden)
            intermediate = layer.intermediate(hidden)
            hidden = layer.output(intermediate, hidden)
            shadow = hidden[:, 0]
            parity_layers.append(shadow.detach())
    context = tr.make_autocast("bf16", device) if use_bf16 else nullcontext()
    with context:
        logits = model.head(shadow.unsqueeze(1)).float().cpu()
    return logits, parity_layers


def flatten_uniform(logits: torch.Tensor) -> torch.Tensor:
    return logits.reshape(logits.shape[0], -1) if logits.ndim == 3 else logits


def probabilities(logits: torch.Tensor, slices: list[tuple[int, int]]) -> torch.Tensor:
    output = torch.empty_like(logits)
    for left, right in slices:
        output[:, left:right] = torch.softmax(logits[:, left:right], dim=-1)
    return output


def policy_records(logits: torch.Tensor, offsets: list[int], slices: list[tuple[int, int]], gold: list[list[int]], scored: list[int], policy: str):
    probs = probabilities(logits, slices)
    records = []
    for row, (left, right) in enumerate(zip(offsets[:-1], offsets[1:])):
        row_logits, row_probs = logits[left:right], probs[left:right]
        full_pred = prediction_from_flat(row_logits[0], slices)
        full_cstar = polarity_argmax_from_flat(row_logits[0], slices)
        full_margin = []
        for lft, rgt in slices:
            slot = row_logits[0, lft:rgt]; full_margin.append(float(slot[1:].max() - slot[0]) if rgt - lft > 2 else float(slot[1] - slot[0]))
        c4_pred, c4_cstar, c4_margin, selected, ties = select_c4(row_probs, row_logits, slices, right - left)
        records.append({"gold": gold[row], "full_prediction": full_pred, "full_cstar": full_cstar,
                        "full_margin": full_margin, "c4_prediction": c4_pred, "c4_cstar": c4_cstar,
                        "c4_margin": c4_margin, "selected_reader": selected, "ties": ties,
                        "reader_count": right - left})
    return records


def metrics_for(records: list[dict[str, Any]], scored: list[int], prediction_key: str, cstar_key: str, margin_key: str):
    full_slots = [{"gold": int(row["gold"][aspect]), "cstar": int(row[cstar_key][aspect]), "margin": float(row[margin_key][aspect])}
                  for row in records for aspect in scored]
    return {"delta0": delta0(full_slots), "curve": curve(full_slots),
            "errors": error_summary(records, scored, prediction_key)}


def run_task(task: str, device: torch.device, smoke_rows: int = 16, smoke_only: bool = False) -> dict[str, Any]:
    start_time = time.time()
    model, tokenizer, bundle, counts, checkpoint, model_dir = load_checkpoint("student_e8", task, device)
    validation = tr.split_field(bundle, "validation")
    texts = list(tr.split_texts(validation)); ids = [str(x) for x in tr.split_ids(validation)]
    gold = [list(x) for x in tr.model_labels(bundle, "validation", validation)]
    scored = list(range(tr.task_num_aspects(bundle)))
    slices = slot_slices(counts, sum(int(x) for x in counts))
    capture = FullCapture(model)
    smoke: dict[str, Any] = {}
    all_shadow_logits: list[torch.Tensor] = []
    all_offsets = [0]
    all_metadata: list[list[dict[str, Any]]] = []
    state_cache: dict[str, list[list[torch.Tensor]]] = {"hidden_inputs": [[] for _ in model.encoder.encoder.layer], "content_k": [[] for _ in model.encoder.encoder.layer], "content_v": [[] for _ in model.encoder.encoder.layer], "full_outputs": [[] for _ in model.encoder.encoder.layer]}
    full_parity_layer: list[float] = []
    full_parity_logits = 0.0
    argmax_mismatches = 0
    cache_unchanged = True
    fp32_smoke: dict[str, Any] = {}
    for batch_start in range(0, len(texts), 16):
        batch_texts = texts[batch_start:batch_start + 16]
        encoded, attention, reader_rows, reader_keeps, metadata = build_reader_batch(tokenizer, batch_texts)
        if batch_start == 0:
            fp_full_logits, fp_hidden, fp_outputs, fp_k, fp_v = full_pass(model, capture, encoded["input_ids"], attention, device, use_bf16=False)
            fp_reader_logits, fp_trajectory = shadow_pass(model, fp_hidden, fp_outputs, fp_k, fp_v, attention, reader_rows, reader_keeps, device, use_bf16=False)
            fp_reader_logits = flatten_uniform(fp_reader_logits)
            fp_layer_errors = [0.0] * len(model.encoder.encoder.layer)
            fp_logit_error = 0.0
            fp_argmax_mismatch = 0
            cursor_fp = 0
            fp_full_flat = flatten_uniform(fp_full_logits)
            for row, meta in enumerate(metadata):
                fp_logit_error = max(fp_logit_error, float((fp_reader_logits[cursor_fp] - fp_full_flat[row]).abs().max()))
                fp_argmax_mismatch += sum(int(torch.argmax(fp_reader_logits[cursor_fp, left:right]) != torch.argmax(fp_full_flat[row, left:right])) for left, right in slices)
                for layer_index, trajectory in enumerate(fp_trajectory):
                    fp_layer_errors[layer_index] = max(fp_layer_errors[layer_index], float((trajectory[cursor_fp].float().cpu() - fp_outputs[layer_index][row].float().cpu()).abs().max()))
                cursor_fp += len(meta)
            fp32_smoke = {"layer_max_abs": fp_layer_errors, "logit_max_abs": fp_logit_error, "argmax_mismatches": fp_argmax_mismatch,
                          "gate_pass": max(fp_layer_errors + [fp_logit_error]) <= 1e-4 and fp_argmax_mismatch == 0}
            if not fp32_smoke["gate_pass"]:
                raise RuntimeError(f"FP32 full-mask parity gate failed: {fp32_smoke}")
        full_logits, hidden, full_outputs, content_k, content_v = full_pass(model, capture, encoded["input_ids"], attention, device, use_bf16=True)
        before_k = {key: value.clone() for key, value in content_k.items()}
        before_v = {key: value.clone() for key, value in content_v.items()}
        reader_logits, parity_layers = shadow_pass(model, hidden, full_outputs, content_k, content_v, attention, reader_rows, reader_keeps, device, use_bf16=True)
        reader_logits = flatten_uniform(reader_logits)
        all_shadow_logits.append(reader_logits)
        for meta in metadata:
            all_offsets.append(all_offsets[-1] + len(meta))
        all_metadata.extend(metadata)
        for index in range(len(model.encoder.encoder.layer)):
            state_cache["hidden_inputs"][index].append(hidden[index].detach().cpu()[..., :])
            state_cache["content_k"][index].append(content_k[index].detach().cpu())
            state_cache["content_v"][index].append(content_v[index].detach().cpu())
            state_cache["full_outputs"][index].append(full_outputs[index].detach().cpu())
        # Full-mask shadow is reader index 0 for every source row.
        cursor = 0
        layer_errors = [0.0] * len(model.encoder.encoder.layer)
        for row, meta in enumerate(metadata):
            count = len(meta); shadow_first = reader_logits[cursor]
            full_row = flatten_uniform(full_logits)[row]
            full_parity_logits = max(full_parity_logits, float((shadow_first - full_row).abs().max()))
            argmax_mismatches += sum(int(torch.argmax(shadow_first[left:right]) != torch.argmax(full_row[left:right])) for left, right in slices)
            for layer_index, trajectory in enumerate(parity_layers):
                expected = full_outputs[layer_index][row].float().cpu()
                layer_errors[layer_index] = max(layer_errors[layer_index], float((trajectory[cursor].float() .cpu() - expected).abs().max()))
            cursor += count
        full_parity_layer = [max(old, new) for old, new in zip(full_parity_layer or [0.0] * len(layer_errors), layer_errors)]
        # Shadow never writes to content caches; this is an explicit fixture check.
        cache_unchanged = cache_unchanged and all(torch.equal(before_k[k], content_k[k]) and torch.equal(before_v[k], content_v[k]) for k in before_k)
        if batch_start == 0:
            for row_index, meta in enumerate(metadata):
                first_length = int(attention[row_index].sum())
                keep = meta[0]["keep_token_mask"]
                if not bool(keep[first_length - 1]) or any(keep[first_length:]):
                    raise AssertionError("full reader mask violates EOS/padding contract")
            smoke = {"rows": min(smoke_rows, len(batch_texts)), "full_mask_layer_max_abs": full_parity_layer,
                     "full_mask_logit_max_abs": full_parity_logits, "full_mask_argmax_mismatches": argmax_mismatches,
                     "content_cache_unchanged": cache_unchanged, "different_masks_present": len(metadata[0]) > 1,
                     "fp32_parity": fp32_smoke,
                     "bf16_gate_pass": bool(all(torch.isfinite(torch.tensor(value)) for value in full_parity_layer + [full_parity_logits]) and argmax_mismatches == 0),
                     "bf16_shape_sensitive_numeric_caveat": "FP32 parity is the correctness gate; BF16 shadow/full GEMM shapes may differ numerically."}
            if not smoke["bf16_gate_pass"] or not cache_unchanged:
                raise RuntimeError(f"BF16 smoke gate failed: {smoke}")
            if smoke_only:
                capture.restore()
                return {"protocol": "shadow-reader-diagnostic-v1", "created_at": now(), "model": "student_e8", "task": task,
                        "validation_only": True, "smoke_only": True, "smoke": smoke,
                        "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint)}
    capture.restore()
    shadow_logits = torch.cat(all_shadow_logits, 0)
    shadow_probs = probabilities(shadow_logits, slices)
    # Reuse the verified recomputed-reader cache for the original C4 policy.
    reader_cache = READER_OUT / "cache" / f"student_e8__{task}__readers.pt"
    reader_index_path = READER_OUT / "cache" / f"student_e8__{task}__index.json"
    reader_report_path = READER_OUT / f"student_e8__{task}.json"
    reader_index = json.loads(reader_index_path.read_text(encoding="utf-8"))
    reader_report = json.loads(reader_report_path.read_text(encoding="utf-8"))
    if reader_report.get("checkpoint_sha256") != sha256(checkpoint):
        raise RuntimeError("reader report checkpoint provenance mismatch")
    index_rows = reader_index.get("rows", [])
    if [str(row.get("id")) for row in index_rows] != ids or [row.get("gold") for row in index_rows] != gold:
        raise RuntimeError("reader cache IDs/gold do not match current frozen validation split")
    if [row.get("readers") for row in index_rows] != all_metadata:
        raise RuntimeError("reader cache clause masks do not match current clause masks")
    reader_state = torch.load(reader_cache, map_location="cpu", weights_only=False)
    original_logits = reader_state["logits"].float()
    offsets = [int(x) for x in reader_state["row_offsets"]]
    if offsets != all_offsets or reader_state.get("provenance", {}).get("checkpoint_sha256") != sha256(checkpoint):
        raise RuntimeError("reader tensor cache offsets/checkpoint provenance mismatch")
    original_records = policy_records(original_logits, offsets, slices, gold, scored, "original")
    shadow_records = policy_records(shadow_logits, all_offsets, slices, gold, scored, "shadow")
    report = {"protocol": "shadow-reader-diagnostic-v1", "created_at": now(), "model": "student_e8", "task": task,
              "validation_only": True, "encoder_forward": True, "checkpoint": str(checkpoint.relative_to(ROOT)),
              "checkpoint_sha256": sha256(checkpoint), "rows": len(texts), "ids": ids,
              "text_sha256": [hashlib.sha256(text.encode()).hexdigest() for text in texts], "scored_aspects": scored,
              "precision": "outer BF16; manual attention FP32; shadow layer modules follow eval BF16 path",
              "smoke": smoke, "full_mask_parity": {"fp32": fp32_smoke, "layer_max_abs": full_parity_layer, "logit_max_abs": full_parity_logits,
                                                        "argmax_mismatches": argmax_mismatches},
              "content_cache_unchanged": cache_unchanged,
              "policies": {"full_shadow": metrics_for(shadow_records, scored, "full_prediction", "full_cstar", "full_margin"),
                           "c4_original": metrics_for(original_records, scored, "c4_prediction", "c4_cstar", "c4_margin"),
                           "c4_shadow": metrics_for(shadow_records, scored, "c4_prediction", "c4_cstar", "c4_margin")},
              "runtime": {"elapsed_s": time.time() - start_time, "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
                          "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device)},
              "source_hashes": {"shadow_script": sha256(Path(__file__)), "reader_script": sha256(Path(__file__).resolve().parent / "reader_operating_point.py")}}
    OUT.mkdir(parents=True, exist_ok=True)
    cache_dir = OUT / "cache"; cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"shadow_logits": shadow_logits, "shadow_probabilities": shadow_probs, "row_offsets": all_offsets,
                "reader_index": all_metadata, "slices": slices, "provenance": report["source_hashes"]},
               cache_dir / f"student_e8__{task}__shadow_readers.pt")
    # Full stack caches are kept batch-wise to avoid padding/truncation rewrites.
    torch.save(state_cache,
               cache_dir / f"student_e8__{task}__full_stack.pt")
    write_json(OUT / f"student_e8__{task}.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    reports = []
    for task in args.tasks:
        reports.append(run_task(task, device, smoke_only=args.smoke_only))
    if args.smoke_only:
        write_json(OUT / "smoke_summary.json", {"protocol": "shadow-reader-diagnostic-v1", "created_at": now(), "reports": reports})
        return 0
    write_json(OUT / "summary.json", {"protocol": "shadow-reader-diagnostic-v1", "created_at": now(), "reports": reports})
    lines = ["# Shadow reader diagnostic", "", "Validation-only shared-content-stack comparison; no training or test access.", "", "| Task | Rows | Full-shadow F1 | Original C4 F1 | Shadow C4 F1 | Full-mask logit max abs |", "|---|---:|---:|---:|---:|---:|"]
    for report in reports:
        lines.append(f"| {report['task']} | {report['rows']} | {100*report['policies']['full_shadow']['delta0']['f1']:.3f}% | {100*report['policies']['c4_original']['delta0']['f1']:.3f}% | {100*report['policies']['c4_shadow']['delta0']['f1']:.3f}% | {report['full_mask_parity']['logit_max_abs']:.6g} |")
    (OUT / "FACTUAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
