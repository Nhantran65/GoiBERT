"""E13/E15 architecture-transfer audit.

This runner is deliberately measurement-only.  It loads the twenty locked
validation checkpoints, runs three inference variants, records the block-6/7
fusion tensors on the first 128 validation rows, and measures a fresh CE
backward probe without an optimizer step.  Test data and training are never
loaded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUT = ROOT / "artifacts" / "architecture_transfer_audit"
SEED = 42
TASKS = ("vsfc_sentiment", "vsfc_topic", "visfd", "absa_restaurant", "absa_hotel")
VARIANTS = ("centered", "uncentered")
FAMILIES = ("e13", "e15")
MODES = ("native", "branch_off", "fp32_fusion")
MICRO_BATCH = 16
TELEMETRY_ROWS = 128
MAX_LENGTH = 256
PROTOCOL = "architecture-transfer-audit-e13-e15-v1"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from goibert import train as tr  # noqa: E402
from goibert import data as data_mod  # noqa: E402
from goibert_e13.modeling import build_e13_student  # noqa: E402
from goibert_e15.modeling import build_e15_classifier  # noqa: E402


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jdefault(value: Any) -> Any:
    return tr._json_default(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=jdefault) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object in {path}")
    return value


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hash_paths(paths: Sequence[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(paths):
        if path.is_file():
            h.update(str(path.resolve().relative_to(ROOT)).encode())
            h.update(sha256(path).encode())
    return h.hexdigest()


def source_hash() -> str:
    paths = [Path(__file__).resolve(), SRC / "goibert" / "train.py", SRC / "goibert" / "data.py", SRC / "goibert" / "metrics.py", *sorted((SRC / "goibert_e13").glob("*.py")), *sorted((SRC / "goibert_e15").glob("*.py"))]
    return hash_paths(paths)


def task_contract(task: str) -> dict[str, Any]:
    # include_test=False is an intentional hard boundary for this script.
    bundle = tr.load_task_bundle(task, ROOT, include_test=False)
    train_split = tr.split_field(bundle, "train")
    valid_split = tr.split_field(bundle, "validation")
    return {
        "bundle": bundle,
        "train": train_split,
        "validation": valid_split,
        "num_aspects": tr.task_num_aspects(bundle),
        "aspect_class_counts": tr.task_aspect_class_counts(bundle),
        "num_classes": tr.task_num_classes(bundle),
    }


def checkpoint_path(family: str, variant: str, task: str) -> Path:
    if family == "e13":
        return ROOT / "artifacts" / "e13" / variant / task / "seed_42" / "best_checkpoint.pt"
    return ROOT / "artifacts" / "e15" / "trials" / variant / task / "seed_42" / "best_checkpoint.pt"


def artifact_dir(family: str, variant: str, task: str) -> Path:
    return OUT / "per_model" / family / variant / task / "seed_42"


def model_and_tokenizer(family: str, variant: str, contract: Mapping[str, Any]):
    if family == "e13":
        model_dir = ROOT / "models" / "goiturn-s-gd"
        model, spec = build_e13_student(model_dir, num_classes=contract["num_classes"], num_aspects=contract["num_aspects"], aspect_class_counts=contract["aspect_class_counts"] if contract["num_aspects"] else None, variant=variant, attention_implementation="sdpa")
    else:
        model_dir = ROOT / "models" / "bamibert-base-original"
        model, spec = build_e15_classifier(model_dir, num_classes=contract["num_classes"], num_aspects=contract["num_aspects"], aspect_class_counts=contract["aspect_class_counts"] if contract["num_aspects"] else None, variant=variant, attention_implementation="sdpa")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    return model, spec, tokenizer, model_dir


def label_overlay() -> dict[str, Any]:
    return {
        "absa_raw_integer_labels": {"0": "absent", "1": "positive", "2": "negative", "3": "neutral"},
        "absa_source_loader_note": "source file integer codes are preserved verbatim in gold; names are the requested audit overlay",
        "visfd_others": {"raw_0": "absent", "raw_4": "present_detection_only", "model_0": "absent", "model_1": "present_detection_only"},
        "numeric_gold_unchanged": True,
    }


def loader_for(family: str, tokenizer: Any, contract: Mapping[str, Any], max_length: int, split_name: str = "validation") -> Any:
    split = contract[split_name]
    return tr._make_loader(tokenizer, split, max_length=max_length, batch_size=MICRO_BATCH, shuffle=False, seed=SEED, labels=tr.model_labels(contract["bundle"], split_name, split), gold_labels=tr.split_labels(split))


def raw_prediction(task: str, prediction: Any) -> Any:
    if task == "visfd":
        return data_mod.visfd_from_model_labels([prediction])[0]
    return prediction


def metric_and_rows(task: str, bundle: Any, ids: list[Any], gold: list[Any], pred_model: list[Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics = tr.evaluate_predictions(task, bundle, gold, pred_model, split="validation")
    rows = []
    for row_id, g, p in zip(ids, gold, pred_model):
        rows.append({"id": row_id, "gold": g, "prediction": raw_prediction(task, p), "prediction_model": p})
    return metrics, rows


def _dtype(t: torch.Tensor) -> str:
    return str(t.dtype).replace("torch.", "")


class Telemetry:
    """Native tensor audit plus same-input output comparisons."""
    def __init__(self, limit: int = TELEMETRY_ROWS) -> None:
        self.limit = limit
        self.rows_seen = 0
        self.batch_shapes: list[list[int]] = []
        self.dtypes: dict[str, str] = {}
        self.sums = {"x2": 0.0, "d2": 0.0, "h2": 0.0, "eff2": 0.0, "err2": 0.0}
        self.counts = {"tokens": 0, "coords": 0, "d_nonzero": 0, "d_zero": 0, "swallowed": 0, "err_tokens": 0}
        self.ratios: dict[str, list[float]] = {"d_over_x": [], "d_over_h": [], "relative_error": []}
        self.native_ln: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.output_stats: dict[str, dict[str, float | int]] = {m: {"tokens": 0, "diff2": 0.0, "ref2": 0.0, "max_abs": 0.0} for m in ("branch_off", "fp32_fusion")}
        self.raw: dict[str, list[torch.Tensor]] = {k: [] for k in ("x", "d", "h", "y", "ln_input", "attention_mask")}
        self.compare_index = 0

    def start_mode(self) -> None:
        self.rows_seen = 0
        self.compare_index = 0

    def record(self, mode: str, hidden: torch.Tensor, x: torch.Tensor, d: torch.Tensor, y: torch.Tensor, ln_input: torch.Tensor, ln_output: torch.Tensor, attention_mask: torch.Tensor) -> None:
        if self.rows_seen >= self.limit:
            return
        take = min(int(attention_mask.shape[0]), self.limit - self.rows_seen)
        mask = attention_mask[:take].detach().to(dtype=torch.bool).cpu()
        self.batch_shapes.append([int(take), int(attention_mask.shape[1])]) if mode == "native" else None
        if mode == "native":
            self.dtypes.update({"x": _dtype(x), "d": _dtype(d), "y": _dtype(y), "h": _dtype(hidden), "ln_input": _dtype(ln_input)})
            self.raw["x"].append(x[:take].detach().cpu())
            self.raw["d"].append(d[:take].detach().cpu())
            self.raw["h"].append(hidden[:take].detach().cpu())
            self.raw["y"].append(y[:take].detach().cpu())
            self.raw["ln_input"].append(ln_input[:take].detach().cpu())
            self.raw["attention_mask"].append(mask)
            self.native_ln.append((ln_output[:take].detach().float().cpu(), mask))
            xf, df, hf, yf = (v[:take].detach().float().cpu() for v in (x, d, hidden, y))
            valid = mask.unsqueeze(-1).expand_as(df)
            dnon = (df != 0) & valid
            dzero = (df == 0) & valid
            sw = dnon & (yf == xf)
            eff = yf - xf
            err = eff - df
            self.counts["tokens"] += int(mask.sum())
            self.counts["coords"] += int(valid.sum())
            self.counts["d_nonzero"] += int(dnon.sum())
            self.counts["d_zero"] += int(dzero.sum())
            self.counts["swallowed"] += int(sw.sum())
            self.sums["x2"] += float((xf[valid] ** 2).sum())
            self.sums["d2"] += float((df[valid] ** 2).sum())
            self.sums["h2"] += float((hf[valid] ** 2).sum())
            self.sums["eff2"] += float((eff[valid] ** 2).sum())
            self.sums["err2"] += float((err[dnon] ** 2).sum())
            for vals, den, name in ((torch.linalg.vector_norm(df, dim=-1), torch.linalg.vector_norm(xf, dim=-1), "d_over_x"), (torch.linalg.vector_norm(df, dim=-1), torch.linalg.vector_norm(hf, dim=-1), "d_over_h"), (torch.linalg.vector_norm(err, dim=-1), torch.linalg.vector_norm(df, dim=-1), "relative_error")):
                finite = valid[..., 0] & (den > 0) & torch.isfinite(vals) & torch.isfinite(den)
                if name == "relative_error":
                    finite &= (torch.linalg.vector_norm(df, dim=-1) > 0)
                ratio = (vals[finite] / den[finite]).tolist()
                self.ratios[name].extend(float(v) for v in ratio)
        else:
            if self.compare_index >= len(self.native_ln):
                raise RuntimeError(f"{mode} telemetry alignment missing native chunk")
            ref, refmask = self.native_ln[self.compare_index]
            got = ln_output[:take].detach().float().cpu()
            common = mask & refmask
            diff = got - ref
            vals = diff[common.unsqueeze(-1).expand_as(diff)]
            rvals = ref[common.unsqueeze(-1).expand_as(ref)]
            self.output_stats[mode]["tokens"] += int(common.sum())
            self.output_stats[mode]["diff2"] += float((vals ** 2).sum())
            self.output_stats[mode]["ref2"] += float((rvals ** 2).sum())
            self.output_stats[mode]["max_abs"] = max(float(self.output_stats[mode]["max_abs"]), float(vals.abs().max()) if vals.numel() else 0.0)
            self.compare_index += 1
        self.rows_seen += take

    def summary(self) -> dict[str, Any]:
        def q(values: list[float]) -> dict[str, Any]:
            if not values:
                return {"q10": None, "q50": None, "q90": None, "q99": None, "count": 0}
            t = torch.tensor(values, dtype=torch.float64)
            return {"q10": float(torch.quantile(t, .10)), "q50": float(torch.quantile(t, .50)), "q90": float(torch.quantile(t, .90)), "q99": float(torch.quantile(t, .99)), "count": len(values)}
        xnorm = self.sums["x2"] ** .5
        dnorm = self.sums["d2"] ** .5
        hnorm = self.sums["h2"] ** .5
        return {
            "rows": min(self.rows_seen, self.limit), "batch_shapes": self.batch_shapes, "dtypes": self.dtypes, "counts": self.counts,
            "frobenius": {"d_over_x": dnorm / xnorm if xnorm else None, "d_over_h": dnorm / hnorm if hnorm else None, "x_norm": xnorm, "d_norm": dnorm, "h_norm": hnorm, "effective_norm": self.sums["eff2"] ** .5, "effective_minus_intended_norm": self.sums["err2"] ** .5},
            "token_quantiles": {name: q(vals) for name, vals in self.ratios.items()},
            "coordinate_swallowing": {"nonzero_delta_coordinates": self.counts["d_nonzero"], "swallowed_coordinates": self.counts["swallowed"], "fraction_among_nonzero": self.counts["swallowed"] / self.counts["d_nonzero"] if self.counts["d_nonzero"] else None, "zero_delta_coordinates_excluded": self.counts["d_zero"]},
            "after_ln_same_input": {mode: {**stats, "rms_diff": (stats["diff2"] / max(1, stats["tokens"])) ** .5, "relative_rms_to_native": (stats["diff2"] / max(stats["ref2"], 1e-30)) ** .5} for mode, stats in self.output_stats.items()},
        }

    def save_raw(self, path: Path) -> None:
        torch.save({"tensors": self.raw, "native_ln": self.native_ln, "summary": self.summary()}, path)


def install_attention_hook(model: Any, family: str, mode: str, telemetry: Telemetry | None):
    index = 6 if family == "e13" else 7
    attention = model.encoder.encoder.layer[index].attention
    original = attention.forward

    def wrapped(self_attention: Any, hidden_states: torch.Tensor, attention_mask: Any = None, head_mask: Any = None, encoder_hidden_states: Any = None, encoder_attention_mask: Any = None, past_key_value: Any = None, output_attentions: Any = False):
        outputs = self_attention.self(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)
        context, delta = outputs[:2]
        x = self_attention.output.dense(context)
        if mode == "branch_off":
            y = x
        elif mode == "fp32_fusion":
            y = x.float() + delta.float()
        else:
            y = x + delta
        dropped = self_attention.output.dropout(y)
        ln_input = dropped + hidden_states
        out = self_attention.output.LayerNorm(ln_input)
        if telemetry is not None:
            telemetry.record(mode, hidden_states, x, delta, y, ln_input, out, getattr(model, "_audit_attention_mask", torch.ones(hidden_states.shape[:2], device=hidden_states.device, dtype=torch.long)))
        return (out,)

    attention.forward = types.MethodType(wrapped, attention)
    return attention, original


def load_checkpoint(model: Any, path: Path) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping) or "model" not in state:
        raise RuntimeError(f"Malformed checkpoint {path}")
    model.load_state_dict(state["model"], strict=True)
    return dict(state)


def run_mode(model: Any, mode: str, task: str, contract: Mapping[str, Any], loader: Any, device: torch.device, telemetry: Telemetry | None) -> tuple[dict[str, Any], list[dict[str, Any]], torch.Tensor, float]:
    if telemetry is not None:
        telemetry.start_mode()
    model.eval()
    ids: list[Any] = []
    gold: list[Any] = []
    pred: list[Any] = []
    logits: list[torch.Tensor] = []
    started = time.time()
    attn, original = install_attention_hook(model, "e13" if model.encoder.config.hidden_size == 384 else "e15", mode, telemetry)
    try:
        with torch.inference_mode():
            for batch in loader:
                moved = tr._move_batch(batch, device)
                model._audit_attention_mask = moved["attention_mask"]
                with tr.make_autocast("bf16", device):
                    out = model(moved["input_ids"], moved["attention_mask"], token_type_ids=moved.get("token_type_ids"))
                logits.append(out.detach().float().cpu())
                pred.extend(tr._predictions(out, aspect_class_counts=contract["aspect_class_counts"]))
                gold.extend(batch["gold_labels"].tolist())
                ids.extend(batch["ids"])
        all_logits = torch.cat(logits, dim=0)
    finally:
        attn.forward = original
        if hasattr(model, "_audit_attention_mask"):
            delattr(model, "_audit_attention_mask")
    metrics, rows = metric_and_rows(task, contract["bundle"], ids, gold, pred)
    return metrics, rows, all_logits, time.time() - started


def prediction_disagreements(task: str, native: list[dict[str, Any]], other: list[dict[str, Any]]) -> dict[str, int]:
    row_count = sum(a["prediction"] != b["prediction"] for a, b in zip(native, other))
    if task in {"visfd", "absa_restaurant", "absa_hotel"}:
        slots = sum(sum(x != y for x, y in zip(a["prediction"], b["prediction"])) for a, b in zip(native, other))
    else:
        slots = row_count
    return {"rows": int(row_count), "slots_or_labels": int(slots)}


def logit_compare(task: str, native: torch.Tensor, other: torch.Tensor, counts: Sequence[int] = ()) -> dict[str, float]:
    diff = (other - native).abs()
    result = {"max_abs_logit_difference": float(diff.max()), "mean_abs_logit_difference": float(diff.mean())}
    if native.ndim == 2 and counts and sum(int(v) for v in counts) == native.shape[-1]:
        pieces: list[torch.Tensor] = []
        offset = 0
        for width in counts:
            width = int(width)
            a, b = native[:, offset:offset + width], other[:, offset:offset + width]
            pieces.append((F.kl_div(F.log_softmax(b, -1), F.softmax(a, -1), reduction="batchmean") + F.kl_div(F.log_softmax(a, -1), F.softmax(b, -1), reduction="batchmean")) / 2)
            offset += width
        result["symmetric_kl"] = float(torch.stack(pieces).mean()) if pieces else 0.0
    else:
        result["symmetric_kl"] = float((F.kl_div(F.log_softmax(other, -1), F.softmax(native, -1), reduction="batchmean") + F.kl_div(F.log_softmax(native, -1), F.softmax(other, -1), reduction="batchmean")) / 2)
    return result


def group_norms(state: Mapping[str, torch.Tensor], layer: int) -> dict[str, Any]:
    def norm(keys: Sequence[str]) -> float:
        return math.sqrt(sum(float(state[k].float().pow(2).sum()) for k in keys)) if keys else 0.0
    branch = {name: norm([k for k in state if k.endswith(f".{name}.weight") and ".attention.self." in k]) for name in ("U", "W", "P")}
    wo = [k for k in state if f"encoder.encoder.layer.{layer}.attention.output.dense" in k]
    ffn = [k for k in state if f"encoder.encoder.layer.{layer}.intermediate.dense" in k or f"encoder.encoder.layer.{layer}.output.dense" in k]
    return {"branch_norms": branch, "p_nonzero": bool(any(torch.count_nonzero(state[k]).item() for k in state if k.endswith(".P.weight"))), "p_nonzero_count": int(sum(torch.count_nonzero(state[k]).item() for k in state if k.endswith(".P.weight"))), "original_Wo_norm": norm(wo), "original_FFN_norm": norm(ffn), "Wo_keys": wo, "FFN_keys": ffn}


def drift_report(family: str, variant: str, task: str, state: Mapping[str, torch.Tensor], contract: Mapping[str, Any], source_dir: Path) -> dict[str, Any]:
    layer = 6 if family == "e13" else 7
    report = group_norms(state, layer)
    source_file = source_dir / "model.safetensors"
    try:
        from safetensors.torch import load_file
        ref = load_file(str(source_file), device="cpu")
        drift: dict[str, float | None] = {}
        initial = initialize_branch(family, variant, contract)
        for n in ("U", "W"):
            keys = [k for k in state if k.endswith(f".{n}.weight")]
            fresh = torch.cat([initial[k].float().reshape(-1) for k in initial if k.endswith(f".{n}.weight")])
            cur = torch.cat([state[k].float().reshape(-1) for k in keys])
            denom = float(fresh.float().norm())
            drift[n] = float((cur - fresh.float().reshape(-1)).norm()) / denom if denom else None
        for group, patterns in {"Wo": [f"encoder.encoder.layer.{layer}.attention.output.dense"], "FFN": [f"encoder.encoder.layer.{layer}.intermediate.dense", f"encoder.encoder.layer.{layer}.output.dense"]}.items():
            keys: list[tuple[str, str]] = []
            for current_key in state:
                if not any(pattern in current_key for pattern in patterns):
                    continue
                if family == "e13":
                    ref_key = current_key.replace("encoder.encoder.", "encoder.", 1)
                else:
                    ref_key = current_key.replace("encoder.encoder.", "roberta.encoder.", 1)
                if ref_key in ref:
                    keys.append((current_key, ref_key))
            num = math.sqrt(sum(float((state[k].float() - ref[r].float()).pow(2).sum()) for k, r in keys))
            den = math.sqrt(sum(float(ref[r].float().pow(2).sum()) for _, r in keys))
            report[f"relative_{group}_displacement_vs_source"] = num / den if den else None
        report["relative_UW_drift_vs_reconstructed_init"] = drift
        report["initializer"] = {"status": "reconstruction_assumption", "hash": initializer_hash(family, variant, contract), "reason": "frozen checkpoints contain no init snapshot; seed-42 constructor path was replayed but not independently proven"}
        report["source_model_sha256"] = sha256(source_file)
    except Exception as exc:
        report["initializer"] = {"status": "unavailable", "error": repr(exc)}
    return report


def initialize_branch(family: str, variant: str, contract: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    torch.manual_seed(SEED)
    model, _, _, _ = model_and_tokenizer(family, variant, contract)
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if name.endswith((".U.weight", ".W.weight", ".P.weight"))}


def initializer_hash(family: str, variant: str, contract: Mapping[str, Any]) -> str:
    values = initialize_branch(family, variant, contract)
    h = hashlib.sha256()
    for key in sorted(values):
        h.update(key.encode()); h.update(values[key].numpy().tobytes())
    return h.hexdigest()


def gradient_probe(family: str, variant: str, task: str, contract: Mapping[str, Any], checkpoint: Path, tokenizer: Any, gradient_device: str) -> dict[str, Any]:
    model, spec, _, _ = model_and_tokenizer(family, variant, contract)
    state = load_checkpoint(model, checkpoint)
    device = torch.device(gradient_device)
    model.to(device).eval()
    loader = loader_for(family, tokenizer, contract, MAX_LENGTH, split_name="train")
    batch = next(iter(loader))
    moved = tr._move_batch(batch, device)
    model.zero_grad(set_to_none=True)
    started = time.time()
    logits = model(moved["input_ids"], moved["attention_mask"], token_type_ids=moved.get("token_type_ids"))
    loss = tr.compute_loss(logits, moved["labels"], num_aspects=contract["num_aspects"], aspect_class_counts=contract["aspect_class_counts"])
    loss.backward()
    groups = {"U": [], "W": [], "P": [], "Wo": [], "FFN": []}
    layer = 6 if family == "e13" else 7
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if name.endswith(".U.weight"): groups["U"].append((name, parameter))
        elif name.endswith(".W.weight"): groups["W"].append((name, parameter))
        elif name.endswith(".P.weight"): groups["P"].append((name, parameter))
        elif f"encoder.encoder.layer.{layer}.attention.output.dense" in name: groups["Wo"].append((name, parameter))
        elif f"encoder.encoder.layer.{layer}.intermediate.dense" in name or f"encoder.encoder.layer.{layer}.output.dense" in name: groups["FFN"].append((name, parameter))
    result: dict[str, Any] = {"objective": "cross_entropy_probe", "training_objective_context": "E13 checkpoints trained with CE+NC; E15 with CE", "device": str(device), "precision": "fp32", "batch_size": len(batch["ids"]), "no_optimizer_step": True, "loss": float(loss.detach().cpu()), "elapsed_seconds": time.time() - started, "groups": {}}
    for group, entries in groups.items():
        weight2 = sum(float(p.detach().float().pow(2).sum()) for _, p in entries)
        grad2 = sum(float(p.grad.detach().float().pow(2).sum()) for _, p in entries)
        result["groups"][group] = {"parameter_names": [n for n, _ in entries], "weight_norm": weight2 ** .5, "raw_grad_norm": grad2 ** .5, "grad_over_weight": grad2 ** .5 / weight2 ** .5 if weight2 else None, "finite": all(bool(torch.isfinite(p.grad).all()) for _, p in entries), "zero_gradient": grad2 == 0.0}
    return result


def source_protocol_table() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dimensions = [
        ("backbone", "goiturn-s-gd", "goiturn-s-gd", "BamiBERT original", "BamiBERT original"),
        ("head", "CLS dense384 tanh dropout0.1", "CLS dense384 tanh dropout0.1", "CLS dense768 tanh dropout0.1", "CLS dense768 tanh dropout0.1"),
        ("dropout", "training recipe dropout; eval disabled", "training recipe dropout; eval disabled", "training recipe dropout; eval disabled", "training recipe dropout; eval disabled"),
        ("optimizer", "AdamW + E7 LLRD gamma0.8", "AdamW + E7 LLRD gamma0.8", "AdamW no LLRD", "AdamW no LLRD"),
        ("rng_path", "seed42; E13 replacement under fork_rng", "seed42 fresh E8 student", "seed42; E15 replacement under fork_rng", "seed42 fresh E14 classifier"),
        ("loss", "CE + NC", "CE + NC", "plain CE", "plain CE"),
        ("train_batch", "effective32; micro32", "effective32; micro32", "effective32; micro16/32", "effective32; micro16/32"),
        ("precision_backend", "BF16/SDPA", "BF16/SDPA", "BF16/SDPA", "BF16/SDPA"),
        ("measurement_scope", "validation; audit fixed micro16", "validation training artifact", "validation; audit fixed micro16", "validation training artifact"),
    ]
    for d, e13, e8, e15, e14 in dimensions:
        rows.extend([{ "comparison": "E13_vs_E8", "dimension": d, "architecture": "E13", "value": e13, "reference": e8 }, { "comparison": "E15_vs_E14", "dimension": d, "architecture": "E15", "value": e15, "reference": e14 }])
    return rows


def compare_outputs(family: str, variant: str, task: str, results: Mapping[str, Any], counts: Sequence[int] = ()) -> dict[str, Any]:
    native = results["native"]
    out: dict[str, Any] = {"family": family, "variant": variant, "task": task, "native_metric": native["metrics"], "modes": {}}
    for mode in ("branch_off", "fp32_fusion"):
        m = results[mode]
        delta = {key: (float(m["metrics"][key]) - float(native["metrics"][key])) for key in ("selection_f1",) if key in m["metrics"] and key in native["metrics"]}
        out["modes"][mode] = {"metrics": m["metrics"], "metric_minus_native": delta, "prediction_disagreements": prediction_disagreements(task, native["rows"], m["rows"]), "logits": logit_compare(task, native["logits"], m["logits"], counts)}
    return out


def audit_validation_alignment(family: str, variant: str, task: str, results: Mapping[str, Any], contract: Mapping[str, Any], checkpoint: Path) -> dict[str, Any]:
    """Check every fresh pass against the frozen validation prediction rows."""
    source_path = checkpoint.parent / "validation_predictions.json"
    saved_metric_path = checkpoint.parent / "best_validation_metrics.json"
    if not source_path.is_file() or not saved_metric_path.is_file():
        raise FileNotFoundError(f"Missing frozen validation artifact beside {checkpoint}")
    source_rows = read_json(source_path)["rows"]
    expected_ids = [str(x["id"]) for x in source_rows]
    expected_gold = [x["gold"] for x in source_rows]
    expected_n = len(tr.split_field(contract["bundle"], "validation"))
    if len(source_rows) != expected_n:
        raise RuntimeError(f"Frozen validation row count mismatch: {len(source_rows)} != {expected_n}")
    modes: dict[str, Any] = {}
    for mode in MODES:
        rows = results[mode]["rows"]
        ids = [str(x["id"]) for x in rows]
        gold = [x["gold"] for x in rows]
        modes[mode] = {"n": len(rows), "n_expected": expected_n, "ids_match_frozen": ids == expected_ids, "gold_match_frozen": gold == expected_gold, "ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()}
        if not (modes[mode]["ids_match_frozen"] and modes[mode]["gold_match_frozen"] and len(rows) == expected_n):
            raise RuntimeError(f"Validation ID/gold/order mismatch in {family}/{variant}/{task}/{mode}")
    saved = read_json(saved_metric_path)
    native_metrics = results["native"]["metrics"]
    saved_selection = float(saved["selection_f1"])
    return {"source_validation_predictions": str(source_path), "source_best_validation_metrics": str(saved_metric_path), "source_rows": len(source_rows), "modes": modes, "native_selection_f1": float(native_metrics["selection_f1"]), "saved_best_selection_f1": saved_selection, "native_minus_saved_best_selection_f1": float(native_metrics["selection_f1"]) - saved_selection, "metric_definition": native_metrics.get("selection_metric")}


def run_job(family: str, variant: str, task: str, device_name: str, gradient_device: str = "cpu") -> dict[str, Any]:
    out = artifact_dir(family, variant, task); out.mkdir(parents=True, exist_ok=True)
    ckpt = checkpoint_path(family, variant, task)
    contract = task_contract(task)
    source_ckpt_sha = sha256(ckpt)
    status_path = out / "status.json"
    write_json(status_path, {"protocol": PROTOCOL, "status": "running", "started_at": now(), "family": family, "variant": variant, "task": task, "checkpoint": str(ckpt), "checkpoint_sha256_before": source_ckpt_sha, "device": device_name})
    device = torch.device(device_name)
    model, spec, tokenizer, source_dir = model_and_tokenizer(family, variant, contract)
    checkpoint_state = load_checkpoint(model, ckpt)
    model.to(device).eval()
    metadata = read_json(ckpt.parent / "metadata.json")
    max_length = int(metadata.get("max_length", MAX_LENGTH))
    loader = loader_for(family, tokenizer, contract, max_length)
    results: dict[str, Any] = {}
    telemetry = Telemetry()
    for mode in MODES:
        mode_json = out / f"{mode}.json"
        pred_json = out / f"{mode}_predictions.json"
        logits_path = out / f"{mode}_logits.pt"
        if mode_json.is_file() and pred_json.is_file() and logits_path.is_file():
            saved = read_json(mode_json); saved["rows"] = read_json(pred_json)["rows"]; saved["logits"] = torch.load(logits_path, map_location="cpu", weights_only=False); results[mode] = saved
            if mode == "native" and (out / "native_telemetry.pt").is_file():
                saved_telemetry = torch.load(out / "native_telemetry.pt", map_location="cpu", weights_only=False)
                telemetry.native_ln = saved_telemetry.get("native_ln", [])
            continue
        metrics, rows, logits, elapsed = run_mode(model, mode, task, contract, loader, device, telemetry)
        saved = {"mode": mode, "metrics": metrics, "elapsed_seconds": elapsed, "rows_count": len(rows), "rows": rows, "logits": logits}
        write_json(pred_json, {"rows": rows, "label_overlay": label_overlay(), "ids_order_sha256": hashlib.sha256(json.dumps([str(r["id"]) for r in rows], separators=(",", ":")).encode()).hexdigest()})
        torch.save(logits, logits_path)
        write_json(mode_json, {k: v for k, v in saved.items() if k not in {"rows", "logits"}})
        results[mode] = saved
        if mode == "native":
            telemetry.save_raw(out / "native_telemetry.pt")
            write_json(out / "native_telemetry.json", telemetry.summary())
        elif telemetry.native_ln:
            write_json(out / "after_ln_same_input.json", telemetry.summary().get("after_ln_same_input", {}))
        write_json(status_path, {"protocol": PROTOCOL, "status": "running", "completed_modes": [m for m in MODES if m in results], "family": family, "variant": variant, "task": task, "device": device_name})
    comparison = compare_outputs(family, variant, task, results, contract["aspect_class_counts"])
    write_json(out / "comparison.json", comparison)
    alignment = audit_validation_alignment(family, variant, task, results, contract, ckpt)
    write_json(out / "validation_alignment.json", alignment)
    drift = drift_report(family, variant, task, checkpoint_state["model"], contract, source_dir)
    write_json(out / "norms_drift.json", drift)
    # The gradient probe is independent of inference mode and uses the first
    # non-shuffled training batch of 16.  It performs backward only.
    try:
        grad = gradient_probe(family, variant, task, contract, ckpt, tokenizer, gradient_device)
    except Exception as exc:
        grad = {"status": "failed", "error": repr(exc), "traceback": traceback.format_exc(), "no_optimizer_step": True}
    write_json(out / "gradient_probe.json", grad)
    if sha256(ckpt) != source_ckpt_sha:
        raise RuntimeError(f"Source checkpoint changed during audit: {ckpt}")
    final_status = "partial" if grad.get("status") == "failed" else "completed"
    payload = {"protocol": PROTOCOL, "status": final_status, "finished_at": now(), "family": family, "variant": variant, "task": task, "checkpoint_sha256_before": source_ckpt_sha, "checkpoint_sha256_after": sha256(ckpt), "source_hash": source_hash(), "model_spec": spec.to_dict(), "source_metadata": metadata, "label_overlay": label_overlay(), "runtime": {"python": platform.python_version(), "torch": torch.__version__, "transformers": __import__("transformers").__version__, "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(), "precision": "bf16", "attention_implementation": "sdpa", "micro_batch_size": MICRO_BATCH, "max_length": max_length, "gradient_device": gradient_device}, "checkpoint_metadata_fields": {k: metadata.get(k) for k in ("learning_rate", "epochs", "best_epoch", "global_step", "train_loss_last", "best_selection_f1", "effective_batch_size", "micro_batch_size", "precision", "loss", "optimizer_coverage", "optimizer_group_contract")}, "modes": {m: {"metrics": results[m]["metrics"], "elapsed_seconds": results[m]["elapsed_seconds"]} for m in MODES}, "comparison": "comparison.json", "validation_alignment": alignment, "norms_drift": "norms_drift.json", "gradient_probe": "gradient_probe.json"}
    write_json(out / "provenance.json", payload)
    write_json(status_path, payload)
    return payload


def gpu_snapshot() -> dict[str, dict[str, int]]:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,memory.used", "--format=csv,noheader,nounits"], text=True, stderr=subprocess.STDOUT, timeout=30)
    except Exception:
        return {}
    result: dict[str, dict[str, int]] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            result[parts[0]] = {"free_mib": int(float(parts[1])), "used_mib": int(float(parts[2]))}
    return result


def jobs() -> list[tuple[str, str, str]]:
    return [(f, v, t) for f in FAMILIES for v in VARIANTS for t in TASKS]


def job_key(f: str, v: str, t: str) -> str:
    return f"{f}__{v}__{t}"


def job_done(f: str, v: str, t: str) -> bool:
    p = artifact_dir(f, v, t) / "status.json"
    if not p.is_file(): return False
    try: return read_json(p).get("status") == "completed"
    except Exception: return False


def run_controller(gradient_device: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "protocol.json", {"protocol": PROTOCOL, "created_at": now(), "source_hash": source_hash(), "families": FAMILIES, "variants": VARIANTS, "tasks": TASKS, "modes": MODES, "micro_batch_size": MICRO_BATCH, "telemetry_rows": TELEMETRY_ROWS, "include_test": False, "optimizer_step": False, "label_overlay": label_overlay(), "gpu_snapshot": gpu_snapshot(), "protocol_comparability": "protocol_comparability.csv/json"})
    pending = [(f, v, t) for f, v, t in jobs() if not job_done(f, v, t)]
    snap = gpu_snapshot()
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable in this process; host execution is required. nvidia-smi={snap}")
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count()) if str(i) in snap and snap[str(i)]["free_mib"] >= 12_000]
    if not devices:
        raise RuntimeError(f"No GPU passed 12000 MiB free guard; refusing CPU/all-GPU fallback. snapshot={snap}")
    state_path = OUT / "execution_state.json"
    running: dict[str, tuple[subprocess.Popen[str], str]] = {}
    free_devices = list(devices)
    completed = [job_key(*j) for j in jobs() if job_done(*j)]
    write_json(state_path, {"protocol": PROTOCOL, "status": "running", "started_at": now(), "pending": [job_key(*j) for j in pending], "completed": completed, "devices": devices, "gpu_snapshot": snap})
    while pending or running:
        while pending and free_devices and len(running) < 4:
            f, v, t = pending.pop(0); key = job_key(f, v, t); device = free_devices.pop(0)
            log = OUT / "logs" / f"{key}.log"; log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("a", encoding="utf-8", buffering=1)
            cmd = [sys.executable, str(Path(__file__).resolve()), "--job", f, v, t, "--device", device, "--gradient-device", gradient_device]
            proc = subprocess.Popen(cmd, cwd=ROOT, env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", "")}, stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, text=True)
            handle.close(); running[key] = (proc, device)
        for key, (proc, device) in list(running.items()):
            rc = proc.poll()
            if rc is None: continue
            if rc != 0: raise RuntimeError(f"worker failed {key}; see {OUT / 'logs' / (key + '.log')}")
            running.pop(key); free_devices.append(device); completed.append(key)
        write_json(state_path, {"protocol": PROTOCOL, "status": "running", "updated_at": now(), "pending": [job_key(*j) for j in pending], "running": list(running), "completed": completed, "devices": devices, "gpu_snapshot": gpu_snapshot()})
        if running: time.sleep(5)
    write_json(state_path, {"protocol": PROTOCOL, "status": "completed", "finished_at": now(), "completed": completed, "devices": devices})
    write_summary()


def write_summary() -> None:
    rows: list[dict[str, Any]] = []
    for f, v, t in jobs():
        p = artifact_dir(f, v, t) / "provenance.json"
        if not p.is_file(): continue
        x = read_json(p); comparison = read_json(artifact_dir(f, v, t) / "comparison.json") if (artifact_dir(f, v, t) / "comparison.json").is_file() else {}; rows.append({"family": f, "variant": v, "task": t, "status": x.get("status"), "native_selection_f1": x.get("modes", {}).get("native", {}).get("metrics", {}).get("selection_f1"), "branch_off_minus_native": comparison.get("modes", {}).get("branch_off", {}).get("metric_minus_native", {}).get("selection_f1"), "fp32_fusion_minus_native": comparison.get("modes", {}).get("fp32_fusion", {}).get("metric_minus_native", {}).get("selection_f1"), "artifact_dir": str(artifact_dir(f, v, t))})
    write_json(OUT / "summary.json", {"protocol": PROTOCOL, "status": "completed" if len(rows) == 20 else "partial", "rows": rows, "count": len(rows), "expected": 20})
    fields = ["family", "variant", "task", "status", "native_selection_f1", "branch_off_minus_native", "fp32_fusion_minus_native", "artifact_dir"]
    with (OUT / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    table = source_protocol_table(); write_json(OUT / "protocol_comparability.json", {"protocol": PROTOCOL, "rows": table})
    with (OUT / "protocol_comparability.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["comparison", "dimension", "architecture", "value", "reference"]); w.writeheader(); w.writerows(table)


def postprocess_job(family: str, variant: str, task: str) -> None:
    """Validate completed files without rebuilding a model or rerunning inference."""
    out = artifact_dir(family, variant, task)
    if not all((out / f"{mode}.json").is_file() and (out / f"{mode}_predictions.json").is_file() and (out / f"{mode}_logits.pt").is_file() for mode in MODES):
        raise RuntimeError(f"Incomplete mode files for {family}/{variant}/{task}")
    contract = task_contract(task)
    results: dict[str, Any] = {}
    for mode in MODES:
        saved = read_json(out / f"{mode}.json")
        saved["rows"] = read_json(out / f"{mode}_predictions.json")["rows"]
        saved["logits"] = torch.load(out / f"{mode}_logits.pt", map_location="cpu", weights_only=False)
        results[mode] = saved
    checkpoint = checkpoint_path(family, variant, task)
    alignment = audit_validation_alignment(family, variant, task, results, contract, checkpoint)
    comparison = compare_outputs(family, variant, task, results, contract["aspect_class_counts"])
    write_json(out / "validation_alignment.json", alignment)
    write_json(out / "comparison.json", comparison)
    for name in ("provenance.json", "status.json"):
        path = out / name
        if path.is_file():
            payload = read_json(path)
            payload["validation_alignment"] = alignment
            payload["comparison"] = "comparison.json"
            write_json(path, payload)


def actual_protocol_table() -> list[dict[str, Any]]:
    """Record task-specific serialized E8/E13 and E14/E15 settings."""
    rows: list[dict[str, Any]] = []
    e14_manifest = read_json(ROOT / "artifacts" / "e14" / "locked_selection_manifest.json")
    e14_selected = {str(x["task_id"]): x for x in e14_manifest.get("selections", [])}
    dimensions = ("learning_rate", "effective_batch_size", "micro_batch_size", "max_length", "precision", "attention_implementation", "loss", "head", "best_epoch", "optimizer_contract")
    for task in TASKS:
        e8m = read_json(ROOT / "artifacts" / "e8" / task / "seed_42" / "metadata.json")
        e13p = read_json(artifact_dir("e13", "centered", task) / "provenance.json").get("source_metadata", {})
        e14_path = Path(str(e14_selected[task]["artifact_dir"])) / "metadata.json"
        e14m = read_json(e14_path)
        e15m = read_json(artifact_dir("e15", "centered", task) / "provenance.json").get("source_metadata", {})
        values = {
            "E8": {"learning_rate": e8m.get("learning_rate"), "effective_batch_size": e8m.get("effective_batch_size"), "micro_batch_size": e8m.get("micro_batch_size"), "max_length": e8m.get("max_length"), "precision": e8m.get("runtime", {}).get("precision", e8m.get("precision")), "attention_implementation": e8m.get("attention_implementation"), "loss": "CE+NC", "head": e8m.get("model", {}).get("head_hidden_size"), "best_epoch": e8m.get("best_epoch"), "optimizer_contract": "E7 LLRD gamma=0.8"},
            "E13": {"learning_rate": e13p.get("learning_rate"), "effective_batch_size": e13p.get("effective_batch_size"), "micro_batch_size": e13p.get("micro_batch_size"), "max_length": e13p.get("max_length"), "precision": e13p.get("runtime", {}).get("precision", e13p.get("precision")), "attention_implementation": e13p.get("attention_implementation"), "loss": "CE+NC", "head": e13p.get("model", {}).get("head_hidden_size"), "best_epoch": e13p.get("best_epoch"), "optimizer_contract": "E7 LLRD gamma=0.8"},
            "E14": {"learning_rate": e14m.get("learning_rate"), "effective_batch_size": e14m.get("effective_batch_size"), "micro_batch_size": e14m.get("micro_batch_size"), "max_length": e14m.get("max_length"), "precision": e14m.get("runtime", {}).get("precision", e14m.get("precision")), "attention_implementation": e14m.get("attention_implementation"), "loss": e14m.get("loss"), "head": e14m.get("model", {}).get("head_hidden_size"), "best_epoch": e14m.get("best_epoch"), "optimizer_contract": "AdamW no LLRD"},
            "E15": {"learning_rate": e15m.get("learning_rate"), "effective_batch_size": e15m.get("effective_batch_size"), "micro_batch_size": e15m.get("micro_batch_size"), "max_length": e15m.get("max_length"), "precision": e15m.get("runtime", {}).get("precision", e15m.get("precision")), "attention_implementation": e15m.get("attention_implementation"), "loss": e15m.get("loss"), "head": e15m.get("model", {}).get("head_hidden_size"), "best_epoch": e15m.get("best_epoch"), "optimizer_contract": "AdamW no LLRD"},
        }
        for dim in dimensions:
            rows.extend([{ "comparison": "E13_vs_E8", "task": task, "dimension": dim, "architecture": "E13", "value": values["E13"].get(dim), "reference": values["E8"].get(dim) }, { "comparison": "E15_vs_E14", "task": task, "dimension": dim, "architecture": "E15", "value": values["E15"].get(dim), "reference": values["E14"].get(dim) }])
    return rows


def actual_variant_table() -> list[dict[str, Any]]:
    """Persist exact per-task settings for every measured variant."""
    rows: list[dict[str, Any]] = []
    fields = ("learning_rate", "effective_batch_size", "micro_batch_size", "max_length", "precision", "attention_implementation", "loss", "head_hidden_size", "best_epoch", "optimizer_contract")
    for family, variants, base in (("e13", VARIANTS, "e13"), ("e15", VARIANTS, "e15")):
        for variant in variants:
            for task in TASKS:
                p = artifact_dir(family, variant, task) / "provenance.json"
                meta = read_json(p).get("source_metadata", {})
                for field in fields:
                    if field == "precision": value = meta.get("runtime", {}).get("precision", meta.get("precision"))
                    elif field == "head_hidden_size": value = meta.get("model", {}).get("head_hidden_size")
                    elif field == "optimizer_contract": value = "E7 LLRD gamma=0.8" if family == "e13" else "AdamW no LLRD"
                    else: value = meta.get(field)
                    rows.append({"family": family, "variant": variant, "task": task, "field": field, "value": value, "metadata_path": str(p)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--postprocess", action="store_true")
    ap.add_argument("--job", nargs=3, metavar=("FAMILY", "VARIANT", "TASK"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gradient-device", default="cpu")
    args = ap.parse_args()
    if args.job:
        f, v, t = args.job
        if f not in FAMILIES or v not in VARIANTS or t not in TASKS: raise ValueError(args.job)
        run_job(f, v, t, args.device, args.gradient_device)
    elif args.postprocess:
        for f, v, t in jobs(): postprocess_job(f, v, t)
        write_summary()
        actual = actual_protocol_table()
        write_json(OUT / "protocol_comparability_actual.json", {"protocol": PROTOCOL, "rows": actual})
        with (OUT / "protocol_comparability_actual.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["comparison", "task", "dimension", "architecture", "value", "reference"]); writer.writeheader(); writer.writerows(actual)
    elif args.all:
        run_controller(args.gradient_device)
    else:
        ap.error("use --all or --job FAMILY VARIANT TASK")


if __name__ == "__main__":
    main()
