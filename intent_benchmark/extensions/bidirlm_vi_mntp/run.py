#!/usr/bin/env python3
"""BidirLM extension of the frozen eight-intent pilot; writes separate artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "intent_benchmark" / "training"))
import train_intent as old  # noqa: E402

MODEL_NAME = "bidirlm_vi_mntp"
MODEL_PATH = ROOT / "artifacts" / "bidirlm_vi_mntp" / "mntp" / "hf_bidirlm"
OUT = ROOT / "intent_benchmark" / "extensions" / MODEL_NAME
old_config = old.load_locked_config
old_provenance = old.pinned_provenance


def config() -> dict:
    value = old_config()
    value["models"][MODEL_NAME] = {
        "path": str(MODEL_PATH.relative_to(ROOT)),
        "description": "BidirLM Vietnamese MNTP encoder, model-native late pooling",
    }
    value["extension"] = {
        "model": MODEL_NAME,
        "pooling": "late",
        "tokenizer_fix_mistral_regex": True,
        "source_training_config_sha256": old.sha256(old.CONFIG_PATH),
        "benchmark_script_sha256": old.sha256(Path(__file__)),
        "test_status": "previously_seen_by_team_not_fresh_holdout",
    }
    return value


class LatePoolIntentClassifier(nn.Module):
    """Same dense/tanh/dropout/output head, logits averaged over valid tokens."""

    def __init__(self, backbone: nn.Module, num_labels: int, dropout: float):
        super().__init__()
        self.backbone = backbone
        width = int(backbone.config.hidden_size)
        self.head_dropout1 = nn.Dropout(dropout)
        self.head_dense = nn.Linear(width, width)
        self.head_activation = nn.Tanh()
        self.head_dropout2 = nn.Dropout(dropout)
        self.head_out = nn.Linear(width, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.head_out(self.head_dropout2(self.head_activation(self.head_dense(self.head_dropout1(hidden)))))
        mask = attention_mask.unsqueeze(-1).to(logits.dtype)
        return (logits * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def load_model(name: str, device: torch.device, cfg: dict | None = None):
    assert name == MODEL_NAME
    cfg = cfg or config()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, local_files_only=True, use_fast=True, trust_remote_code=True,
        fix_mistral_regex=True,
    )
    backbone, loading = AutoModel.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=True,
        output_loading_info=True, attn_implementation="sdpa",
    )
    missing = list(loading.get("missing_keys", []))
    unexpected = list(loading.get("unexpected_keys", []))
    mismatched = list(loading.get("mismatched_keys", []))
    if missing or unexpected or mismatched:
        raise RuntimeError(f"Backbone load mismatch: {missing=}, {unexpected=}, {mismatched=}")
    model = LatePoolIntentClassifier(backbone, len(cfg["labels"]), float(cfg["head_dropout"])).to(device)
    count = sum(p.numel() for p in model.parameters())
    info = {
        "model_name": name,
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "model_files": {k: old.sha256(MODEL_PATH / k) for k in ("config.json", "model.safetensors", "tokenizer.json")},
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": len(tokenizer),
        "hidden_size": backbone.config.hidden_size,
        "backbone_class": type(backbone).__name__,
        "attention_implementation": "sdpa",
        "pooling": "late",
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "mismatched_keys": mismatched,
        "allowed_missing_only": True,
        "allowed_unexpected_only": True,
        "trainable_parameters": count,
        "total_parameters": count,
    }
    return tokenizer, model, info


def provenance(cfg: dict) -> dict:
    value = old_provenance(cfg)
    value["extension"] = cfg["extension"]
    value["extension"]["source_model_path"] = str(MODEL_PATH.relative_to(ROOT))
    value["extension"]["source_model_sha256"] = old.sha256(MODEL_PATH / "model.safetensors")
    return value


old.load_locked_config = config
old.load_tokenizer_and_model = load_model
old.pinned_provenance = provenance


def evaluate_test(run_dir: Path, gpu: int) -> dict:
    cfg = config()
    device = old.resolve_device(gpu)
    tokenizer, model, loading = load_model(MODEL_NAME, device, cfg)
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    rows = old.build_dataset_rows(old.TEST_PATH)
    dataset = old.IntentDataset(rows, tokenizer, cfg["labels"], int(cfg["max_length"]))
    loader = DataLoader(
        dataset, batch_size=int(cfg["batch_size"]), shuffle=False,
        collate_fn=lambda examples: old.collate_examples(examples, tokenizer.pad_token_id),
    )
    precision = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    metrics, predictions = old.evaluate(model, loader, device, cfg["labels"], precision)
    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "model": MODEL_NAME,
        "selected_run": str(run_dir.relative_to(ROOT)),
        "selection": json.loads((OUT / "selection.json").read_text()),
        "metrics": metrics,
        "n_test": len(rows),
        "loading": loading,
        "provenance": provenance(cfg),
    }
    (OUT / "test_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (OUT / "test_predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["smoke", "trial", "select", "test"])
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--lr", type=float)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == "smoke":
        result = old.run_smoke_model(MODEL_NAME, args.gpu, OUT / "smoke")
    elif args.command == "trial":
        if args.lr not in (2e-5, 5e-5):
            raise ValueError("Use only the two learning rates in the original pilot")
        slug = "2e-5" if args.lr == 2e-5 else "5e-5"
        result = old.run_trial(MODEL_NAME, args.lr, 42, args.gpu, OUT / f"lr_{slug}")
    elif args.command == "select":
        options = []
        for slug in ("2e-5", "5e-5"):
            run = OUT / f"lr_{slug}"
            if not (run / "COMPLETED").exists():
                raise RuntimeError(f"Incomplete trial: {run}")
            summary = json.loads((run / "summary.json").read_text())
            options.append((float(summary["best_dev_macro_f1"]), slug, summary))
        winner = max(options, key=lambda row: (row[0], row[1] == "2e-5"))
        result = {
            "criterion": "dev_macro_f1; tie -> 2e-5",
            "winner": winner[1],
            "best_dev_macro_f1": winner[0],
            "best_epoch": winner[2]["best_epoch"],
            "trials": {slug: score for score, slug, _ in options},
            "checkpoint_sha256": old.sha256(OUT / f"lr_{winner[1]}" / "best.pt"),
        }
        (OUT / "selection.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    else:
        selection = json.loads((OUT / "selection.json").read_text())
        result = evaluate_test(OUT / f"lr_{selection['winner']}", args.gpu)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
