#!/usr/bin/env python3
"""Step 1.3 — prove the vocabulary trim costs nothing on a model that is already fine-tuned.

Takes the fine-tuned XLM-R-large Restaurant checkpoint (val 77.81), (A) trims the embedding rows and maps ids
with the original tokenizer, (B) swaps in the exported trimmed encoder and the trimmed fast tokenizer, and
re-evaluates the same validation split with the same loader/metric.  (A) must reproduce the baseline F1
exactly (logits delta recorded); (B) must stay within +-0.1.  Appends to trim_report.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import sources as S

XL_SCRIPTS = S.ROOT / "scripts" / "xlmr_large_baseline"
if str(XL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(XL_SCRIPTS))

CHECKPOINT = S.ROOT / "artifacts" / "xlmr_large_baseline" / "xlmr_large" / "absa_restaurant" / "lr_2e-05" / "seed_42" / "best_checkpoint.pt"


class MappedTokenizer:
    """Original tokenizer + (A) id mapping; exposes exactly what ``goibert.train._make_loader`` uses."""

    def __init__(self, base, old_to_new: np.ndarray):
        self.base = base
        self.old_to_new = old_to_new
        self.is_fast = True
        self.pad_token_id = int(old_to_new[base.pad_token_id])
        self.pad_token = base.pad_token
        self.name_or_path = base.name_or_path

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        encoded = self.base(texts, **kwargs)
        encoded["input_ids"] = [self.old_to_new[np.asarray(ids, dtype=np.int64)].tolist() for ids in encoded["input_ids"]]
        return encoded

    def pad(self, features, **kwargs):
        return self.base.pad(features, **kwargs)

    def __len__(self):
        return len(self.base)

    def __getattr__(self, name):
        return getattr(self.base, name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--keep", default=str(S.ART / "keep_set.json"))
    ap.add_argument("--trimmed-model", default=str(S.ROOT / "models" / "trimmed" / "xlmr_large_vi"))
    ap.add_argument("--report", default=str(S.ART / "trim_report.json"))
    args = ap.parse_args()
    started = time.time()
    device = torch.device(args.device)
    keep = json.loads(Path(args.keep).read_text())
    keep_ordered = [int(i) for i in keep["old_ids"]]
    size = len(keep_ordered)
    old_to_new = np.full(S.VOCAB_SIZE, 3, dtype=np.int64)
    old_to_new[np.asarray(keep_ordered, dtype=np.int64)] = np.arange(size, dtype=np.int64)

    import common as xl_common  # scripts/xlmr_large_baseline/common.py
    from goibert import train as e1_train

    bundle = e1_train.load_task_bundle("absa_restaurant", S.ROOT, include_test=False)
    valid_split = e1_train.split_field(bundle, "validation")
    num_aspects = e1_train.task_num_aspects(bundle)
    counts = e1_train.task_aspect_class_counts(bundle)

    def build(with_ckpt: bool = True):
        model, spec = xl_common.build_model("xlmr_large", num_classes=e1_train.task_num_classes(bundle),
                                            num_aspects=num_aspects, aspect_class_counts=counts)
        if with_ckpt:
            state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
            missing, unexpected = model.load_state_dict(state["model"], strict=True)
            if missing or unexpected:
                raise RuntimeError(f"checkpoint mismatch: {missing} {unexpected}")
        return model, spec

    def evaluate(model, tokenizer) -> dict:
        loader = e1_train._make_loader(tokenizer, valid_split, max_length=256, batch_size=args.batch, shuffle=False, seed=42,
                                       labels=e1_train.model_labels(bundle, "validation", valid_split),
                                       gold_labels=e1_train.split_labels(valid_split))
        metrics, rows = e1_train.run_validation(model, loader, device, "bf16", "absa_restaurant", bundle)
        return {"f1": e1_train.selection_score(metrics), "rows": rows, "loader": loader}

    tokenizer = S.xlmr_tokenizer()
    base_model, spec = build()
    base_model.to(device).eval()
    base_result = evaluate(base_model, tokenizer)
    print(f"[verify] untrimmed F1 {100*base_result['f1']:.4f} (checkpoint val {100*json.loads((CHECKPOINT.parent/'metadata.json').read_text())['best_selection_f1']:.4f})", flush=True)

    # ---- (A) id mapping + row-trimmed embedding -------------------------------------------------
    model_a = copy.deepcopy(base_model)
    hidden = int(model_a.encoder.config.hidden_size)
    old_weight = model_a.encoder.embeddings.word_embeddings.weight.data.clone()
    new_embedding = nn.Embedding(size, hidden, padding_idx=1)
    with torch.no_grad():
        new_embedding.weight.copy_(old_weight[torch.as_tensor(keep_ordered, dtype=torch.long)])
    model_a.encoder.embeddings.word_embeddings = new_embedding
    model_a.encoder.config.vocab_size = size
    model_a.to(device).eval()
    mapped_tokenizer = MappedTokenizer(tokenizer, old_to_new)
    a_result = evaluate(model_a, mapped_tokenizer)
    print(f"[verify] (A) F1 {100*a_result['f1']:.4f}", flush=True)

    # logits delta on the same validation batches -------------------------------------------------
    max_delta = 0.0
    loader = e1_train._make_loader(tokenizer, valid_split, max_length=256, batch_size=args.batch, shuffle=False, seed=42,
                                   labels=e1_train.model_labels(bundle, "validation", valid_split),
                                   gold_labels=e1_train.split_labels(valid_split))
    with torch.inference_mode():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits_base = base_model(ids, mask)
            mapped_ids = torch.from_numpy(old_to_new[ids.cpu().numpy()]).to(device)
            logits_a = model_a(mapped_ids, mask)
            max_delta = max(max_delta, float((logits_base.float() - logits_a.float()).abs().max().item()))

    # ---- (B) the same fine-tuned model, but inputs re-tokenised with the trimmed tokenizer -------
    from transformers import AutoTokenizer

    # the trimmed tokenizer ships inside every trimmed model directory (config.json makes it resolvable)
    tok_b = AutoTokenizer.from_pretrained(args.trimmed_model, local_files_only=True, use_fast=True)
    if len(tok_b) != size:
        raise RuntimeError(f"trimmed tokenizer size {len(tok_b)} != |K| {size}")
    b_result = evaluate(model_a, tok_b)
    print(f"[verify] (B) F1 {100*b_result['f1']:.4f}", flush=True)

    # the exported (pretrained, trimmed) directory must load and expose |K| rows --------------------
    from transformers import AutoModel

    exported = AutoModel.from_pretrained(args.trimmed_model, local_files_only=True, add_pooling_layer=False)
    export_check = {"encoder_class": type(exported).__name__,
                    "vocab_size": int(exported.config.vocab_size),
                    "embedding_rows": int(exported.get_input_embeddings().weight.shape[0]),
                    "matches_keep_size": int(exported.get_input_embeddings().weight.shape[0]) == size}
    del exported

    report = json.loads(Path(args.report).read_text()) if Path(args.report).exists() else {}
    report.setdefault("equivalence", {}).update({
        "checkpoint": str(CHECKPOINT.relative_to(S.ROOT)),
        "checkpoint_val_f1": json.loads((CHECKPOINT.parent / "metadata.json").read_text())["best_selection_f1"],
        "untrimmed_f1": base_result["f1"], "A_f1": a_result["f1"], "B_f1": b_result["f1"],
        "A_minus_untrimmed": a_result["f1"] - base_result["f1"], "B_minus_untrimmed": b_result["f1"] - base_result["f1"],
        "max_abs_logits_delta_A_vs_untrimmed": max_delta,
        "A_identical": abs(a_result["f1"] - base_result["f1"]) < 1e-12,
        "B_within_0.1": abs(b_result["f1"] - base_result["f1"]) <= 0.001,
        "export_check": export_check, "validation_rows": len(base_result["rows"]), "device": str(device),
        "elapsed_seconds": time.time() - started,
    })
    S.write_json(Path(args.report), report)
    ok = report["equivalence"]["A_identical"] and report["equivalence"]["B_within_0.1"]
    print(f"[verify] {'PASSED' if ok else 'FAILED'} (max|Δ logits| = {max_delta:.3e}) -> {args.report}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
