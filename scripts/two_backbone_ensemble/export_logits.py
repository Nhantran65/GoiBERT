#!/usr/bin/env python3
"""Strict checkpoint replay on validation only; one model/task per process."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from goibert import train

OUT = ROOT / "artifacts/two_backbone_ensemble"
TASKS = ("absa_restaurant", "visfd", "vsfc_topic")
RUNS = {
    "cafe": ["artifacts/cdbk_pilot/runs/T_absa_restaurant_seed42", "artifacts/sealion_benchmark/cafe_runs/visfd/lr_2e-05/seed_42", "artifacts/sealion_benchmark/cafe_runs/vsfc_topic/lr_2e-05/seed_42"],
    "cafe_s43": ["artifacts/cafebert_aspect_pair_confirmation/seed43/T_absa_restaurant_seed43", None, None],
    "xlmr_large": ["artifacts/xlmr_large_baseline/xlmr_large/absa_restaurant/lr_1e-05/seed_42", "artifacts/xlmr_large_baseline/xlmr_large/visfd/lr_2e-05/seed_42", "artifacts/xlmr_large_baseline/xlmr_large/vsfc_topic/lr_2e-05/seed_42"],
    "p_vi": [f"artifacts/bidirlm_vi_mntp/ft/P_vi/{t}/lr_1e-05/seed_42" for t in TASKS],
    "o": [f"artifacts/qwen3_encoder_probe/B2/{t}/lr_1e-05/seed_42" for t in TASKS],
    "b1b": [f"artifacts/bidirlm_1b_probe/phase2/ft/B1B/{t}/lr_5e-06/seed_42" for t in TASKS],
}
SOURCE = {"cafe": "cdbk_pilot", "cafe_s43": "cdbk_pilot", "xlmr_large": "xlmr_large_baseline", "p_vi": "bidirlm_vi_mntp", "o": "qwen3_encoder_probe", "b1b": "bidirlm_1b_probe"}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def common_for(key):
    name = SOURCE[key]
    spec = importlib.util.spec_from_file_location("ensemble_" + name, ROOT / "scripts" / name / "common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(key, bundle):
    c = common_for(key)
    counts = train.task_aspect_class_counts(bundle)
    kw = dict(num_classes=train.task_num_classes(bundle), num_aspects=train.task_num_aspects(bundle), aspect_class_counts=counts)
    if key in ("cafe", "cafe_s43"):
        model, spec = c.build_model("cafebert", {**kw, "bundle": bundle})
        tokenizer, _ = c.load_tokenizer("cafebert")
    elif key == "xlmr_large":
        model, spec = c.build_model("xlmr_large", **kw)
        tokenizer = c.load_tokenizer("xlmr_large")
    else:
        arm = {"p_vi": "P_vi", "o": "B2", "b1b": "B1B"}[key]
        model, spec = c.build_model(arm, **kw)
        tokenizer = c.load_tokenizer(arm)
    return model, tokenizer, spec


def infer(model, tokenizer, bundle, cfg, device):
    split = train.split_field(bundle, "validation")
    loader = train._make_loader(tokenizer, split, max_length=int(cfg["max_length"]), batch_size=int(cfg["micro_batch_size"]), shuffle=False,
                                seed=int(cfg["seed"]), labels=train.model_labels(bundle, "validation", split), gold_labels=train.split_labels(split))
    ids, gold, logits = [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            moved = train._move_batch(batch, device)
            with train.make_autocast(cfg["precision"], device):
                result = model(moved["input_ids"], moved["attention_mask"], token_type_ids=moved.get("token_type_ids"))
            logits.append(result.float().cpu())
            ids.extend(batch["ids"])
            gold.extend(batch["gold_labels"].tolist())
    return ids, gold, torch.cat(logits)


def score(task, bundle, ids, gold, logits):
    counts = train.task_aspect_class_counts(bundle)
    pred = train._predictions(logits, aspect_class_counts=counts)
    metrics = train.evaluate_predictions(task, bundle, gold, pred, split="validation")
    return pred, float(train.selection_score(metrics)), metrics


def trim_check(model, tokenizer, bundle, cfg, device, base_ids, base_gold, base_logits, base_pred, key, task):
    if key != "cafe":
        return None
    from torch import nn
    keep = read(ROOT / "artifacts/vocab_trim_student/keep_set.json")["old_ids"]
    xlmr_cfg = read(ROOT / "artifacts/xlmr_large_baseline/xlmr_large" / task / ("lr_1e-05" if task == "absa_restaurant" else "lr_2e-05") / "seed_42/resolved_config.json")
    cafe_cfg = read(ROOT / RUNS["cafe"][TASKS.index(task)] / "resolved_config.json")
    cafe_spm = cafe_cfg["tokenizer"]["files"]["sentencepiece.bpe.model"]
    xlmr_spm = xlmr_cfg["model_manifest"]["files"]["sentencepiece.bpe.model"]
    if cafe_spm != xlmr_spm or not cafe_cfg["tokenizer"].get("sentencepiece_identical_to_cafebert", False):
        raise RuntimeError("Cafe tokenizer differs from XLM-R-large")
    trim_tok = PreTrainedTokenizerFast(tokenizer_file=str(ROOT / "artifacts/vocab_trim_student/tokenizer_vi/tokenizer.json"),
                                      unk_token="<unk>", pad_token="<pad>", bos_token="<s>", eos_token="</s>", mask_token="<mask>")
    assert len(tokenizer) == 250002 and len(keep) == len(trim_tok)
    old_to_new = np.full(len(tokenizer), 3, dtype=np.int64)
    old_to_new[np.asarray(keep, dtype=np.int64)] = np.arange(len(keep), dtype=np.int64)
    split = train.split_field(bundle, "validation")
    texts = list(train.split_texts(split))
    original = train.encode_texts(tokenizer, texts, int(cfg["max_length"]))
    trimmed = train.encode_texts(trim_tok, texts, int(cfg["max_length"]))
    same_ids = sum(old_to_new[np.asarray(a)].tolist() == b for a, b in zip(original, trimmed))
    original_count = sum(p.numel() for p in model.parameters())
    embedding = model.encoder.embeddings.word_embeddings
    new_emb = nn.Embedding(len(keep), embedding.embedding_dim, padding_idx=1, device=device, dtype=embedding.weight.dtype)
    with torch.no_grad():
        new_emb.weight.copy_(embedding.weight[torch.as_tensor(keep, device=device, dtype=torch.long)])
    model.encoder.embeddings.word_embeddings = new_emb
    model.encoder.config.vocab_size = len(keep)
    del embedding
    gc.collect()
    torch.cuda.empty_cache()
    ids, gold, logits = infer(model, trim_tok, bundle, cfg, device)
    pred, f1, _ = score(task, bundle, ids, gold, logits)
    assert ids == base_ids and gold == base_gold
    result = {"task": task, "tokenizer_original_vocab": len(tokenizer), "tokenizer_trim_vocab": len(trim_tok),
              "tokenizer_cafe_xlmr_identical": True, "sentencepiece_sha256": cafe_spm,
              "sequences": len(texts), "token_ids_identical_count": same_ids,
              "token_ids_identical_rate": same_ids / len(texts), "max_abs_logits_delta": float((logits - base_logits).abs().max()),
              "predictions_identical": pred == base_pred, "f1_original": score(task, bundle, base_ids, base_gold, base_logits)[1],
              "f1_trim": f1, "parameters_original": original_count, "parameters_trim": sum(p.numel() for p in model.parameters())}
    write(OUT / "trim" / f"{task}.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=RUNS)
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    key, task = args.model, args.task
    rel = RUNS[key][TASKS.index(task)]
    if rel is None:
        raise ValueError("model/task absent")
    path = ROOT / rel
    cfg = read(path / "resolved_config.json")
    meta = read(path / "metadata.json")
    saved = read(path / "validation_predictions.json")["rows"]
    assert cfg["include_test"] is False
    assert cfg["task_id"] == task
    device = torch.device(args.device)
    started = time.monotonic()
    bundle = train.load_task_bundle(task, ROOT, include_test=False)
    model, tokenizer, spec = build(key, bundle)
    state = torch.load(path / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state["model"], strict=True)
    assert not missing and not unexpected
    model.to(device)
    gpu_started = time.monotonic()
    ids, gold, logits = infer(model, tokenizer, bundle, cfg, device)
    pred, f1, metrics = score(task, bundle, ids, gold, logits)
    saved_by_id = {r["id"]: r for r in saved}
    exact_ids = set(ids) == set(saved_by_id) and len(ids) == len(saved_by_id)
    exact_gold = exact_ids and all(g == saved_by_id[i]["gold"] for i, g in zip(ids, gold))
    exact_pred = exact_ids and all(p == saved_by_id[i]["prediction"] for i, p in zip(ids, pred))
    saved_f1 = float(meta["best_selection_f1"])
    check = {"model": key, "task": task, "strict_load": True, "missing": list(missing), "unexpected": list(unexpected),
             "ids_identical": exact_ids, "gold_identical": exact_gold, "predictions_identical": exact_pred,
             "f1": f1, "saved_f1": saved_f1, "f1_abs_delta": abs(f1 - saved_f1), "f1_identical": abs(f1 - saved_f1) < 1e-9,
             "validation_rows": len(ids), "logits_shape": list(logits.shape), "aspect_class_counts": list(train.task_aspect_class_counts(bundle))}
    write(OUT / "checks" / key / f"{task}.json", check)
    if not (exact_ids and exact_gold and exact_pred and check["f1_identical"]):
        raise RuntimeError(f"reproduction blocker: {check}")
    (OUT / "logits" / key).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "logits" / key / f"{task}.npz", ids=np.asarray(ids), logits=logits.numpy(), gold=np.asarray(gold), aspect_class_counts=np.asarray(train.task_aspect_class_counts(bundle), dtype=np.int32))
    if key == "cafe":
        trim_check(model, tokenizer, bundle, cfg, device, ids, gold, logits, pred, key, task)
    gpu_seconds = time.monotonic() - gpu_started
    manifest = {"model": key, "task": task, "run_dir": rel, "checkpoint_sha256": sha(path / "best_checkpoint.pt"),
                "venv": sys.prefix, "python": platform.python_version(), "torch": torch.__version__,
                "transformers": __import__("transformers").__version__, "numpy": np.__version__, "device": str(device),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "precision": cfg["precision"],
                "max_length": cfg["max_length"], "batch_size": cfg["micro_batch_size"], "model_spec": spec,
                "gpu_seconds": gpu_seconds, "elapsed_seconds": time.monotonic() - started}
    write(OUT / "manifest_parts" / key / f"{task}.json", manifest)
    print(json.dumps({"model": key, "task": task, "f1": f1, "trim": key == "cafe", "gpu_seconds": gpu_seconds}), flush=True)


if __name__ == "__main__":
    main()
