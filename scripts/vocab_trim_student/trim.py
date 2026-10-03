#!/usr/bin/env python3
"""Step 1.2 + exports (docs/VOCAB_TRIM_STUDENT_PLAN.md).

Builds (A) the id map (original tokenizer + row-trimmed embedding) and (B) the trimmed fast tokenizer
(Unigram vocab filtered in original order, specials 0-3, <mask> last, unk_id 3), measures `<unk>` rates on
validation + test *text* and the (B)-vs-(A) tokenisation agreement, and exports the four trimmed HF encoder
directories to models/trimmed/.  Writes artifacts/vocab_trim_student/trim_report.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

import sources as S

TRIMMED_DIR = S.ROOT / "models" / "trimmed"
EXPORTS = {
    "xlmr_large_vi": {"repo": S.XLM_R_LARGE[0], "commit": S.XLM_R_LARGE[1], "pooling": False},
    "xlmr_base_vi": {"repo": S.XLM_R_BASE[0], "commit": S.XLM_R_BASE[1], "pooling": False},
    "mminilmv2_l12_vi": {"repo": S.MINILM12[0], "commit": S.MINILM12[1], "pooling": False},
    "mminilmv2_l6_vi": {"repo": S.MINILM6[0], "commit": S.MINILM6[1], "pooling": False},
}


def build_trimmed_tokenizer(src_dir: Path, keep_ordered: list[int], out_dir: Path) -> dict:
    """(B): filter the Unigram vocab, keep original order/score, specials at 0-3, <mask> last, unk_id = 3."""
    from tokenizers import Tokenizer

    raw = json.loads((src_dir / "tokenizer.json").read_text())
    vocab = raw["model"]["vocab"]
    new_vocab = [vocab[i] for i in keep_ordered]
    raw["model"]["vocab"] = new_vocab
    raw["model"]["unk_id"] = 3
    last_id = len(keep_ordered) - 1
    keep_content = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "<mask>": last_id}
    added = [t for t in raw.get("added_tokens", []) if t.get("content") in keep_content]
    for entry in added:
        entry["id"] = keep_content[entry["content"]]
    raw["added_tokens"] = added
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tokenizer.json").write_text(json.dumps(raw, ensure_ascii=False))
    for name in ("tokenizer_config.json", "special_tokens_map.json"):
        if (src_dir / name).exists():
            shutil.copyfile(src_dir / name, out_dir / name)
    config_path = out_dir / "tokenizer_config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        cfg.setdefault("model_max_length", 512)
        config_path.write_text(json.dumps(cfg, ensure_ascii=False))
    else:
        config_path.write_text(json.dumps({"model_max_length": 512}))
    tokenizer = Tokenizer.from_str(json.dumps(raw))
    fast = __import__("transformers").PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", pad_token="<pad>", bos_token="<s>", eos_token="</s>", mask_token="<mask>")
    return fast, {"fast_tokenizer_size": len(fast), "mask_id": fast.mask_token_id, "unk_id": fast.unk_token_id}


def export_encoder(name: str, spec: dict, keep_ordered: list[int], tokenizer_dir: Path, models_dir: Path) -> dict:
    import torch
    from transformers import AutoConfig, AutoModel

    src = S.snapshot(spec["repo"], spec["commit"])
    model = AutoModel.from_pretrained(str(src), local_files_only=True, add_pooling_layer=False, attn_implementation="sdpa")
    before = int(sum(p.numel() for p in model.parameters()))
    embedding_before = int(model.get_input_embeddings().weight.numel())
    old_weight = model.get_input_embeddings().weight.data.clone()
    model.resize_token_embeddings(len(keep_ordered))
    with torch.no_grad():
        model.get_input_embeddings().weight.copy_(old_weight[torch.as_tensor(keep_ordered, dtype=torch.long)])
    model.config.vocab_size = len(keep_ordered)
    out = models_dir / name
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    for file in tokenizer_dir.iterdir():
        shutil.copyfile(file, out / file.name)
    after = int(sum(p.numel() for p in model.parameters()))
    config = AutoConfig.from_pretrained(str(out))
    return {"source": f"{spec['repo']}@{spec['commit']}", "encoder_class": type(model).__name__,
            "parameters_before": before, "parameters_after": after,
            "embedding_parameters_before": embedding_before, "embedding_parameters_after": len(keep_ordered) * int(model.config.hidden_size),
            "hidden_size": int(model.config.hidden_size), "num_hidden_layers": int(model.config.num_hidden_layers),
            "vocab_size_after": len(keep_ordered), "config_class": type(config).__name__, "dir": str(out.relative_to(S.ROOT))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep", default=str(S.ART / "keep_set.json"))
    ap.add_argument("--out", default=str(S.ART / "trim_report.json"))
    ap.add_argument("--models-dir", default=str(TRIMMED_DIR))
    ap.add_argument("--examples", type=int, default=20)
    ap.add_argument("--skip-exports", action="store_true")
    args = ap.parse_args()
    models_dir = Path(args.models_dir)
    started = time.time()
    keep = json.loads(Path(args.keep).read_text())
    keep_ordered = [int(i) for i in keep["old_ids"]]
    size = len(keep_ordered)

    tokenizer = S.xlmr_tokenizer()
    specials = {tok: tokenizer.convert_tokens_to_ids(tok) for tok in ("<s>", "<pad>", "</s>", "<unk>", "<mask>")}
    print(f"[trim] keep size {size}; special ids {specials}", flush=True)
    old_to_new = np.full(S.VOCAB_SIZE, 3, dtype=np.int64)
    old_to_new[np.asarray(keep_ordered, dtype=np.int64)] = np.arange(size, dtype=np.int64)

    # ---- (B) trimmed tokenizer ------------------------------------------------------------------
    src_snapshot = S.snapshot(S.XLM_R_LARGE[0], S.XLM_R_LARGE[1])
    trimmed_dir = S.ART / "tokenizer_vi"
    tok_b, info_b = build_trimmed_tokenizer(src_snapshot, keep_ordered, trimmed_dir)
    report: dict = {"protocol": "vocab-trim-v1", "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "keep_set": {"size": size, "source": args.keep}, "special_ids": specials,
                    "trimmed_tokenizer": {**info_b, "dir": str(trimmed_dir.relative_to(S.ROOT)), "fast_only": True,
                                          "note": "sentencepiece.bpe.model is not shipped: the trimmed fast tokenizer is authoritative; "
                                                  "using the slow tokenizer would fall back to the original 250k vocab"}}

    # ---- unk + agreement ------------------------------------------------------------------------
    def unk_stats(texts, tokenizer_fn, map_fn=None) -> dict:
        total = unknown = sentences = affected = 0
        for text in texts:
            ids = tokenizer_fn(text)
            if map_fn is not None:
                ids = map_fn(ids)
            sentences += 1
            total += len(ids)
            n_unk = int(np.sum(np.asarray(ids) == 3))
            unknown += n_unk
            affected += 1 if n_unk else 0
        return {"tokens": total, "unk_tokens": unknown, "unk_token_rate": unknown / max(total, 1),
                "sentences": sentences, "sentences_with_unk": affected,
                "sentence_affected_rate": affected / max(sentences, 1)}

    val_texts = [t for t in S.benchmark_texts(splits=("validation",))]
    test_texts = [t for t in S.benchmark_test_texts()]
    social_test = [t for t in S.social_test_texts()]
    report["unk"] = {
        "A_validation_5task": unk_stats(val_texts, lambda t: tokenizer(t, add_special_tokens=False)["input_ids"], lambda ids: old_to_new[ids]),
        "B_validation_5task": unk_stats(val_texts, lambda t: tok_b(t, add_special_tokens=False)["input_ids"]),
        "A_test_text_5task": unk_stats(test_texts, lambda t: tokenizer(t, add_special_tokens=False)["input_ids"], lambda ids: old_to_new[ids]),
        "B_test_text_5task": unk_stats(test_texts, lambda t: tok_b(t, add_special_tokens=False)["input_ids"]),
        "A_test_text_social": unk_stats(social_test, lambda t: tokenizer(t, add_special_tokens=False)["input_ids"], lambda ids: old_to_new[ids]),
        "B_test_text_social": unk_stats(social_test, lambda t: tok_b(t, add_special_tokens=False)["input_ids"]),
    }
    agreements = 0
    differences: list[dict] = []
    for text in val_texts:
        ids_orig = np.asarray(tokenizer(text, add_special_tokens=False)["input_ids"])
        mapped = old_to_new[ids_orig]
        ids_trim = np.asarray(tok_b(text, add_special_tokens=False)["input_ids"])
        if mapped.shape == ids_trim.shape and np.array_equal(mapped, ids_trim):
            agreements += 1
        elif len(differences) < args.examples:
            first = int(np.argmax((mapped != ids_trim) | (mapped.shape != ids_trim.shape))) if mapped.shape == ids_trim.shape else 0
            differences.append({"text": text[:160], "orig_len": int(ids_orig.size), "trim_len": int(ids_trim.size),
                                "first_diff": first,
                                "orig_tokens": [tokenizer.convert_ids_to_tokens(int(i)) for i in ids_orig[max(0, first-2):first+3]],
                                "trim_tokens": [tok_b.convert_ids_to_tokens(int(i)) for i in ids_trim[max(0, first-2):first+3]]})
    report["tokenisation_agreement"] = {"validation_sequences": len(val_texts), "identical": agreements,
                                        "identical_rate": agreements / max(len(val_texts), 1), "examples": differences}

    # ---- exports --------------------------------------------------------------------------------
    if not args.skip_exports:
        report["exports"] = {name: export_encoder(name, spec, keep_ordered, trimmed_dir, models_dir) for name, spec in EXPORTS.items()}
        for name, info in report["exports"].items():
            print(f"[trim] {name}: {info['parameters_before']/1e6:.1f}M -> {info['parameters_after']/1e6:.1f}M "
                  f"(emb {info['embedding_parameters_before']/1e6:.1f}M -> {info['embedding_parameters_after']/1e6:.1f}M)", flush=True)
    report["elapsed_seconds"] = time.time() - started
    S.write_json(Path(args.out), report)
    print(f"[trim] unk A/B (val): {100*report['unk']['A_validation_5task']['unk_token_rate']:.5f}% / "
          f"{100*report['unk']['B_validation_5task']['unk_token_rate']:.5f}% ; agreement "
          f"{100*report['tokenisation_agreement']['identical_rate']:.2f}% -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
