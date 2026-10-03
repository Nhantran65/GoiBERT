#!/usr/bin/env python3
"""Pinned, validation-free Vietnamese decoder BPB gate for BidirLM-1B."""

from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "bidirlm_1b_probe" / "phase1"
DATA = ROOT / "data" / "bidirlm_vi_mntp"
OLD_MANIFEST = ROOT / "artifacts" / "bidirlm_vi_mntp" / "data_manifest.json"
POOL_CACHE = Path("/tmp/bidirlm_vi_mntp_general_docs.json")
QWEN_REPO, QWEN_COMMIT = "Qwen/Qwen3-0.6B-Base", "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
GEMMA_REPO, GEMMA_COMMIT = "unsloth/gemma-3-1b-pt", "34a98bf3eaecc08c699516f625eb5e56e4c5d99d"
BIDIR_REPO, BIDIR_COMMIT = "BidirLM/BidirLM-0.6B-Base", "29794508bbb4e378b4789eebff42ae4c2d47d524"
EXPECTED_GEMMA = {"model.safetensors": "ee5250f6eb1aa7cfb729dfd4dc8d9964fd772324776c6d00bf2bc674c069cb27",
                  "tokenizer.model": "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c"}
TASKS = ("vsfc_sentiment", "vsfc_topic", "visfd", "absa_restaurant", "absa_hotel")
EOS_ID, SEQ, SEED = 151643, 512, 42


def jwrite(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def snapshot(repo: str, commit: str) -> Path:
    home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    path = home / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / commit
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def file_manifest(repo: str, commit: str, license_name: str) -> dict:
    path = snapshot(repo, commit)
    files = {p.name: sha_file(p.resolve()) for p in sorted(path.iterdir()) if p.is_file()}
    return {"repo": repo, "commit": commit, "snapshot": str(path), "license": license_name, "sha256": files}


def _pool_helpers():
    for p in (ROOT / "scripts" / "mmbert_vi_cpt_pilot" / "v3",
              ROOT / "scripts" / "mmbert_vi_cpt_pilot" / "v2",
              ROOT / "scripts" / "mmbert_vi_cpt_pilot"):
        sys.path.insert(0, str(p))
    import audit_v3 as A3
    import common_v2 as C2
    import common_v3 as C3
    return A3, C2, C3


def load_pool():
    A3, C2, C3 = _pool_helpers()
    if POOL_CACHE.exists():
        general = json.loads(POOL_CACHE.read_text(encoding="utf-8"))
    else:
        bucket = [json.loads(line) for line in (C3.DATA_V3 / "informal_bucket" / "informal_docs.jsonl").open(encoding="utf-8")]
        general = A3.build_general_v3({r["pool_index"] for r in bucket})
        POOL_CACHE.write_text(json.dumps(general), encoding="utf-8")
    by_part = collections.defaultdict(list)
    for row in general["docs"]:
        by_part[row["part"]].append(row)
    available = []
    for n, part in enumerate(sorted(by_part), 1):
        rows = sorted(by_part[part], key=lambda r: r["pool_index"])
        texts = C2.part_texts(part, {r["kh64"] for r in rows})
        available.extend(r for r in rows if texts.get(r["kh64"]) is not None)
        if n % 25 == 0:
            print(f"[pool] {n}/{len(by_part)} parts, {len(available)} non-null docs", flush=True)
    return general, available, C2


def fetch_selected(rows: list[dict], C2) -> list[str]:
    by_part = collections.defaultdict(list)
    for i, row in enumerate(rows):
        by_part[row["part"]].append((i, row))
    result = [None] * len(rows)
    for part, entries in by_part.items():
        texts = C2.part_texts(part, {row["kh64"] for _, row in entries})
        for i, row in entries:
            result[i] = texts.get(row["kh64"])
            if result[i] is None:
                raise RuntimeError(f"missing selected text {row['pool_index']}")
    return result


def eos_positions(tokens: np.ndarray) -> np.ndarray:
    flat = tokens.reshape(-1)
    chunks = []
    stride = 4_000_000
    for start in range(0, len(flat), stride):
        pos = np.flatnonzero(flat[start:start + stride] == EOS_ID)
        if len(pos):
            chunks.append(pos.astype(np.int64) + start)
        if start and start % 200_000_000 == 0:
            print(f"[eos] scanned {start}/{len(flat)} tokens", flush=True)
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)


def prepare_domain_a() -> tuple[list[str], dict, dict]:
    started = time.time()
    old = json.loads(OLD_MANIFEST.read_text())
    expected_docs = int(old["dataset"]["documents_tokenized"])
    for name in ("tokens.npy", "order.npy"):
        actual = sha_file(DATA / name)
        expected = old["files"][name]["sha256"]
        if actual != expected:
            raise RuntimeError(f"{name} checksum mismatch: {actual} != {expected}")
    tokens = np.load(DATA / "tokens.npy", mmap_mode="r")
    order = np.load(DATA / "order.npy", mmap_mode="r")
    if tokens.ndim != 2 or tokens.shape[1] != SEQ or len(order) != len(tokens):
        raise RuntimeError("tokens/order shape mismatch")
    if list(tokens.shape) != old["files"]["tokens.npy"]["shape"]:
        raise RuntimeError("tokens manifest shape mismatch")
    general, available, C2 = load_pool()
    if len(available) != expected_docs or int(general["documents"]) != expected_docs:
        raise RuntimeError(f"pool doc mismatch: non-null={len(available)} manifest={expected_docs}")
    eos = eos_positions(tokens)
    if len(eos) > expected_docs:
        raise RuntimeError(f"too many EOS: {len(eos)} > {expected_docs}")
    flat = tokens.reshape(-1)
    tail_docs = expected_docs - len(eos)
    trailing = len(flat) - (int(eos[-1]) + 1 if len(eos) else 0)
    if tail_docs:
        if trailing <= 0 or tail_docs > trailing:
            raise RuntimeError(f"unexplained missing EOS: {tail_docs} docs, {trailing} trailing tokens")
        tail_rows = available[len(eos):]
        tail_texts = fetch_selected(tail_rows, C2)
        tok = AutoTokenizer.from_pretrained(str(snapshot(BIDIR_REPO, BIDIR_COMMIT)), use_fast=True, local_files_only=True)
        tail_stream = [x for ids in tok(tail_texts, add_special_tokens=False)["input_ids"] for x in (ids + [EOS_ID])]
        expected_cut = int(old["dataset"]["tokens"]) - len(flat)
        if (list(map(int, flat[-trailing:])) != tail_stream[:trailing]
                or len(tail_stream) - trailing != expected_cut):
            raise RuntimeError("truncated tail does not match retokenized final docs")
    starts = np.concatenate((np.array([0], dtype=np.int64), eos[:-1] + 1))
    first_seq, last_seq = starts // SEQ, eos // SEQ
    used = np.zeros(len(tokens), dtype=np.uint8)
    n_train = int(old["dataset"]["train_sequences"])
    if n_train != 488281:
        raise RuntimeError(f"unexpected train sequence count: {n_train}")
    used[order[:n_train]] = 1
    prefix = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(used, dtype=np.int64)))
    eligible = np.flatnonzero(prefix[last_seq + 1] == prefix[first_seq])
    if len(eligible) < 2000:
        raise RuntimeError(f"too few eligible docs: {len(eligible)}")
    sampled = np.random.default_rng(SEED).choice(eligible, 2000, replace=False)
    sample_rows = [available[int(i)] for i in sampled]
    full_texts = fetch_selected(sample_rows, C2)
    tok = AutoTokenizer.from_pretrained(str(snapshot(BIDIR_REPO, BIDIR_COMMIT)), use_fast=True, local_files_only=True)
    retok_checks = []
    for j in range(20):
        i = int(sampled[j])
        ids = tok(full_texts[j], add_special_tokens=False)["input_ids"]
        exact = np.array_equal(np.asarray(ids, dtype=np.uint32), flat[int(starts[i]):int(eos[i])])
        retok_checks.append({"sample_position": j, "pool_index": sample_rows[j]["pool_index"],
                             "token_count": len(ids), "exact_match": bool(exact)})
    if not all(x["exact_match"] for x in retok_checks):
        raise RuntimeError("20-doc retokenization check failed")
    texts = []
    for s in full_texts:
        if len(s) > 2000:
            cut = next((i for i in range(1999, -1, -1) if s[i].isspace()), -1)
            if cut < 0:
                raise RuntimeError("2,000-character cut has no preceding space")
            s = s[:cut]
        texts.append(s)
    text_hash = hashlib.sha256(json.dumps(texts, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    sample_manifest = {
        "seed": SEED, "sampling": "numpy.random.default_rng(42).choice(eligible_doc_indices, 2000, replace=False)",
        "text_list_sha256": text_hash, "text_hash_canonicalization": "UTF-8 JSON list, ensure_ascii=False, separators=(',', ':')",
        "documents": [{"pool_index": int(r["pool_index"]), "kh64": int(r["kh64"])} for r in sample_rows],
        "source_documents": expected_docs, "observed_eos": int(len(eos)), "truncated_tail_docs_excluded": tail_docs,
        "truncated_tail_tokens": trailing, "eligible_documents": int(len(eligible)), "retokenization_checks": retok_checks,
        "train_sequence_count": n_train, "elapsed_seconds": time.time() - started,
    }
    jwrite(ART / "sample_manifest.json", sample_manifest)
    return texts, sample_manifest, {"nfc_fraction": sum(unicodedata.normalize("NFC", t) == t for t in texts) / len(texts),
                                    "count": len(texts), "sha256": text_hash}


def prepare_domain_b() -> tuple[list[str], dict]:
    sys.path.insert(0, str(ROOT / "src"))
    from goibert import train
    raw = []
    counts = {}
    for task in TASKS:
        bundle = train.load_task_bundle(task, ROOT, include_test=False)
        texts = list(train.split_texts(train.split_field(bundle, "train")))
        counts[task] = len(texts)
        raw.extend(texts)
        print(f"[train] {task}: {len(texts)} text rows", flush=True)
        del bundle
    texts = list(dict.fromkeys(raw))
    hash_value = hashlib.sha256(json.dumps(texts, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return texts, {"count": len(texts), "rows_before_exact_dedup": len(raw), "task_counts": counts,
                   "nfc_fraction": sum(unicodedata.normalize("NFC", t) == t for t in texts) / len(texts),
                   "sha256": hash_value}


def batches(ids: list[list[int]], budget: int = 1536, max_batch: int = 8):
    indices = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    group, maxlen = [], 0
    for i in indices:
        new_max = max(maxlen, len(ids[i]) + 1)
        if group and (new_max * (len(group) + 1) > budget or len(group) >= max_batch):
            yield group
            group, maxlen = [], 0
        group.append(i)
        maxlen = max(maxlen, len(ids[i]) + 1)
    if group:
        yield group


def measure_model(key: str, source: Path, texts: list[list[str]], device: torch.device) -> tuple[dict, list[dict]]:
    started = time.time()
    use_fast = key == "qwen3_base"
    tok = AutoTokenizer.from_pretrained(str(source), use_fast=use_fast, local_files_only=True)
    if key == "gemma3_1b" and type(tok).__name__ != "GemmaTokenizer":
        raise RuntimeError(f"Gemma tokenizer is not slow SentencePiece: {type(tok).__name__}")
    start_id = 151643 if key == "qwen3_base" else tok.bos_token_id
    if key == "qwen3_base" and tok.convert_tokens_to_ids("<|endoftext|>") != start_id:
        raise RuntimeError("Qwen start id mismatch")
    if start_id is None:
        raise RuntimeError("missing start token")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    print(f"[{key}] loading fp32 model / tokenizer {type(tok).__name__} start={start_id}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(str(source), local_files_only=True, dtype=torch.float32,
                                                attn_implementation="sdpa").to(device).eval()
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError(f"{key}: weights are not all fp32")
    torch.cuda.reset_peak_memory_stats(device)
    result = [None] * sum(map(len, texts))
    per_domain = {}
    offset = 0
    for domain, domain_texts in (("A", texts[0]), ("B", texts[1])):
        ids = [tok.encode(t, add_special_tokens=False) for t in domain_texts]
        if not ids or any(len(x) == 0 for x in ids):
            raise RuntimeError(f"{key}/{domain}: empty tokenization")
        local_nll = np.empty(len(ids), dtype=np.float64)
        first_five = []
        n_batches = 0
        t0 = time.time()
        for group in batches(ids):
            width = max(len(ids[i]) + 1 for i in group)
            x = torch.full((len(group), width), pad_id, dtype=torch.long, device=device)
            mask = torch.zeros_like(x)
            labels = torch.full_like(x, -100)
            for j, i in enumerate(group):
                n = len(ids[i])
                x[j, 0] = start_id
                x[j, 1:n + 1] = torch.tensor(ids[i], dtype=torch.long, device=device)
                mask[j, :n + 1] = 1
                labels[j, 1:n + 1] = x[j, 1:n + 1]
            with torch.inference_mode():
                out = model(input_ids=x, attention_mask=mask, use_cache=False)
                losses = F.cross_entropy(out.logits[:, :-1, :].contiguous().view(-1, out.logits.shape[-1]),
                                         labels[:, 1:].contiguous().view(-1), reduction="none", ignore_index=-100)
                losses = losses.view(len(group), width - 1).sum(dim=1)
            for j, i in enumerate(group):
                local_nll[i] = float(losses[j].item())
            n_batches += 1
            if n_batches % 500 == 0:
                print(f"[{key}/{domain}] {n_batches} batches / {len(ids)} texts; {time.time()-t0:.0f}s", flush=True)
            del out, losses, x, mask, labels
        # Independently check the exact five-text NLL contract against the model's own loss.
        for i in range(min(5, len(ids))):
            x = torch.tensor([[start_id] + ids[i]], dtype=torch.long, device=device)
            labels = x.clone()
            labels[:, 0] = -100
            with torch.inference_mode():
                out = model(input_ids=x, attention_mask=torch.ones_like(x), labels=labels, use_cache=False)
                manual = F.cross_entropy(out.logits[:, :-1, :].reshape(-1, out.logits.shape[-1]),
                                         labels[:, 1:].reshape(-1), reduction="sum")
            model_total = float(out.loss.item()) * len(ids[i])
            relative = abs(float(manual.item()) - model_total) / max(abs(model_total), 1e-12)
            first_five.append({"index": i, "manual_nll": float(manual.item()), "model_nll": model_total,
                               "relative_error": relative, "pass": relative < 1e-4})
        if not all(c["pass"] for c in first_five):
            raise RuntimeError(f"{key}/{domain}: model loss mismatch")
        per_domain[domain] = {"count": len(ids), "batches": n_batches, "seconds": time.time()-t0,
                              "max_tokens_in_text": max(map(len, ids)), "model_loss_checks": first_five}
        for i, (nll, token_ids) in enumerate(zip(local_nll, ids)):
            result[offset + i] = {"nll_nats": float(nll), "tokens": len(token_ids)}
        offset += len(ids)
        del ids
    detail = {"tokenizer_class": type(tok).__name__, "start_id": start_id, "pad_id": pad_id,
              "dtype": "float32", "tf32": False, "eval": True, "attention_implementation": "sdpa",
              "parameters": sum(p.numel() for p in model.parameters()), "domains": per_domain,
              "elapsed_seconds": time.time() - started,
              "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    del model
    torch.cuda.empty_cache()
    return detail, result


def summarize(texts_a: list[str], texts_b: list[str], measurements: dict) -> dict:
    all_texts = texts_a + texts_b
    rows = []
    for idx, text in enumerate(all_texts):
        domain = "A" if idx < len(texts_a) else "B"
        b = len(text.encode("utf-8"))
        w = len(text.split())
        row = {"domain": domain, "index": idx if domain == "A" else idx-len(texts_a),
               "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "bytes": b, "words": w,
               "nfc": unicodedata.normalize("NFC", text) == text}
        for key in ("qwen3_base", "gemma3_1b"):
            datum = measurements[key][idx]
            row[key] = {**datum, "bpb": datum["nll_nats"] / math.log(2) / b if b else None}
        rows.append(row)
    with (ART / "per_text_bpb.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    domains = {}
    rng = np.random.default_rng(SEED)
    for domain in ("A", "B"):
        drows = [r for r in rows if r["domain"] == domain]
        bytes_arr = np.array([r["bytes"] for r in drows], dtype=np.int64)
        q = np.array([r["qwen3_base"]["nll_nats"] for r in drows])
        g = np.array([r["gemma3_1b"]["nll_nats"] for r in drows])
        nbytes = int(bytes_arr.sum())
        block = {"texts": len(drows), "bytes": nbytes, "words": sum(r["words"] for r in drows)}
        for key, nll in (("qwen3_base", q), ("gemma3_1b", g)):
            tokens = sum(r[key]["tokens"] for r in drows)
            bpbs = [r[key]["bpb"] for r in drows if r[key]["bpb"] is not None]
            block[key] = {"bpb": float(nll.sum() / math.log(2) / nbytes), "nll_nats": float(nll.sum()),
                          "tokens": tokens, "tokens_per_byte": tokens/nbytes,
                          "tokens_per_word": tokens/block["words"] if block["words"] else None,
                          "per_text_bpb_quantiles": {str(p): float(np.quantile(bpbs, p)) for p in (0, .05, .25, .5, .75, .95, 1)}}
        rs = np.empty(1000, dtype=np.float64)
        for i in range(1000):
            sample = rng.integers(0, len(drows), size=len(drows))
            rs[i] = float(g[sample].sum() / q[sample].sum() - 1)
        block["r"] = float(g.sum() / q.sum() - 1)
        block["r_bootstrap_95ci"] = list(map(float, np.quantile(rs, [.025, .975])))
        block["bootstrap_replicates"] = 1000
        block["bootstrap_seed"] = SEED
        domains[domain] = block
    return domains


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expect CUDA_VISIBLE_DEVICES to expose exactly one eligible GPU")
    device = torch.device("cuda:0")
    free, total = torch.cuda.mem_get_info(device)
    if free < 12 * (1 << 30):
        raise RuntimeError(f"phase1 requires >=12 GiB free, found {free / (1<<30):.3f} GiB")
    started = time.time()
    manifests = {
        "qwen3_base": file_manifest(QWEN_REPO, QWEN_COMMIT, "apache-2.0"),
        "gemma3_1b": file_manifest(GEMMA_REPO, GEMMA_COMMIT, "Gemma Terms of Use (unsloth mirror)"),
        "bidirlm_tokenizer": file_manifest(BIDIR_REPO, BIDIR_COMMIT, "apache-2.0"),
    }
    for name, expected in EXPECTED_GEMMA.items():
        found = manifests["gemma3_1b"]["sha256"].get(name)
        if found != expected:
            raise RuntimeError(f"Gemma {name} SHA256 mismatch: {found} != {expected}")
    print("[manifest] Gemma required SHA256 checks passed", flush=True)
    texts_a, sample_manifest, a_detail = prepare_domain_a()
    texts_b, b_detail = prepare_domain_b()
    manifest = {"protocol": "bidirlm-1b-original-decoder-bpb-v1", "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "models": manifests, "environment": {"torch": torch.__version__, "transformers": __import__("transformers").__version__,
                    "sentencepiece": __import__("sentencepiece").__version__, "gpu_visible_name": torch.cuda.get_device_name(device),
                    "gpu_original_id": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu_free_bytes_at_launch": free,
                    "gpu_total_bytes": total}, "domains": {"A": a_detail, "B": b_detail},
                "sample_manifest_sha256": sha_file(ART / "sample_manifest.json"),
                "normalization": "original string; no normalization", "domain_a_cut": "at last ASCII space before 2,000 characters",
                "domain_b_dedup": "exact string", "qwen_start_id": 151643,
                "gemma_tokenizer": "slow GemmaTokenizer from tokenizer.model"}
    jwrite(ART / "manifest.json", manifest)
    texts = [texts_a, texts_b]
    measurements = {}
    model_details = {}
    for key, repo, commit in (("qwen3_base", QWEN_REPO, QWEN_COMMIT), ("gemma3_1b", GEMMA_REPO, GEMMA_COMMIT)):
        model_details[key], measurements[key] = measure_model(key, snapshot(repo, commit), texts, device)
        print(f"[{key}] completed in {model_details[key]['elapsed_seconds']/60:.2f} min", flush=True)
    domains = summarize(texts_a, texts_b, measurements)
    results = {"protocol": manifest["protocol"], "domains": domains, "models": model_details,
               "elapsed_seconds": time.time()-started, "gpu_id": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "gpu_hours": (time.time()-started)/3600}
    jwrite(ART / "bpb_results.json", results)
    r_a, r_b = domains["A"]["r"], domains["B"]["r"]
    decision = "CLOSE" if r_a > .05 else "PROCEED" if r_b <= .05 else "STOP-CONFLICT"
    gate = {"decision": decision, "r_A": r_a, "r_B": r_b, "threshold": .05,
            "reason": "r_A > 0.05" if decision == "CLOSE" else "r_A <= 0.05 and r_B <= 0.05" if decision == "PROCEED" else "r_A <= 0.05 and r_B > 0.05",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "bpb_results_sha256": sha_file(ART / "bpb_results.json")}
    jwrite(ART / "gate.json", gate)
    print(f"[gate] {decision} r_A={r_a:.8f} r_B={r_b:.8f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
