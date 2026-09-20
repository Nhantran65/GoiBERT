#!/usr/bin/env python3
"""v2 §2 "vá 1": document-level filter for spun / machine-translated Vietnamese text (S1 + S2).

  S1 — syllable-level Kneser-Ney bigram/trigram LM (pure Python/NumPy; kenlm is not installed in this venv) trained on a
       clean reference corpus: HPLT pool documents from the mainstream-news whitelist in bins 7-8 plus the CC-BY v3
       Vietnamese review sentences.  Documents are scored by mean log-probability per syllable and the unseen-bigram rate.
  S2 — logistic regression on char 3-5-gram TF-IDF plus the S1 features, trained on the 800 v1 audit labels
       (`machine_translated_or_generated`) plus >= 400 documents labelled by hand for v2 (seed 42).

Both thresholds are locked before the whole pool is scored:
  * S2 threshold = the probability whose 5-fold cross-validated recall(spun) is >= 0.90 (precision reported);
  * S1 threshold = the 5th percentile of the S1 score of the clean reference documents.
A document is dropped when `S2 >= threshold` OR `S1 < reference percentile`.  If the filter drops more than 15 % of the
audited-true UGC documents (`--side-effect`), the S1 threshold is relaxed to the 2nd percentile and recorded.

Subcommands: build-reference | fit | score-pool | score-texts
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v2 as C  # noqa: E402

import numpy as np  # noqa: E402

MODEL_DIR = C.ART_V2 / "spun_filter"
UNK = "<unk>"
MAX_SYLLABLES = 40_000_000        # reference corpus size for the LM
VOCAB_CAP = 400_000
D2_DEFAULT, D3_DEFAULT = 0.75, 0.75


# ------------------------------------------------------------------ reference corpus
def _reference_part(job) -> list[str]:
    part, rows = job
    need = {r["kh64"]: r for r in rows}
    out: list[str] = []
    for kh64, text in C.part_texts(part, set(need)).items():
        out.append(text)
    return out


def reference_texts(workers: int = 32) -> tuple[list[str], dict]:
    """Mainstream-news pool documents (bins 7-8) + v3 Vietnamese review sentences."""
    import pyarrow.parquet as pq
    pool = {r["pool_index"]: r for r in C.pool_table()}
    wanted: dict[str, list[dict]] = collections.defaultdict(list)
    for r in pool.values():
        if r["split"] == "pool" and r["bin"] in ("7", "8") and C.host_matches(r["host"], C.MAINSTREAM_NEWS_HOSTS):
            wanted[r["part"]].append(r)
    texts: list[str] = []
    with mp.Pool(workers) as pool_exec:
        for chunk in pool_exec.imap_unordered(_reference_part, sorted(wanted.items())):
            texts.extend(chunk)
    docs_used = len(texts)
    v3_sentences = 0
    if C.V3_PARQUET.is_file():
        pf = pq.ParquetFile(str(C.V3_PARQUET))
        for rg in range(pf.num_row_groups):
            tb = pf.read_row_group(rg, columns=["language", "sentence"])
            for lang, sent in zip(tb.column("language").to_pylist(), tb.column("sentence").to_pylist()):
                if lang == "vi" and sent and len(sent) > 20 and C.common.is_vietnamese(sent):
                    texts.append(sent)
                    v3_sentences += 1
    meta = {"mainstream_news_documents": docs_used, "v3_vi_sentences": v3_sentences,
            "hosts": sorted(C.MAINSTREAM_NEWS_HOSTS), "v3_parquet": str(C.V3_PARQUET.name), "v3_revision": C.V3_REVISION,
            "license_v3": "CC BY 4.0 (phamluan/vn_msa_sar_v3 @ 6f3f17b3)"}
    return texts, meta


def build_vocab(texts: list[str], cap: int = VOCAB_CAP):
    counter: collections.Counter = collections.Counter()
    for t in texts:
        counter.update(C.syllables(t))
    vocab = {w: i for i, (w, _) in enumerate(counter.most_common(cap))}
    return vocab, counter


def encode(texts: list[str], vocab: dict[str, int], max_syllables: int = MAX_SYLLABLES):
    ids = np.empty(max_syllables, dtype=np.int64)
    n = 0
    unk = len(vocab)
    for t in texts:
        s = C.syllables(t)
        if len(s) < 2:
            continue
        arr = np.fromiter((vocab.get(w, unk) for w in s), dtype=np.int64, count=len(s))
        take = min(len(arr), max_syllables - n)
        ids[n:n + take] = arr[:take]
        n += take
        if n >= max_syllables:
            break
    return ids[:n]


def build_counts(ids: np.ndarray, vocab_size: int):
    """Bigram/trigram count tables (sorted keys + counts) and the continuation statistics Kneser-Ney needs."""
    V = np.int64(vocab_size + 1)
    b = ids[:-1] * V + ids[1:]
    t = (ids[:-2] * V + ids[1:-1]) * V + ids[2:]
    b_keys, b_counts = np.unique(b, return_counts=True)
    t_keys, t_counts = np.unique(t, return_counts=True)
    c1_ctx = np.bincount((b_keys // V).astype(np.int64), weights=b_counts, minlength=vocab_size + 1).astype(np.float64)
    n1p_first = np.bincount((b_keys // V).astype(np.int64), minlength=vocab_size + 1).astype(np.float64)
    pcont_num = np.bincount((b_keys % V).astype(np.int64), weights=np.ones_like(b_counts, dtype=np.float64), minlength=vocab_size + 1)
    pcont = pcont_num / max(1.0, pcont_num.sum())
    # distinct successors per bigram context, for the trigram interpolation weight
    uv = t_keys // V
    uv_uniq, uv_counts = np.unique(uv, return_counts=True)
    return {"V": V, "vocab_size": vocab_size, "bigram_keys": b_keys, "bigram_counts": b_counts.astype(np.float64),
            "trigram_keys": t_keys, "trigram_counts": t_counts.astype(np.float64), "c1_ctx": c1_ctx,
            "n1p_first": n1p_first, "pcont": pcont, "tri_ctx_keys": uv_uniq, "tri_ctx_n1p": uv_counts.astype(np.float64),
            "n_bigrams": int(b.size), "n_trigrams": int(t.size), "unique_bigrams": int(b_keys.size), "unique_trigrams": int(t_keys.size)}


def discount(counts: np.ndarray) -> float:
    n1 = float(np.sum(counts == 1))
    n2 = float(np.sum(counts == 2))
    return n1 / (n1 + 2 * n2) if (n1 + 2 * n2) > 0 else 0.75


def lookup(sorted_keys: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Vectorised count lookup; 0 for keys that are absent."""
    if queries.size == 0:
        return np.zeros(0, dtype=np.float64)
    pos = np.searchsorted(sorted_keys, queries)
    pos = np.minimum(pos, max(0, sorted_keys.size - 1))
    hit = (sorted_keys.size > 0) & (sorted_keys[pos] == queries)
    return hit.astype(np.float64)


def scores_from_ids(ids: np.ndarray, m: dict, d2: float, d3: float) -> tuple[float, float]:
    """(mean log-probability per syllable, unseen-bigram rate) under the interpolated Kneser-Ney model."""
    if ids.size < 3:
        return float("nan"), float("nan")
    V = m["V"]
    b = ids[:-1] * V + ids[1:]
    pos = np.searchsorted(m["bigram_keys"], b)
    pos = np.minimum(pos, m["bigram_keys"].size - 1)
    c2 = np.where(m["bigram_keys"][pos] == b, m["bigram_counts"][pos], 0.0)
    ctx = ids[:-1]
    c_ctx = m["c1_ctx"][ctx]
    lam2 = d2 * m["n1p_first"][ctx] / np.maximum(c_ctx, 1e-12)
    p2 = np.where(c_ctx > 0, (np.maximum(c2 - d2, 0.0) + lam2 * m["pcont"][ids[1:]]) / np.maximum(c_ctx, 1e-12), m["pcont"][ids[1:]])
    t = (ids[:-2] * V + ids[1:-1]) * V + ids[2:]
    pos3 = np.searchsorted(m["trigram_keys"], t)
    pos3 = np.minimum(pos3, m["trigram_keys"].size - 1)
    c3 = np.where(m["trigram_keys"][pos3] == t, m["trigram_counts"][pos3], 0.0)
    uv = b[:-1]
    posc = np.searchsorted(m["tri_ctx_keys"], uv)
    posc = np.minimum(posc, m["tri_ctx_keys"].size - 1)
    n1p_uv = np.where(m["tri_ctx_keys"][posc] == uv, m["tri_ctx_n1p"][posc], 0.0)
    c_uv = c2[:-1]
    lam3 = d3 * n1p_uv / np.maximum(c_uv, 1e-12)
    p3 = np.where(c_uv > 0, (np.maximum(c3 - d3, 0.0) + lam3 * p2[:-1]) / np.maximum(c_uv, 1e-12), p2[:-1])
    logp = np.log(np.maximum(np.concatenate([p2, p3]), 1e-12))
    unseen = float(np.mean(c2 == 0))
    return float(logp.mean()), unseen


_MODEL_CACHE: dict | None = None


def load_model() -> dict:
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    m = dict(np.load(MODEL_DIR / "lm_counts.npz", allow_pickle=False))
    m.update(C.read_json(MODEL_DIR / "lm_meta.json"))
    vocab = C.read_json(MODEL_DIR / "vocab.json")["vocab"]
    m["vocab"] = vocab
    _MODEL_CACHE = m
    return m


def s1_score(texts: list[str], model: dict | None = None) -> list[tuple[float, float]]:
    m = model or load_model()
    vocab = m["vocab"]
    unk = len(vocab)
    out = []
    for t in texts:
        s = C.syllables(t)
        if len(s) < 3:
            out.append((float("nan"), float("nan")))
            continue
        ids = np.fromiter((vocab.get(w, unk) for w in s), dtype=np.int64, count=len(s))
        out.append(scores_from_ids(ids, m, float(m["d2"]), float(m["d3"])))
    return out


# ------------------------------------------------------------------ subcommands
def cmd_build_reference(args) -> int:
    t0 = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    texts, meta = reference_texts(args.workers)
    vocab, counter = build_vocab(texts)
    ids = encode(texts, vocab, args.max_syllables)
    np.save(MODEL_DIR / "reference_ids.npy", ids)
    C.write_json(MODEL_DIR / "vocab.json", {"vocab": vocab, "size": len(vocab), "unk_id": len(vocab),
                                            "vocab_cap": VOCAB_CAP, "distinct_syllables": len(counter)})
    C.write_json(MODEL_DIR / "reference_meta.json", {**meta, "documents": len(texts), "syllables": int(ids.size),
                                                     "vocab_size": len(vocab), "elapsed_s": round(time.time() - t0, 1)})
    print(f"[s1] reference: {len(texts)} documents, {ids.size} syllables, vocab {len(vocab)} ({time.time()-t0:.0f}s)")
    return 0


def cmd_fit(args) -> int:
    t0 = time.time()
    ids = np.load(MODEL_DIR / "reference_ids.npy")
    vocab = C.read_json(MODEL_DIR / "vocab.json")["vocab"]
    m = build_counts(ids, len(vocab))
    d2, d3 = discount(m["bigram_counts"]), discount(m["trigram_counts"])
    m["d2"], m["d3"] = d2, d3
    np.savez(MODEL_DIR / "lm_counts.npz", V=m["V"], bigram_keys=m["bigram_keys"], bigram_counts=m["bigram_counts"],
             trigram_keys=m["trigram_keys"], trigram_counts=m["trigram_counts"], c1_ctx=m["c1_ctx"], n1p_first=m["n1p_first"],
             pcont=m["pcont"], tri_ctx_keys=m["tri_ctx_keys"], tri_ctx_n1p=m["tri_ctx_n1p"])
    C.write_json(MODEL_DIR / "lm_meta.json", {k: v for k, v in m.items() if isinstance(v, (int, float, str))})
    print(f"[s1] LM built: {m['unique_bigrams']} bigrams, {m['unique_trigrams']} trigrams, D2={d2:.3f} D3={d3:.3f} ({time.time()-t0:.0f}s)")
    return 0


# ------------------------------------------------------------------ S2: supervised classifier on top of S1
EXCERPT_CHARS = 1200      # same window as the v1 hand audit and as the v2 hand labels


def training_rows() -> list[dict]:
    """800 v1 audit labels + the v2 hand labels (both keyed by their 1 200-character excerpt)."""
    rows: list[dict] = []
    samples = {}
    for r in C.iter_jsonl(C.ART_V1 / "data_audit" / "samples.jsonl"):
        samples[(r["cell"], r["index"])] = r
    for lab in C.iter_jsonl(C.ART_V1 / "data_audit" / "labels.jsonl"):
        s = samples.get((lab["cell"], lab["index"]))
        if s is None:
            continue
        rows.append({"source": f"v1:{lab['cell']}", "id": lab.get("id"), "text": s.get("excerpt", "")[:EXCERPT_CHARS],
                     "spun": int(bool(lab["machine_translated_or_generated"])), "real_vietnamese": bool(lab["real_vietnamese"]),
                     "boilerplate": bool(lab["boilerplate_majority"]), "spam": bool(lab["adult_or_spam"])})
    v2_path = C.ART_V2 / "data_audit_v2" / "s2_labels.jsonl"
    if v2_path.is_file():
        for r in C.iter_jsonl(v2_path):
            rows.append({"source": f"v2:{r.get('stratum','random')}", "id": r.get("id"), "text": r.get("excerpt", "")[:EXCERPT_CHARS],
                         "spun": int(bool(r["spun"])), "real_vietnamese": bool(r.get("real_vietnamese", True)),
                         "boilerplate": bool(r.get("boilerplate", False)), "spam": bool(r.get("spam", False))})
    return rows


def cmd_fit_s2(args) -> int:
    t0 = time.time()
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    import scipy.sparse as sp

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    rows = training_rows()
    texts = [r["text"] for r in rows]
    y = np.array([r["spun"] for r in rows], dtype=np.int64)
    s1 = s1_score(texts)
    s1_mean = np.array([s[0] for s in s1], dtype=np.float64)
    s1_unseen = np.array([s[1] for s in s1], dtype=np.float64)
    s1_mean = np.nan_to_num(s1_mean, nan=np.nanmedian(s1_mean))
    s1_unseen = np.nan_to_num(s1_unseen, nan=1.0)
    extra = np.c_[s1_mean, s1_unseen, np.log1p([len(t) for t in texts])]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=300_000, sublinear_tf=True)
    X = sp.hstack([vec.fit_transform(texts), sp.csr_matrix((extra - extra.mean(0)) / (extra.std(0) + 1e-9))]).tocsr()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.SEED)
    oof = np.zeros(len(y))
    aucs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", solver="liblinear")
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], oof[te]))
    auc = float(roc_auc_score(y, oof))
    # threshold: recall(spun) >= 0.90 on the pooled out-of-fold predictions
    order = np.argsort(-oof)
    ys = y[order]
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    recall = tp / max(1, y.sum())
    ok = np.where(recall >= args.target_recall)[0]
    thr_idx = int(ok[0]) if ok.size else len(ys) - 1
    threshold = float(oof[order][thr_idx])
    precision = float(tp[thr_idx] / k[thr_idx])
    recall_at = float(recall[thr_idx])
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", solver="liblinear")
    clf.fit(X, y)

    import pickle
    with (MODEL_DIR / "s2_model.pkl").open("wb") as fh:
        pickle.dump({"vectorizer": vec, "clf": clf, "extra_mean": extra.mean(0), "extra_std": extra.std(0) + 1e-9,
                     "excerpt_chars": EXCERPT_CHARS}, fh)
    # S1 threshold = 5th percentile of the clean reference documents
    ref_ids = np.load(MODEL_DIR / "reference_ids.npy")
    m = load_model()
    # score reference documents individually (they were concatenated for training; re-read a sample of them)
    ref_texts, _ = reference_texts()
    idx = np.linspace(0, len(ref_texts) - 1, num=min(4000, len(ref_texts))).astype(int)
    ref_scores = [s[0] for s in s1_score([ref_texts[i] for i in idx], m)]
    ref_scores = np.array([s for s in ref_scores if not math.isnan(s)])
    p5, p2 = float(np.percentile(ref_scores, 5)), float(np.percentile(ref_scores, 2))
    report = {"created_at": C.common.now(), "rows": len(rows), "positives": int(y.sum()),
              "sources": {s: int(sum(1 for r in rows if r["source"].startswith(s))) for s in ("v1", "v2")},
              "cv_auc_folds": [round(a, 4) for a in aucs], "cv_auc": round(auc, 4),
              "target_recall": args.target_recall, "threshold_s2": threshold, "precision_at_threshold": precision, "recall_at_threshold": recall_at,
              "n_features": int(X.shape[1]), "s1_reference_docs_scored": int(len(ref_scores)),
              "threshold_s1_p5": p5, "threshold_s1_p2": p2, "s1_reference_mean": float(ref_scores.mean()),
              "rule": "drop when S2 >= threshold_s2 OR S1_full < threshold_s1_p5 (p2 if the UGC side-effect check fails)",
              "elapsed_s": round(time.time() - t0, 1)}
    C.write_json(MODEL_DIR / "cv_report.json", report)
    C.write_json(MODEL_DIR / "thresholds.json", {"threshold_s2": threshold, "threshold_s1_p5": p5, "threshold_s1_p2": p2,
                                                 "locked_at": C.common.now(), "locked_before_pool_scoring": True})
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


_S2 = None


def load_s2():
    global _S2
    if _S2 is None:
        import pickle
        import scipy.sparse as sp
        with (MODEL_DIR / "s2_model.pkl").open("rb") as fh:
            _S2 = pickle.load(fh)
        _S2["sp"] = sp
    return _S2


def s2_prob(texts: list[str], s1: list[tuple[float, float]] | None = None) -> np.ndarray:
    model = load_s2()
    sp = model["sp"]
    X = model["vectorizer"].transform(texts)
    if s1 is None:
        s1 = s1_score(texts)
    mean = np.array([0.0 if math.isnan(a) else a for a, _ in s1])
    unseen = np.array([1.0 if math.isnan(b) else b for _, b in s1])
    extra = np.c_[mean, unseen, np.log1p([len(t) for t in texts])]
    z = (extra - model["extra_mean"]) / model["extra_std"]
    X = sp.hstack([X, sp.csr_matrix(z)]).tocsr()
    return model["clf"].predict_proba(X)[:, 1]


def score_texts(texts: list[str], s1_override: list[tuple[float, float]] | None = None) -> dict:
    """Full v2 vá-1 decision for a batch of texts (S1 on the full text, S2 on the first 1200 characters)."""
    thr = C.read_json(MODEL_DIR / "thresholds.json")
    s1_full = s1_override or s1_score(texts)
    excerpts = [t[:EXCERPT_CHARS] for t in texts]
    s1_ex = s1_score(excerpts)
    prob = s2_prob(excerpts, s1_ex)
    mean = np.array([np.nan if math.isnan(a) else a for a, _ in s1_full])
    unseen = np.array([np.nan if math.isnan(b) else b for _, b in s1_full])
    s1_thr = float(thr.get("threshold_s1", thr["threshold_s1_p5"]))
    drop = (prob >= float(thr["threshold_s2"])) | (np.nan_to_num(mean, nan=0.0) < s1_thr)
    return {"s1_mean": mean, "s1_unseen": unseen, "s2_prob": prob, "drop": drop,
            "threshold_s2": float(thr["threshold_s2"]), "threshold_s1": s1_thr}


def _score_part(job) -> dict:
    part, rows = job
    needed = {r["kh64"]: r for r in rows}
    out = []
    texts = []
    metas = []
    for kh64, text in C.part_texts(part, set(needed)).items():
        r = needed[kh64]
        if r is not None:
            texts.append(text)
            metas.append(r)
            if len(texts) >= 256:
                res = score_texts(texts)
                out.extend({"pool_index": m["pool_index"], "s1_mean": float(a) if not math.isnan(a) else None,
                            "s1_unseen": float(b) if not math.isnan(b) else None, "s2_prob": float(p), "drop": bool(d)}
                           for m, a, b, p, d in zip(metas, res["s1_mean"], res["s1_unseen"], res["s2_prob"], res["drop"]))
                texts, metas = [], []
    if texts:
        res = score_texts(texts)
        out.extend({"pool_index": m["pool_index"], "s1_mean": float(a) if not math.isnan(a) else None,
                    "s1_unseen": float(b) if not math.isnan(b) else None, "s2_prob": float(p), "drop": bool(d)}
                   for m, a, b, p, d in zip(metas, res["s1_mean"], res["s1_unseen"], res["s2_prob"], res["drop"]))
    return {"part": part, "rows": out}


def cmd_score_pool(args) -> int:
    t0 = time.time()
    pool = [r for r in C.pool_table() if r["split"] == "pool"]
    by_part: dict[str, list[dict]] = collections.defaultdict(list)
    for r in pool:
        by_part[r["part"]].append(r)
    jobs = [(part, rows) for part, rows in sorted(by_part.items())]
    out_dir = C.DATA_V2 / "spun_scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with mp.Pool(args.workers) as pool_exec:
        for i, res in enumerate(pool_exec.imap_unordered(_score_part, jobs), 1):
            with (out_dir / f"{res['part']}.jsonl").open("w", encoding="utf-8") as fh:
                for row in res["rows"]:
                    fh.write(json.dumps(row) + "\n")
            results.append({"part": res["part"], "rows": len(res["rows"])})
            if i % 32 == 0:
                print(f"[s2] scored {i}/{len(jobs)} parts ({time.time()-t0:.0f}s)", flush=True)
    n_rows = sum(r["rows"] for r in results)
    C.write_json(C.DATA_V2 / "spun_scores_manifest.json", {"created_at": C.common.now(), "parts": len(results), "rows": n_rows,
                                                           "workers": args.workers, "elapsed_s": round(time.time() - t0, 1),
                                                           "rule": C.read_json(MODEL_DIR / "thresholds.json")})
    print(f"[s2] scored {n_rows} pool documents in {time.time()-t0:.0f}s -> {out_dir}")
    return 0


def cmd_score_texts(args) -> int:
    texts = [json.loads(l)["text"] for l in Path(args.input).open(encoding="utf-8")]
    res = score_texts(texts)
    out = [{"i": i, "s1_mean": None if math.isnan(a) else float(a), "s1_unseen": None if math.isnan(b) else float(b),
            "s2_prob": float(p), "drop": bool(d)} for i, (a, b, p, d) in enumerate(zip(res["s1_mean"], res["s1_unseen"], res["s2_prob"], res["drop"]))]
    C.write_json(Path(args.output), {"thresholds": {"s2": res["threshold_s2"], "s1": res["threshold_s1"]}, "rows": out})
    print(json.dumps({"rows": len(out), "drop_rate": float(np.mean([r["drop"] for r in out])) if out else None}))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-reference"); b.add_argument("--max-syllables", type=int, default=MAX_SYLLABLES)
    b.add_argument("--workers", type=int, default=32); b.set_defaults(func=cmd_build_reference)
    f = sub.add_parser("fit"); f.set_defaults(func=cmd_fit)
    f2 = sub.add_parser("fit-s2"); f2.add_argument("--target-recall", type=float, default=0.90); f2.set_defaults(func=cmd_fit_s2)
    sp = sub.add_parser("score-pool"); sp.add_argument("--workers", type=int, default=48); sp.set_defaults(func=cmd_score_pool)
    st = sub.add_parser("score-texts"); st.add_argument("--input", required=True); st.add_argument("--output", required=True); st.set_defaults(func=cmd_score_texts)
    args = ap.parse_args()
    raise SystemExit(args.func(args))
