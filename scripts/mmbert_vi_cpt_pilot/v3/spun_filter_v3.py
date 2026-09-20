#!/usr/bin/env python3
"""v3 §3.1: rebuild the spun/spam filter so that genuine informal Vietnamese is not dropped.

  * S1 reference = v2 reference (mainstream-news bin 7-8 documents + CC-BY v3 review sentences) **plus** audited
    informal documents (v1 `discussion`), training the same syllable-level interpolated Kneser-Ney LM (pure NumPy).
  * S2 = logistic regression on char 3-5-gram TF-IDF + S1 features, trained on the 1 200 v2 labels **plus** the newly
    hand-labelled informal documents (v3 §3.1), which are all labelled *not* spun/spam.
  * Thresholds are locked before the pool is scored: S2 at the probability whose 5-fold recall(spun) >= 0.90, S1 at the
    5th percentile of the reference.  The same rule as v2 applies (`S2 >= t2` OR `S1 < t1`).

Subcommands: build | score-pool
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v3 as C3  # noqa: E402
import spun_filter as S  # noqa: E402  (v2 implementation, reused read-only)

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

MODEL_DIR = C3.ART_V3 / "spun_filter_v3"
MAX_SYLLABLES = 70_000_000
EXCERPT_CHARS = S.EXCERPT_CHARS
V3_INFORMAL_LABELS = C3.ART_V3 / "data_audit_v3" / "informal_labels.jsonl"


def informal_reference_texts(limit: int = 4000) -> list[str]:
    """Audited/informal Vietnamese documents used to broaden the S1 reference (§3.1)."""
    id_reg = C3.id_register_by_pool_index()
    pool = [r for r in C3.C2.pool_table() if r["split"] == "pool" and r["group"] == "register_discussion"]
    pool.sort(key=lambda r: (-(id_reg.get(r["pool_index"], 0.0)), r["pool_index"]))
    picked = pool[:limit]
    texts: list[str] = []
    by_part: dict[str, list[dict]] = collections.defaultdict(list)
    for r in picked:
        by_part[r["part"]].append(r)
    for part, rows in by_part.items():
        need = {r["kh64"]: r for r in rows}
        for kh64, text in C3.C2.part_texts(part, set(need)).items():
            texts.append(text)
    return texts


def v3_rows() -> list[dict]:
    rows = S.training_rows()          # 800 v1 audit labels + 400 v2 hand labels
    n_before = len(rows)
    if V3_INFORMAL_LABELS.is_file():
        for r in C3.iter_jsonl(V3_INFORMAL_LABELS):
            if not r.get("informal_real"):
                continue          # only the hand-labelled *real informal* documents are added as not-spun examples
            rows.append({"source": "v3:informal", "id": r.get("id"), "text": (r.get("excerpt") or "")[:EXCERPT_CHARS],
                         "spun": 0, "real_vietnamese": True, "boilerplate": bool(r.get("boilerplate", False)),
                         "spam": bool(r.get("spam", False))})
    return rows, n_before


def build_v3_model(model_dir: Path, reuse_s1: bool = False) -> dict:
    t0 = time.time()
    model_dir.mkdir(parents=True, exist_ok=True)
    S.MODEL_DIR = model_dir
    S._MODEL_CACHE = None
    # ---- S1 reference: v2 reference + informal documents
    if reuse_s1 and (model_dir / "lm_counts.npz").is_file():
        m = S.load_model()
        vocab = m["vocab"]
        ref_meta = C3.read_json(model_dir / "reference_meta.json")
        all_texts = None
        print(f"[s1v3] reusing the existing S1 model ({m['unique_bigrams']} bigrams / {m['unique_trigrams']} trigrams)", flush=True)
    else:
        ref_texts, ref_meta = S.reference_texts(32)
        informal = informal_reference_texts()
        all_texts = ref_texts + informal
        vocab, counter = S.build_vocab(all_texts)
        ids = S.encode(all_texts, vocab, MAX_SYLLABLES)
        np.save(model_dir / "reference_ids.npy", ids)
        C3.write_json(model_dir / "vocab.json", {"vocab": vocab, "size": len(vocab), "unk_id": len(vocab),
                                                 "distinct_syllables": len(counter)})
        m = S.build_counts(ids, len(vocab))
        d2, d3 = S.discount(m["bigram_counts"]), S.discount(m["trigram_counts"])
        m["d2"], m["d3"] = d2, d3
        np.savez(model_dir / "lm_counts.npz", V=m["V"], bigram_keys=m["bigram_keys"], bigram_counts=m["bigram_counts"],
                 trigram_keys=m["trigram_keys"], trigram_counts=m["trigram_counts"], c1_ctx=m["c1_ctx"], n1p_first=m["n1p_first"],
                 pcont=m["pcont"], tri_ctx_keys=m["tri_ctx_keys"], tri_ctx_n1p=m["tri_ctx_n1p"])
        C3.write_json(model_dir / "lm_meta.json", {k: v for k, v in m.items() if isinstance(v, (int, float, str))})
        C3.write_json(model_dir / "reference_meta.json", {**ref_meta, "v2_reference_documents": len(ref_texts),
                                                          "informal_reference_documents": len(informal), "syllables": int(ids.size),
                                                          "vocab_size": len(vocab), "D2": d2, "D3": d3})
        print(f"[s1v3] reference {len(all_texts)} docs / {ids.size} syllables, vocab {len(vocab)}, "
              f"{m['unique_bigrams']} bigrams / {m['unique_trigrams']} trigrams ({time.time()-t0:.0f}s)", flush=True)

    # ---- S2 on the augmented labels
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    rows, n_v2 = v3_rows()
    texts = [r["text"] for r in rows]
    y = np.array([r["spun"] for r in rows], dtype=np.int64)
    s1 = S.s1_score(texts)
    s1_mean = np.nan_to_num(np.array([a for a, _ in s1], dtype=np.float64), nan=-5.0)
    s1_unseen = np.nan_to_num(np.array([b for _, b in s1], dtype=np.float64), nan=1.0)
    extra = np.c_[s1_mean, s1_unseen, np.log1p([len(t) for t in texts])]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=300_000, sublinear_tf=True)
    X = sp.hstack([vec.fit_transform(texts), sp.csr_matrix((extra - extra.mean(0)) / (extra.std(0) + 1e-9))]).tocsr()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=C3.SEED)
    oof = np.zeros(len(y))
    aucs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", solver="liblinear")
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], oof[te]))
    auc = float(roc_auc_score(y, oof))
    order = np.argsort(-oof)
    ys = y[order]
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    recall = tp / max(1, y.sum())
    ok = np.where(recall >= 0.90)[0]
    thr_idx = int(ok[0]) if ok.size else len(ys) - 1
    threshold = float(oof[order][thr_idx])
    precision = float(tp[thr_idx] / k[thr_idx])
    recall_at = float(recall[thr_idx])
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", solver="liblinear")
    clf.fit(X, y)
    with (model_dir / "s2_model.pkl").open("wb") as fh:
        pickle.dump({"vectorizer": vec, "clf": clf, "extra_mean": extra.mean(0), "extra_std": extra.std(0) + 1e-9,
                     "excerpt_chars": EXCERPT_CHARS}, fh)
    # ---- S1 threshold from the reference documents
    if all_texts is None:
        thr_prev = C3.read_json(model_dir / "thresholds.json")
        p5, p2 = float(thr_prev["threshold_s1_p5"]), float(thr_prev["threshold_s1_p2"])
        n_ref_scored = thr_prev.get("s1_reference_docs_scored", 4000)
    else:
        idx = np.linspace(0, len(all_texts) - 1, num=min(4000, len(all_texts))).astype(int)
        ref_scores = np.array([s[0] for s in S.s1_score([all_texts[i] for i in idx])])
        ref_scores = ref_scores[~np.isnan(ref_scores)]
        p5, p2 = float(np.percentile(ref_scores, 5)), float(np.percentile(ref_scores, 2))
        n_ref_scored = len(ref_scores)
    report = {"created_at": C3.common.now(), "rows": len(rows), "rows_v2": n_v2, "rows_v3_informal": len(rows) - n_v2,
              "positives": int(y.sum()), "cv_auc": round(auc, 4), "cv_auc_folds": [round(a, 4) for a in aucs],
              "threshold_s2": threshold, "precision_at_threshold": precision, "recall_at_threshold": recall_at,
              "threshold_s1_p5": p5, "threshold_s1_p2": p2, "s1_reference_docs_scored": int(n_ref_scored),
              "elapsed_s": round(time.time() - t0, 1),
              "rule": "drop when S2 >= threshold_s2 OR S1_full < threshold_s1_p5"}
    C3.write_json(model_dir / "cv_report.json", report)
    C3.write_json(model_dir / "thresholds.json", {"threshold_s2": threshold, "threshold_s1_p5": p5, "threshold_s1_p2": p2,
                                                  "locked_at": C3.common.now(), "locked_before_pool_scoring": True})
    print(json.dumps(report, ensure_ascii=False, indent=1), flush=True)
    return report


def _score_part_v3(job):
    part, rows = job
    out = []
    texts, metas = [], []
    by_kh = {r["kh64"]: r for r in rows}
    for kh64, text in C3.C2.part_texts(part, set(by_kh)).items():
        meta = by_kh[kh64]
        texts.append(text)
        metas.append(meta)
        if len(texts) >= 256:
            res = S.score_texts(texts)
            out.extend({"pool_index": m["pool_index"], "s1": (None if a != a else float(a)), "s2": float(p), "drop": bool(d)}
                       for m, a, p, d in zip(metas, res["s1_mean"], res["s2_prob"], res["drop"]))
            texts, metas = [], []
    if texts:
        res = S.score_texts(texts)
        out.extend({"pool_index": m["pool_index"], "s1": (None if a != a else float(a)), "s2": float(p), "drop": bool(d)}
                   for m, a, p, d in zip(metas, res["s1_mean"], res["s2_prob"], res["drop"]))
    return part, out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--reuse-s1", action="store_true")
    sp_ = sub.add_parser("score-pool"); sp_.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()
    if args.cmd == "build":
        build_v3_model(MODEL_DIR, reuse_s1=args.reuse_s1)
        return 0
    # score-pool
    t0 = time.time()
    S.MODEL_DIR = MODEL_DIR
    S._MODEL_CACHE = None
    pool = [r for r in C3.C2.pool_table() if r["split"] == "pool"]
    by_part: dict[str, list[dict]] = collections.defaultdict(list)
    for r in pool:
        by_part[r["part"]].append(r)
    out_dir = C3.DATA_V3 / "spun_scores_v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = sorted(by_part.items())
    n_rows = 0
    with mp.Pool(args.workers) as pool_exec:
        for i, (part, rows) in enumerate(pool_exec.imap_unordered(_score_part_v3, jobs), 1):
            with (out_dir / f"{part}.jsonl").open("w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            n_rows += len(rows)
            if i % 48 == 0:
                print(f"[s2v3] {i}/{len(jobs)} parts ({time.time()-t0:.0f}s)", flush=True)
    C3.write_json(out_dir / "manifest.json", {"created_at": C3.common.now(), "rows": n_rows, "parts": len(jobs),
                                              "workers": args.workers, "elapsed_s": round(time.time() - t0, 1),
                                              "thresholds": C3.read_json(MODEL_DIR / "thresholds.json")})
    print(f"[s2v3] scored {n_rows} pool documents in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
