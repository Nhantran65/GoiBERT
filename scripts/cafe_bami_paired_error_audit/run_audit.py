#!/usr/bin/env python3
"""Read-only paired error audit CafeBERT vs BamiBERT on validation (docs/CAFEBERT_BAMIBERT_PAIRED_ERROR_AUDIT_PLAN.md).

No model, no GPU, no test.  Inputs: six saved validation_predictions.json (E14 BamiBERT / CafeBERT seed 42) and the
frozen validation text via goibert.data.load_task.  Outputs (artifacts/cafe_bami_paired_error_audit/):
  manifest.json   provenance/hashes, correctness gate, six recomputed F1 (evaluator + independent count)
  aggregate.json  full-validation tables (§3)
  events.jsonl    one record per scored unit that carries information (Topic: every sentence; ABSA: every gold-present
                  pair-scored slot, every slot where >=1 model is spurious, and OTHERS detection events)
  packet.jsonl / packet.md   72 fixed-quota cases (§4) with empty root-review columns
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from goibert import data as gdata  # noqa: E402
from goibert import metrics as gmetrics  # noqa: E402
from goibert import train as e1_train  # noqa: E402

OUT = ROOT / "artifacts" / "cafe_bami_paired_error_audit"
SEED = 42
TASKS = ("absa_restaurant", "vsfc_topic", "visfd")
SOURCES = {
    "absa_restaurant": {"bami": "artifacts/e14/trials/absa_restaurant/lr_5e-05/seed_42", "cafe": "artifacts/cdbk_pilot/runs/T_absa_restaurant_seed42", "n": 771, "expected": {"bami": 0.7474178403755869, "cafe": 0.7771971496437055}},
    "vsfc_topic": {"bami": "artifacts/e14/trials/vsfc_topic/lr_1e-05/seed_42", "cafe": "artifacts/sealion_benchmark/cafe_runs/vsfc_topic/lr_2e-05/seed_42", "n": 1583, "expected": {"bami": 0.7953280821882547, "cafe": 0.8174463175313963}},
    "visfd": {"bami": "artifacts/e14/trials/visfd/lr_5e-05/seed_42", "cafe": "artifacts/sealion_benchmark/cafe_runs/visfd/lr_2e-05/seed_42", "n": 1112, "expected": {"bami": 0.8438338152304786, "cafe": 0.8546649922543563}},
}
ABSA_NAMES = {0: "absent", 1: "positive", 2: "negative", 3: "neutral"}   # docs/E12_LABEL_MAPPING_AUDIT.md; data.py names are known-wrong
TOPIC_NAMES = dict(enumerate(gdata.VSFC_TOPIC_LABEL_NAMES))
OTHERS_IDX = gdata.VISFD_OTHERS_ASPECT_INDEX
PACKET_QUOTA = {"cafe_only": 8, "bami_only": 8, "both_wrong": 4, "both_correct": 4}
EVENT_QUOTA = {"miss": 3, "wrong_polarity": 3, "spurious": 2}
EVENT_ORDER = ("miss", "wrong_polarity", "spurious")
MAX_PER_ASPECT = 2


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- independent metric counts

def topic_macro_f1(gold, pred) -> tuple[float, dict]:
    per = {}
    f1s = []
    for c in range(4):
        tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        per[TOPIC_NAMES[c]] = {"code": c, "support": tp + fn, "predicted": tp + fp, "tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}
        f1s.append(f1)
    return sum(f1s) / 4, per


def pair_counts(gold_rows, pred_rows, pair_mask) -> dict:
    tp = fp = fn = 0
    for g, p in zip(gold_rows, pred_rows):
        for j, scored in enumerate(pair_mask):
            if not scored:
                continue
            gv, pv = g[j], p[j]
            if gv > 0 and pv > 0 and gv == pv:
                tp += 1
            if pv > 0 and not (gv > 0 and gv == pv):
                fp += 1
            if gv > 0 and not (pv > 0 and gv == pv):
                fn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0}


# ----------------------------------------------------------------------------- load + gate

def load_task(task: str) -> dict:
    src = SOURCES[task]
    bundle = e1_train.load_task_bundle(task, ROOT, include_test=False)
    val = e1_train.split_field(bundle, "validation")
    texts = list(e1_train.split_texts(val)); ids = [str(i) for i in e1_train.split_ids(val)]; gold = [list(g) if not isinstance(g, int) else int(g) for g in e1_train.split_labels(val)]
    preds = {}
    prov = {"bundle_provenance": e1_train.bundle_provenance(bundle, ROOT), "loader": "goibert.data.load_task via goibert.train.load_task_bundle(include_test=False)"}
    for model in ("bami", "cafe"):
        d = ROOT / src[model]
        pj = d / "validation_predictions.json"
        j = read_json(pj)
        rows = j["rows"]
        meta = read_json(d / "metadata.json") if (d / "metadata.json").exists() else {}
        bvm = read_json(d / "best_validation_metrics.json") if (d / "best_validation_metrics.json").exists() else {}
        preds[model] = {"rows": rows, "path": str(pj.relative_to(ROOT)), "sha256": sha256_file(pj), "file_epoch": j.get("epoch"),
                        "best_epoch_metadata": meta.get("best_epoch"), "best_selection_f1_metadata": meta.get("best_selection_f1"), "best_selection_f1_metrics_file": bvm.get("selection_f1"),
                        "checkpoint_sha256_metadata": meta.get("checkpoint_sha256"), "metadata_sha256": sha256_file(d / "metadata.json") if (d / "metadata.json").exists() else None,
                        "model_source": meta.get("model_source") or meta.get("model", {}).get("backbone") or meta.get("init")}
    return {"task": task, "bundle": bundle, "texts": texts, "ids": ids, "gold": gold, "preds": preds, "provenance": prov, "n_expected": src["n"]}


def gate(t: dict) -> dict:
    task = t["task"]; ids = t["ids"]; n = t["n_expected"]
    checks = {"loader_rows": len(ids), "expected_rows": n, "rows_ok": len(ids) == n, "ids_unique": len(set(ids)) == len(ids)}
    by_id = {}
    for model in ("bami", "cafe"):
        rows = t["preds"][model]["rows"]
        rid = [str(r["id"]) for r in rows]
        checks[f"{model}_rows"] = len(rows); checks[f"{model}_rows_ok"] = len(rows) == n
        checks[f"{model}_ids_unique"] = len(set(rid)) == len(rid)
        checks[f"{model}_same_id_set_as_loader"] = set(rid) == set(ids)
        checks[f"{model}_same_order_as_loader"] = rid == ids
        by_id[model] = {str(r["id"]): r for r in rows}
    gold_ok = True; shape_ok = True
    for i, g in zip(ids, t["gold"]):
        for model in ("bami", "cafe"):
            r = by_id[model][i]
            if task == "vsfc_topic":
                if int(r["gold"]) != int(g) or int(r["prediction"]) not in (0, 1, 2, 3) or int(r["gold"]) not in (0, 1, 2, 3):
                    gold_ok = False
            else:
                if list(r["gold"]) != list(g):
                    gold_ok = False
                width = 12 if task == "absa_restaurant" else 11
                if len(r["gold"]) != width or len(r["prediction"]) != width:
                    shape_ok = False
                if task == "visfd":
                    if r["gold"][OTHERS_IDX] not in (0, 4) or r["prediction"][OTHERS_IDX] not in (0, 1):
                        shape_ok = False
                    if any(v not in (0, 1, 2, 3) for k, v in enumerate(r["prediction"]) if k != OTHERS_IDX) or any(v not in (0, 1, 2, 3) for k, v in enumerate(r["gold"]) if k != OTHERS_IDX):
                        shape_ok = False
                else:
                    if any(v not in (0, 1, 2, 3) for v in r["gold"]) or any(v not in (0, 1, 2, 3) for v in r["prediction"]):
                        shape_ok = False
    checks["gold_identical_loader_and_both_files"] = gold_ok
    checks["shape_and_code_range_ok"] = shape_ok
    checks["bami_cafe_gold_identical"] = all(by_id["bami"][i]["gold"] == by_id["cafe"][i]["gold"] for i in ids)
    # six F1 recomputed
    f1 = {}
    for model in ("bami", "cafe"):
        rows = [by_id[model][i] for i in ids]
        gold = [r["gold"] for r in rows]; pred = [r["prediction"] for r in rows]
        ev = e1_train.evaluate_predictions(task, t["bundle"], gold, pred, split="validation")
        if task == "vsfc_topic":
            ind, _ = topic_macro_f1([int(g) for g in gold], [int(p) for p in pred])
        else:
            pair_mask = [True] * 12 if task == "absa_restaurant" else [k != OTHERS_IDX for k in range(11)]
            ind = pair_counts(gold, pred, pair_mask)["f1"]
        exp = SOURCES[task]["expected"][model]
        meta_f1 = t["preds"][model]["best_selection_f1_metadata"]; file_f1 = t["preds"][model]["best_selection_f1_metrics_file"]
        f1[model] = {"evaluator": ev["selection_f1"], "independent": ind, "metric_name": ev["selection_metric"], "plan_reference": exp,
                     "artifact_metadata_best_selection_f1": meta_f1, "artifact_best_validation_metrics_selection_f1": file_f1,
                     "match_evaluator_vs_independent": abs(ev["selection_f1"] - ind) <= 1e-12,
                     "match_evaluator_vs_artifact": (meta_f1 is not None and abs(ev["selection_f1"] - meta_f1) <= 1e-12) or (file_f1 is not None and abs(ev["selection_f1"] - file_f1) <= 1e-12),
                     "match_vs_plan_reference_1e-6": abs(ev["selection_f1"] - exp) <= 1e-6}
    checks["f1"] = f1
    checks["all_ok"] = all(v for k, v in checks.items() if k.endswith("_ok") or k.endswith("unique") or k.startswith("bami_same") or k.startswith("cafe_same") or k in ("gold_identical_loader_and_both_files", "bami_cafe_gold_identical")) \
        and all(f1[m]["match_evaluator_vs_independent"] and f1[m]["match_evaluator_vs_artifact"] for m in f1)
    t["by_id"] = by_id
    return checks


# ----------------------------------------------------------------------------- aggregates + events

def outcome_present(gold_v: int, pred_v: int) -> str:
    if pred_v == gold_v:
        return "correct"
    if pred_v == 0:
        return "miss"
    return "wrong_polarity"


def paired_category(gold_v, b, c) -> str:
    bc, cc = (b == gold_v), (c == gold_v)
    if bc and cc:
        return "both_correct"
    if cc and not bc:
        return "cafe_only"
    if bc and not cc:
        return "bami_only"
    return "both_wrong_same" if b == c else "both_wrong_different"


def analyse(t: dict, events: list) -> dict:
    task = t["task"]; ids = t["ids"]; by_id = t["by_id"]; texts = t["texts"]
    agg: dict = {"task": task, "n_rows": len(ids)}
    if task == "vsfc_topic":
        gold = [int(by_id["bami"][i]["gold"]) for i in ids]; pb = [int(by_id["bami"][i]["prediction"]) for i in ids]; pc = [int(by_id["cafe"][i]["prediction"]) for i in ids]
        agg["label_names"] = TOPIC_NAMES
        agg["support"] = {TOPIC_NAMES[c]: gold.count(c) for c in range(4)}
        for name, p in (("bami", pb), ("cafe", pc)):
            m, per = topic_macro_f1(gold, p)
            conf = [[sum(1 for g, q in zip(gold, p) if g == a and q == b) for b in range(4)] for a in range(4)]
            agg[name] = {"macro_f1": m, "accuracy": sum(g == q for g, q in zip(gold, p)) / len(gold), "per_class": per, "confusion_rows_gold_cols_pred": conf}
        paired = {TOPIC_NAMES[c]: Counter() for c in range(4)}; overall = Counter(); crosstab = defaultdict(Counter)
        for i, g, b, c in zip(ids, gold, pb, pc):
            cat = paired_category(g, b, c); paired[TOPIC_NAMES[g]][cat] += 1; overall[cat] += 1
            crosstab[TOPIC_NAMES[g]][f"bami={TOPIC_NAMES[b]}|cafe={TOPIC_NAMES[c]}"] += 1
            events.append({"task": task, "id": i, "aspect_index": -1, "aspect": None, "gold": g, "gold_name": TOPIC_NAMES[g], "bami": b, "bami_name": TOPIC_NAMES[b], "cafe": c, "cafe_name": TOPIC_NAMES[c],
                           "bami_outcome": "correct" if b == g else "wrong_class", "cafe_outcome": "correct" if c == g else "wrong_class", "paired": cat})
        agg["paired_by_gold_class"] = {k: dict(v) for k, v in paired.items()}; agg["paired_overall"] = dict(overall)
        agg["crosstab_by_gold_class"] = {k: dict(sorted(v.items(), key=lambda kv: -kv[1])) for k, v in crosstab.items()}
        return agg
    names = list(gdata.RESTAURANT_ASPECT_NAMES) if task == "absa_restaurant" else list(gdata.VISFD_ASPECT_NAMES)
    pair_mask = [True] * len(names) if task == "absa_restaurant" else [k != OTHERS_IDX for k in range(len(names))]
    agg["aspect_names"] = names; agg["pair_scored"] = pair_mask; agg["polarity_names"] = ABSA_NAMES
    gold_rows = [list(by_id["bami"][i]["gold"]) for i in ids]; pb_rows = [list(by_id["bami"][i]["prediction"]) for i in ids]; pc_rows = [list(by_id["cafe"][i]["prediction"]) for i in ids]
    for name, p in (("bami", pb_rows), ("cafe", pc_rows)):
        agg[name] = {"pair_micro": pair_counts(gold_rows, p, pair_mask), "per_aspect": {}}
        for j, an in enumerate(names):
            if not pair_mask[j]:
                continue
            pc_ = pair_counts([[g[j]] for g in gold_rows], [[q[j]] for q in p], [True])
            gp = sum(1 for g in gold_rows if g[j] > 0)
            ev = Counter(outcome_present(g[j], q[j]) for g, q in zip(gold_rows, p) if g[j] > 0)
            sp = sum(1 for g, q in zip(gold_rows, p) if g[j] == 0 and q[j] > 0)
            agg[name]["per_aspect"][an] = {"support_gold_present": gp, **pc_, "correct": ev["correct"], "miss": ev["miss"], "wrong_polarity": ev["wrong_polarity"], "spurious": sp}
        agg[name]["events_total"] = {k: sum(v[k] for v in agg[name]["per_aspect"].values()) for k in ("correct", "miss", "wrong_polarity", "spurious")}
        agg[name]["by_gold_polarity"] = {}
        for pol in (1, 2, 3):
            ev = Counter(outcome_present(g[j], q[j]) for g, q in zip(gold_rows, p) for j in range(len(names)) if pair_mask[j] and g[j] == pol)
            agg[name]["by_gold_polarity"][ABSA_NAMES[pol]] = {"support": sum(ev.values()), **{k: ev[k] for k in ("correct", "miss", "wrong_polarity")}}
        agg[name]["spurious_by_predicted_polarity"] = {ABSA_NAMES[pol]: sum(1 for g, q in zip(gold_rows, p) for j in range(len(names)) if pair_mask[j] and g[j] == 0 and q[j] == pol) for pol in (1, 2, 3)}
        if task == "visfd":
            gp = sum(1 for g in gold_rows if g[OTHERS_IDX] == 4); tp = sum(1 for g, q in zip(gold_rows, p) if g[OTHERS_IDX] == 4 and q[OTHERS_IDX] == 1)
            sp = sum(1 for g, q in zip(gold_rows, p) if g[OTHERS_IDX] == 0 and q[OTHERS_IDX] == 1)
            agg[name]["others_detection_only"] = {"gold_present": gp, "detected": tp, "missed": gp - tp, "spurious": sp, "note": "raw code 4 <-> model code 1; excluded from pair F1 and from polarity packet"}
    paired_aspect_pol = defaultdict(Counter); paired_overall_present = Counter(); paired_absent = Counter(); cross_present = defaultdict(Counter)
    n_gold_present_rows = Counter()
    for i, g, b, c in zip(ids, gold_rows, pb_rows, pc_rows):
        n_gp = sum(1 for j in range(len(names)) if pair_mask[j] and g[j] > 0)
        for j, an in enumerate(names):
            if not pair_mask[j]:
                if task == "visfd":
                    gb = g[j] == 4; qb, qc = b[j] == 1, c[j] == 1
                    if gb or qb or qc:
                        events.append({"task": task, "id": i, "aspect_index": j, "aspect": an, "gold": g[j], "gold_name": "others_present" if gb else "absent", "bami": b[j], "bami_name": "others_present" if qb else "absent", "cafe": c[j], "cafe_name": "others_present" if qc else "absent",
                                       "bami_outcome": ("correct" if qb else "miss") if gb else ("spurious" if qb else "correct_absent"), "cafe_outcome": ("correct" if qc else "miss") if gb else ("spurious" if qc else "correct_absent"),
                                       "paired": paired_category(int(gb), int(qb), int(qc)), "detection_only": True, "n_gold_present_scored": n_gp})
                continue
            gv, bv, cv = g[j], b[j], c[j]
            if gv > 0:
                cat = paired_category(gv, bv, cv); paired_aspect_pol[f"{an}|{ABSA_NAMES[gv]}"][cat] += 1; paired_overall_present[cat] += 1
                cross_present[an][f"bami={outcome_present(gv, bv)}|cafe={outcome_present(gv, cv)}"] += 1
                events.append({"task": task, "id": i, "aspect_index": j, "aspect": an, "gold": gv, "gold_name": ABSA_NAMES[gv], "bami": bv, "bami_name": ABSA_NAMES[bv], "cafe": cv, "cafe_name": ABSA_NAMES[cv],
                               "bami_outcome": outcome_present(gv, bv), "cafe_outcome": outcome_present(gv, cv), "paired": cat, "n_gold_present_scored": n_gp})
            elif bv > 0 or cv > 0:
                cat = paired_category(0, bv, cv); paired_absent[cat] += 1
                events.append({"task": task, "id": i, "aspect_index": j, "aspect": an, "gold": 0, "gold_name": "absent", "bami": bv, "bami_name": ABSA_NAMES[bv], "cafe": cv, "cafe_name": ABSA_NAMES[cv],
                               "bami_outcome": "spurious" if bv > 0 else "correct_absent", "cafe_outcome": "spurious" if cv > 0 else "correct_absent", "paired": cat, "n_gold_present_scored": n_gp})
            else:
                paired_absent["both_correct_absent"] += 1
    agg["paired_gold_present_overall"] = dict(paired_overall_present)
    agg["paired_gold_present_by_aspect_polarity"] = {k: dict(v) for k, v in sorted(paired_aspect_pol.items())}
    agg["paired_gold_absent"] = dict(paired_absent)
    agg["crosstab_gold_present_by_aspect"] = {k: dict(sorted(v.items(), key=lambda kv: -kv[1])) for k, v in cross_present.items()}
    agg["gold_present_slots_total"] = sum(paired_overall_present.values())
    agg["gold_present_per_row"] = dict(Counter(sum(1 for j in range(len(names)) if pair_mask[j] and g[j] > 0) for g in gold_rows))
    return agg


# ----------------------------------------------------------------------------- packet

def hkey(task: str, bucket: str, error_type: str, sid: str, aidx: int) -> str:
    return sha256_text(f"{SEED}|{task}|{bucket}|{error_type}|{sid}|{aidx}")


def wrong_model_event(e: dict, bucket: str) -> str:
    """Event type of the model that is wrong in an exclusive bucket."""
    return e["bami_outcome"] if bucket == "cafe_only" else e["cafe_outcome"]


def select_packet(task: str, events: list, n_by_id: dict) -> tuple[list, dict]:
    ev = [e for e in events if e["task"] == task and not e.get("detection_only")]
    used_ids: set = set()
    picked: list = []; shortfall: dict = {}

    column_aspect_count: dict = defaultdict(Counter)   # <=2 events per aspect per column (bucket), shared across event types

    def take(cands, k, bucket, error_type, per_aspect_cap=True):
        out = []
        aspect_count = column_aspect_count[bucket]
        cands = sorted(cands, key=lambda e: hkey(task, bucket, error_type, e["id"], e["aspect_index"]))
        for e in cands:
            if len(out) >= k:
                break
            if e["id"] in used_ids:
                continue
            if per_aspect_cap and task != "vsfc_topic" and aspect_count[e["aspect"]] >= MAX_PER_ASPECT:
                continue
            out.append((e, bucket, error_type)); used_ids.add(e["id"]); aspect_count[e["aspect"]] += 1
        if len(out) < k:   # relax the per-aspect cap only if still short
            for e in cands:
                if len(out) >= k:
                    break
                if e["id"] in used_ids:
                    continue
                out.append((e, bucket, error_type)); used_ids.add(e["id"])
        return out

    if task == "vsfc_topic":
        for bucket, k in PACKET_QUOTA.items():
            pool = [e for e in ev if (e["paired"] == bucket if bucket in ("cafe_only", "bami_only", "both_correct") else e["paired"].startswith("both_wrong"))]
            # round-robin over gold classes in hash order to cover the four classes
            by_class = {c: sorted([e for e in pool if e["gold"] == c], key=lambda e: hkey(task, bucket, "class", e["id"], -1)) for c in range(4)}
            out = []; ptr = {c: 0 for c in range(4)}
            while len(out) < k and any(ptr[c] < len(by_class[c]) for c in range(4)):
                for c in range(4):
                    if len(out) >= k:
                        break
                    while ptr[c] < len(by_class[c]):
                        e = by_class[c][ptr[c]]; ptr[c] += 1
                        if e["id"] not in used_ids:
                            out.append((e, bucket, "wrong_class" if bucket != "both_correct" else "correct")); used_ids.add(e["id"]); break
            picked.extend(out)
            classes_covered = sorted({e["gold"] for e, _, _ in out})
            shortfall[bucket] = {"quota": k, "got": len(out), "candidates": len(pool), "gold_classes_covered": [TOPIC_NAMES[c] for c in classes_covered], "candidates_by_class": {TOPIC_NAMES[c]: len(by_class[c]) for c in range(4)}}
        return picked, shortfall

    for bucket in ("cafe_only", "bami_only"):
        pool = [e for e in ev if e["paired"] == bucket]
        got = []
        for et in EVENT_ORDER:
            got.extend(take([e for e in pool if wrong_model_event(e, bucket) == et], EVENT_QUOTA[et], bucket, et))
        if len(got) < PACKET_QUOTA[bucket]:
            for et in EVENT_ORDER:
                if len(got) >= PACKET_QUOTA[bucket]:
                    break
                got.extend(take([e for e in pool if wrong_model_event(e, bucket) == et], PACKET_QUOTA[bucket] - len(got), bucket, et))
        picked.extend(got)
        shortfall[bucket] = {"quota": PACKET_QUOTA[bucket], "got": len(got), "candidates": len(pool), "by_event": dict(Counter(et for _, _, et in got)), "candidates_by_event": dict(Counter(wrong_model_event(e, bucket) for e in pool))}
    pool = [e for e in ev if e["paired"].startswith("both_wrong")]
    present = [e for e in pool if e["gold"] > 0]; absent = [e for e in pool if e["gold"] == 0]
    got = take(present, PACKET_QUOTA["both_wrong"], "both_wrong", "gold_present")
    if len(got) < PACKET_QUOTA["both_wrong"]:
        got.extend(take(absent, PACKET_QUOTA["both_wrong"] - len(got), "both_wrong", "both_spurious"))
    picked.extend(got); shortfall["both_wrong"] = {"quota": PACKET_QUOTA["both_wrong"], "got": len(got), "candidates_gold_present": len(present), "candidates_gold_absent": len(absent)}
    pool = [e for e in ev if e["paired"] == "both_correct" and e["gold"] > 0]
    got = take(pool, PACKET_QUOTA["both_correct"], "both_correct", "correct")
    picked.extend(got); shortfall["both_correct"] = {"quota": PACKET_QUOTA["both_correct"], "got": len(got), "candidates_gold_present": len(pool)}
    return picked, shortfall


def packet_rows(t: dict, picked: list) -> list:
    task = t["task"]; idx = {i: k for k, i in enumerate(t["ids"])}
    rows = []
    for e, bucket, et in picked:
        k = idx[e["id"]]; text = t["texts"][k]; rb = t["by_id"]["bami"][e["id"]]; rc = t["by_id"]["cafe"][e["id"]]
        row = {"task": task, "id": e["id"], "bucket": bucket, "event_type": et, "text": text, "text_sha256": sha256_text(text),
               "target_aspect_index": e["aspect_index"], "target_aspect": e["aspect"], "gold": e["gold"], "gold_name": e["gold_name"],
               "bami_prediction": e["bami"], "bami_name": e["bami_name"], "cafe_prediction": e["cafe"], "cafe_name": e["cafe_name"], "bami_outcome": e["bami_outcome"], "cafe_outcome": e["cafe_outcome"],
               "gold_vector": rb["gold"] if task != "vsfc_topic" else None, "bami_vector": rb["prediction"] if task != "vsfc_topic" else None, "cafe_vector": rc["prediction"] if task != "vsfc_topic" else None,
               "n_gold_present_scored": e.get("n_gold_present_scored"), "sources": {"bami": t["preds"]["bami"]["path"], "bami_sha256": t["preds"]["bami"]["sha256"], "cafe": t["preds"]["cafe"]["path"], "cafe_sha256": t["preds"]["cafe"]["sha256"]},
               "selection_key_sha256": hkey(task, bucket, et, e["id"], e["aspect_index"]),
               "gold_supported": "", "evidence_span": "", "possible_failure": "", "ambiguity_note": ""}
        rows.append(row)
    return rows


def vec_str(task: str, vec, names) -> str:
    if vec is None:
        return ""
    if task == "visfd":
        return ", ".join(f"{names[j]}={'OTHERS' if (j == OTHERS_IDX and v in (1, 4)) else ABSA_NAMES[v]}" for j, v in enumerate(vec) if v)
    return ", ".join(f"{names[j]}={ABSA_NAMES[v]}" for j, v in enumerate(vec) if v)


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"created_at": now(), "seed": SEED, "read_only": True, "no_model_no_gpu_no_test": True, "label_mapping": {"absa_polarity": ABSA_NAMES, "topic": TOPIC_NAMES, "visfd_others": "slot 5 detection-only: gold raw 4 <-> model 1; excluded from pair F1"},
                      "evaluator_sha256": sha256_file(ROOT / "src/goibert/metrics.py"), "loader_sha256": sha256_file(ROOT / "src/goibert/data.py"), "train_helpers_sha256": sha256_file(ROOT / "src/goibert/train.py"), "audit_source_sha256": sha256_file(Path(__file__)), "tasks": {}}
    events: list = []; aggregate: dict = {}; packet: list = []; shortfalls: dict = {}
    all_ok = True
    for task in TASKS:
        t = load_task(task)
        g = gate(t)
        manifest["tasks"][task] = {"predictions": {m: {k: v for k, v in t["preds"][m].items() if k != "rows"} for m in ("bami", "cafe")}, "provenance": t["provenance"], "gate": g}
        all_ok &= g["all_ok"]
        print(f"[{task}] gate all_ok={g['all_ok']} F1 bami {g['f1']['bami']['evaluator']:.6f} cafe {g['f1']['cafe']['evaluator']:.6f}", flush=True)
        if not g["all_ok"]:
            continue
        aggregate[task] = analyse(t, events)
        picked, sf = select_packet(task, events, None)
        shortfalls[task] = sf
        packet.extend(packet_rows(t, picked))
    manifest["gate_all_ok"] = all_ok
    if not all_ok:
        write_json(OUT / "manifest.json", manifest); print("BLOCKER: correctness gate failed; see manifest.json"); return 1
    with (OUT / "events.jsonl").open("w", encoding="utf-8") as fh:
        for e in events:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    with (OUT / "packet.jsonl").open("w", encoding="utf-8") as fh:
        for r in packet:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    # packet.md
    md = ["# Packet 72 trường hợp — CafeBERT vs BamiBERT (validation, seed 42)", "", "Chọn theo quota §4 và thứ tự SHA256 cố định; **không** là mẫu ngẫu nhiên. Mã ABSA `0/1/2/3 = absent/positive/negative/neutral`. Cột `gold_supported / evidence_span / possible_failure / ambiguity_note` để root điền.", ""]
    for task in TASKS:
        names = list(gdata.RESTAURANT_ASPECT_NAMES) if task == "absa_restaurant" else (list(gdata.VISFD_ASPECT_NAMES) if task == "visfd" else None)
        md.append(f"## {task}"); md.append("")
        for bucket in ("cafe_only", "bami_only", "both_wrong", "both_correct"):
            rows = [r for r in packet if r["task"] == task and r["bucket"] == bucket]
            md.append(f"### {bucket} ({len(rows)})"); md.append("")
            for n, r in enumerate(rows, 1):
                md.append(f"**{n}. `{r['id']}`** — event `{r['event_type']}`" + (f" — target **{r['target_aspect']}** (slot {r['target_aspect_index']})" if names else ""))
                md.append(f"> {r['text']}")
                md.append(f"- gold: `{r['gold']}` {r['gold_name']} · Bami: `{r['bami_prediction']}` {r['bami_name']} ({r['bami_outcome']}) · Cafe: `{r['cafe_prediction']}` {r['cafe_name']} ({r['cafe_outcome']})")
                if names:
                    md.append(f"- gold vector: {vec_str(task, r['gold_vector'], names) or '(all absent)'}  ·  gold-present scored: {r['n_gold_present_scored']}")
                    md.append(f"- Bami vector: {vec_str(task, r['bami_vector'], names) or '(all absent)'}")
                    md.append(f"- Cafe vector: {vec_str(task, r['cafe_vector'], names) or '(all absent)'}")
                md.append("- gold_supported: ☐  evidence_span: ____  possible_failure: ____  ambiguity_note: ____"); md.append("")
    (OUT / "packet.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    manifest["packet"] = {"rows": len(packet), "quota_total": 72, "shortfalls": shortfalls, "by_task_bucket": {t: dict(Counter(r["bucket"] for r in packet if r["task"] == t)) for t in TASKS},
                          "unique_ids_per_task": {t: len({r["id"] for r in packet if r["task"] == t}) for t in TASKS}, "selection_rule": "sha256('42|task|bucket|error_type|stable_id|aspect_index') ascending; event quotas 3/3/2 miss/wrong_polarity/spurious with fallback miss->wrong_polarity->spurious; <=2 per aspect per exclusive column when enough candidates; unique review id per task; both_wrong prefers gold-present; both_correct requires a correct gold-present pair; Topic round-robin over gold classes"}
    manifest["outputs"] = {n: sha256_file(OUT / n) for n in ("events.jsonl", "packet.jsonl", "packet.md")}
    manifest["events_rows"] = len(events); manifest["elapsed_s"] = round(time.time() - t0, 1)
    write_json(OUT / "aggregate.json", aggregate)
    manifest["outputs"]["aggregate.json"] = sha256_file(OUT / "aggregate.json")
    write_json(OUT / "manifest.json", manifest)
    print(json.dumps({"packet_rows": len(packet), "shortfalls": shortfalls}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
