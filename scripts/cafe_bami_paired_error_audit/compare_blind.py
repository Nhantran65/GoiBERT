#!/usr/bin/env python3
"""Join blind-reader labels (Sonnet 5 subagents, text-only) with gold / Bami / Cafe and compute:
  * reader agreement with gold on control groups (both_correct, both_wrong_same) and on disagreement units
  * for every exclusive-disagreement slot: reader sides with gold (gold-supported), with the wrong model (gold-disputed), or neither
  * 'adjudicated gap': Cafe-Bami F1 recomputed on slots where the reader agrees with gold
Usage: compare_blind.py <blind_dir> <mapping_dir>"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "cafe_bami_paired_error_audit"
BLIND, MAP = Path(sys.argv[1]), Path(sys.argv[2])
NAMES = {"absent": 0, "positive": 1, "negative": 2, "neutral": 3}
REST = ["AMBIENCE#GENERAL", "DRINKS#PRICES", "DRINKS#QUALITY", "DRINKS#STYLE&OPTIONS", "FOOD#PRICES", "FOOD#QUALITY", "FOOD#STYLE&OPTIONS", "LOCATION#GENERAL", "RESTAURANT#GENERAL", "RESTAURANT#MISCELLANEOUS", "RESTAURANT#PRICES", "SERVICE#GENERAL"]
VISFD = ["BATTERY", "CAMERA", "DESIGN", "FEATURES", "GENERAL", "OTHERS", "PERFORMANCE", "PRICE", "SCREEN", "SER&ACC", "STORAGE"]
TOPIC = {"lecturer": 0, "training_program": 1, "facility": 2, "others": 3}
ev = [json.loads(l) for l in (ART / "events.jsonl").open(encoding="utf-8")]
by_task_rev = defaultdict(lambda: defaultdict(list))
for e in ev:
    by_task_rev[e["task"]][e["id"]].append(e)


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def pair_f1(units, key):
    tp = fp = fn = 0
    for u in units:
        g, p = u["gold"], u[key]
        tp += int(g > 0 and p == g); fp += int(p > 0 and not (g > 0 and p == g)); fn += int(g > 0 and not (p > 0 and p == g))
    return f1(tp, fp, fn), tp, fp, fn


report: dict = {}
for task, names in (("absa_restaurant", REST), ("vsfc_topic", None), ("visfd", VISFD)):
    out_path = BLIND / "out" / f"{task}_reader1.jsonl"
    if not out_path.exists():
        report[task] = {"status": "reader output missing"}; continue
    reader = {}
    bad = 0
    for line in out_path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line); reader[r["uid"]] = r
        except Exception:
            bad += 1
    mapping = {m["uid"]: m for m in json.loads((MAP / f"{task}_mapping_DO_NOT_GIVE_TO_READER.json").read_text(encoding="utf-8"))}
    missing = [u for u in mapping if u not in reader]
    res: dict = {"units_expected": len(mapping), "units_read": len(reader), "missing": len(missing), "malformed_lines": bad, "uncertain_rate_by_group": {}, "by_group": {}}
    groups = defaultdict(list)
    for uid, m in mapping.items():
        if uid in reader:
            groups[m["group"]].append((uid, m["id"], reader[uid]))
    if task == "vsfc_topic":
        for grp, items in groups.items():
            agree_gold = agree_b = agree_c = 0; n = 0; unc = 0; sides = Counter()
            for uid, rid, r in items:
                u = by_task_rev[task][rid][0]; lab = TOPIC.get(r.get("label"), -1); n += 1; unc += int(bool(r.get("uncertain")))
                agree_gold += int(lab == u["gold"]); agree_b += int(lab == u["bami"]); agree_c += int(lab == u["cafe"])
                if grp == "disagreement":
                    wrong = "bami" if u["paired"] == "cafe_only" else "cafe"
                    sides["gold_supported" if lab == u["gold"] else ("gold_disputed_sides_with_wrong_model" if lab == u[wrong] else "neither")] += 1
            res["by_group"][grp] = {"n": n, "reader=gold": agree_gold / n, "reader=bami": agree_b / n, "reader=cafe": agree_c / n, "uncertain": unc / n, **({"disagreement_sides": dict(sides)} if grp == "disagreement" else {})}
        # adjudicated gap: sentences where reader agrees with gold, across ALL read units (disagreement+controls) and on disagreement only
        for scope in ("disagreement", "all_read"):
            units = []
            for grp, items in groups.items():
                if scope == "disagreement" and grp != "disagreement":
                    continue
                for uid, rid, r in items:
                    u = by_task_rev[task][rid][0]
                    if TOPIC.get(r.get("label"), -1) == u["gold"]:
                        units.append(u)
            acc_b = sum(u["bami"] == u["gold"] for u in units) / max(1, len(units)); acc_c = sum(u["cafe"] == u["gold"] for u in units) / max(1, len(units))
            res[f"adjudicated_{scope}"] = {"units_gold_supported": len(units), "bami_acc": acc_b, "cafe_acc": acc_c, "gap_pp": 100 * (acc_c - acc_b)}
        # full-validation macro-F1 with disputed sentences removed
        disputed = {rid for uid, rid, r in groups.get("disagreement", []) if TOPIC.get(r.get("label"), -1) != by_task_rev[task][rid][0]["gold"]}
        keep = [by_task_rev[task][rid][0] for rid in by_task_rev[task] if rid not in disputed]
        def macro(units, key):
            f = []
            for c in range(4):
                tp = sum(u["gold"] == c and u[key] == c for u in units); fp = sum(u["gold"] != c and u[key] == c for u in units); fn = sum(u["gold"] == c and u[key] != c for u in units); f.append(f1(tp, fp, fn))
            return sum(f) / 4
        res["full_validation_minus_disputed_disagreements"] = {"removed": len(disputed), "n": len(keep), "bami_macro_f1": macro(keep, "bami"), "cafe_macro_f1": macro(keep, "cafe"), "gap_pp": 100 * (macro(keep, "cafe") - macro(keep, "bami"))}
        report[task] = res
        continue
    # ABSA: reader vector per review
    def reader_vec(r):
        vec = {a: 0 for a in names}
        for lab in r.get("labels", []):
            a = lab.get("aspect"); p = NAMES.get(lab.get("polarity"), None)
            if a in vec and p:
                vec[a] = p
        if task == "visfd":
            vec["OTHERS"] = 4 if r.get("others") else 0
        return vec
    slot_stats = {grp: Counter() for grp in groups}
    disputed_slots = set(); supported_slots = set()
    for grp, items in groups.items():
        n_rev = 0; exact = 0; unc = 0; gp_agree = gp_total = 0; spur_reader = 0; sides = Counter()
        for uid, rid, r in items:
            vec = reader_vec(r); n_rev += 1; unc += int(bool(r.get("uncertain")))
            units = by_task_rev[task][rid]
            gold_vec = {u["aspect"]: u["gold"] for u in units}
            # exact match on scored slots
            scored = [a for a in names if a != "OTHERS"]
            ok = all(vec[a] == gold_vec.get(a, 0) for a in scored); exact += int(ok)
            for u in units:
                if u.get("detection_only"):
                    continue
                a = u["aspect"]; rl = vec[a]
                if u["gold"] > 0:
                    gp_total += 1; gp_agree += int(rl == u["gold"])
                if u["gold"] == 0 and rl > 0:
                    spur_reader += 1
                slot_stats[grp][f"reader_eq_gold"] += int(rl == u["gold"]); slot_stats[grp]["slots"] += 1
                slot_stats[grp]["reader_eq_bami"] += int(rl == u["bami"]); slot_stats[grp]["reader_eq_cafe"] += int(rl == u["cafe"])
                if u["paired"] in ("cafe_only", "bami_only"):
                    wrong = "bami" if u["paired"] == "cafe_only" else "cafe"
                    key = (rid, a)
                    if rl == u["gold"]:
                        sides[f"{u['paired']}:gold_supported"] += 1; supported_slots.add(key)
                    elif rl == u[wrong]:
                        sides[f"{u['paired']}:gold_disputed_sides_with_wrong_model"] += 1; disputed_slots.add(key)
                    else:
                        sides[f"{u['paired']}:neither"] += 1; disputed_slots.add(key)
            # reader-spurious slots that are not in events (gold absent & both models absent) -> count as reader extra aspects
            for a in scored:
                if vec[a] > 0 and a not in gold_vec:
                    slot_stats[grp]["reader_extra_aspect_not_in_gold_or_models"] += 1
        s = slot_stats[grp]
        res["by_group"][grp] = {"reviews": n_rev, "exact_vector_match_rate": exact / n_rev, "gold_present_slot_agreement": gp_agree / max(1, gp_total), "gold_present_slots": gp_total,
                                "reader_spurious_on_event_slots": spur_reader, "reader_extra_aspects": s.get("reader_extra_aspect_not_in_gold_or_models", 0), "uncertain": unc / n_rev,
                                "event_slots": s["slots"], "reader=gold_slot_rate": s["reader_eq_gold"] / max(1, s["slots"]), "reader=bami_slot_rate": s["reader_eq_bami"] / max(1, s["slots"]), "reader=cafe_slot_rate": s["reader_eq_cafe"] / max(1, s["slots"]),
                                **({"disagreement_sides": dict(sides)} if grp == "disagreement" else {})}
    # adjudicated gap on disagreement slots + full validation minus disputed
    all_units = [u for rid in by_task_rev[task] for u in by_task_rev[task][rid] if not u.get("detection_only")]
    dis_units = [u for u in all_units if u["paired"] in ("cafe_only", "bami_only")]
    sup_units = [u for u in dis_units if (u["id"], u["aspect"]) in supported_slots]
    fb, *_ = pair_f1(all_units, "bami"); fc, *_ = pair_f1(all_units, "cafe")
    keep = [u for u in all_units if (u["id"], u["aspect"]) not in disputed_slots]
    kb, *_ = pair_f1(keep, "bami"); kc, *_ = pair_f1(keep, "cafe")
    res["disagreement_slots"] = {"total": len(dis_units), "gold_supported": len(sup_units), "disputed_or_neither": len(disputed_slots)}
    res["full_validation_pair_f1"] = {"bami": fb, "cafe": fc, "gap_pp": 100 * (fc - fb)}
    res["full_validation_minus_disputed_disagreement_slots"] = {"removed_slots": len(disputed_slots), "bami": kb, "cafe": kc, "gap_pp": 100 * (kc - kb)}
    # among supported disagreement slots: who is right
    res["supported_disagreement_slots_by_winner"] = dict(Counter(u["paired"] for u in sup_units))
    res["disputed_disagreement_slots_by_winner"] = dict(Counter(u["paired"] for u in dis_units if (u["id"], u["aspect"]) in disputed_slots))
    report[task] = res
(ART / "blind_reading_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=1))
