#!/usr/bin/env python3
"""Build blinded reading sets: text + task guideline only (no gold, no predictions, anonymised shuffled ids).
Units: every review with >=1 exclusive-disagreement slot (cafe_only / bami_only), plus 40 'both_wrong_same' and 40
'both_correct' control reviews per task (seed 42).  The id mapping is written OUTSIDE the repo (scratchpad) so the reader
agent cannot join it back; it is copied into artifacts only after the reading is done."""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from goibert import train as e1_train  # noqa: E402

ART = ROOT / "artifacts" / "cafe_bami_paired_error_audit"
BLIND = Path(sys.argv[1])          # scratchpad dir for the reader
BLIND.mkdir(parents=True, exist_ok=True)
SEED = 42
N_CTRL = 40
ev = [json.loads(l) for l in (ART / "events.jsonl").open(encoding="utf-8")]
summary = {}
for task in ("absa_restaurant", "vsfc_topic", "visfd"):
    bundle = e1_train.load_task_bundle(task, ROOT, include_test=False)
    val = e1_train.split_field(bundle, "validation")
    texts = dict(zip([str(i) for i in e1_train.split_ids(val)], e1_train.split_texts(val)))
    by_rev = defaultdict(list)
    for e in ev:
        if e["task"] == task:
            by_rev[e["id"]].append(e)
    dis = sorted(r for r, us in by_rev.items() if any(u["paired"] in ("cafe_only", "bami_only") and not u.get("detection_only") for u in us))
    bws = sorted(r for r, us in by_rev.items() if r not in dis and any(u["paired"] == "both_wrong_same" and not u.get("detection_only") for u in us))
    if task == "vsfc_topic":
        bc = sorted(r for r, us in by_rev.items() if all(u["paired"] == "both_correct" for u in us))
    else:
        bc = sorted(r for r, us in by_rev.items() if r not in dis and r not in bws and all(u["paired"] == "both_correct" for u in us if not u.get("detection_only")) and any(u["gold"] > 0 for u in us if not u.get("detection_only")))
    rng = random.Random(SEED)
    ctrl_bws = rng.sample(bws, min(N_CTRL, len(bws))); ctrl_bc = rng.sample(bc, min(N_CTRL, len(bc)))
    units = [(r, "disagreement") for r in dis] + [(r, "ctrl_both_wrong_same") for r in ctrl_bws] + [(r, "ctrl_both_correct") for r in ctrl_bc]
    rng.shuffle(units)
    prefix = {"absa_restaurant": "R", "vsfc_topic": "T", "visfd": "V"}[task]
    mapping = []
    with (BLIND / f"{task}_blind.jsonl").open("w", encoding="utf-8") as fh:
        for k, (rid, grp) in enumerate(units, 1):
            uid = f"{prefix}{k:04d}"
            fh.write(json.dumps({"uid": uid, "text": texts[rid]}, ensure_ascii=False) + "\n")
            mapping.append({"uid": uid, "id": rid, "group": grp})
    (BLIND / f"{task}_mapping_DO_NOT_GIVE_TO_READER.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=0), encoding="utf-8")
    summary[task] = {"disagreement_reviews": len(dis), "ctrl_both_wrong_same": len(ctrl_bws), "ctrl_both_correct": len(ctrl_bc), "total_units": len(units), "pool_bws": len(bws), "pool_bc": len(bc)}
print(json.dumps(summary, indent=1))
