"""Pilot-B summary table: per arm, per task, mean over seeds of clean best F1 and stripped F1 at the clean-best epoch."""
import json, glob, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; P = ROOT / "artifacts" / "pilot_b" / "probe"
arms = [a for a in ["cafebert_0tok", "a", "c", "b2", "b2_hlr", "b"] if (P / a).exists()]
tasks = ["vsfc_sentiment", "vsfc_topic", "visfd", "absa_restaurant", "absa_hotel"]
res = {a: defaultdict(list) for a in arms}
for a in arms:
    for f in glob.glob(str(P / a / "*.json")):
        d = json.load(open(f)); t = d["args"]["task"]
        if len(d["curves"]) < 20: continue
        res[a][t].append((d["best_f1"], d.get("strip_f1_at_best", float("nan")), d["args"]["seed"]))
def fmt(v): return " ".join(f"{x[0]:.2f}" for x in sorted(v, key=lambda x: x[2]))
def mean(v, i): return sum(x[i] for x in v) / len(v) if v else float("nan")
names = {"cafebert_0tok": "CafeBERT (0 tok)", "a": "a plain CPT", "c": "c +dropout", "b2": "b2 additive", "b2_hlr": "b2 additive lr×40", "b": "b constrained r64"}
print("CLEAN (mean over seeds; per-seed in brackets)")
print(f"{'arm':22s}" + "".join(f"{t[:14]:>22s}" for t in tasks) + f"{'mean':>8s}")
for a in arms:
    row = [mean(res[a][t], 0) for t in tasks]
    print(f"{names[a]:22s}" + "".join(f"{m:8.2f} [{fmt(res[a][t]):>11s}]" for m, t in zip(row, tasks)) + f"{sum(row)/len(row):8.2f}")
print("\nSTRIPPED (all diacritics removed from validation text; at clean-best epoch)")
print(f"{'arm':22s}" + "".join(f"{t[:14]:>22s}" for t in tasks) + f"{'mean':>8s}")
for a in arms:
    row = [mean(res[a][t], 1) for t in tasks]
    print(f"{names[a]:22s}" + "".join(f"{m:22.2f}" for m in row) + f"{sum(r for r in row if r == r)/len(row):8.2f}")
json.dump({a: {t: v for t, v in res[a].items()} for a in arms}, open(ROOT / "artifacts" / "pilot_b" / "summary_table.json", "w"), indent=1)
