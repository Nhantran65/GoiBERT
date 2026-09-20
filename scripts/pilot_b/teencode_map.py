"""Build a teencode/abbreviation variant -> canonical word dictionary from ViLexNorm (train+dev), then measure how much
of the social / benchmark text it covers.  Pairs with equal whitespace-token counts are aligned 1:1 (79% of pairs).
Kept variants: >= 3 occurrences, dominant canonical >= 80%, the variant is changed in >= 50% of its occurrences (so
real words like 'qua' are excluded), variant != canonical after diacritic stripping (those are the diacritic block's job).
Output: data/cpt/teencode_map.json {variant: canonical}; coverage table printed + artifacts/research_diagnostics/teencode_coverage.md.
"""
from __future__ import annotations
import json, re, sys, unicodedata
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from goibert import train as tr
MARKS = set("̛̣̀́̃̉̂̆")
def strip(s): return unicodedata.normalize("NFC", "".join(c for c in unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D")) if c not in MARKS))
def words(s): return re.findall(r"[^\W\d_]+", s.lower())

pairs = pd.concat([pd.read_csv(ROOT / f"data/raw/social/ViLexNorm/{s}.csv") for s in ("train", "dev")])
sub = Counter(); occ = Counter(); usable = 0
for o, n in zip(pairs["original"].astype(str), pairs["normalized"].astype(str)):
    ow, nw = o.lower().split(), n.lower().split()
    if len(ow) != len(nw): continue
    usable += 1
    for a, b in zip(ow, nw):
        a, b = a.strip(".,!?:;()\"'"), b.strip(".,!?:;()\"'")
        if not a or not re.fullmatch(r"[^\W\d_]+", a): continue
        occ[a] += 1
        if a != b and re.fullmatch(r"[^\W\d_]+", b): sub[(a, b)] += 1
by_var = defaultdict(Counter)
for (a, b), c in sub.items(): by_var[a][b] += c
tmap = {}
for a, cands in by_var.items():
    b, c = cands.most_common(1)[0]; tot = sum(cands.values())
    if c >= 3 and c / tot >= 0.8 and tot / occ[a] >= 0.5 and strip(a) != strip(b): tmap[a] = b
print(f"usable pairs {usable}/{len(pairs)}, variant types {len(by_var)}, kept {len(tmap)}")
print("examples:", sorted(tmap.items(), key=lambda kv: -sub[kv])[:40])
json.dump(tmap, open(ROOT / "data/cpt/teencode_map.json", "w"), ensure_ascii=False, indent=0)

rows = []
SOC = ROOT / "data/raw/social"
sets = {"vihsd dev": pd.read_csv(SOC / "vihsd/dev.csv")["free_text"], "vispam dev": pd.read_csv(SOC / "vispamdetection/dev.csv")["Comment"],
        "vsmec valid": pd.read_csv(SOC / "uit-vsmec/valid.csv")["Sentence"], "vilexnorm test (orig)": pd.read_csv(SOC / "ViLexNorm/test.csv")["original"]}
for t in ["vsfc_sentiment", "visfd", "absa_restaurant", "absa_hotel"]:
    b = tr.load_task_bundle(t, ROOT, include_test=False); sets[t + " val"] = list(tr.split_texts(tr.split_field(b, "validation")))
for name, texts in sets.items():
    nt = nv = nr = 0; hit = Counter()
    for s in texts:
        w = words(str(s)); nt += len(w); v = [x for x in w if x in tmap]; nv += len(v); nr += bool(v); hit.update(v)
    rows.append((name, len(texts), nt / max(1, len(texts)), 100 * nv / max(1, nt), 100 * nr / max(1, len(texts)), ", ".join(f"{k}({c})" for k, c in hit.most_common(6))))
md = "| text | rows | words/row | % words that are teencode variants | % rows with >=1 variant | top variants |\n|---|---|---|---|---|---|\n"
md += "\n".join(f"| {n} | {r} | {w:.1f} | {pv:.2f} | {pr:.1f} | {top} |" for n, r, w, pv, pr, top in rows)
print(md); (ROOT / "artifacts/research_diagnostics/teencode_coverage.md").write_text(md)
