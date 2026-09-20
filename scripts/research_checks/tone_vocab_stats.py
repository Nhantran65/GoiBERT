"""No-train check for the tone-factorized embedding idea (Idea 2 / pilot b).

For a tokenizer: decompose every vocab token into (skeleton, tone) where skeleton = token with Vietnamese tone marks
removed (grave/acute/tilde/hook/dot-below; vowel-quality marks like â ă ê ô ơ ư đ are kept) and tone in
{none, huyen, sac, nga, hoi, nang}.  Report (i) how many vocab tokens carry a tone, (ii) how many skeleton groups have
>= 2 members (sharing possible), (iii) on task train text: share of toned token occurrences, and how many *rare* toned
token types (train count <= 5) have a *frequent* sibling (count >= 50) in their skeleton group ("rescue potential").
"""
from __future__ import annotations
import argparse, sys, unicodedata
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from transformers import AutoTokenizer
from goibert import train as tr

TONES = {"̀": "huyen", "́": "sac", "̃": "nga", "̉": "hoi", "̣": "nang"}

def decompose(s: str):
    d = unicodedata.normalize("NFD", s); tone = "none"; out = []
    for ch in d:
        if ch in TONES: tone = TONES[ch] if tone == "none" else "multi"
        else: out.append(ch)
    return unicodedata.normalize("NFC", "".join(out)), tone

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tokenizers", nargs="+", default=["FacebookAI/xlm-roberta-large", "jhu-clsp/mmBERT-base"])
    ap.add_argument("--tasks", nargs="+", default=["absa_restaurant", "absa_hotel", "visfd", "vsfc_sentiment"]); args = ap.parse_args()
    for name in args.tokenizers:
        tok = AutoTokenizer.from_pretrained(name); V = tok.get_vocab(); groups = defaultdict(list); tone_of = {}
        for t, i in V.items():
            s = tok.convert_tokens_to_string([t]) if not t.startswith("<") else t
            sk, tn = decompose(s); tone_of[i] = tn; groups[sk].append(i)
        toned = [i for i, tn in tone_of.items() if tn not in ("none", "multi")]
        multi = sum(1 for tn in tone_of.values() if tn == "multi")
        shared = {i for g in groups.values() if len(g) >= 2 for i in g}
        print(f"\n== {name}: vocab {len(V)}, toned tokens {len(toned)} ({100*len(toned)/len(V):.1f}%), multi-tone {multi}, "
              f"skeleton groups>=2: {sum(1 for g in groups.values() if len(g)>=2)}, toned tokens in a shared group: {len(shared & set(toned))}")
        for task in args.tasks:
            b = tr.load_task_bundle(task, ROOT, include_test=False); texts = list(tr.split_texts(tr.split_field(b, "train")))
            cnt = Counter(); 
            for i in range(0, len(texts), 256): 
                for ids in tok(texts[i:i+256], add_special_tokens=False)["input_ids"]: cnt.update(ids)
            total = sum(cnt.values()); toned_occ = sum(c for i, c in cnt.items() if tone_of[i] not in ("none", "multi"))
            rare = [i for i, c in cnt.items() if c <= 5 and tone_of[i] not in ("none", "multi")]
            gid = {i: sk for sk, g in groups.items() for i in g}
            def sib_max(i): return max((cnt.get(j, 0) for j in groups[gid[i]] if j != i), default=0)
            rescued = sum(1 for i in rare if sib_max(i) >= 50)
            rare_occ = sum(cnt[i] for i in rare); rescued_occ = sum(cnt[i] for i in rare if sib_max(i) >= 50)
            # also: validation tokens unseen in train (types) that have a frequent skeleton sibling
            vtexts = list(tr.split_texts(tr.split_field(b, "validation"))); vcnt = Counter()
            for i in range(0, len(vtexts), 256):
                for ids in tok(vtexts[i:i+256], add_special_tokens=False)["input_ids"]: vcnt.update(ids)
            unseen = [i for i in vcnt if cnt.get(i, 0) == 0 and tone_of[i] not in ("none", "multi")]
            unseen_resc = sum(1 for i in unseen if sib_max(i) >= 50)
            print(f"  {task:16s} tok/sent {total/len(texts):5.1f} | toned occ {100*toned_occ/total:4.1f}% | rare toned types {len(rare):4d} "
                  f"(occ {100*rare_occ/total:.2f}%) of which w/ frequent skeleton sibling {rescued} ({100*rescued/max(1,len(rare)):.0f}%) "
                  f"| val toned types unseen in train {len(unseen)}, w/ frequent sibling {unseen_resc}")

if __name__ == "__main__":
    main()
