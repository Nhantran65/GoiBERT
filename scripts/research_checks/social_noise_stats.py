"""How noisy is real Vietnamese social-media text, compared to our existing task train splits?

For each raw social dataset's train split (and ViLexNorm's *original* side), reports:
  - % rows typed with NO diacritics at all (proxy: >=3 alphabetic words, 0 of them diacritic-bearing).
  - % rows with MIXED diacritics (0% < share of diacritic-bearing alphabetic words < 60%).
  - % rows containing >=1 whole-word teencode/abbreviation token from a fixed list.
  - mean whitespace-split words per row.
For ViLexNorm additionally: among original/normalized row pairs whose whitespace token counts match, the
per-differing-token split between "diacritics only" (stripping diacritics from the normalized token recovers
the original token) and "abbreviation/teencode" (everything else), plus the coverage (% of pairs usable this way).

Also runs the identical 4 basic stats on our existing task train text (vsfc_sentiment, visfd, absa_restaurant,
absa_hotel) via goibert.train.load_task_bundle for a same-table comparison. No GPU / model loading involved.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from goibert import train as tr  # noqa: E402

SOCIAL_DIR = ROOT / "data" / "raw" / "social"
OUT_DIR = ROOT / "artifacts" / "research_diagnostics"

# --- diacritic detection -----------------------------------------------------------------
# Tone marks (huyen, sac, nga, hoi, nang) + vowel-quality marks (circumflex, breve, horn):
# treated together as "diacritics" here per the task's own definition, since typing without
# any of these (i.e. plain telex/VNI with no accent keys at all) is the realistic proxy for
# "typed without diacritics", and it also strips the vowel-quality marks (a-vs-a^/a-breve/etc).
DIACRITIC_MARKS = {"̀", "́", "̃", "̉", "̣", "̂", "̆", "̛"}
DJ_CHARS = {"đ", "Đ"}

ALPHA_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def word_has_diacritic(word: str) -> bool:
    if any(ch in DJ_CHARS for ch in word):
        return True
    nfd = unicodedata.normalize("NFD", word)
    return any(ch in DIACRITIC_MARKS for ch in nfd)


def strip_diacritics(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s)
    stripped = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    stripped = unicodedata.normalize("NFC", stripped)
    return stripped.replace("đ", "d").replace("Đ", "D")


# --- teencode / abbreviation whole-word token list ---------------------------------------
TEENCODE_TOKENS = [
    "ko", "k", "hok", "hong", "dc", "đc", "đk", "bt", "bth", "vs", "ng", "ngta", "mn", "mng",
    "ae", "bn", "cx", "cg", "j", "z", "ntn", "sao z", "vcl", "vl", "vkl", "wa", "qua", "roi",
    "r", "nhe", "nha", "thik", "iu", "e", "a", "ck", "vk", "bik", "bít", "hem", "hông",
]
_TEENCODE_PATTERN = re.compile(
    "|".join(rf"(?<!\w){re.escape(tok)}(?!\w)" for tok in TEENCODE_TOKENS),
    re.IGNORECASE | re.UNICODE,
)


def row_has_teencode(text: str) -> bool:
    return _TEENCODE_PATTERN.search(text) is not None


# --- per-row / per-corpus stats -----------------------------------------------------------
def row_stats(text: str) -> dict:
    words = ALPHA_WORD_RE.findall(text)
    n_alpha = len(words)
    n_diac = sum(1 for w in words if word_has_diacritic(w))
    frac_diac = (n_diac / n_alpha) if n_alpha else None
    no_diacritic = n_alpha >= 3 and n_diac == 0
    mixed = frac_diac is not None and 0.0 < frac_diac < 0.60
    return {
        "n_words_ws": len(text.split()),
        "no_diacritic": no_diacritic,
        "mixed": mixed,
        "teencode": row_has_teencode(text),
    }


def corpus_stats(name: str, texts: Sequence[str]) -> dict:
    texts = [t if isinstance(t, str) else "" for t in texts]
    n = len(texts)
    rows = [row_stats(t) for t in texts]
    mean_words = sum(r["n_words_ws"] for r in rows) / n if n else 0.0
    pct_no_diac = 100.0 * sum(r["no_diacritic"] for r in rows) / n if n else 0.0
    pct_mixed = 100.0 * sum(r["mixed"] for r in rows) / n if n else 0.0
    pct_teencode = 100.0 * sum(r["teencode"] for r in rows) / n if n else 0.0
    return {
        "dataset": name,
        "rows": n,
        "words_per_row": round(mean_words, 2),
        "pct_no_diacritic_rows": round(pct_no_diac, 2),
        "pct_mixed_rows": round(pct_mixed, 2),
        "pct_teencode_rows": round(pct_teencode, 2),
    }


# --- ViLexNorm alignment analysis ---------------------------------------------------------
def vilexnorm_alignment_stats(orig: Sequence[str], norm: Sequence[str]) -> dict:
    assert len(orig) == len(norm)
    n_pairs = len(orig)
    n_used = 0
    diac_only = 0
    other = 0
    for o, n in zip(orig, norm):
        o = o if isinstance(o, str) else ""
        n = n if isinstance(n, str) else ""
        o_toks = o.split()
        n_toks = n.split()
        if len(o_toks) != len(n_toks):
            continue
        n_used += 1
        for ot, nt in zip(o_toks, n_toks):
            if ot == nt:
                continue
            if strip_diacritics(nt) == ot:
                diac_only += 1
            else:
                other += 1
    total_diffs = diac_only + other
    return {
        "row_pairs": n_pairs,
        "row_pairs_used_matching_token_count": n_used,
        "coverage_pct": round(100.0 * n_used / n_pairs, 2) if n_pairs else 0.0,
        "differing_token_pairs": total_diffs,
        "pct_diacritics_only": round(100.0 * diac_only / total_diffs, 2) if total_diffs else 0.0,
        "pct_abbreviation_other": round(100.0 * other / total_diffs, 2) if total_diffs else 0.0,
    }


# --- dataset loaders ------------------------------------------------------------------------
def load_social_train_texts() -> dict[str, Sequence[str]]:
    out: dict[str, Sequence[str]] = {}

    df = pd.read_csv(SOCIAL_DIR / "vihsd" / "train.csv")
    out["vihsd"] = df["free_text"].astype(str).tolist()

    df = pd.read_csv(SOCIAL_DIR / "vispamdetection" / "train.csv")
    out["vispamdetection"] = df["Comment"].astype(str).tolist()

    df = pd.read_csv(SOCIAL_DIR / "ViCTSD" / "ViCTSD_train.csv")
    out["ViCTSD"] = df["Comment"].astype(str).tolist()

    df = pd.read_csv(SOCIAL_DIR / "uit-vsmec" / "train.csv")
    out["uit-vsmec"] = df["Sentence"].astype(str).tolist()

    df = pd.read_csv(SOCIAL_DIR / "ViLexNorm" / "train.csv")
    out["ViLexNorm(original)"] = df["original"].astype(str).tolist()

    return out


def load_vilexnorm_pairs() -> tuple[Sequence[str], Sequence[str]]:
    df = pd.read_csv(SOCIAL_DIR / "ViLexNorm" / "train.csv")
    return df["original"].astype(str).tolist(), df["normalized"].astype(str).tolist()


GOIBERT_TASKS = ("vsfc_sentiment", "visfd", "absa_restaurant", "absa_hotel")


def load_goibert_task_texts(tasks: Sequence[str]) -> dict[str, Sequence[str]]:
    out: dict[str, Sequence[str]] = {}
    for task in tasks:
        bundle = tr.load_task_bundle(task, ROOT, include_test=False)
        texts = list(tr.split_texts(tr.split_field(bundle, "train")))
        out[task] = texts
    return out


# --- reporting ------------------------------------------------------------------------------
TABLE_COLS = ["dataset", "rows", "words_per_row", "pct_no_diacritic_rows", "pct_mixed_rows", "pct_teencode_rows"]
TABLE_HEADER = ["dataset", "rows", "words/row", "%no-diacritic rows", "%mixed", "%teencode rows"]


def render_markdown_table(rows: list[dict]) -> str:
    lines = ["| " + " | ".join(TABLE_HEADER) + " |", "|" + "---|" * len(TABLE_HEADER)]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                str(r[c]) for c in TABLE_COLS
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(GOIBERT_TASKS))
    args = ap.parse_args()

    social_texts = load_social_train_texts()
    social_rows = [corpus_stats(name, texts) for name, texts in social_texts.items()]

    task_texts = load_goibert_task_texts(args.tasks)
    task_rows = [corpus_stats(name, texts) for name, texts in task_texts.items()]

    all_rows = social_rows + task_rows

    orig, norm = load_vilexnorm_pairs()
    vln_alignment = vilexnorm_alignment_stats(orig, norm)

    table_md = render_markdown_table(all_rows)
    print(table_md)
    print()
    print("ViLexNorm original->normalized token-diff alignment (train split):")
    for k, v in vln_alignment.items():
        print(f"  {k}: {v}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = OUT_DIR / "social_noise_stats.md"
    json_path = OUT_DIR / "social_noise_stats.json"

    md_lines = [
        "# Social-media noise stats vs. existing task train text",
        "",
        "Diacritic definition: a word is diacritic-bearing if its NFD form contains any of the tone marks "
        "U+0300/U+0301/U+0303/U+0309/U+0323, or the vowel-quality marks U+0302/U+0306/U+031B, or the letter đ/Đ.",
        "",
        "\"No-diacritic\" row: >=3 alphabetic words, none diacritic-bearing. "
        "\"Mixed\" row: 0% < share of diacritic-bearing alphabetic words < 60%. "
        "\"Teencode\" row: contains >=1 whole-word token from a fixed teencode/abbreviation list "
        "(case-insensitive).",
        "",
        table_md,
        "",
        "## ViLexNorm original->normalized token-diff alignment (train split)",
        "",
        "Whitespace-aligned original/normalized row pairs where token counts match; among differing token pairs, "
        "share that is diacritics-only (`strip_diacritics(normalized_token) == original_token`) vs. "
        "abbreviation/teencode (everything else).",
        "",
    ]
    for k, v in vln_alignment.items():
        md_lines.append(f"- **{k}**: {v}")
    md_lines.append("")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    json_path.write_text(
        json.dumps(
            {
                "social": social_rows,
                "goibert_tasks": task_rows,
                "vilexnorm_alignment": vln_alignment,
                "teencode_tokens": TEENCODE_TOKENS,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {md_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
