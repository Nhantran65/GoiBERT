"""5 EuroBERT Table 1 multilingual tasks: XNLI, PAWS-X, AmazonReviews,
MassiveIntent, SeaHorse. Mirrors language sets from Table 7."""
from __future__ import annotations
import json, os
from datasets import load_dataset as _hf_load_dataset, DatasetDict


# --- dataset revision pinning ----------------------------------------------
# Datasets on the Hub are mutable: a new revision of GLUE or PAWS-X changes your
# numbers with nothing in the result to explain why. Pin them here (or via the
# ENCODER_EVAL_DATASET_REVISIONS env var, a JSON object of {repo: revision}) for
# citable runs. Whatever is in effect is recorded in each result's provenance.
#
#   ENCODER_EVAL_DATASET_REVISIONS='{"nyu-mll/glue": "<sha>"}'
DATASET_REVISIONS: dict[str, str] = {}

try:
    DATASET_REVISIONS.update(json.loads(os.environ["ENCODER_EVAL_DATASET_REVISIONS"]))
except (KeyError, ValueError):
    pass

_USED_REVISIONS: dict[str, str | None] = {}


def load_dataset(path, name=None, **kw):
    """`datasets.load_dataset` with an optional pinned revision per repo.

    Records what was actually requested so the run can report it.
    """
    rev = kw.pop("revision", None) or DATASET_REVISIONS.get(path)
    _USED_REVISIONS[path] = rev
    if name is None:
        return _hf_load_dataset(path, revision=rev, **kw)
    return _hf_load_dataset(path, name, revision=rev, **kw)


def dataset_revisions_used() -> dict:
    """Which dataset repos were loaded this run, and at which pinned revision."""
    return dict(_USED_REVISIONS)


EURO_LANGS = {"en", "de", "es", "fr", "it", "nl", "pl", "pt"}

# Primary reported metric per task — the SINGLE source of truth shared by
# run.py (which metric to aggregate) and summarize.py (which metric to read).
# Anything not listed defaults to "accuracy".
#   matthews : CoLA (the standard GLUE metric for it)
#   f1       : MRPC / QQP (standard GLUE metric)
#   f1_macro : CB (standard SuperGLUE metric)
#   spearman : STS-B, SeaHorse (regression targets)
TASK_METRIC = {
    "glue_cola": "matthews",
    "glue_mrpc": "f1",
    "glue_qqp":  "f1",
    "glue_stsb": "spearman",
    "sg_cb":     "f1_macro",
    "seahorse":  "spearman",
}

# Tasks with a LABELED test split — these report test; everything else reports dev
# (GLUE/SuperGLUE test labels are hidden by the benchmark).
TASKS_WITH_TEST_SPLIT = {"xnli", "paws_x", "amazon", "massive_intent", "seahorse"}


def task_metric(task: str) -> str:
    """Primary metric key for a task (see TASK_METRIC)."""
    return TASK_METRIC.get(task, "accuracy")

TASK_LANGS = {
    "xnli":           ["en", "de", "es", "fr", "ar", "hi", "ru", "tr", "vi", "zh"],
    "paws_x":         ["en", "de", "es", "fr"],
    "amazon":         ["en", "de", "es", "fr", "ja", "zh"],
    "massive_intent": ["en", "de", "es", "fr", "it", "nl", "pl", "pt",
                       "ar", "hi", "ja", "ru", "tr", "vi", "zh"],
    "seahorse":       ["en", "de", "es", "ru", "tr", "vi"],
    # GLUE — English-only single-config tasks
    "glue_cola":      ["en"], "glue_sst2": ["en"], "glue_mrpc": ["en"],
    "glue_stsb":      ["en"], "glue_qqp":  ["en"], "glue_mnli": ["en"],
    "glue_qnli":      ["en"], "glue_rte":  ["en"],
    # SuperGLUE — harness-compatible subset (no COPA / ReCoRD)
    "sg_boolq":       ["en"], "sg_cb":     ["en"], "sg_wic":   ["en"],
    "sg_wsc":         ["en"],
}


def load_xnli(langs):
    out_train, per_lang_val, per_lang_test = [], {}, {}
    for L in langs:
        d = load_dataset("facebook/xnli", L)
        for s in d.keys():
            d[s] = d[s].map(
                lambda x: {"text_a": x["premise"], "text_b": x["hypothesis"],
                           "labels": x["label"], "lang": L},
                remove_columns=d[s].column_names,
            )
        out_train.append(d["train"])
        per_lang_val[L] = d["validation"]
        if "test" in d:
            per_lang_test[L] = d["test"]
    return {"train": out_train}, per_lang_val, per_lang_test, 3, "single_label_classification"


def load_pawsx(langs):
    out_train, per_lang_val, per_lang_test = [], {}, {}
    for L in langs:
        d = load_dataset("google-research-datasets/paws-x", L)
        for s in d.keys():
            d[s] = d[s].map(
                lambda x: {"text_a": x["sentence1"], "text_b": x["sentence2"],
                           "labels": x["label"], "lang": L},
                remove_columns=d[s].column_names,
            )
        out_train.append(d["train"])
        per_lang_val[L] = d["validation"]
        if "test" in d:
            per_lang_test[L] = d["test"]
    return {"train": out_train}, per_lang_val, per_lang_test, 2, "single_label_classification"


def load_amazon(langs):
    """SetFit/amazon_reviews_multi_<lang>: 5-class star rating."""
    out_train, per_lang_val, per_lang_test = [], {}, {}
    for L in langs:
        d = load_dataset(f"SetFit/amazon_reviews_multi_{L}")
        for s in d.keys():
            d[s] = d[s].map(
                lambda x: {"text_a": x["text"], "text_b": None,
                           "labels": x["label"], "lang": L},
                remove_columns=d[s].column_names,
            )
        out_train.append(d["train"])
        per_lang_val[L] = d["validation"]
        if "test" in d:
            per_lang_test[L] = d["test"]
    return {"train": out_train}, per_lang_val, per_lang_test, 5, "single_label_classification"


def load_massive_intent(langs):
    """mteb/amazon_massive_intent: 60-class intent string → global label map."""
    en = load_dataset("mteb/amazon_massive_intent", "en")
    all_labels = sorted({r["label"] for r in en["train"]})
    label2id = {l: i for i, l in enumerate(all_labels)}
    n_labels = len(all_labels)
    out_train, per_lang_val, per_lang_test = [], {}, {}
    for L in langs:
        cfg = {"zh": "zh-CN"}.get(L, L)
        d = load_dataset("mteb/amazon_massive_intent", cfg)
        for s in d.keys():
            d[s] = d[s].map(
                lambda x: {"text_a": x["text"], "text_b": None,
                           "labels": label2id[x["label"]], "lang": L},
                remove_columns=d[s].column_names,
            )
        out_train.append(d["train"])
        per_lang_val[L] = d["validation"]
        if "test" in d:
            per_lang_test[L] = d["test"]
    return {"train": out_train}, per_lang_val, per_lang_test, n_labels, "single_label_classification"


def load_seahorse(langs):
    """hgissbkh/seahorse: regression target = mean of 6 binary dims, summary first."""
    DIMS = ["comprehensible", "repetition", "grammar", "attribution",
            "main_ideas", "conciseness"]
    raw = load_dataset("hgissbkh/seahorse")
    out_train, per_lang_val, per_lang_test = [], {}, {}
    for L in langs:
        sub = DatasetDict({s: raw[s].filter(lambda x: x.get("lang") == L)
                           for s in raw.keys()})
        for s in sub.keys():
            sub[s] = sub[s].map(
                lambda x: {"text_a": x["summary"], "text_b": x["text"],
                           "labels": float(sum(x[d] for d in DIMS) / len(DIMS)),
                           "lang": L},
                remove_columns=sub[s].column_names,
            )
        if len(sub["train"]) > 0:
            out_train.append(sub["train"])
        per_lang_val[L] = sub["validation"]
        if "test" in sub and len(sub["test"]) > 0:
            per_lang_test[L] = sub["test"]
    return {"train": out_train}, per_lang_val, per_lang_test, 1, "regression"


# ---------- GLUE (English) ----------
def _glue_single(name, text_key, num_labels, problem="single_label_classification",
                 val_key="validation"):
    """Single-sentence GLUE task (CoLA, SST-2)."""
    def loader(_langs):
        d = load_dataset("nyu-mll/glue", name)
        for s in d.keys():
            d[s] = d[s].map(
                lambda x: {"text_a": x[text_key], "text_b": None,
                           "labels": float(x["label"]) if problem == "regression" else int(x["label"]),
                           "lang": "en"},
                remove_columns=d[s].column_names,
            )
        return {"train": [d["train"]]}, {"en": d[val_key]}, {}, num_labels, problem
    return loader


def _glue_pair(name, key_a, key_b, num_labels, problem="single_label_classification",
               val_key="validation"):
    """Sentence-pair GLUE task (MRPC, STS-B, QQP, MNLI, QNLI, RTE)."""
    def loader(_langs):
        d = load_dataset("nyu-mll/glue", name)
        for s in d.keys():
            d[s] = d[s].map(
                lambda x: {"text_a": x[key_a], "text_b": x[key_b],
                           "labels": float(x["label"]) if problem == "regression" else int(x["label"]),
                           "lang": "en"},
                remove_columns=d[s].column_names,
            )
        return {"train": [d["train"]]}, {"en": d[val_key]}, {}, num_labels, problem
    return loader


# ---------- SuperGLUE (English) ----------
def _sg_pair(name, key_a, key_b, num_labels):
    def loader(_langs):
        d = load_dataset("aps/super_glue", name)
        for s in d.keys():
            d[s] = d[s].map(
                lambda x: {"text_a": x[key_a], "text_b": x[key_b],
                           "labels": int(x["label"]), "lang": "en"},
                remove_columns=d[s].column_names,
            )
        return {"train": [d["train"]]}, {"en": d["validation"]}, {}, num_labels, "single_label_classification"
    return loader


def load_sg_wic(_langs):
    """WiC: same-sense binary classification of a target word in two sentences.

    Encode as: text_a = "<word>: <sentence1>", text_b = "<word>: <sentence2>"
    so the model sees the target word in both segments.
    """
    d = load_dataset("aps/super_glue", "wic")
    for s in d.keys():
        d[s] = d[s].map(
            lambda x: {"text_a": f"{x['word']}: {x['sentence1']}",
                       "text_b": f"{x['word']}: {x['sentence2']}",
                       "labels": int(x["label"]), "lang": "en"},
            remove_columns=d[s].column_names,
        )
    return {"train": [d["train"]]}, {"en": d["validation"]}, {}, 2, "single_label_classification"


def load_sg_wsc(_langs):
    """WSC: coreference resolution. Format as (text, "<span1> ↔ <span2>") pair."""
    d = load_dataset("aps/super_glue", "wsc.fixed")
    for s in d.keys():
        d[s] = d[s].map(
            lambda x: {"text_a": x["text"],
                       "text_b": f"{x['span1_text']} ↔ {x['span2_text']}",
                       "labels": int(x["label"]), "lang": "en"},
            remove_columns=d[s].column_names,
        )
    return {"train": [d["train"]]}, {"en": d["validation"]}, {}, 2, "single_label_classification"


LOADERS = {
    "xnli": load_xnli,
    "paws_x": load_pawsx,
    "amazon": load_amazon,
    "massive_intent": load_massive_intent,
    "seahorse": load_seahorse,
    # GLUE
    "glue_cola": _glue_single("cola", "sentence", 2),
    "glue_sst2": _glue_single("sst2", "sentence", 2),
    "glue_mrpc": _glue_pair("mrpc", "sentence1", "sentence2", 2),
    "glue_stsb": _glue_pair("stsb", "sentence1", "sentence2", 1, problem="regression"),
    "glue_qqp":  _glue_pair("qqp",  "question1", "question2", 2),
    "glue_mnli": _glue_pair("mnli", "premise",   "hypothesis", 3, val_key="validation_matched"),
    "glue_qnli": _glue_pair("qnli", "question",  "sentence",  2),
    "glue_rte":  _glue_pair("rte",  "sentence1", "sentence2", 2),
    # SuperGLUE
    "sg_boolq":  _sg_pair("boolq", "passage", "question", 2),
    "sg_cb":     _sg_pair("cb",    "premise", "hypothesis", 3),
    "sg_wic":    load_sg_wic,
    "sg_wsc":    load_sg_wsc,
}

# Recommended per-task knobs (paper recipe — bsz=32, 10k steps, except SeaHorse 5k)
TASK_KNOBS = {
    "xnli":           {"max_steps": 10000, "bsz": 32, "max_seq_len": 256},
    "paws_x":         {"max_steps": 10000, "bsz": 32, "max_seq_len": 256},
    "amazon":         {"max_steps": 10000, "bsz": 32, "max_seq_len": 256},
    "massive_intent": {"max_steps": 10000, "bsz": 32, "max_seq_len": 128},
    "seahorse":       {"max_steps": 5000,  "bsz": 4,  "max_seq_len": 1024},
    # GLUE — short English text, bsz=32, 10k steps (small tasks just over-train; that's fine)
    "glue_cola":      {"max_steps": 10000, "bsz": 32, "max_seq_len": 128},
    "glue_sst2":      {"max_steps": 10000, "bsz": 32, "max_seq_len": 128},
    "glue_mrpc":      {"max_steps": 10000, "bsz": 32, "max_seq_len": 128},
    "glue_stsb":      {"max_steps": 10000, "bsz": 32, "max_seq_len": 128},
    "glue_qqp":       {"max_steps": 10000, "bsz": 32, "max_seq_len": 128},
    "glue_mnli":      {"max_steps": 10000, "bsz": 32, "max_seq_len": 128},
    "glue_qnli":      {"max_steps": 10000, "bsz": 32, "max_seq_len": 128},
    "glue_rte":       {"max_steps": 10000, "bsz": 32, "max_seq_len": 128},
    # SuperGLUE
    "sg_boolq":       {"max_steps": 10000, "bsz": 32, "max_seq_len": 256},
    "sg_cb":          {"max_steps": 10000, "bsz": 32, "max_seq_len": 256},
    "sg_wic":         {"max_steps": 10000, "bsz": 32, "max_seq_len": 128},
    "sg_wsc":         {"max_steps": 10000, "bsz": 32, "max_seq_len": 128},
}
