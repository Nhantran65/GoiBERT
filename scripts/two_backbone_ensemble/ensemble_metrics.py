#!/usr/bin/env python3
"""Validation-only probability ensemble, paired bootstrap and locked gate."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from goibert import train

OUT = ROOT / "artifacts/two_backbone_ensemble"
TASKS = ("absa_restaurant", "visfd", "vsfc_topic")
MODELS = ("cafe", "cafe_s43", "xlmr_large", "p_vi", "o", "b1b")
COMBOS = {"PAIR": ("cafe", "p_vi"), "CTRL_FAM": ("cafe", "xlmr_large"),
          "CTRL_SEED": ("cafe", "cafe_s43"), "cafe+o": ("cafe", "o"),
          "cafe+b1b": ("cafe", "b1b"), "p_vi+b1b": ("p_vi", "b1b"),
          "cafe+p_vi+b1b": ("cafe", "p_vi", "b1b")}


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read(name):
    return json.loads((OUT / name).read_text())


def load(model, task):
    path = OUT / "logits" / model / f"{task}.npz"
    with np.load(path) as z:
        return {k: z[k].copy() for k in z.files}


def aligned(ref, other):
    lookup = {str(v): i for i, v in enumerate(other["ids"])}
    idx = [lookup[str(v)] for v in ref["ids"]]
    assert len(idx) == len(lookup)
    assert np.array_equal(ref["gold"], other["gold"][idx])
    assert np.array_equal(ref["aspect_class_counts"], other["aspect_class_counts"])
    return other["logits"][idx]


def probs(logits, counts):
    if not len(counts) or logits.ndim == 3:
        x = logits.astype(np.float64)
        x -= x.max(axis=-1, keepdims=True)
        p = np.exp(x)
        return p / p.sum(axis=-1, keepdims=True)
    parts = []
    start = 0
    for width in counts:
        x = logits[:, start:start + width].astype(np.float64)
        x -= x.max(axis=-1, keepdims=True)
        p = np.exp(x)
        parts.append(p / p.sum(axis=-1, keepdims=True))
        start += width
    assert start == logits.shape[1]
    return np.concatenate(parts, axis=1)


def predict(prob, counts):
    return train._predictions(torch.from_numpy(np.log(np.maximum(prob, 1e-300))), aspect_class_counts=counts.tolist())


def f1(task, bundle, gold, pred):
    return float(train.selection_score(train.evaluate_predictions(task, bundle, gold, pred, split="validation"))) * 100


def bootstrap(task, bundle, gold, pred_pair, pred_cafe, pred_ctrl):
    rng = np.random.default_rng(42)
    n = len(gold)
    arr = np.empty((1000, 2), dtype=np.float64)
    for i in range(1000):
        idx = rng.integers(0, n, n)
        g = [gold[j] for j in idx]
        a = f1(task, bundle, g, [pred_pair[j] for j in idx])
        b = f1(task, bundle, g, [pred_cafe[j] for j in idx])
        c = f1(task, bundle, g, [pred_ctrl[j] for j in idx])
        arr[i] = (a - b, a - c)
    return {name: {"point": None, "ci95": np.quantile(arr[:, k], [0.025, 0.975]).tolist(),
                   "fraction_gt_zero": float(np.mean(arr[:, k] > 0)), "replicates": 1000, "seed": 42}
            for k, name in enumerate(("PAIR_minus_cafe", "PAIR_minus_CTRL_FAM"))}


def correction(gold, cafe, pair):
    g = np.asarray(gold)
    c = np.asarray(cafe)
    p = np.asarray(pair)
    if g.ndim == 1:
        g, c, p = g[:, None], c[:, None], p[:, None]
    fix = (c != g) & (p == g)
    harm = (c == g) & (p != g)
    return {"fixed_slots": int(fix.sum()), "harmed_slots": int(harm.sum()),
            "examples_with_fix": int(fix.any(axis=1).sum()), "examples_with_harm": int(harm.any(axis=1).sum()),
            "examples_with_both": int((fix.any(axis=1) & harm.any(axis=1)).sum())}


def main():
    singles, results, boots, trims, counts, manifests, checks = {}, {}, {}, {}, {}, {}, {}
    for model in MODELS:
        manifests[model] = {}
        checks[model] = {}
        for task in TASKS:
            p = OUT / "manifest_parts" / model / f"{task}.json"
            if p.exists():
                manifests[model][task] = json.loads(p.read_text())
                checks[model][task] = json.loads((OUT / "checks" / model / f"{task}.json").read_text())
    write("manifest.json", {"runs": manifests, "gpu_hours": sum(v["gpu_seconds"] for x in manifests.values() for v in x.values()) / 3600})
    write("reproduction_checks.json", checks)
    for task in TASKS:
        bundle = train.load_task_bundle(task, ROOT, include_test=False)
        ref = load("cafe", task)
        ids = [str(x) for x in ref["ids"]]
        gold = ref["gold"].tolist()
        class_counts = ref["aspect_class_counts"]
        p = {}
        pred = {}
        singles[task] = {}
        for model in MODELS:
            if model == "cafe_s43" and task != "absa_restaurant":
                continue
            entry = ref if model == "cafe" else load(model, task)
            logits = entry["logits"] if model == "cafe" else aligned(ref, entry)
            p[model] = probs(logits, class_counts)
            pred[model] = predict(p[model], class_counts)
            singles[task][model] = f1(task, bundle, gold, pred[model])
            assert abs(singles[task][model] / 100 - checks[model][task]["f1"]) < 1e-9
        results[task] = {"single": singles[task], "ensemble": {}, "weight_scan": {}, "corrections": {}}
        all_preds = {}
        for name, members in COMBOS.items():
            if any(m not in p for m in members):
                continue
            avg = np.mean([p[m] for m in members], axis=0)
            pr = predict(avg, class_counts)
            all_preds[name] = pr
            value = f1(task, bundle, gold, pr)
            results[task]["ensemble"][name] = {"f1": value, "delta_cafe": value - singles[task]["cafe"]}
        ctrl = results[task]["ensemble"]["CTRL_FAM"]["f1"]
        for v in results[task]["ensemble"].values():
            v["delta_CTRL_FAM"] = v["f1"] - ctrl
        for weight in (0.3, 0.4, 0.5, 0.6, 0.7):
            avg = weight * p["cafe"] + (1 - weight) * p["p_vi"]
            results[task]["weight_scan"][str(weight)] = f1(task, bundle, gold, predict(avg, class_counts))
        results[task]["corrections"] = correction(gold, pred["cafe"], all_preds["PAIR"])
        boots[task] = bootstrap(task, bundle, gold, all_preds["PAIR"], pred["cafe"], all_preds["CTRL_FAM"])
        boots[task]["PAIR_minus_cafe"]["point"] = results[task]["ensemble"]["PAIR"]["delta_cafe"]
        boots[task]["PAIR_minus_CTRL_FAM"]["point"] = results[task]["ensemble"]["PAIR"]["delta_CTRL_FAM"]
        trims[task] = json.loads((OUT / "trim" / f"{task}.json").read_text())
        counts[task] = {"cafe_trim": trims[task]["parameters_trim"], "p_vi": manifests["p_vi"][task]["model_spec"]["total_parameters_with_head"]}
        counts[task]["total"] = counts[task]["cafe_trim"] + counts[task]["p_vi"]
    write("ensemble_results.json", results)
    write("bootstrap.json", boots)
    write("trim_check.json", trims)
    counts["max_head_total"] = max(v["total"] for v in counts.values())
    counts["within_1b"] = counts["max_head_total"] <= 1_000_000_000
    write("param_counts.json", counts)
    gate = {"decision": "PASS", "tasks": {}, "trim_identical_all": all(v["predictions_identical"] and v["f1_trim"] == v["f1_original"] for v in trims.values()),
            "parameter_within_1b": counts["within_1b"], "max_head_total": counts["max_head_total"]}
    for task in TASKS:
        pair = results[task]["ensemble"]["PAIR"]["f1"]
        cafe = singles[task]["cafe"]
        ctrl = results[task]["ensemble"]["CTRL_FAM"]["f1"]
        gate["tasks"][task] = {"PAIR": pair, "cafe": cafe, "CTRL_FAM": ctrl, "delta_cafe": pair - cafe,
                                "delta_CTRL_FAM": pair - ctrl, "gain_ge_0p8": pair >= cafe + 0.8,
                                "beats_CTRL_FAM": pair > ctrl}
    if not (gate["trim_identical_all"] and gate["parameter_within_1b"] and all(v["gain_ge_0p8"] and v["beats_CTRL_FAM"] for v in gate["tasks"].values())):
        gate["decision"] = "CLOSE"
    write("gate.json", gate)
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
