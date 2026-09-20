#!/usr/bin/env python3
"""v4: build the gate table (O / P_web / P_inf / C_inf vs Bami/Cafe), the deltas Q1-Q3 and the report tables.

Writes `artifacts/mmbert_vi_cpt_pilot/v4/summary.json` and `artifacts/mmbert_vi_cpt_pilot/v4/report_tables.md`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_v4 as C4  # noqa: E402

TASKS = ("absa_restaurant", "visfd", "vsfc_topic")
LABEL = {"absa_restaurant": "Restaurant", "visfd": "ViSFD", "vsfc_topic": "Topic"}
ARMS = ("O", "P_web", "P_inf", "C_inf")
REFERENCE = {"Cafe": {"absa_restaurant": 0.7772, "visfd": 0.8547, "vsfc_topic": 0.8174},
             "Bami": {"absa_restaurant": 0.7474, "visfd": 0.8438, "vsfc_topic": 0.7953}}
BASELINE_O42 = {"absa_restaurant": C4.ROOT / "artifacts/base_encoder_baseline/mmbert_base/absa_restaurant/lr_5e-05/metadata.json",
                "visfd": C4.ROOT / "artifacts/base_encoder_baseline/mmbert_base/visfd/lr_5e-05/metadata.json",
                "vsfc_topic": C4.ROOT / "artifacts/base_encoder_baseline/mmbert_base/vsfc_topic/lr_2e-05/metadata.json"}


def arm_dir(arm: str) -> str:
    return arm + (C4.NC_SUFFIX if arm in C4.ARMS_NC else "")


def ft_row(arm: str, task: str, seed: int):
    if arm == "O" and seed == 42:
        p = BASELINE_O42[task]
        if not p.is_file():
            return None
        m = json.loads(p.read_text(encoding="utf-8"))
        return {"f1": m["best_selection_f1"], "epoch": m["best_epoch"], "status": m["status"], "minutes": m["elapsed_seconds"] / 60,
                "source": str(p.relative_to(C4.ROOT)), "seed": 42, "lr": m["learning_rate"]}
    d = C4.ART_V4 / "ft" / arm_dir(arm) / task / f"seed_{seed}"
    if not (d / "metadata.json").is_file():
        return None
    m = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    return {"f1": m["best_selection_f1"], "epoch": m["best_epoch"], "status": m["status"], "minutes": m["elapsed_seconds"] / 60,
            "source": str(d.relative_to(C4.ROOT)), "seed": seed, "lr": m["learning_rate"],
            "reload_ok": m.get("postrun_checks", {}).get("reload_matches_best"), "ids_ok": m.get("postrun_checks", {}).get("ids_stable_and_complete"),
            "curve": [{"epoch": e["epoch"], "f1": e["selection_f1"]} for e in m.get("epoch_curves", [])]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(C4.ART_V4))
    args = ap.parse_args()
    out = Path(args.out)
    rows = {}
    for arm in ARMS:
        for task in TASKS:
            seeds = (42, 43) if (task == "absa_restaurant" or arm == "O") else (42,)
            rows[(arm, task)] = {s: ft_row(arm, task, s) for s in seeds}

    def pairmean(arm: str, task: str):
        vals = [v["f1"] for v in rows[(arm, task)].values() if v]
        return statistics.mean(vals) if vals else None

    deltas = {q: {t: (pairmean(a, t) - pairmean(b, t)) if (pairmean(a, t) is not None and pairmean(b, t) is not None) else None
                  for t in TASKS}
              for q, (a, b) in {"Q1": ("P_web", "O"), "Q2": ("P_inf", "P_web"), "Q3": ("C_inf", "P_inf")}.items()}
    line = []
    line.append("| Arm | Restaurant (42 / 43 / pair-mean) | ViSFD (42) | Topic (42) |")
    line.append("|---|---|---|---|")
    for arm in ARMS:
        r = rows[(arm, "absa_restaurant")]
        parts = " / ".join("n/a" if not v else f"{100*v['f1']:.2f}" for v in r.values())
        pm = pairmean(arm, "absa_restaurant")
        cell_r = f"{parts}" + (f" / **{100*pm:.2f}**" if pm else "")
        line.append(f"| {arm} | {cell_r} | "
                    + (f"{100*rows[(arm,'visfd')][42]['f1']:.2f}" if rows[(arm, 'visfd')][42] else "n/a") + " | "
                    + (f"{100*rows[(arm,'vsfc_topic')][42]['f1']:.2f}" if rows[(arm, 'vsfc_topic')][42] else "n/a") + " |")
    for name, ref in REFERENCE.items():
        line.append(f"| {name} (tham chiếu) | {100*ref['absa_restaurant']:.2f} | {100*ref['visfd']:.2f} | {100*ref['vsfc_topic']:.2f} |")
    line.append("")
    line.append("| Delta | Restaurant | ViSFD | Topic |")
    line.append("|---|---|---|---|")
    for q, d in deltas.items():
        line.append(f"| {q} | " + " | ".join("n/a" if d[t] is None else f"{100*d[t]:+.2f}" for t in TASKS) + " |")
    gate = {"restaurant_pairmean": {a: pairmean(a, "absa_restaurant") for a in ARMS},
            "deltas": deltas}
    q2_rest = deltas["Q2"]["absa_restaurant"]
    safe = all((pairmean(a, t) or 0) >= (pairmean("O", t) or 0) - 0.005 for a in ("P_inf", "C_inf") for t in ("visfd", "vsfc_topic"))
    gate["Q2_pass"] = bool(q2_rest is not None and q2_rest >= 0.005 and safe)
    q3 = deltas["Q3"]["absa_restaurant"]
    gate["Q3_pass"] = bool(gate["Q2_pass"] and q3 is not None and q3 >= 0.005)
    q1 = deltas["Q1"]
    gate["Q1_pass"] = bool(any(v is not None and v >= 0.005 for v in q1.values()) and (q1["absa_restaurant"] is not None and q1["absa_restaurant"] >= -0.005))
    (out / "report_tables.md").write_text("\n".join(line) + "\n", encoding="utf-8")
    C4.write_json(out / "summary.json", {"created_at": C4.common.now(), "rows": {f"{k[0]}/{k[1]}": {str(s): v for s, v in r.items()} for k, r in rows.items()},
                                         "deltas": deltas, "gate": gate, "reference": REFERENCE})
    print("\n".join(line))
    print(json.dumps(gate, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
