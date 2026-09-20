#!/usr/bin/env python3
"""Evaluate the amended §5 gate (plan Phụ lục A, protocol v2) on the v1 preflight and profile evidence — written BEFORE M/G launch.

v2 criteria: L_MLM / L_rel and their gradients finite; encoder gradient of L_rel non-zero on the fixed minibatches; no
instability during the ~100-step profile (finite losses, no divergence, ratio > 0 throughout).  No ratio floor/ceiling.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def main() -> int:
    pf = common.read_json(common.ART_DIR / "preflight.json")
    prof = common.read_json(common.ART_DIR / "cpt" / "profile_G_micro64.json")
    prof_m = common.read_json(common.ART_DIR / "cpt" / "profile_M_micro64.json")
    diag = json.loads((common.ART_DIR / "grad_ratio_diagnostic.json").read_text())
    log_lines = [l for l in (common.ART_DIR / "cpt" / "profile_G.log").read_text(encoding="utf-8").splitlines() if l.startswith("[G] step ")]
    losses = [float(l.split(" loss ")[1].split()[0]) for l in log_lines]
    rels = [float(l.split(" rel ")[1].split()[0]) for l in log_lines]
    per = pf["grad_ratio"]["per_batch"]
    finite = all(r["finite"] and math.isfinite(r["mlm"]) and math.isfinite(r["rel"]) for r in per) and all(math.isfinite(r["enc_ratio"]) for r in diag)
    nonzero = all(r["grad_norm_rel"] > 0 for r in per) and all(r["gr"]["layers_1_6"] > 0 and r["gr"]["layers_7_12"] > 0 for r in diag) and pf["gradients"]["student_encoder_grad_norm_from_rel"] > 0
    stable = all(math.isfinite(x) for x in losses) and max(losses) < 2 * losses[0] and all(r["ratio"] > 0 for r in prof["grad_ratios"]) and rels[-1] < rels[0] and math.isfinite(prof_m["last_loss"])
    decision = {
        "protocol": common.PROTOCOL, "stage": "gate_v2_before_launch", "created_at": common.now(), "source_hash": common.source_hash(),
        "amendment": "plan Phụ lục A (2026-09-18): §5 gate = finite, non-zero, stable; ratio floor 0.05 removed; lambda 1.0, T 0.2, layer pairs, corpus, budget, §6 gate unchanged",
        "evidence": {"preflight_file": "artifacts/cafe_bami_repdistill_pilot/preflight.json", "profile_G": "artifacts/cafe_bami_repdistill_pilot/cpt/profile_G_micro64.json", "profile_M": "artifacts/cafe_bami_repdistill_pilot/cpt/profile_M_micro64.json",
                     "preflight_ratios": [r["ratio"] for r in per], "preflight_rel_grad_norms": [r["grad_norm_rel"] for r in per], "diagnostic_16_batches_ratio_min": min(r["enc_ratio"] for r in diag),
                     "profile_G_loss_first_last": [losses[0], losses[-1]], "profile_G_rel_first_last": [rels[0], rels[-1]], "profile_G_ratio_trajectory": [(r["step"], round(r["ratio"], 4)) for r in prof["grad_ratios"]],
                     "profile_M_loss_first_last": [prof_m["first_loss"], prof_m["last_loss"]]},
        "criteria": {"finite": finite, "rel_gradient_nonzero": nonzero, "stable_in_profile": stable},
        "v1_gate_for_record": {"ratio_range_0.05_20": False, "note": "v1 preflight ratio 0.015-0.035 recorded in gate_decision.json; no longer a stop condition under v2"},
        "data_gate": common.read_json(common.DATA_DIR / "manifest.json")["gate"], "alignment_gate_pass": pf["gate"]["alignment_pass_95"] and pf["gate"]["no_out_of_range_spans"] and pf["gate"]["manual_cases_reviewed"],
        "launch_allowed": bool(finite and nonzero and stable),
        "runs_launched_at_decision_time": {"cpt_M": False, "cpt_G": False, "finetunes": False},
    }
    decision["decision"] = "LAUNCH_M_AND_G" if decision["launch_allowed"] else "STOP"
    common.write_json(common.ART_DIR / "gate_decision_v2.json", decision)
    print(json.dumps({k: decision[k] for k in ("decision", "criteria", "launch_allowed", "source_hash")}, indent=1))
    return 0 if decision["launch_allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
