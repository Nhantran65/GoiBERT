"""Collect M1 (T/B/K) + D1 (TD/KD_D) into artifacts/cdbk_dapt_pilot/summary.json and print the report table."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_d1  # noqa: E402

common_d1.install()
import common  # noqa: E402
from common import ROOT  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts" / "cdbk_pilot"))
from summarize import load_run  # noqa: E402  (M1 loader; works on any run dir)

M1_RUNS = ROOT / "artifacts" / "cdbk_pilot" / "runs"
LABELS = {"T": ("CafeBERT", "CE", "2e-05"), "B": ("BamiBERT", "CE", "5e-05"), "K": ("BamiBERT", "CE+KD(T)", "5e-05"),
          "TD": ("CafeBERT-DAPT", "CE", "2e-05"), "KD_D": ("BamiBERT", "CE+KD(TD)", "5e-05")}


def main() -> int:
    runs = {a: load_run(M1_RUNS / f"{a}_absa_restaurant_seed42") for a in ("T", "B", "K")}
    runs.update({a: load_run(common_d1.run_dir(a)) for a in ("TD", "KD_D")})
    f = lambda a: runs[a].get("selection_f1") if runs.get(a) else None  # noqa: E731
    deltas = {}
    for name, (x, y) in {"T_minus_B": ("T", "B"), "K_minus_B": ("K", "B"), "TD_minus_T": ("TD", "T"), "KD_D_minus_K": ("KD_D", "K"), "KD_D_minus_B": ("KD_D", "B"), "KD_D_minus_TD": ("KD_D", "TD")}.items():
        if f(x) is not None and f(y) is not None:
            deltas[name + "_pp"] = 100 * (f(x) - f(y))
    dapt = common_d1.dapt_summary() if (common_d1.DAPT_DIR / "summary.json").is_file() else None
    checks_dir = common_d1.D1_ROOT / "checks"
    checks = {p.stem: common.read_json(p).get("all_ok") for p in checks_dir.glob("*.json")} if checks_dir.is_dir() else {}
    summary = {"protocol": common_d1.PROTOCOL_D1, "task": common.TASK_ID, "seed": common.SEED, "runs": runs, "deltas_pp": deltas,
               "dapt": None if dapt is None else {k: dapt.get(k) for k in ("tokens_processed", "passes_over_train", "wall_time_s", "tokens_per_s", "holdout", "train_loss_first_50_mean", "train_loss_last_50_mean", "export", "corpus_tokens", "masked_positions_total", "world", "micro", "accum", "peak_gpu_memory_gb_rank0")},
               "checks_all_ok": checks, "created_at": common.now()}
    common.write_json(common_d1.D1_ROOT / "summary.json", summary)
    print("| Arm | Init | Loss | LR | Best epoch | Val pair micro-F1 | Runtime (min) | Status |")
    print("|---|---|---|---|---|---|---|---|")
    for a in ("T", "B", "K", "TD", "KD_D"):
        r = runs[a]; init, loss, lr = LABELS[a]
        if r and r.get("selection_f1") is not None:
            print(f"| {a} | {init} | {loss} | {lr} | {r['best_epoch']} | {100*r['selection_f1']:.2f} | {r['elapsed_minutes']:.1f} | {r['status']} |")
        else:
            print(f"| {a} | {init} | {loss} | {lr} | – | – | – | {r.get('status') if r else 'missing'} |")
    for k, v in deltas.items():
        print(f"{k}: {v:+.2f}")
    if dapt:
        print(f"DAPT: {dapt['tokens_processed']} tokens, {dapt['passes_over_train']:.2f} passes, {dapt['wall_time_s']/3600:.2f} h, holdout {dapt['holdout']['initial']:.4f} -> {dapt['holdout']['final']:.4f}")
    print("checks:", checks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
