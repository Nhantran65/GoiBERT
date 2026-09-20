#!/usr/bin/env python3
"""Grid table, per-task LR selection (max validation, tie -> smaller LR), comparison with BamiBERT E14 (same grid) and
CafeBERT (fixed 2e-5) validation references, pre-registered gate, and docs/BASE_ENCODER_BASELINE_RESULT.md."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

TASK_LABEL = {"absa_restaurant": "Restaurant", "vsfc_topic": "Topic", "visfd": "ViSFD", "absa_hotel": "Hotel", "vsfc_sentiment": "Sentiment"}
CAFE = {"absa_restaurant": common.ROOT / "artifacts/cdbk_pilot/runs/T_absa_restaurant_seed42/metadata.json"}
for t in ("vsfc_topic", "visfd", "absa_hotel", "vsfc_sentiment"):
    CAFE[t] = common.ROOT / f"artifacts/sealion_benchmark/cafe_runs/{t}/lr_2e-05/seed_42/metadata.json"


def rj(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def cell(model: str, task: str, lr: float):
    d = common.ART / model / task / f"lr_{lr:.0e}".replace("+", "")
    if not (d / "metadata.json").exists():
        return None
    m = rj(d / "metadata.json")
    return {"f1": m["best_selection_f1"], "epoch": m["best_epoch"], "status": m["status"], "minutes": m["elapsed_seconds"] / 60, "checks_ok": m["status"] == "completed", "dir": str(d.relative_to(common.ROOT))}


def select(cells: dict[float, dict | None]):
    done = {lr: c for lr, c in cells.items() if c}
    if len(done) < len(cells):
        return None
    best = max(done.values(), key=lambda c: c["f1"])["f1"]
    lr = min(l for l, c in done.items() if c["f1"] == best)
    return {"lr": lr, **done[lr]}


def main() -> int:
    bami_grid = rj(common.ROOT / "artifacts/e14/validation_grid.json")["grid"]
    bami = {}
    for t in common.TASKS:
        rows = [r for r in bami_grid if r["task_id"] == t]
        best = max(r["selection_f1"] for r in rows); lr = min(r["learning_rate"] for r in rows if r["selection_f1"] == best)
        bami[t] = {"f1": best, "lr": lr, "epoch": [r["best_epoch"] for r in rows if r["learning_rate"] == lr][0], "grid": {r["learning_rate"]: r["selection_f1"] for r in rows}}
    cafe = {t: {"f1": rj(p)["best_selection_f1"], "epoch": rj(p)["best_epoch"], "lr": 2e-5} for t, p in CAFE.items()}
    out = {"protocol": common.PROTOCOL, "created_at": common.now(), "models": {}, "references": {"bami_e14": bami, "cafe_fixed_2e-5": cafe}}
    for model in common.MODELS:
        grid = {t: {lr: cell(model, t, lr) for lr in common.LR_GRID} for t in common.TASKS}
        sel = {t: select(grid[t]) for t in common.TASKS}
        out["models"][model] = {"grid": {t: {f"{lr:.0e}": c for lr, c in g.items()} for t, g in grid.items()}, "selected": sel, "complete": all(sel.values()),
                                "manifest": common.model_manifest(model)}
        if all(sel.values()):
            d = {t: 100 * (sel[t]["f1"] - bami[t]["f1"]) for t in common.TASKS}
            dc = {t: 100 * (sel[t]["f1"] - cafe[t]["f1"]) for t in common.TASKS}
            out["models"][model]["delta_vs_bami_pp"] = d; out["models"][model]["delta_vs_cafe_pp"] = dc
            gate = {"ge_bami_minus_0.5_on_4_of_5": sum(v >= -0.5 for v in d.values()) >= 4, "restaurant_ge_bami_minus_0.5": d["absa_restaurant"] >= -0.5,
                    "topic_or_visfd_plus_0.5": d["vsfc_topic"] >= 0.5 or d["visfd"] >= 0.5, "borderline_tasks": [t for t, v in d.items() if abs(v) <= 0.5]}
            gate["pass"] = gate["ge_bami_minus_0.5_on_4_of_5"] and gate["restaurant_ge_bami_minus_0.5"] and gate["topic_or_visfd_plus_0.5"]
            out["models"][model]["gate"] = gate
    common.ART.mkdir(parents=True, exist_ok=True)
    (common.ART / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    # report
    L = ["# Baseline mmBERT-base / XLM-R-base trên 5 task — kết quả (validation seed 42)", "",
         f"Plan `docs/BASE_ENCODER_BASELINE_PLAN.md`; tạo {out['created_at']}; recipe E14 (3 LR, 30 epoch, head CLS E14, eff. batch 32, bf16); không test. Artifacts `artifacts/base_encoder_baseline/`. F1×100; ABSA = pair micro F1, VSFC = fixed-label macro F1.", ""]
    for model, r in out["models"].items():
        m = r["manifest"]
        spec_any = None
        for t in common.TASKS:
            for lr in common.LR_GRID:
                c = r["grid"][t].get(f"{lr:.0e}")
                if c:
                    spec_any = rj(common.ROOT / c["dir"] / "metadata.json")["model_spec"]; break
            if spec_any:
                break
        L.append(f"## {model} — `{m['repo']} @ {m['commit'][:12]}`" + (f" ({spec_any['encoder_class']}, {spec_any['num_hidden_layers']}L/{spec_any['hidden_size']}, {spec_any['encoder_non_embedding_parameters']/1e6:.1f}M không-embedding / {spec_any['encoder_parameters']/1e6:.1f}M encoder, vocab {spec_any['vocab_size']})" if spec_any else ""))
        L.append(""); L.append("| Task | LR 1e-5 | LR 2e-5 | LR 5e-5 | Chọn (LR, epoch) | Bami E14 (LR) | Cafe 2e-5 | Δ vs Bami | Δ vs Cafe |"); L.append("|---|---:|---:|---:|---|---:|---:|---:|---:|")
        for t in common.TASKS:
            g = r["grid"][t]; s = r["selected"][t]
            cells = [f"{100*g[f'{lr:.0e}']['f1']:.2f}@{g[f'{lr:.0e}']['epoch']}" if g.get(f"{lr:.0e}") else "—" for lr in common.LR_GRID]
            L.append(f"| {TASK_LABEL[t]} | {cells[0]} | {cells[1]} | {cells[2]} | " + (f"**{100*s['f1']:.2f}** ({s['lr']:.0e}, {s['epoch']})" if s else "—") + f" | {100*bami[t]['f1']:.2f} ({bami[t]['lr']:.0e}) | {100*cafe[t]['f1']:.2f} | " + (f"{r['delta_vs_bami_pp'][t]:+.2f}" if s and "delta_vs_bami_pp" in r else "—") + " | " + (f"{r['delta_vs_cafe_pp'][t]:+.2f}" if s and "delta_vs_cafe_pp" in r else "—") + " |")
        L.append("")
        if "gate" in r:
            gt = r["gate"]
            L.append(f"Gate (đăng ký trước): ≥ Bami−0,5 trên ≥4/5: **{gt['ge_bami_minus_0.5_on_4_of_5']}**; Restaurant ≥ Bami−0,5: **{gt['restaurant_ge_bami_minus_0.5']}**; Topic hoặc ViSFD ≥ +0,5: **{gt['topic_or_visfd_plus_0.5']}** → **{'PASS' if gt['pass'] else 'FAIL'}**. Task sát biên (|Δ| ≤ 0,5): {', '.join(TASK_LABEL[t] for t in gt['borderline_tasks']) or 'không'}.")
            L.append("")
        else:
            L.append(f"Lưới chưa đủ ({sum(1 for t in common.TASKS for lr in common.LR_GRID if r['grid'][t].get(f'{lr:.0e}'))}/15 ô).")
            L.append("")
    L.append("Ghi chú: Bami E14 chọn LR theo cùng grid; CafeBERT chỉ chạy LR 2e-5 (Restaurant reuse từ CDBK). Một seed; |Δ| < 1 điểm nằm trong nhiễu run-to-run đã quan sát. Mọi run có reload/prediction/ID checks trong `metadata.json`.")
    (common.ROOT / "docs" / "BASE_ENCODER_BASELINE_RESULT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    for model, r in out["models"].items():
        print(model, "complete" if r["complete"] else "partial", {TASK_LABEL[t]: (round(100 * s["f1"], 2), f"{s['lr']:.0e}") for t, s in r["selected"].items() if s}, r.get("gate", {}).get("pass"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
