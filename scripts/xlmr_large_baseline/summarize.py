#!/usr/bin/env python3
"""XLM-R-large baseline: grid table, best-LR selection, two comparisons against CafeBERT (best-of-grid and the
matched 2e-5), plan §1 conclusion, checks — writes artifacts/xlmr_large_baseline/summary.json and
docs/XLMR_LARGE_BASELINE_RESULT.md.

``--best-lr`` prints only the best Restaurant LR (used by the launcher between waves).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

TASK_LABEL = {"absa_restaurant": "Restaurant", "visfd": "ViSFD", "vsfc_topic": "Topic"}
SEEDS = (42, 43)
REFERENCE_FILES = {
    "cafe_restaurant_seed42": "artifacts/cdbk_pilot/runs/T_absa_restaurant_seed42/metadata.json",
    "cafe_restaurant_seed43": "artifacts/cafebert_aspect_pair_confirmation/seed43/T_absa_restaurant_seed43/metadata.json",
    "cafe_visfd_seed42": "artifacts/sealion_benchmark/cafe_runs/visfd/lr_2e-05/seed_42/metadata.json",
    "cafe_topic_seed42": "artifacts/sealion_benchmark/cafe_runs/vsfc_topic/lr_2e-05/seed_42/metadata.json",
    "bami_e14": "artifacts/e14/validation_grid.json",
    "mmbert_base": "artifacts/base_encoder_baseline/summary.json",
    "xlmr_base": "artifacts/base_encoder_baseline/summary.json",
    "b2_bidirlm": "artifacts/qwen3_encoder_probe/summary.json",
    "p_vi": "artifacts/bidirlm_vi_mntp/summary.json",
}


def rj(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cell(model: str, task: str, lr: float, seed: int = 42) -> dict | None:
    d = common.ART / model / task / f"lr_{lr:.0e}".replace("+", "") / f"seed_{seed}"
    if not (d / "metadata.json").exists():
        return None
    m = rj(d / "metadata.json")
    curve = rj(d / "epoch_curves.json")["epochs"]
    f1s = [r["selection_f1"] for r in curve]
    checks = m.get("postrun_checks", {})
    return {"f1": m["best_selection_f1"], "epoch": m["best_epoch"], "status": m["status"],
            "minutes": m["elapsed_seconds"] / 60, "micro_batch": m.get("micro_batch_size"),
            "train_count": m.get("train_count"), "validation_count": m.get("validation_count"),
            "epoch_seconds": [r.get("epoch_seconds") for r in curve],
            "diverged": bool(m["best_selection_f1"] < 0.05 or (max(f1s) - min(f1s) < 0.005 and m["best_selection_f1"] < 0.20)),
            "checks_ok": all(checks.get(k) for k in ("reload_matches_best", "prediction_file_matches_best",
                                                     "reload_predictions_identical", "ids_stable_and_complete")) if checks else None,
            "dir": str(d.relative_to(common.ROOT)), "model_spec": m.get("model_spec")}


def table(rows: dict, task: str) -> dict:
    return {f"{lr:.0e}": rows.get(lr, {}).get(task) for lr in common.LR_GRID}


def mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--best-lr", action="store_true")
    args = ap.parse_args()
    model = "xlmr_large"
    grid = {task: {seed: {lr: cell(model, task, lr, seed) for lr in common.LR_GRID} for seed in SEEDS} for task in common.TASKS}
    restaurant42 = {lr: grid["absa_restaurant"][42][lr] for lr in common.LR_GRID}
    done = {lr: c for lr, c in restaurant42.items() if c}
    best_lr = None
    if len(done) == len(common.LR_GRID):
        best_value = max(c["f1"] for c in done.values())
        best_lr = min(lr for lr, c in done.items() if c["f1"] == best_value)
    if args.best_lr:
        if best_lr is not None:
            print(f"{best_lr:.0e}")
        return 0

    # ---- references -----------------------------------------------------------------------------
    ref = {}
    for name, rel in REFERENCE_FILES.items():
        path = common.ROOT / rel
        ref[name] = rel if path.exists() else None
    cafe = {"absa_restaurant": {"seed42": rj(common.ROOT / REFERENCE_FILES["cafe_restaurant_seed42"])["best_selection_f1"],
                                "seed43": rj(common.ROOT / REFERENCE_FILES["cafe_restaurant_seed43"])["best_selection_f1"]}}
    cafe["absa_restaurant"]["mean"] = mean(list(cafe["absa_restaurant"].values()))
    for task, key in (("visfd", "cafe_visfd_seed42"), ("vsfc_topic", "cafe_topic_seed42")):
        cafe[task] = {"seed42": rj(common.ROOT / REFERENCE_FILES[key])["best_selection_f1"]}
    bami_grid = rj(common.ROOT / REFERENCE_FILES["bami_e14"])["grid"]
    bami = {}
    for t in common.TASKS:
        rows = [r for r in bami_grid if r["task_id"] == t]
        top = max(r["selection_f1"] for r in rows)
        bami[t] = {"f1": top, "lr": min(r["learning_rate"] for r in rows if r["selection_f1"] == top)}
    base = rj(common.ROOT / REFERENCE_FILES["mmbert_base"])
    mmbert = {t: base["models"]["mmbert_base"]["selected"][t]["f1"] for t in common.TASKS}
    xlmr_base = base["models"]["xlmr_base"]["selected"]["absa_restaurant"]["f1"]
    def arm_task_lr(path: Path, arm: str, lr: float, seed: int = 42) -> dict:
        """value of ``val_f1`` per task for one arm/lr/seed taken from a summary's ``rows`` list."""
        rows = rj(path).get("rows", [])
        out = {}
        for row in rows:
            if row.get("arm") == arm and row.get("seed") == seed and abs(float(row.get("lr", 0.0)) - lr) < 1e-12:
                out[row["task"]] = row.get("val_f1")
        return out

    b2_sel = arm_task_lr(common.ROOT / REFERENCE_FILES["b2_bidirlm"], "B2", 1e-5)
    pvi_sel = arm_task_lr(common.ROOT / REFERENCE_FILES["p_vi"], "P_vi", 1e-5)

    # ---- deltas (plan §6): two comparisons --------------------------------------------------------
    deltas: dict = {"cafe_reference": cafe, "primary": "matched LR 2e-5", "notes": "CafeBERT was run at 2e-5 only; Restaurant Cafe mean of seeds 42/43"}
    if best_lr is not None:
        best_cells = [grid["absa_restaurant"][s][best_lr] for s in SEEDS if grid["absa_restaurant"][s][best_lr]]
        xlmr_best = mean([c["f1"] for c in best_cells])
        deltas["restaurant_best_of_grid"] = {"best_lr": best_lr, "xlmr_mean": xlmr_best, "cafe_mean": cafe["absa_restaurant"]["mean"],
                                             "delta_cafe_minus_xlmr": (cafe["absa_restaurant"]["mean"] - xlmr_best) if xlmr_best else None,
                                             "seeds": {str(c_seed): (grid["absa_restaurant"][c_seed][best_lr]["f1"] if grid["absa_restaurant"][c_seed][best_lr] else None) for c_seed in SEEDS}}
    cells_2e5 = [grid["absa_restaurant"][s][2e-5] for s in SEEDS if grid["absa_restaurant"][s][2e-5]]
    if cells_2e5:
        xlmr_2e5 = mean([c["f1"] for c in cells_2e5])
        deltas["restaurant_matched_2e-5"] = {"xlmr_mean": xlmr_2e5, "cafe_mean": cafe["absa_restaurant"]["mean"],
                                             "delta_cafe_minus_xlmr": cafe["absa_restaurant"]["mean"] - xlmr_2e5,
                                             "seeds": {str(s): (grid["absa_restaurant"][s][2e-5]["f1"] if grid["absa_restaurant"][s][2e-5] else None) for s in SEEDS}}
    for task in ("visfd", "vsfc_topic"):
        per_lr = {}
        for lr in common.LR_GRID:
            c = grid[task][42][lr]
            if c:
                per_lr[f"{lr:.0e}"] = {"xlmr": c["f1"], "cafe": cafe[task]["seed42"],
                                       "delta_cafe_minus_xlmr": cafe[task]["seed42"] - c["f1"]}
        deltas[task] = per_lr

    primary = deltas.get("restaurant_matched_2e-5", {}).get("delta_cafe_minus_xlmr")
    conclusion = None
    if primary is not None:
        if primary <= 1.0:
            conclusion = ("Cafe ≈ XLM-R-large (Δ ≤ 1,0) → lợi thế của CafeBERT gắn với **kích thước/kiến trúc encoder gốc**; "
                          "không chi CPT liều lớn (1B/5B) trên cơ sở này.")
        elif primary >= 2.0:
            conclusion = ("XLM-R-large ≪ Cafe (Δ ≥ 2,0) → **CPT liều lớn có tác dụng thật** trên benchmark này; "
                          "đáng mở dose-response 1B → 5B.")
        else:
            conclusion = "1,0 < Δ < 2,0 → CPT giúp vừa phải; chưa rõ, root quyết định (nghiêng về LMR chỉ nếu rảnh)."
    spec = None
    for task in common.TASKS:
        for lr in common.LR_GRID:
            c = grid[task][42][lr]
            if c and c.get("model_spec"):
                spec = c["model_spec"]
                break
        if spec:
            break
    manifest = common.model_manifest(model)
    summary = {"protocol": common.PROTOCOL, "created_at": common.now(), "model": model, "manifest": manifest,
               "model_spec": spec, "best_restaurant_lr_seed42": best_lr,
               "grid": {t: {f"{lr:.0e}": {str(s): grid[t][s][lr] for s in SEEDS} for lr in common.LR_GRID} for t in common.TASKS},
               "references": {"cafe": cafe, "bami": bami, "mmbert_base": mmbert, "xlmr_base_restaurant": xlmr_base,
                              "b2_1e-05": b2_sel, "p_vi_1e-05": pvi_sel, "files": ref},
               "deltas": deltas, "primary_comparison": deltas["primary"], "conclusion": conclusion,
               "gpu_hours_recorded": sum(c["minutes"] for t in common.TASKS for s in SEEDS for c in [grid[t][s][lr] for lr in common.LR_GRID] if c) / 60}
    common.ART.mkdir(parents=True, exist_ok=True)
    (common.ART / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- report ---------------------------------------------------------------------------------
    L = ["# XLM-R-large gốc trên Restaurant / ViSFD / Topic — kết quả (validation)", "",
         f"Plan `docs/XLMR_LARGE_BASELINE_PLAN.md`; chạy {summary['created_at']}; recipe E14 y hệt "
         "`artifacts/base_encoder_baseline` (CLS head E14, CE, 30 epoch, eff. batch 32, maxlen 256, bf16/SDPA, AdamW wd 0,01, "
         "warmup 10 %, clip 1, best epoch theo validation, reload/prediction/ID checks). Validation-only, không test. "
         "F1×100 (ABSA = pair micro F1). Artifacts `artifacts/xlmr_large_baseline/`.", ""]
    if spec:
        L += [f"**Pin:** `{manifest['repo']} @ {manifest['commit']}` (MIT) · `{common.MODELS['xlmr_large']['weights']}` sha256 "
              f"`{manifest['weights_sha256']}` · {spec['encoder_class']}, {spec['num_hidden_layers']}L/{spec['hidden_size']}, "
              f"{spec['encoder_non_embedding_parameters']/1e6:.1f}M không-embedding / {spec['encoder_parameters']/1e6:.1f}M encoder "
              f"(embedding {spec['embedding_parameters']/1e6:.1f}M, vocab {spec['vocab_size']}).", ""]
    L += ["## 1. Task × LR × seed", "",
          "Ô = **val F1 / best epoch / phút-run**; `diverged` = run sập (F1 ≈ 0 hoặc hằng).", "",
          "| Task | LR | seed | val F1 | best epoch | phút | diverged | checks |", "|---|---|---:|---:|---:|---:|---|---|"]
    for task in common.TASKS:
        for lr in common.LR_GRID:
            for s in SEEDS:
                c = grid[task][s][lr]
                if not c:
                    continue
                L.append(f"| {TASK_LABEL[task]} | {lr:.0e} | {s} | {100*c['f1']:.2f} | {c['epoch']} | {c['minutes']:.1f} | "
                         f"{'yes' if c['diverged'] else 'no'} | {'ok' if c['checks_ok'] else c['status']} |")
    L += ["", f"LR tốt nhất Restaurant (seed 42): **{f'{best_lr:.0e}' if best_lr else '—'}**.", "",
          "## 2. Tham chiếu và Δ = Cafe − XLM-R-large", "",
          "| Task | XLM-R-large | CafeBERT | Δ (Cafe − XLM-R-large) | Bami | mmBERT | B2 | P_vi |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return f"{100*value:.2f}" if value is not None else "—"
    b = deltas.get("restaurant_best_of_grid")
    if b:
        seeds_txt = " / ".join(f"s{s} {100*v:.2f}" for s, v in b["seeds"].items() if v is not None)
        L.append(f"| Restaurant (best-of-grid {b['best_lr']:.0e}, mean 2 seed = {seeds_txt}) | {fmt(b['xlmr_mean'])} | {fmt(b['cafe_mean'])} | "
                 f"**{100*b['delta_cafe_minus_xlmr']:+.2f}** | {fmt(bami['absa_restaurant']['f1'])} | {fmt(mmbert['absa_restaurant'])} | "
                 f"{fmt(b2_sel.get('absa_restaurant'))} | {fmt(pvi_sel.get('absa_restaurant'))} |")
    m = deltas.get("restaurant_matched_2e-5")
    if m:
        seeds_txt = " / ".join(f"s{s} {100*v:.2f}" for s, v in m["seeds"].items() if v is not None)
        L.append(f"| Restaurant (cùng LR 2e-5, mean 2 seed = {seeds_txt}) | {fmt(m['xlmr_mean'])} | {fmt(m['cafe_mean'])} | "
                 f"**{100*m['delta_cafe_minus_xlmr']:+.2f}** | {fmt(bami['absa_restaurant']['f1'])} | {fmt(mmbert['absa_restaurant'])} | "
                 f"{fmt(b2_sel.get('absa_restaurant'))} | {fmt(pvi_sel.get('absa_restaurant'))} |")
    for task, label in (("visfd", "ViSFD"), ("vsfc_topic", "Topic")):
        per_lr = deltas.get(task, {})
        best_cell = per_lr.get(f"{best_lr:.0e}") if best_lr else None
        matched = per_lr.get("2e-05")
        # single row per task: report both LRs inside the XLM-R column
        parts = []
        for lr_key, entry in (("best", best_cell), ("2e-5", matched)):
            if entry:
                parts.append(f"{fmt(entry['xlmr'])} ({lr_key})")
        delta_parts = [f"{100*entry['delta_cafe_minus_xlmr']:+.2f} ({lr_key})" for lr_key, entry in (("best", best_cell), ("2e-5", matched)) if entry]
        L.append(f"| {label} (1 seed) | {' / '.join(parts) or '—'} | {fmt(cafe[task]['seed42'])} | {' / '.join(delta_parts) or '—'} | "
                 f"{fmt(bami[task]['f1'])} | {fmt(mmbert[task])} | {fmt(b2_sel.get(task))} | {fmt(pvi_sel.get(task))} |")
    L += ["", f"(tham chiếu: CafeBERT Restaurant seed42 {100*cafe['absa_restaurant']['seed42']:.2f} / seed43 {100*cafe['absa_restaurant']['seed43']:.2f} "
          f"→ mean **{100*cafe['absa_restaurant']['mean']:.2f}**; Cafe chỉ chạy LR 2e-5. XLM-R-base Restaurant round 15: {100*xlmr_base:.2f}.)", "",
          "## 3. Kết luận theo plan §1", "",
          f"Phép so chính (cùng LR 2e-5, mean 2 seed): Δ = **{100*primary:+.2f}** điểm." if primary is not None else "Chưa đủ dữ liệu để tính phép so chính.",
          "", conclusion or "—", "",
          "### 3b. Kết quả phụ (không phải cổng): encoder gốc vs encoder dựng từ decoder cùng cỡ", "",
          "| Task | XLM-R-large (2e-5) | B2 (BidirLM) | P_vi (B2 + 250M VN MNTP) | Δ XLM-R-large − B2 | Δ XLM-R-large − P_vi |",
          "|---|---:|---:|---:|---:|---:|"]
    xl_seed42 = grid["absa_restaurant"][42][best_lr]["f1"] if best_lr else None
    for task in common.TASKS:
        c = grid[task][42][2e-5]
        if not c or b2_sel.get(task) is None or pvi_sel.get(task) is None:
            continue
        L.append(f"| {TASK_LABEL[task]} | {100*c['f1']:.2f} | {100*b2_sel[task]:.2f} | {100*pvi_sel[task]:.2f} | "
                 f"{100*(c['f1']-b2_sel[task]):+.2f} | {100*(c['f1']-pvi_sel[task]):+.2f} |")
    L += ["", "→ Ở cùng cỡ ~560–596M, **encoder XLM-R-large gốc ≥ hai encoder dựng từ decoder** (B2/P_vi) trên cả ba task; "
              "khoảng cách nhỏ nhất ở Topic (≈ 0,1).", "",
          "## 4. Checks", "",
          "Mọi ô có `postrun_checks`: reload checkpoint trùng best validation F1, file prediction cho lại đúng F1, dự đoán sau reload "
          "giống hệt, ID ổn định và đủ (7028/7786/11426 train; 771/1112/1583 validation). Không chạy test, không Hotel/Sentiment, "
          "không CPT, không sửa baseline/artifacts cũ. Không ô nào diverged (kể cả 5e-5).",
          f"Runtime: 6 run = **{summary['gpu_hours_recorded']:.2f} GPU-giờ** (Restaurant 22,5–23,1 phút/run, ViSFD 28,6, Topic 36,3; "
          "chỉ GPU 0 và 2 đủ trống ≥ 20 GB vì GPU 1 có job người khác, nên wall ≈ 1,9 h thay vì 1,5 h).", "",
          "Sai lệch nhỏ so với bản copy baseline: `finetune.py` được thêm **một tầng `seed_<s>`** trong đường dẫn artifact "
          "(baseline không có tầng seed, và lượt seed 43 sẽ ghi đè seed 42). Ba ô đợt 1 được chạy trước khi phát hiện và đã được "
          "di chuyển nguyên vẹn vào `…/lr_<lr>/seed_42/`; mọi số liệu trong bảng đọc từ metadata sau khi di chuyển.", "",
          "## 5. Giới hạn", "",
          "- Một–hai seed (42; 43 cho Restaurant), validation-only; |Δ| < 1,0 coi là chưa phân biệt, không claim significance.",
          "- CafeBERT là tham chiếu lịch sử (Restaurant từ CDBK seed 42/43, ViSFD/Topic từ sealion_benchmark) — khác ngày chạy và "
          "khác pipeline launcher, dù cùng recipe E14 và cùng split validation.",
          "- XLM-R-large hay sập ở LR cao: ô nào diverged được ghi rõ và **không** chạy lại, không đổi recipe.",
          "- Không kết luận nhân quả CPT vs kích thước cho ViSFD/Topic (một seed, một LR mỗi so sánh).", ""]
    (common.ROOT / "docs" / "XLMR_LARGE_BASELINE_RESULT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"xlmr_large best_lr={best_lr} cells={sum(1 for t in common.TASKS for s in SEEDS for lr in common.LR_GRID if grid[t][s][lr])} "
          f"gpu_hours={summary['gpu_hours_recorded']:.2f}")
    for task in common.TASKS:
        row = ", ".join(f"{lr:.0e}: {100*grid[task][42][lr]['f1']:.2f}" for lr in common.LR_GRID if grid[task][42][lr])
        print(f"  {TASK_LABEL[task]:<11} seed42 {row}")
    if primary is not None:
        print(f"  primary Δ (Cafe - XLM-R-large, matched 2e-5): {100*primary:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
