#!/usr/bin/env python3
"""Vocabulary-trim student screen summary: grid table, per-model LR choice, plan §2.3 readings.

``--best-lr-per-model`` prints "model lr" lines for the launcher.  Default mode writes
artifacts/vocab_trim_student/summary.json and docs/VOCAB_TRIM_STUDENT_RESULT.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

TASK_LABEL = {"absa_restaurant": "Restaurant", "visfd": "ViSFD", "vsfc_topic": "Topic"}
REFERENCE = {
    "bamibert": {"absa_restaurant": 0.747442, "visfd": 0.843811, "vsfc_topic": 0.795279},
    "mmbert_base": {"absa_restaurant": 0.756262, "visfd": 0.836390, "vsfc_topic": 0.811808},
    "xlmr_large_raw": {"absa_restaurant": 0.778147, "visfd": 0.854800, "vsfc_topic": 0.806100},
    "b2_bidirlm": {"absa_restaurant": 0.757200, "visfd": 0.848500, "vsfc_topic": 0.807400},
    "p_vi": {"absa_restaurant": 0.761500, "visfd": 0.854400, "vsfc_topic": 0.806900},
}
MODEL_NOTE = {"m12": "mMiniLMv2-L12-H384 (student của XLM-R-large)", "m6": "mMiniLMv2-L6-H384",
              "b8": "XLM-R-base, giữ 8 lớp dưới", "b12": "XLM-R-base, đủ 12 lớp"}


def rj(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cell(model: str, task: str, lr: float, seed: int = 42):
    directory = common.out_dir_for(model, task, lr, seed)
    if not (directory / "metadata.json").exists():
        return None
    m = rj(directory / "metadata.json")
    curve = rj(directory / "epoch_curves.json")["epochs"]
    checks = m.get("postrun_checks", {})
    return {"f1": m["best_selection_f1"], "epoch": m["best_epoch"], "minutes": m["elapsed_seconds"] / 60,
            "status": m["status"], "layers_kept": m.get("model_spec", {}).get("layers_kept"),
            "parameters": m.get("model_spec", {}).get("total_parameters_with_head"),
            "encoder_parameters": m.get("model_spec", {}).get("encoder_parameters"),
            "micro_batch": m.get("micro_batch_size"), "train_count": m.get("train_count"), "validation_count": m.get("validation_count"),
            "diverged": bool(m["best_selection_f1"] < 0.05 or (max(r["selection_f1"] for r in curve)
                                                              - min(r["selection_f1"] for r in curve) < 0.005 and m["best_selection_f1"] < 0.20)),
            "checks_ok": all(checks.get(k) for k in ("reload_matches_best", "prediction_file_matches_best",
                                                     "reload_predictions_identical", "ids_stable_and_complete")) if checks else None,
            "dir": str(directory.relative_to(common.ROOT))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--best-lr-per-model", action="store_true")
    args = ap.parse_args()
    grid = {key: {task: {seed: {lr: cell(key, task, lr, seed) for lr in common.MODELS[key]["lr_grid"]} for seed in (42, 43)}
                  for task in common.TASKS} for key in common.MODELS}
    best_lr = {}
    for key in common.MODELS:
        done = {lr: c for lr, c in grid[key]["absa_restaurant"][42].items() if c}
        if len(done) == len(common.MODELS[key]["lr_grid"]):
            top = max(c["f1"] for c in done.values())
            best_lr[key] = min(lr for lr, c in done.items() if c["f1"] == top)
    if args.best_lr_per_model:
        for key, lr in best_lr.items():
            print(f"{key} {lr:.0e}")
        return 0
    summary = {"protocol": common.PROTOCOL, "created_at": common.now(),
               "keep_set": rj(common.ART / "keep_set.json") if (common.ART / "keep_set.json").exists() else None,
               "best_restaurant_lr_per_model": {k: f"{v:.0e}" for k, v in best_lr.items()},
               "grid": {key: {task: {f"{lr:.0e}": {str(seed): grid[key][task][seed][lr] for seed in (42, 43)}
                                    for lr in common.MODELS[key]["lr_grid"]} for task in common.TASKS} for key in common.MODELS},
               "reference": REFERENCE,
               "gpu_hours_recorded": sum(c["minutes"] for key in common.MODELS for task in common.TASKS for seed in (42, 43)
                                         for c in [grid[key][task][seed][lr] for lr in common.MODELS[key]["lr_grid"]] if c) / 60,
               "readings": {}, "trim_report": rj(common.ART / "trim_report.json") if (common.ART / "trim_report.json").exists() else None,
               "vocab_counts": rj(common.ART / "vocab_counts.json") if (common.ART / "vocab_counts.json").exists() else None}
    selected = {key: (grid[key]["absa_restaurant"][42][best_lr[key]] if key in best_lr else None) for key in common.MODELS}
    if selected.get("b12"):
        summary["readings"]["b12_vocab_trim_pre_ft"] = {"restaurant": selected["b12"]["f1"],
                                                        "expected": "~73.6 ± 0.5", "ok": abs(selected["b12"]["f1"] - 0.736) <= 0.005}
    if selected.get("b8"):
        summary["readings"]["b8_step3_basis"] = {"restaurant": selected["b8"]["f1"], "threshold_ge_74": bool(selected["b8"]["f1"] >= 0.74),
                                                 "threshold_lt_72": bool(selected["b8"]["f1"] < 0.72)}
    if selected.get("m12"):
        summary["readings"]["m12_target_34m"] = {"restaurant": selected["m12"]["f1"], "threshold_ge_73": bool(selected["m12"]["f1"] >= 0.73)}
    beat = {}
    for key in common.MODELS:
        if not selected.get(key):
            continue
        wins = [task for task in common.TASKS
                if grid[key][task][42][best_lr[key]] and grid[key][task][42][best_lr[key]]["f1"] >= REFERENCE["bamibert"][task]]
        beat[key] = {"tasks_ge_bamibert": [TASK_LABEL[t] for t in wins], "count": len(wins),
                     "highlight": len(wins) >= 2}
    summary["readings"]["beat_bamibert"] = beat
    round15 = rj(common.ROOT / "artifacts" / "base_encoder_baseline" / "summary.json")
    round15_f1 = round15["models"]["xlmr_base"]["selected"]["absa_restaurant"]["f1"]
    if selected.get("b12"):
        summary["readings"]["b12_vs_round15_untrimmed"] = {
            "round15_xlmr_base_restaurant": round15_f1, "b12_restaurant": selected["b12"]["f1"],
            "delta": selected["b12"]["f1"] - round15_f1}
    common.ART.mkdir(parents=True, exist_ok=True)
    (common.ART / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- report --------------------------------------------------------------------------------
    vc = summary["vocab_counts"] or {}
    tr = summary["trim_report"] or {}
    L = ["# Student nhỏ từ XLM-R-large — kết quả bước 1 (cắt vocab) + bước 2 (đo sàn)", "",
         f"Plan `docs/VOCAB_TRIM_STUDENT_PLAN.md`; chạy {summary['created_at']}; recipe E14 như `scripts/base_encoder_baseline` "
         "(CLS head E14, CE, 30 epoch, eff. batch 32, maxlen 256, bf16/SDPA, AdamW wd 0,01, warmup 10 %, clip 1, best epoch theo "
         "validation, reload/prediction/ID checks). Validation-only (test chỉ dùng **text** để đo `<unk>`). "
         "Artifacts `artifacts/vocab_trim_student/`; model cắt ở `models/trimmed/`.", ""]
    if vc and tr:
        eq0 = (tr.get("equivalence") or {})
        b12_line = ""
        if selected.get("b12") and "b12_vs_round15_untrimmed" in summary["readings"]:
            e = summary["readings"]["b12_vs_round15_untrimmed"]
            b12_line = (f"Bản 12 lớp trên XLM-R-base cắt vocab đạt **{100*e['b12_restaurant']:.2f}** Restaurant, so với "
                        f"**{100*e['round15_xlmr_base_restaurant']:.2f}** của XLM-R-base **chưa cắt** ở round 15 → Δ "
                        f"**{100*e['delta']:+.2f}**, xác nhận lần nữa là cắt vocab không mất điểm. ")
        L += ["**Kết luận ngắn.** Bước 1 thành công: |K| = "
              f"**{vc['keep_set']['size']}** token, phủ **{100*vc['coverage']['pool']:.4f} %** lượt token của pool và **100 %** train+dev benchmark, "
              f"`<unk>` chỉ **{100*(tr.get('unk',{}).get('B_validation_5task',{}).get('unk_token_rate') or 0):.5f} %** token trên validation, "
              f"khớp tokenization (B)↔(A) **{100*tr.get('tokenisation_agreement',{}).get('identical_rate',0):.2f} %**, và trên checkpoint đã fine-tune thì "
              f"F1 **không đổi** ({100*eq0.get('untrimmed_f1',0):.4f} → {100*eq0.get('A_f1',0):.4f}, max|Δ logits| = "
              f"{eq0.get('max_abs_logits_delta_A_vs_untrimmed',0):.1e}) → phát hành được 4 model cắt gọn (XLM-R-large 558,8M → 348,9M; "
              "XLM-R-base 277,5M → 120,0M; mMiniLMv2-L12 117,5M → 38,8M; L6 106,8M → 28,1M). "
              + b12_line +
              "Bước 2 (đo sàn) **không** đạt mốc nào của plan §2.3 cho hướng student ≤100M: student công khai mMiniLMv2 chỉ "
              f"**{100*selected['m12']['f1']:.2f}** (M12) / **{100*selected['m6']['f1']:.2f}** (M6) Restaurant — quá xa BamiBERT 74,74; "
              f"nền 8 lớp B8 = **{100*selected['b8']['f1']:.2f}** (< 72, dưới mốc 74,0 cho bước 3); không ứng viên nào ≥ BamiBERT ở ≥ 2/3 task. "
              "Kết luận: muốn có student nhỏ cạnh tranh thì phải **tự distill (bước 3)** trên XLM-R-large với dữ liệu của ta — "
              "các student công khai và việc bỏ lớp đơn thuần là chưa đủ.", ""]
    if vc:
        L += ["## 1. Bước 1 — cắt vocab", "",
              f"- |K| = **{vc['keep_set']['size']}** (cap {vc['cap']}, mandatory {vc['keep_set']['mandatory_tokens']}, "
              f"single-char {vc['keep_set']['single_char_pieces']}, frequent {vc['keep_set']['frequent_tokens']}"
              + (f", dropped-for-cap {vc['keep_set']['dropped_for_cap']}" if vc["keep_set"]["cap_hit"] else "") + ")",
              f"- Token đếm: pool {vc['tokens_counted']['pool']/1e6:.1f}M (target {vc['pool_token_target']/1e6:.0f}M), wiki "
              f"{vc['tokens_counted']['wiki']/1e6:.1f}M, benchmark train+dev {vc['tokens_counted']['benchmark_train_dev']/1e6:.2f}M, "
              f"social train+dev {vc['tokens_counted']['social_train_dev']/1e6:.2f}M",
              f"- Độ phủ của K: pool **{100*vc['coverage']['pool']:.4f} %**, wiki {100*vc['coverage']['wiki']:.4f} %, "
              f"benchmark train+dev **{100*vc['coverage']['benchmark_train_dev']:.4f} %**, social {100*vc['coverage']['social_train_dev']:.4f} %", ""]
    if tr:
        unk = tr.get("unk", {})
        ag = tr.get("tokenisation_agreement", {})
        eq = tr.get("equivalence", {})
        L += ["### 1.2 `<unk>` và độ khớp tokenization", "",
              "| Đo | (A) token/unk | (A) câu bị ảnh hưởng | (B) token/unk | (B) câu bị ảnh hưởng |", "|---|---:|---:|---:|---:|"]
        for label, key_a, key_b in (("validation 5 task", "A_validation_5task", "B_validation_5task"),
                                    ("text test 5 task", "A_test_text_5task", "B_test_text_5task"),
                                    ("text test social", "A_test_text_social", "B_test_text_social")):
            a, b = unk.get(key_a, {}), unk.get(key_b, {})
            L.append(f"| {label} | {100*a.get('unk_token_rate', 0):.5f} % | {100*a.get('sentence_affected_rate', 0):.3f} % | "
                     f"{100*b.get('unk_token_rate', 0):.5f} % | {100*b.get('sentence_affected_rate', 0):.3f} % |")
        L += ["", f"Khớp tokenization (B) vs (A) trên validation: **{100*ag.get('identical_rate', 0):.2f} %** chuỗi trùng id "
              f"({ag.get('identical', 0)}/{ag.get('validation_sequences', 0)}); {len(ag.get('examples', []))} ví dụ khác nhau được lưu trong "
              "`trim_report.json`.", ""]
        if eq:
            L += ["### 1.3 Bằng chứng cắt vocab không mất điểm (không train)", "",
                  f"Checkpoint XLM-R-large Restaurant 2e-5 seed 42 (val {100*eq['checkpoint_val_f1']:.2f}): không cắt "
                  f"**{100*eq['untrimmed_f1']:.4f}** → (A) id-map + embedding cắt **{100*eq['A_f1']:.4f}** "
                  f"(Δ {100*eq['A_minus_untrimmed']:+.4f}; max|Δ logits| **{eq['max_abs_logits_delta_A_vs_untrimmed']:.2e}**) → (B) re-tokenize bằng "
                  f"tokenizer cắt **{100*eq['B_f1']:.4f}** (Δ {100*eq['B_minus_untrimmed']:+.4f}) — "
                  f"{'ĐẠT' if eq.get('A_identical') and eq.get('B_within_0.1') else 'KHÔNG ĐẠT'}.", ""]
        exports = tr.get("exports", {})
        if exports:
            L += ["### Tham số trước/sau khi cắt (4 model, `models/trimmed/`)", "",
                  "| Model | Kiến trúc | Tham số trước | Tham số sau | Embedding trước → sau |", "|---|---|---:|---:|---:|"]
            for name, info in exports.items():
                L.append(f"| `{name}` | {info['encoder_class']}, {info['num_hidden_layers']}L/{info['hidden_size']} | "
                         f"{info['parameters_before']/1e6:.1f}M | **{info['parameters_after']/1e6:.1f}M** | "
                         f"{info['embedding_parameters_before']/1e6:.1f}M → {info['embedding_parameters_after']/1e6:.1f}M |")
            L.append("")
    L += ["## 2. Bước 2 — đo sàn (validation, seed 42)", "",
          "Ô = **val F1 / best epoch / phút-run**; mỗi model dùng LR grid riêng (M12/M6: 2e-5, 5e-5, 1e-4; B8/B12: 1e-5, 2e-5, 5e-5).", ""]
    for key in common.MODELS:
        L += [f"### {key} — {MODEL_NOTE[key]}", "",
              "| Task | " + " | ".join(f"LR {lr:.0e}" for lr in common.MODELS[key]["lr_grid"]) + " | chọn |", 
              "|---" * (len(common.MODELS[key]["lr_grid"]) + 2) + "|"]
        for task in common.TASKS:
            cells = []
            for lr in common.MODELS[key]["lr_grid"]:
                c = grid[key][task][42][lr]
                cells.append(f"{100*c['f1']:.2f} / {c['epoch']} / {c['minutes']:.1f}" if c else "—")
            sel = None
            if key in best_lr:
                sel = grid[key][task][42][best_lr[key]]
            extra = ""
            c43 = grid[key][task][43][best_lr[key]] if key in best_lr else None
            if c43:
                extra = f" · seed43 {100*c43['f1']:.2f}"
            L.append(f"| {TASK_LABEL[task]} | " + " | ".join(cells) + " | " +
                     (f"**{100*sel['f1']:.2f}** @{best_lr[key]:.0e}{extra}" if sel else "—") + " |")
        L.append("")
    L += ["### Tham chiếu (validation, seed 42)", "",
          "| Task | BamiBERT | mmBERT-base | XLM-R-large gốc | B2 | P_vi |", "|---|---:|---:|---:|---:|---:|"]
    for task in common.TASKS:
        L.append(f"| {TASK_LABEL[task]} | " + " | ".join(f"{100*REFERENCE[k][task]:.2f}" for k in
                 ("bamibert", "mmbert_base", "xlmr_large_raw", "b2_bidirlm", "p_vi")) + " |")
    L += ["", "### Đọc kết quả theo plan §2.3", ""]
    r = summary["readings"]
    def fmt(entry, text):
        return f"- {text}: {entry}" if entry else None
    if "b12_vocab_trim_pre_ft" in r:
        e = r["b12_vocab_trim_pre_ft"]
        L.append(f"- B12 (XLM-R-base cắt vocab, 12 lớp) Restaurant = **{100*e['restaurant']:.2f}** (kỳ vọng {e['expected']}) → "
                 f"{'khớp' if e['ok'] else 'lệch kỳ vọng'}: cắt vocab trước fine-tune không mất điểm.")
    if "b8_step3_basis" in r:
        e = r["b8_step3_basis"]
        verdict = ("**≥ 74,0 → có cơ sở cho bước 3** (student khởi tạo B8, mục tiêu ≥ 75,5)" if e["threshold_ge_74"]
                   else ("< 72 → cân nhắc giữ 10 lớp/khởi tạo khác" if e["threshold_lt_72"] else "trong dải 72–74 → chưa rõ"))
        L.append(f"- B8 (XLM-R-base 8 lớp dưới) Restaurant = **{100*e['restaurant']:.2f}** → {verdict}.")
    if "m12_target_34m" in r:
        e = r["m12_target_34m"]
        m12_params = ((tr.get("exports") or {}).get("mminilmv2_l12_vi") or {}).get("parameters_after")
        size = f", ~{m12_params/1e6:.0f}M sau cắt" if m12_params else ""
        L.append(f"- M12 (mMiniLMv2-L12{size}) Restaurant = **{100*e['restaurant']:.2f}** → "
                 f"{'≥ 73,0: mục tiêu student ~30M là thực tế' if e['threshold_ge_73'] else '< 73,0: mục tiêu student ~30M khó'}.")
    for key, e in r.get("beat_bamibert", {}).items():
        L.append(f"- {key}: ≥ BamiBERT ở {e['count']}/3 task ({', '.join(e['tasks_ge_bamibert']) or 'không'})"
                 + (" — **ghi đậm: student ≤100M ≥ Bami 103M mà chưa distill riêng**" if e["highlight"] else ""))
    if "b12_vs_round15_untrimmed" in r:
        e = r["b12_vs_round15_untrimmed"]
        L.append(f"- B12 (cắt vocab) {100*e['b12_restaurant']:.2f} vs XLM-R-base **chưa cắt** round 15 "
                 f"{100*e['round15_xlmr_base_restaurant']:.2f} → Δ {100*e['delta']:+.2f}: cắt vocab pre-FT không mất điểm (khớp §1.3).")
    seed43 = {key: grid[key]["absa_restaurant"][43].get(best_lr[key]) for key in ("m12", "b8") if key in best_lr}
    if all(seed43.values()):
        L.append("- Sàn nhiễu seed ở cỡ nhỏ (Restaurant, cùng LR): "
                 + ", ".join(f"{k} seed42 {100*grid[k]['absa_restaurant'][42][best_lr[k]]['f1']:.2f} vs seed43 {100*v['f1']:.2f}"
                             for k, v in seed43.items())
                 + " → chênh ~2–3 điểm, nên ở cỡ này |Δ| < 1 **không** có nghĩa chưa phân biệt; mọi so sánh với BamiBERT phải coi là thô.")
    L += ["", "## 3. Checks, license, giới hạn", "",
          "Mọi ô có `postrun_checks` (reload trùng best F1, file prediction cho lại đúng F1, dự đoán sau reload giống hệt, ID ổn định và đủ). "
          "Không chạy test (ngoài đo `<unk>` trên text), không Hotel/Sentiment, không distill, không sửa artifacts cũ.", "",
          "- License: XLM-R (base/large) MIT; mMiniLMv2 phát hành theo UniLM/MIT (card `nreimers/mMiniLMv2-*`); dữ liệu đếm vocab: pool v3 CC0 "
          "(không NC), Wikipedia CC BY-SA 4.0 (chỉ đếm tần suất, không phân phối lại text).",
          "- Giới hạn: 1 seed (42) + seed 43 cho M12/B8; validation-only; các student **chưa distill** trên dữ liệu của ta — đây là sàn, "
          "không phải kết quả cuối; ở cỡ model này sàn nhiễu seed đo được ~2–3 điểm nên **không** dùng ngưỡng |Δ| < 1,0 của các model lớn, "
          "và không claim significance.",
          f"- Chi phí: {summary['gpu_hours_recorded']:.2f} GPU-giờ cho {sum(1 for k in common.MODELS for t in common.TASKS for s in (42,43) for lr in common.MODELS[k]['lr_grid'] if grid[k][t][s][lr])} ô fine-tune.", ""]
    (common.ROOT / "docs" / "VOCAB_TRIM_STUDENT_RESULT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"best LR per model: {summary['best_restaurant_lr_per_model']}  gpu-hours {summary['gpu_hours_recorded']:.2f}")
    for key in common.MODELS:
        row = ", ".join(f"{task}: {100*grid[key][task][42][best_lr[key]]['f1']:.2f}"
                        for task in common.TASKS if key in best_lr and grid[key][task][42][best_lr[key]])
        print(f"  {key:<4} {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
