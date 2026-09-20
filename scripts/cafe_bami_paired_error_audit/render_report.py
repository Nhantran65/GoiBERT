#!/usr/bin/env python3
"""Render docs/CAFEBERT_BAMIBERT_PAIRED_ERROR_AUDIT_PACKET.md mechanically from manifest.json / aggregate.json (no interpretation)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "cafe_bami_paired_error_audit"
m = json.loads((ART / "manifest.json").read_text(encoding="utf-8"))
a = json.loads((ART / "aggregate.json").read_text(encoding="utf-8"))
L: list[str] = []
P = L.append
pct = lambda x: f"{100 * x:.2f}"  # noqa: E731

P("# Audit lỗi ghép cặp CafeBERT–BamiBERT trên validation — packet bàn giao (cơ học)")
P("")
P(f"Plan `docs/CAFEBERT_BAMIBERT_PAIRED_ERROR_AUDIT_PLAN.md`; tạo {m['created_at']}; seed {m['seed']}; **read-only** (không model/GPU/test, không sửa nhãn). Source `scripts/cafe_bami_paired_error_audit/run_audit.py` (sha256 `{m['audit_source_sha256'][:16]}…`); artifacts `artifacts/cafe_bami_paired_error_audit/{{manifest.json, aggregate.json, events.jsonl, packet.jsonl, packet.md}}`. Báo cáo này chỉ trình bày số liệu và phép kiểm; **diễn giải nguyên nhân và chọn giả thuyết là việc của root**.")
P("")
P("Mã nhãn: Topic `0 lecturer / 1 training_program / 2 facility / 3 others`; ABSA `0 absent / 1 positive / 2 negative / 3 neutral` (theo `docs/E12_LABEL_MAPPING_AUDIT.md`, không dùng tên trong `data.py`); ViSFD `OTHERS` slot 5 detection-only (gold raw 4 ↔ model 1), loại khỏi pair F1 và packet polarity.")
P("")
P("## 1. Nguồn và gate correctness")
P("")
P("| Task | Model | Prediction file | sha256 | best epoch | F1 evaluator | F1 đếm độc lập | Khớp artifact ≤1e-12 |")
P("|---|---|---|---|---:|---:|---:|:--:|")
for t, v in m["tasks"].items():
    for mo in ("bami", "cafe"):
        p = v["predictions"][mo]; f = v["gate"]["f1"][mo]
        P(f"| {t} | {'BamiBERT E14' if mo == 'bami' else 'CafeBERT'} | `{p['path']}` | `{p['sha256'][:12]}…` | {p['best_epoch_metadata']} | {f['evaluator']:.10f} | {f['independent']:.10f} | {'✓' if f['match_evaluator_vs_artifact'] and f['match_evaluator_vs_independent'] else '✗'} |")
P("")
P("Gate: mỗi task 771/1.583/1.112 hàng ở loader và cả hai file; ID duy nhất, cùng tập và **cùng thứ tự** với loader; gold giống hệt giữa loader và hai file; shape 12/11 slot, mã trong tập cho phép; ViSFD OTHERS gold ∈ {0,4}, prediction ∈ {0,1}. Sáu F1 tính lại bằng `goibert.metrics.evaluate_task` và bằng phép đếm TP/FP/FN độc lập trong script: khớp nhau và khớp `metadata.json`/`best_validation_metrics.json` của từng run ≤1e-12; khớp mốc 4 chữ số trong plan ≤1e-6. `gate_all_ok = " + str(m["gate_all_ok"]) + "`. Evaluator sha256 `" + m["evaluator_sha256"][:12] + "…`, loader `" + m["loader_sha256"][:12] + "…`. Không đọc `test_predictions.json`.")
P("")

# ---------------- Topic
t = a["vsfc_topic"]
P("## 2. VSFC Topic — toàn validation (1.583 câu)")
P("")
P("| Lớp | Support | Bami P / R / F1 | Cafe P / R / F1 | Bami TP/FP/FN | Cafe TP/FP/FN |")
P("|---|---:|---|---|---|---|")
for c in ("lecturer", "training_program", "facility", "others"):
    b = t["bami"]["per_class"][c]; k = t["cafe"]["per_class"][c]
    P(f"| {c} | {b['support']} | {pct(b['precision'])} / {pct(b['recall'])} / {pct(b['f1'])} | {pct(k['precision'])} / {pct(k['recall'])} / {pct(k['f1'])} | {b['tp']}/{b['fp']}/{b['fn']} | {k['tp']}/{k['fp']}/{k['fn']} |")
P(f"| **macro F1 / accuracy** | | **{pct(t['bami']['macro_f1'])} / {pct(t['bami']['accuracy'])}** | **{pct(t['cafe']['macro_f1'])} / {pct(t['cafe']['accuracy'])}** | | |")
P("")
names = ["lecturer", "training_program", "facility", "others"]
for mo in ("bami", "cafe"):
    P(f"Confusion {mo} (hàng = gold, cột = dự đoán, thứ tự {', '.join(names)}):")
    P("")
    P("| gold \\ pred | " + " | ".join(names) + " |"); P("|---|" + "---:|" * 4)
    for i, row in enumerate(t[mo]["confusion_rows_gold_cols_pred"]):
        P(f"| {names[i]} | " + " | ".join(str(x) for x in row) + " |")
    P("")
P("Paired outcomes theo gold class (both_correct / cafe_only / bami_only / both_wrong_same / both_wrong_different):")
P("")
P("| Gold class | both_correct | cafe_only | bami_only | both_wrong_same | both_wrong_different |"); P("|---|---:|---:|---:|---:|---:|")
for c in names:
    v = t["paired_by_gold_class"][c]
    P(f"| {c} | {v.get('both_correct', 0)} | {v.get('cafe_only', 0)} | {v.get('bami_only', 0)} | {v.get('both_wrong_same', 0)} | {v.get('both_wrong_different', 0)} |")
o = t["paired_overall"]
P(f"| **tổng** | {o['both_correct']} | {o['cafe_only']} | {o['bami_only']} | {o['both_wrong_same']} | {o['both_wrong_different']} |")
P("")
P("Cross-tab (bami_pred | cafe_pred) theo gold class, các ô ≥5:")
P("")
for c in names:
    items = [(k, v) for k, v in t["crosstab_by_gold_class"][c].items() if v >= 5]
    P(f"- **{c}** (n={t['support'][c]}): " + "; ".join(f"`{k}` {v}" for k, v in items))
P("")

# ---------------- ABSA
for task, title in (("absa_restaurant", "UIT-ABSA Restaurant (771 review, 12 aspect)"), ("visfd", "ViSFD (1.112 comment, 10 slot polarity + OTHERS)")):
    r = a[task]
    P(f"## {'3' if task == 'absa_restaurant' else '4'}. {title} — toàn validation")
    P("")
    P(f"Gold-present pair-scored slots: **{r['gold_present_slots_total']}**; phân bố số aspect gold-present/review: {json.dumps(r['gold_present_per_row'])}.")
    P("")
    P("| Model | TP | FP | FN | Pair micro F1 | correct | miss | wrong_polarity | spurious |"); P("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for mo in ("bami", "cafe"):
        pm = r[mo]["pair_micro"]; ev = r[mo]["events_total"]
        P(f"| {'Bami' if mo == 'bami' else 'Cafe'} | {pm['tp']} | {pm['fp']} | {pm['fn']} | {pct(pm['f1'])} | {ev['correct']} | {ev['miss']} | {ev['wrong_polarity']} | {ev['spurious']} |")
    P("")
    P("Theo gold polarity (gold-present slots): support · correct / miss / wrong_polarity — Bami vs Cafe; spurious theo polarity dự đoán:")
    P("")
    P("| Gold polarity | Support | Bami correct/miss/wrong | Cafe correct/miss/wrong | Bami spurious→pol | Cafe spurious→pol |"); P("|---|---:|---|---|---:|---:|")
    for pol in ("positive", "negative", "neutral"):
        b = r["bami"]["by_gold_polarity"][pol]; k = r["cafe"]["by_gold_polarity"][pol]
        P(f"| {pol} | {b['support']} | {b['correct']}/{b['miss']}/{b['wrong_polarity']} | {k['correct']}/{k['miss']}/{k['wrong_polarity']} | {r['bami']['spurious_by_predicted_polarity'][pol]} | {r['cafe']['spurious_by_predicted_polarity'][pol]} |")
    P("")
    P("Per-aspect (support = gold-present; F1 = pair F1 của aspect; events của từng model). Support thấp (<30) chỉ để đếm, không xếp hạng:")
    P("")
    P("| Aspect | Support | Bami F1 | Bami miss/wrong/spur | Cafe F1 | Cafe miss/wrong/spur | Δ F1 (Cafe−Bami, pp) |"); P("|---|---:|---:|---|---:|---|---:|")
    for an in r["aspect_names"]:
        if an not in r["bami"]["per_aspect"]:
            continue
        b = r["bami"]["per_aspect"][an]; k = r["cafe"]["per_aspect"][an]
        P(f"| {an} | {b['support_gold_present']} | {pct(b['f1'])} | {b['miss']}/{b['wrong_polarity']}/{b['spurious']} | {pct(k['f1'])} | {k['miss']}/{k['wrong_polarity']}/{k['spurious']} | {100 * (k['f1'] - b['f1']):+.2f} |")
    P("")
    pp = r["paired_gold_present_overall"]; pa = r["paired_gold_absent"]
    P(f"Paired outcomes trên gold-present slots: both_correct {pp.get('both_correct', 0)}, cafe_only {pp.get('cafe_only', 0)}, bami_only {pp.get('bami_only', 0)}, both_wrong_same {pp.get('both_wrong_same', 0)}, both_wrong_different {pp.get('both_wrong_different', 0)}. Trên gold-absent slots: both_correct_absent {pa.get('both_correct_absent', 0)} (không dùng để nói model \"đúng hầu hết\"), cafe_only (Bami spurious) {pa.get('cafe_only', 0)}, bami_only (Cafe spurious) {pa.get('bami_only', 0)}, cả hai spurious cùng polarity {pa.get('both_wrong_same', 0)}, khác polarity {pa.get('both_wrong_different', 0)}.")
    P("")
    if task == "visfd":
        for mo in ("bami", "cafe"):
            od = r[mo]["others_detection_only"]
            P(f"- OTHERS detection-only, {mo}: gold-present {od['gold_present']}, detected {od['detected']}, missed {od['missed']}, spurious {od['spurious']}.")
        P("")
    P("Paired outcomes theo aspect × gold polarity (cafe_only / bami_only / both_correct / both_wrong_same / both_wrong_different):")
    P("")
    P("| Aspect | polarity | cafe_only | bami_only | both_correct | bw_same | bw_diff |"); P("|---|---|---:|---:|---:|---:|---:|")
    for key, v in r["paired_gold_present_by_aspect_polarity"].items():
        an, pol = key.split("|")
        P(f"| {an} | {pol} | {v.get('cafe_only', 0)} | {v.get('bami_only', 0)} | {v.get('both_correct', 0)} | {v.get('both_wrong_same', 0)} | {v.get('both_wrong_different', 0)} |")
    P("")

# ---------------- packet
P("## 5. Packet 72 trường hợp (`packet.jsonl`, `packet.md`)")
P("")
P(f"Quy tắc chọn: {m['packet']['selection_rule']}.")
P("")
P("| Task | cafe_only | bami_only | both_wrong | both_correct | ID duy nhất |"); P("|---|---:|---:|---:|---:|---:|")
for t_, v in m["packet"]["by_task_bucket"].items():
    P(f"| {t_} | {v['cafe_only']} | {v['bami_only']} | {v['both_wrong']} | {v['both_correct']} | {m['packet']['unique_ids_per_task'][t_]} |")
P("")
P("Ứng viên và shortfall theo bucket:")
P("")
for t_, sf in m["packet"]["shortfalls"].items():
    for b, v in sf.items():
        extra = ""
        if "by_event" in v:
            extra = f"; event lấy {v['by_event']}; ứng viên theo event {v['candidates_by_event']}"
        if "gold_classes_covered" in v:
            extra = f"; gold classes phủ {v['gold_classes_covered']}; ứng viên theo class {v['candidates_by_class']}"
        if "candidates_gold_present" in v and "candidates_gold_absent" in v:
            extra = f"; ứng viên gold-present {v['candidates_gold_present']}, gold-absent {v['candidates_gold_absent']}"
        elif "candidates_gold_present" in v:
            extra = f"; ứng viên gold-present {v['candidates_gold_present']}"
        P(f"- {t_}/{b}: quota {v['quota']}, lấy {v['got']}" + (f", ứng viên {v['candidates']}" if "candidates" in v else "") + extra + ".")
P("")
P("Shortfall về phủ lớp: Topic `cafe_only` không có ứng viên gold `facility` (0 câu Cafe đúng–Bami sai ở lớp này), nên cột này chỉ phủ 3/4 lớp; các cột Topic khác phủ đủ 4 lớp. Mọi quota số lượng đều đủ. Mỗi hàng packet có text nguyên văn, gold/pred hai model (số + tên), event target, vector đầy đủ (ABSA), số aspect gold-present, path/sha256 nguồn, khóa chọn, và bốn cột trống `gold_supported / evidence_span / possible_failure / ambiguity_note` cho root.")
P("")
P("## 6. Giới hạn diễn giải")
P("")
P("- Packet chọn **có điều kiện** theo đúng/sai và theo quota event; không dùng để ước lượng tỷ lệ nguyên nhân lỗi trong dân số. Bảng toàn validation ở §2–§4 mới là số mô tả đầy đủ.")
P("- Một seed, một checkpoint mỗi model (best validation epoch), validation đã dùng để chọn LR/epoch; disagreement cafe_only/bami_only gồm cả nhiễu run-to-run (~1 điểm F1 quan sát trong project). Không có test.")
P("- Đơn vị ABSA là (review, slot); `both_correct` chỉ nói về slot target; các ô support thấp (DRINKS#PRICES 19, STORAGE 14, FOOD#PRICES 45…) chỉ để đếm.")
P("- Macro F1 Topic và micro F1 ABSA không cộng trung bình. Tên polarity ABSA theo E12 (`1 positive / 2 negative / 3 neutral`); gold có thể sai — cột `gold_supported` để root đánh giá, subagent không relabel.")
P("- Không đề xuất kiến trúc hay pilot từ packet này.")
(ROOT / "docs" / "CAFEBERT_BAMIBERT_PAIRED_ERROR_AUDIT_PACKET.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("rendered", len(L), "lines")
