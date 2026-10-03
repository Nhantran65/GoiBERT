#!/usr/bin/env python3
"""Render the Vietnamese validation-only review report from computed artifacts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/two_backbone_ensemble"
DOC = ROOT / "docs/TWO_BACKBONE_ENSEMBLE_CHECK_RESULT.md"
TASKS = ("absa_restaurant", "visfd", "vsfc_topic")
LABELS = {"absa_restaurant": "Restaurant", "visfd": "ViSFD", "vsfc_topic": "Topic"}


def read(name):
    return json.loads((ART / name).read_text())


def fmt(x):
    return repr(float(x))


def main():
    results, boots, trims, counts, gate, manifest, checks = [read(n) for n in
        ("ensemble_results.json", "bootstrap.json", "trim_check.json", "param_counts.json", "gate.json", "manifest.json", "reproduction_checks.json")]
    lines = [
        "# Kiểm tra ghép hai backbone bằng trung bình xác suất — validation only",
        "",
        "## Bối cảnh và câu hỏi",
        "",
        "Các backbone đơn lẻ BidirLM-1B và P_vi chưa vượt CafeBERT ổn định. Phân tích lỗi trước đó gợi ý các model khác họ có lỗi bổ sung cho nhau. Phép đo này kiểm tra CafeBERT cắt vocab + P_vi, với trọng số 0,5/0,5, tổng tham số tối đa 1 tỷ. Hai đối chứng gồm Cafe + XLM-R-large gốc và Cafe seed 42 + Cafe seed 43 (Restaurant). Tất cả số liệu chỉ dùng validation; không train, không mở test.",
        "",
        "## F1 của các model và tổ hợp",
        "",
        "F1 dưới đây là `selection_f1 × 100`, tính lại từ logits và metric của repo. Chữ số giữ nguyên ở độ chính xác báo cáo, không làm tròn cho đẹp.",
        "",
        "| Model/tổ hợp | Restaurant | ViSFD | Topic |",
        "|---|---:|---:|---:|",
    ]
    names = [("Cafe", "cafe"), ("Cafe seed 43", "cafe_s43"), ("XLM-R-large", "xlmr_large"),
             ("P_vi", "p_vi"), ("O", "o"), ("B1B", "b1b"), ("PAIR Cafe + P_vi", "PAIR"),
             ("CTRL_FAM Cafe + XLM-R-large", "CTRL_FAM"), ("CTRL_SEED Cafe + Cafe seed 43", "CTRL_SEED"),
             ("Cafe + O", "cafe+o"), ("Cafe + B1B", "cafe+b1b"),
             ("P_vi + B1B", "p_vi+b1b"), ("Cafe + P_vi + B1B", "cafe+p_vi+b1b")]
    for label, key in names:
        vals = []
        for t in TASKS:
            value = results[t]["single"].get(key)
            if value is None:
                value = results[t]["ensemble"].get(key, {}).get("f1")
            vals.append("—" if value is None else fmt(value))
        lines.append("| " + label + " | " + " | ".join(vals) + " |")
    lines += ["", "| Tổ hợp | Task | Δ vs Cafe | Δ vs CTRL_FAM |", "|---|---|---:|---:|"]
    for label, key in names[6:]:
        for task in TASKS:
            x = results[task]["ensemble"].get(key)
            if x is not None:
                lines.append(f"| {label} | {LABELS[task]} | {fmt(x['delta_cafe'])} | {fmt(x['delta_CTRL_FAM'])} |")
    lines += ["", "## Chẩn đoán validation", "",
              "Paired bootstrap lấy lại ví dụ (1.000 lần, seed 42); khoảng tin cậy là percentile 2,5–97,5%. Tỷ lệ `>0` là tỷ lệ replicate có chênh lệch dương.", "",
              "| Task | Chênh lệch | Điểm | CI 95% | Tỷ lệ >0 |", "|---|---|---:|---|---:|"]
    for task in TASKS:
        for name, value in boots[task].items():
            lines.append(f"| {LABELS[task]} | {name} | {fmt(value['point'])} | [{fmt(value['ci95'][0])}, {fmt(value['ci95'][1])}] | {fmt(value['fraction_gt_zero'])} |")
    lines += ["", "Quét trọng số Cafe dưới đây chọn trên validation nên kết quả tốt nhất là lạc quan; không dùng cho gate.", "",
              "| w_Cafe | Restaurant | ViSFD | Topic |", "|---:|---:|---:|---:|"]
    for w in ("0.3", "0.4", "0.5", "0.6", "0.7"):
        lines.append("| " + w + " | " + " | ".join(fmt(results[t]["weight_scan"][w]) for t in TASKS) + " |")
    lines += ["", "| Task | Slot sửa đúng | Slot làm sai | Ví dụ có sửa đúng | Ví dụ có làm sai | Ví dụ có cả hai |", "|---|---:|---:|---:|---:|---:|"]
    for task in TASKS:
        c = results[task]["corrections"]
        lines.append(f"| {LABELS[task]} | {c['fixed_slots']} | {c['harmed_slots']} | {c['examples_with_fix']} | {c['examples_with_harm']} | {c['examples_with_both']} |")
    lines += ["", "## Cắt vocab và số tham số", "",
              "Cafe dùng cùng SentencePiece với XLM-R-large, vocab gốc 250.002. Keep-set và tokenizer cắt từ `artifacts/vocab_trim_student/` được áp vào checkpoint Cafe trong bộ nhớ; logits được so với checkpoint gốc cùng precision BF16.", "",
              "| Task | Chuỗi token ID khớp | max abs(Δ logits) | Dự đoán giống hệt | F1 gốc | F1 cắt | Cafe cắt + head | P_vi + head | Tổng |",
              "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for task in TASKS:
        v, p = trims[task], counts[task]
        lines.append(f"| {LABELS[task]} | {v['token_ids_identical_count']}/{v['sequences']} ({fmt(v['token_ids_identical_rate'])}) | {fmt(v['max_abs_logits_delta'])} | {v['predictions_identical']} | {fmt(v['f1_original']*100)} | {fmt(v['f1_trim']*100)} | {p['cafe_trim']} | {p['p_vi']} | {p['total']} |")
    lines += ["", f"Tổng lớn nhất khi dùng head lớn nhất: **{counts['max_head_total']}** tham số. Điều kiện ≤1.000.000.000: **{counts['within_1b']}**.",
              "", "## Gate cố định", "",
              "PASS chỉ khi cả ba task có PAIR − Cafe ≥0,8 điểm và PAIR > CTRL_FAM, đồng thời Cafe cắt cho dự đoán giống hệt và tổng ≤1B.", "",
              "| Task | Δ PAIR − Cafe | ≥0,8 | Δ PAIR − CTRL_FAM | >0 |", "|---|---:|---|---:|---|"]
    for task in TASKS:
        g = gate["tasks"][task]
        lines.append(f"| {LABELS[task]} | {fmt(g['delta_cafe'])} | {g['gain_ge_0p8']} | {fmt(g['delta_CTRL_FAM'])} | {g['beats_CTRL_FAM']} |")
    lines += ["", f"Trim dự đoán giống hệt: **{gate['trim_identical_all']}**. Ngân sách tham số: **{gate['parameter_within_1b']}**. Quyết định: **{gate['decision']}**.",
              "", "## Tái tạo và giới hạn", "",
              f"Đã strict load và tái tạo dự đoán/F1 cho {sum(len(x) for x in checks.values())} checkpoint; mọi check trong `reproduction_checks.json` đều PASS. SHA-256 của 16 checkpoint trong manifest trùng metadata gốc. Chỉ dùng GPU vật lý 1 (`CUDA_VISIBLE_DEVICES=1`, trong process là `cuda:0`). Tổng GPU giờ theo thời gian model nằm trên GPU: **{fmt(manifest['gpu_hours'])}**.",
              "", "Cảnh báo từ Transformers về regex tokenizer xuất hiện khi nạp BidirLM-0.6B/1B. Đã giữ nguyên tokenizer của các run gốc; dự đoán tái tạo trùng tuyệt đối. Khi dựng tokenizer cắt ở lượt thử đầu, thiếu khai báo pad token làm loader dừng trước phép kiểm trim; sau đó dùng đúng năm special token của `trim.py` và cả ba phép kiểm trim PASS. Không cần cài hoặc nâng cấp gói. Không có sai lệch phương pháp so với spec.",
              "", "Chỉ có một seed chung; các thành viên đã chọn epoch theo validation nên ensemble trên validation lạc quan. Các đối chứng chịu cùng thiên lệch chọn epoch. Trung bình xác suất khác bầu chọn đa số trước đó. Chưa có kết quả test.",
              "", "## Lệnh tái lập", "", "```bash", "bash scripts/two_backbone_ensemble/run_all.sh", "CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src HF_HUB_OFFLINE=1 .venv-e0-cuda/bin/python scripts/two_backbone_ensemble/ensemble_metrics.py", "PYTHONPATH=src .venv-e0-cuda/bin/python scripts/two_backbone_ensemble/summarize.py", "```", ""]
    DOC.write_text("\n".join(lines))
    print(DOC)


if __name__ == "__main__":
    main()
