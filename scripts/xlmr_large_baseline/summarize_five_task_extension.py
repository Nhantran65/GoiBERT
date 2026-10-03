#!/usr/bin/env python3
"""Audit the two new validation grids and report the XLM-R-large five-task table."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from goibert import train as e1_train  # noqa: E402

ART = ROOT / "artifacts" / "xlmr_large_baseline"
TASKS = ("vsfc_sentiment", "vsfc_topic", "visfd", "absa_restaurant", "absa_hotel")
NEW = ("vsfc_sentiment", "absa_hotel")
LR = (1e-5, 2e-5, 5e-5)
LABEL = {"vsfc_sentiment": "Sentiment", "vsfc_topic": "Topic", "visfd": "ViSFD", "absa_restaurant": "Restaurant", "absa_hotel": "Hotel"}


def read(path: Path):
    return json.loads(path.read_text())


def cell(task: str, lr: float) -> dict:
    path = ART / "xlmr_large" / task / f"lr_{lr:.0e}" / "seed_42"
    recovery = ART / "recovery" / task / f"lr_{lr:.0e}" / "seed_42"
    if recovery.joinpath("metadata.json").exists() and read(recovery / "metadata.json").get("status") == "completed":
        path = recovery
    meta = read(path / "metadata.json")
    metrics = read(path / "best_validation_metrics.json")
    rows = read(path / "validation_predictions.json")["rows"]
    curves = read(path / "epoch_curves.json")["epochs"]
    checks = meta["postrun_checks"]
    assert meta["status"] == "completed" and all(checks[k] for k in ("reload_matches_best", "prediction_file_matches_best", "reload_predictions_identical", "ids_stable_and_complete"))
    assert len(curves) == 30
    assert meta["model_manifest"]["commit"] == "c23d21b0620b635a76227c604d44e43a9f0ee389"
    assert meta["task_id"] == task and int(meta["seed"]) == 42 and abs(float(meta["learning_rate"]) - lr) < 1e-12
    bundle = e1_train.load_task_bundle(task, ROOT, include_test=False)
    recalculated = e1_train.selection_score(e1_train.evaluate_predictions(task, bundle, [r["gold"] for r in rows], [r["prediction"] for r in rows], split="validation"))
    assert abs(recalculated - float(meta["best_selection_f1"])) < 1e-12
    assert abs(recalculated - float(metrics["selection_f1"])) < 1e-12
    return {"f1": recalculated, "epoch": int(meta["best_epoch"]), "minutes": float(meta["elapsed_seconds"]) / 60, "path": str(path.relative_to(ROOT)), "checks_ok": True}


def main() -> None:
    original = read(ART / "summary.json")
    refs = read(ROOT / "artifacts" / "base_encoder_baseline" / "summary.json")["references"]
    grid = {task: {f"{lr:.0e}": cell(task, lr) for lr in LR} for task in NEW}
    best = {task: max(grid[task].items(), key=lambda kv: kv[1]["f1"]) for task in NEW}
    all_five = {}
    for task in TASKS:
        if task in NEW:
            key, result = best[task]
            selected = {"f1": result["f1"], "lr": float(key), "epoch": result["epoch"]}
        elif task == "absa_restaurant":
            selected = {"f1": original["grid"][task]["2e-05"]["42"]["f1"], "lr": 2e-5, "epoch": original["grid"][task]["2e-05"]["42"]["epoch"]}
        else:
            selected = {"f1": original["grid"][task]["2e-05"]["42"]["f1"], "lr": 2e-5, "epoch": original["grid"][task]["2e-05"]["42"]["epoch"]}
        all_five[task] = {"xlmr": selected, "bami": refs["bami_e14"][task]["f1"], "cafe": refs["cafe_fixed_2e-5"][task]["f1"]}

    payload = {"protocol": "xlmr-large-five-task-extension-v1", "validation_only": True, "seed": 42, "new_grid": grid, "five_task_table": all_five, "restaurant_two_seed_means": {"xlmr": 0.7701, "cafe": 0.7706}, "source_original_summary": "artifacts/xlmr_large_baseline/summary.json", "runtime_recovery": {"failed_attempts": ["artifacts/xlmr_large_baseline/xlmr_large/absa_hotel/lr_2e-05/seed_42", "artifacts/xlmr_large_baseline/xlmr_large/vsfc_sentiment/lr_5e-05/seed_42"], "complete_reruns": ["artifacts/xlmr_large_baseline/recovery/absa_hotel/lr_2e-05/seed_42", "artifacts/xlmr_large_baseline/recovery/vsfc_sentiment/lr_5e-05/seed_42"], "observation": "CUDA illegal instruction / illegal memory access occurred while GPU workloads overlapped; root cause not established. Both cells were rerun from the start on GPU 1/2 with the same seed and recipe."}}
    out = ART / "five_task_extension_summary.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    lines = ["# XLM-R-large: Sentiment và Hotel, hoàn chỉnh bảng năm task", "", "Validation-only, seed 42; recipe E14 và checkpoint giống baseline XLM-R-large ba task. Hai task mới chạy grid ba LR riêng; ba task cũ tái dùng, không chạy lại.", "", "| Task | XLM-R-large | BamiBERT | CafeBERT | Large−Cafe |", "|---|---:|---:|---:|---:|"]
    for task in TASKS:
        row = all_five[task]
        lines.append(f"| {LABEL[task]} | {100*row['xlmr']['f1']:.2f} | {100*row['bami']:.2f} | {100*row['cafe']:.2f} | {100*(row['xlmr']['f1']-row['cafe']):+.2f} |")
    lines += ["", "Các điểm trong bảng đều là validation seed 42. Bami được chọn trong grid LR riêng; Cafe dùng LR 2e-5; XLM-R-large Sentiment/Hotel chọn trong ba LR. Restaurant XLM-R-large/Cafe mean hai seed lần lượt 77,01/77,06, nên không diễn giải chênh seed 42 là thắng ổn định.", "", "| Task mới | 1e-5 | 2e-5 | 5e-5 | Chọn |", "|---|---:|---:|---:|---|"]
    for task in NEW:
        cells = grid[task]
        best_lr, best_cell = best[task]
        lines.append(f"| {LABEL[task]} | {100*cells['1e-05']['f1']:.2f} | {100*cells['2e-05']['f1']:.2f} | {100*cells['5e-05']['f1']:.2f} | {best_lr}, epoch {best_cell['epoch']} |")
    lines += ["", "Mỗi ô mới hoàn thành 30 epoch, reload best checkpoint, tái tính F1 từ predictions, đối chiếu IDs/gold trong runner; script tổng hợp độc lập tính lại F1 từ các prediction files. Cả sáu ô phải đạt trước khi báo cáo được ghi.", "", "**Sentiment LR 5e-5 bất ổn:** validation rơi xuống ~22,47 từ epoch 5 và không hồi phục; checkpoint tốt nhất 82,35 ở epoch 3 của run recovery, thấp hơn rõ hai LR còn lại. Vì vậy không coi đây là một LR cạnh tranh ổn định.", "", "**Recovery lỗi thiết bị:** lượt Hotel LR2e-5 trên GPU3 lỗi CUDA illegal instruction; lượt Sentiment LR5e-5 trên GPU0 lỗi CUDA illegal memory access. Không dùng điểm của hai run dở. Cả hai được chạy lại từ đầu cùng seed/recipe trong thư mục `recovery/` trên GPU1/2. Người dùng đã mở rộng quyền sử dụng từ GPU2/3 sang cả GPU0–3; không dừng job khác.", "", f"GPU-giờ tính từ **sáu run hoàn tất**: {sum(v['minutes'] for row in grid.values() for v in row.values())/60:.2f}; chưa gồm hai run CUDA lỗi. Không mở test, không sửa baseline cũ.", "", "Giới hạn: một seed cho hai task mới; việc chọn best epoch và best LR trên validation làm điểm lạc quan; chưa kiểm định khác biệt nhỏ. Hotel cùng LR2e-5 so Cafe là 73,57 vs 73,34 (+0,23), nhỏ hơn chênh best-of-grid +0,32. Xem artifacts máy đọc ở `artifacts/xlmr_large_baseline/five_task_extension_summary.json`.", ""]
    (ROOT / "docs" / "XLMR_LARGE_FIVE_TASK_EXTENSION_RESULT.md").write_text("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()
