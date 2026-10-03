#!/usr/bin/env python3
"""Independent F1 audit and locked screening gate for the Cafe cross-head pilot."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from goibert import train as e1_train  # noqa: E402

ART = ROOT / "artifacts" / "cafe_cross_head"
TASKS = ("absa_restaurant", "visfd", "vsfc_topic")
ARMS = ("mha", "diagonal", "full")
LABELS = {"absa_restaurant": "Restaurant", "visfd": "ViSFD", "vsfc_topic": "Topic"}


def read(path: Path):
    return json.loads(path.read_text())


def cell(task: str, arm: str) -> dict:
    directory = ART / arm / task / "lr_2e-05" / "seed_42"
    metadata = read(directory / "metadata.json")
    predictions = read(directory / "validation_predictions.json")["rows"]
    curves = read(directory / "epoch_curves.json")["epochs"]
    bundle = e1_train.load_task_bundle(task, ROOT, include_test=False)
    f1 = e1_train.selection_score(e1_train.evaluate_predictions(task, bundle, [r["gold"] for r in predictions], [r["prediction"] for r in predictions], split="validation"))
    checks = metadata["postrun_checks"]
    assert metadata["status"] == "completed" and metadata["seed"] == 42 and metadata["model_kind"] == arm and metadata["task_id"] == task
    assert len(curves) == 30 and all(checks[k] for k in ("reload_matches_best", "prediction_file_matches_best", "reload_ids_identical", "reload_predictions_identical"))
    assert abs(f1 - metadata["best_selection_f1"]) < 1e-12
    return {"f1": f1, "best_epoch": metadata["best_epoch"], "minutes": metadata["elapsed_seconds"] / 60, "peak_gib": metadata["peak_gpu_memory_allocated_bytes"] / 1024**3, "source_hash": metadata["source_hash"], "dir": str(directory.relative_to(ROOT))}


def main() -> None:
    grid = {task: {arm: cell(task, arm) for arm in ARMS} for task in TASKS}
    hashes = {v["source_hash"] for rows in grid.values() for v in rows.values()}
    assert len(hashes) == 1, f"source changed between runs: {hashes}"
    deltas = {task: {"full_minus_mha": grid[task]["full"]["f1"] - grid[task]["mha"]["f1"], "full_minus_diagonal": grid[task]["full"]["f1"] - grid[task]["diagonal"]["f1"], "diagonal_minus_mha": grid[task]["diagonal"]["f1"] - grid[task]["mha"]["f1"]} for task in TASKS}
    strong = [task for task, row in deltas.items() if row["full_minus_mha"] >= 0.005 and row["full_minus_diagonal"] >= 0.005]
    no_large_drop = all(row["full_minus_mha"] >= -0.005 and row["full_minus_diagonal"] >= -0.005 for row in deltas.values())
    passed = len(strong) >= 2 and no_large_drop
    payload = {"protocol": "cafe-cross-head-v1", "stage": "three-task-validation-screen", "seed": 42, "grid": grid, "deltas": deltas, "gate": {"strong_tasks": strong, "no_other_drop_gt_0.5pp": no_large_drop, "passed": passed}, "gpu_hours": sum(v["minutes"] for rows in grid.values() for v in rows.values()) / 60, "preflight": "artifacts/cafe_cross_head/preflight.json", "no_test_opened": True}
    (ART / "screen_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    lines = ["# CafeBERT cross-head — screening Restaurant / ViSFD / Topic", "", "Validation-only, seed42, LR2e-5, 30epoch, ba arm matched implementation. Không test. [Plan](CAFEBERT_CROSS_HEAD_GENERAL_PILOT_PLAN_20260922.md).", "", "| Task | MHA matched | Diagonal | Full cross-head | Full−MHA | Full−Diagonal |", "|---|---:|---:|---:|---:|---:|"]
    for task in TASKS:
        row, delta = grid[task], deltas[task]
        lines.append(f"| {LABELS[task]} | {100*row['mha']['f1']:.2f} | {100*row['diagonal']['f1']:.2f} | {100*row['full']['f1']:.2f} | {100*delta['full_minus_mha']:+.2f} | {100*delta['full_minus_diagonal']:+.2f} |")
    lines += ["", f"**Gate đã khóa: {'ĐẠT' if passed else 'KHÔNG ĐẠT'}** — full phải hơn cả MHA và diagonal ≥0,5 điểm ở ít nhất hai nguồn, không giảm nguồn còn lại >0,5 điểm. Đây là heuristic đầu tư, không significance.", "", "## Đọc đường cong và quyết định", "", "| Task | Arm | Trung bình top 3 epoch | Trung bình 10 epoch cuối |", "|---|---|---:|---:|"]
    for task in TASKS:
        for arm in ARMS:
            path = ROOT / grid[task][arm]["dir"] / "epoch_curves.json"
            values = [100 * row["selection_f1"] for row in read(path)["epochs"]]
            lines.append(f"| {LABELS[task]} | {arm} | {sum(sorted(values, reverse=True)[:3])/3:.2f} | {sum(values[-10:])/10:.2f} |")
    lines += ["", "Restaurant diagonal hơn MHA trên cả top-3 và 10 epoch cuối, nhưng full không hơn diagonal. ViSFD ba arm gần ngang nhau. Topic full có best cao hơn một ít, còn trung bình 10 epoch cuối gần bằng diagonal. Không có bằng chứng lợi ích rộng, riêng của trộn chéo head; dừng ở screening theo plan, không mở Sentiment/Hotel, seed mới, CPT hoặc test.", "", "## Correctness và chi phí", "", "Preflight 100 bước: max|Δ logits| official SDPA→MHA matched = 2,38e-7 FP32; diagonal/full ở identity khớp MHA chính xác trên batch kiểm; đổi nội dung pad không đổi logits; mọi ma trận trộn có gradient hữu hạn khác 0. Ba arm ~0,19 giây/bước, peak allocated ~10,5 GiB trên Restaurant. Kết quả: `artifacts/cafe_cross_head/preflight.json`.", "", f"Tổng {payload['gpu_hours']:.2f} GPU-giờ cho 9 ô hoàn tất, chỉ GPU 0/1. 9/9 reload/prediction checks và tính lại F1 độc lập đạt; source hash trùng. Không test, không sửa baseline cũ.", "", "MHA matched không dùng đúng kernel SDPA ở bốn block trên cùng, nên Restaurant MHA 76,81 thấp hơn baseline Cafe lịch sử 77,72. Dù eval logits tại initialization gần trùng, quỹ đạo fine-tune có thể khác do kernel/RNG. Chênh kiến trúc chính phải so **trong ba arm matched**; không lấy Cafe lịch sử thay control.", "", "Giới hạn: một seed; chọn best epoch trên validation; ba task đầu đại diện ba nguồn text nhưng chưa bao gồm Sentiment/Hotel. Một operator có lợi trong fine-tune riêng mỗi task chưa tự tạo checkpoint encoder tổng quát. Không suy thắng Cafe/Bami trên test.", ""]
    (ROOT / "docs" / "CAFEBERT_CROSS_HEAD_SCREEN_RESULT.md").write_text("\n".join(lines))
    print(json.dumps(payload["gate"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
