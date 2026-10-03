# CafeBERT cross-head — screening Restaurant / ViSFD / Topic

Validation-only, seed42, LR2e-5, 30epoch, ba arm matched implementation. Không test. [Plan](CAFEBERT_CROSS_HEAD_GENERAL_PILOT_PLAN_20260922.md).

| Task | MHA matched | Diagonal | Full cross-head | Full−MHA | Full−Diagonal |
|---|---:|---:|---:|---:|---:|
| Restaurant | 76.81 | 77.39 | 76.93 | +0.12 | -0.46 |
| ViSFD | 85.54 | 85.53 | 85.50 | -0.04 | -0.03 |
| Topic | 80.92 | 80.99 | 81.17 | +0.25 | +0.18 |

**Gate đã khóa: KHÔNG ĐẠT** — full phải hơn cả MHA và diagonal ≥0,5 điểm ở ít nhất hai nguồn, không giảm nguồn còn lại >0,5 điểm. Đây là heuristic đầu tư, không significance.

## Đọc đường cong và quyết định

| Task | Arm | Trung bình top 3 epoch | Trung bình 10 epoch cuối |
|---|---|---:|---:|
| Restaurant | mha | 76.39 | 75.74 |
| Restaurant | diagonal | 77.28 | 76.75 |
| Restaurant | full | 76.86 | 76.50 |
| ViSFD | mha | 85.51 | 85.30 |
| ViSFD | diagonal | 85.45 | 85.29 |
| ViSFD | full | 85.49 | 85.32 |
| Topic | mha | 80.50 | 79.10 |
| Topic | diagonal | 80.95 | 80.52 |
| Topic | full | 81.05 | 80.51 |

Restaurant diagonal hơn MHA trên cả top-3 và 10 epoch cuối, nhưng full không hơn diagonal. ViSFD ba arm gần ngang nhau. Topic full có best cao hơn một ít, còn trung bình 10 epoch cuối gần bằng diagonal. Không có bằng chứng lợi ích rộng, riêng của trộn chéo head; dừng ở screening theo plan, không mở Sentiment/Hotel, seed mới, CPT hoặc test.

## Correctness và chi phí

Preflight 100 bước: max|Δ logits| official SDPA→MHA matched = 2,38e-7 FP32; diagonal/full ở identity khớp MHA chính xác trên batch kiểm; đổi nội dung pad không đổi logits; mọi ma trận trộn có gradient hữu hạn khác 0. Ba arm ~0,19 giây/bước, peak allocated ~10,5 GiB trên Restaurant. Kết quả: `artifacts/cafe_cross_head/preflight.json`.

Tổng 4.50 GPU-giờ cho 9 ô hoàn tất, chỉ GPU 0/1. 9/9 reload/prediction checks và tính lại F1 độc lập đạt; source hash trùng. Không test, không sửa baseline cũ.

MHA matched không dùng đúng kernel SDPA ở bốn block trên cùng, nên Restaurant MHA 76,81 thấp hơn baseline Cafe lịch sử 77,72. Dù eval logits tại initialization gần trùng, quỹ đạo fine-tune có thể khác do kernel/RNG. Chênh kiến trúc chính phải so **trong ba arm matched**; không lấy Cafe lịch sử thay control.

Giới hạn: một seed; chọn best epoch trên validation; ba task đầu đại diện ba nguồn text nhưng chưa bao gồm Sentiment/Hotel. Một operator có lợi trong fine-tune riêng mỗi task chưa tự tạo checkpoint encoder tổng quát. Không suy thắng Cafe/Bami trên test.
