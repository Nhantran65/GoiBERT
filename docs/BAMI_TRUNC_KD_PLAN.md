# Plan: BamiBERT cắt lớp + KD từ CafeBERT (pilot Sentiment + Restaurant)

Ngày chốt: 2026-09-26. Người dùng duyệt ý tưởng; triển khai qua Codex (gpt-6-sol, reasoning high), root review.

## Mục tiêu và lý do

Mục tiêu dự án: một model **≤ BamiBERT (103M) mà không thua Bami**, hoặc nhỉnh hơn.

Kết quả trước ([CAFE_OOF_KD_STUDENT_RESULT.md](CAFE_OOF_KD_STUDENT_RESULT.md)):

- GoiTurn-S 26M + KD từ Cafe ngang Bami ở Topic và Restaurant, kém 0,6–0,9 điểm ở Hotel và ViSFD, và **kém 2,48 điểm ở Sentiment**.
- KD không giúp Sentiment/Topic. Mọi recipe cho student 26M đều kẹt quanh 84,5–84,8 ở Sentiment, nên nhiều khả năng đây là **giới hạn năng lực/pretraining**.

**Giả thuyết:** giữ pretraining của Bami nhưng cắt bớt các lớp trên cùng (còn 10 hoặc 8 lớp), rồi fine-tune có KD từ nhãn mềm OOF của Cafe (đã có sẵn), sẽ cho model nhỏ hơn Bami mà không thua Bami ở cả Sentiment lẫn Restaurant.

## Thiết kế

- **Model:** `models/bamibert-base-original`, dựng qua `goibert_e14.modeling.build_bamibert_classifier`, đúng như E14.
  - Cắt lớp = **giữ k lớp encoder dưới cùng**, bỏ các lớp trên, cập nhật `config.num_hidden_layers = k`.
  - Làm sau khi nạp weights pretrained và **trước** khi fine-tune. Head E14 giữ nguyên.
  - Ước tính tham số: L12 khoảng 103M, **L10 khoảng 88,9M**, **L8 khoảng 74,7M**. Phải đếm thực và báo cáo.
- **Recipe:** giống hệt E14 (`scripts/run_e14.py::_run_trial`):
  - AdamW của `goibert.train.make_optimizer`, 30 epoch, batch 32, max length 256, bf16, warmup 10%, clip 1;
  - chọn best epoch trên validation;
  - **LR đã khóa của Bami**: Sentiment 2e-5, Restaurant 5e-5. Bản cắt dùng cùng LR, không quét.
- **KD:** giống `scripts/cafe_oof_kd/student.py`:
  - loss = 0,5 · CE(gold) + 0,5 · T² · KL(teacher_T ‖ student_T) trên từng slot, T = 2;
  - nhãn mềm lấy từ `artifacts/cafe_oof_kd/soft/{vsfc_sentiment,absa_restaurant}.npz`, tra theo id ví dụ;
  - dùng `label_widths` / `slot_segments` của `scripts/cafe_oof_kd/common.py` cho cả task một nhãn lẫn ABSA.
- **Các arm** (task `vsfc_sentiment` và `absa_restaurant`, seed 42/43/44):

  | Arm | Lớp | Loss |
  |---|---:|---|
  | L12-KD | 12 | CE + KD |
  | L10-CE | 10 | CE |
  | L10-KD | 10 | CE + KD |
  | L8-CE | 8 | CE |
  | L8-KD | 8 | CE + KD |
  | **Tái tạo:** L12-CE seed 42 | 12 | CE, chạy bằng loop mới, phải gần E14 seed 42 (86,33 / 74,74) |

  Tổng 5 × 2 × 3 + 2 = **32 run**.
- **Mốc Bami L12-CE 3 seed** (đã có, không chạy lại):
  - seed 42: `artifacts/e14/trials/<task>/lr_<LR>/seed_42`;
  - seed 43/44: `artifacts/cafe_oof_kd/bami/<task>/seed_{43,44}`.
  - Trung bình: Sentiment **86,83** (86,33 / 86,12 / 88,04), Restaurant **73,36** (74,74 / 71,83 / 73,51).

## Gate (chốt trước khi đo; trung bình 3 seed validation)

- Một độ sâu k ∈ {10, 8} **đạt** nếu `mean(Lk-KD) ≥ mean(Bami) − 0,5` ở **cả Sentiment lẫn Restaurant**.
- Pilot **PASS** nếu có ít nhất một k đạt. Khi đó đề xuất mở rộng 5 task với độ sâu nhỏ nhất đạt.
- Báo thêm, không phải gate:
  - `Lk-KD − Lk-CE` (tác dụng của KD);
  - `Lk-CE − Bami` (mất mát do cắt lớp);
  - `L12-KD − Bami` (Bami cùng cỡ có nhỉnh hơn nhờ KD không).

## Ràng buộc

- Validation-only, `include_test=False`, không mở test. Không commit hay push. Không sửa script/artifact cũ.
- Chỉ ghi vào `scripts/bami_trunc_kd/`, `artifacts/bami_trunc_kd/`, `docs/BAMI_TRUNC_KD_RESULT.md`, và `/tmp` riêng.
- GPU theo danh sách người dùng duyệt:
  - mỗi GPU một job của mình, cần ≥ 12 GB trống lúc launch;
  - không kill, dừng hay renice process không do mình tạo.
- Launcher phải chạy **tách khỏi phiên** (`setsid nohup`) để không chết khi phiên agent kết thúc.
- Ngân sách trần: **8 GPU-giờ**.
