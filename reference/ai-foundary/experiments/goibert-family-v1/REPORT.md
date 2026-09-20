# GoiBERT Family v1 — Báo cáo kết quả (4 datasets, multi-seed, NC ablation)

Cập nhật: 2026-09-04 (thay bản 2026-09-02 — bổ sung multi-seed VLSP, PhoATIS, noise
ablation; RÚT LẠI claim thắng VLSP của bản trước). LUMI MI250X; chọn theo validation, chấm
test 1 lần; cùng protocol + lr grid hai phía. Delivery level: **implementation-complete**.

## Kết luận chính (đã kiểm chứng multi-seed ở nơi cần)

1. **Mục tiêu Line (model mạnh nhất ≤5% dưới BamiBERT): ĐẠT** — GoiBERT-B DAPT: VSFC
   2.73%, ViSFD 0.86%, PhoATIS 2.85%, VLSP trung bình 3 seeds **tốt hơn** baseline +2.6
   điểm (nhưng xem mục variance).
2. **Kết quả giá trị nhất: PhoATIS (hội thoại giao dịch — gần chat đặt món nhất):
   GoiTurn-S 26.5M THẮNG NHẸ baseline 103M (−0.29%), XS 10.6M hòa (0.45%, accuracy bằng
   nhau 0.9731).** Với mục tiêu sản phẩm, model nhỏ đã đạt parity đúng domain cần.
3. **RÚT LẠI claim "thắng BamiBERT +10.7 điểm trên VLSP"** của báo cáo 09-02: multi-seed
   cho thấy đó là seed may mắn (chi tiết dưới). Xu hướng trung bình vẫn dương (+2.6 điểm)
   nhưng variance chi phối — không claim công khai.
4. Backbone GD nâng turn-model sản phẩm: val macro F1 0.9395 → 0.9607 (test đang chờ số).

## Bảng tổng seed 42 (macro F1; gap âm = mình hơn)

| Model | Params | VSFC | ViSFD | VLSP-Rest | PhoATIS |
|---|---|---|---|---|---|
| BamiBERT (thước đo) | ~103M | 0.8274 / 0% | 0.7548 / 0% | 0.3942 / 0% | 0.5629 / 0% |
| GoiBERT-B DAPT | ~103M | 0.8048 / 2.73% | 0.7483 / 0.86% | 0.5017 / −27.3%¹ | 0.5468 / 2.85% |
| GoiTurn-S GD | ~26.5M | 0.7916 / 4.32% | 0.6990 / 7.40% | 0.4372 / −10.9%¹ | **0.5645 / −0.29%** |
| GoiTurn-XS GD | ~10.6M | 0.7865 / 4.94% | 0.6538 / 13.39% | 0.4239 / −7.5%¹ | 0.5604 / 0.45% |

¹ VLSP seed 42 KHÔNG tái lập — xem mục multi-seed. Không dùng cột này làm claim.

## Multi-seed VLSP (bài học variance — lý do benchmark nhỏ phải chạy nhiều seed)

| Seed | Baseline | DAPT-B | GoiTurn-S | GoiTurn-XS |
|---|---|---|---|---|
| 42 | 0.3942 | 0.5017 | 0.4372 | 0.4239 |
| 43 | 0.4576 | 0.4578 | 0.3659 | 0.3563 |
| 44 | 0.4492 | 0.4187 | 0.4151 | 0.3946 |
| **Mean** | **0.4337** | **0.4594** | 0.4061 | 0.3916 |

- Baseline tự dao động ±3 điểm giữa seeds → dataset 4.7K, macro F1 lớp thưa = nhiễu lớn.
- DAPT-B: hơn baseline trung bình +2.6 điểm, thắng/hòa 2/3 seeds — xu hướng dương, chưa
  đủ n để nói significance. Muốn claim: cần ≥5 seeds + kiểm định.
- Students trên VLSP: trung bình DƯỚI baseline (~−2.8 điểm) — chiến thắng seed 42 là may.

## Noise-consistency: PHÁN QUYẾT GIỮ — kết quả mạnh nhất của dự án (đo 2026-09-04)

Chấm trên 4 slices (clean / không dấu / teencode / combined), VSFC seed 42:

| Model | Clean | **Không dấu** | Combined | Suy giảm (clean−combined) |
|---|---|---|---|---|
| BamiBERT 103M | 0.8274 | 0.4988 | 0.6211 | 0.206 |
| BamiBERT + NC | 0.8064 | **0.7430** | 0.7662 | **0.040** |
| GoiTurn-S GD | 0.7916 | 0.4053 | 0.5799 | 0.212 |
| **GoiTurn-S GD + NC** | 0.7964 | **0.7354** | 0.7587 | **0.038** |
| GoiTurn-XS GD + NC | 0.7511 | 0.7273 | 0.7383 | 0.013 |

- **NC biến "sập khi mất dấu" (−17..−21 điểm) thành "gần miễn nhiễm" (−1..−5 điểm)**, giá
  chỉ 0–2 điểm trên clean. VLSP cùng mẫu hình (suy giảm 0.06–0.09 → 0.017–0.029).
- **Headline: GoiTurn-S 26.5M + NC đạt 0.7354 trên VSFC không dấu — hơn BamiBERT 103M
  (0.4988) tới +23.7 điểm** trên đúng kiểu input khách Việt gõ mobile. Vượt dự đoán §13.8
  (+4–6) gấp ~5 lần. Differentiator thật sự đầu tiên mà BamiBERT không có.
- Slice teencode gần như vô hại với mọi model (vocab PhoGPT đã chứa ko/dc/vs) — thủ phạm
  robustness là MẤT DẤU, và NC là thuốc.
- **Xác nhận trên CẢ 4 dataset (đo đủ 2026-09-04)** — macro F1 slice "không dấu",
  thường → +NC: VSFC 0.40–0.50 → 0.73–0.74; PhoATIS 0.15–0.20 → 0.49–0.53; ViSFD
  0.39–0.49 → 0.64–0.71; VLSP 0.28–0.36 → 0.37–0.42. Suy giảm clean→combined co từ
  0.13–0.21 xuống 0.016–0.041 ở mọi model, mọi dataset; giá trên clean ≈ 0 (PhoATIS
  baseline+NC còn TĂNG clean 0.5629→0.5701; ViSFD S 7.40%→6.50% gap, XS 13.39%→10.02%).
- DAPT + NC cộng hưởng: trên ViSFD không dấu, DAPT-B+NC (0.7136) vượt baseline+NC
  (0.6933). Bảng đầy đủ: `results/noisy_eval_<dataset>.md` (4 file).

**Khuyến nghị chốt: NC bật mặc định cho MỌI model production của dòng Goi.** Turn-model
v1 (`variant_b_gdinit_nc`, test 0.9582 / false-cut 5.32%) đã có NC — là model turn
khuyến nghị.

## Turn-model v0.5 (sản phẩm) — Verified 2026-09-04

Variant B + init từ backbone `goiturn_s_gd`, test synthetic held-out (8,019 câu):
**macro F1 0.9331 → 0.9577** (+2.5 điểm), **false-cut 8.09% → 5.11%** (giảm 37%),
đạt 99.2% chất lượng bản BamiBERT finetune 103M (0.9658/3.65%) với model 26.5M — chưa
dùng asymmetric loss/calibration (Phase 4). Val còn tăng ở epoch cuối → v1 nên train dài
hơn. v0.5 là model turn khuyến nghị hiện tại: `variant_b_gdinit__scratch.pt`.

## Phương pháp (không đổi so với bản trước)

Corpus tự dựng 1.67GB (CC100-vi + domain ×20) → DAPT MLM 30% 40K steps (loss 3.57→1.876)
→ TinyBERT general distillation 30K steps → task stage per dataset. Chống bùa: chọn val,
test 1 lần, seed suffix tách file, baseline cùng grid.

## Caveats

1. VSFC/ViSFD/PhoATIS mới 1 seed — variance thấy được ở VLSP nhắc rằng các chênh lệch
   <1–2 điểm ở dataset khác cũng chưa chắc bền. Multi-seed thêm nếu cần claim mạnh.
2. PhoATIS macro F1 thấp (~0.56) do 28 lớp nhiều lớp hiếm; accuracy 97.3% khớp anchor
   JointIDSF — sanity pass.
3. NC chưa được chấm đúng đề (thiếu noisy slices).
4. GoiBERT-B là derivative BamiBERT — legal check trước khi public. Checkpoints trên
   /scratch LUMI (dọn ~90 ngày; Line sẽ cấp acc HF team để lưu dài hạn).

## Việc kế tiếp (theo giá trị)

1. **Turn-model v1 = v0.5 + NC + train dài hơn** — gom 2 phát hiện lớn nhất vào model
   sản phẩm (backbone GD đã +2.5 điểm; NC sẽ chống chat không dấu).
2. Train NC variants cho PhoATIS + ViSFD (hoàn thiện bảng robustness 4 dataset).
3. P2 data thật (messenger_messages 1 năm) — nguồn đã audit sẵn, giá trị lớn nhất còn lại.
4. HF upload khi có acc (private trước, legal check license BamiBERT cho DAPT-B).
5. (tùy chọn) VLSP ≥5 seeds nếu team muốn theo đuổi claim thắng tuyệt đối.
