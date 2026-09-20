# Deep research 2026-09-20 — còn hướng nào khả thi để vượt BamiBERT và CafeBERT (100–600M)?

Root tự tổng hợp từ 18 vòng thí nghiệm local + tài liệu 2025–2026. Không training mới. Mục đích: liệt kê **mọi** hướng còn lại, chấm bằng bằng chứng (local trước, literature sau), chi phí, kỳ vọng và tính mới; xếp hạng để root quyết.

## 1. Những ràng buộc đã được chứng minh (không mở lại)

| Ràng buộc | Bằng chứng local |
|---|---|
| Cơ chế thêm vào encoder (reader/slot routing, pair-value residual, gated FFN, CTI, label-attention…) ≤ +1 validation, không giữ trên test | E9–E17, E20 (`docs/ARCHITECTURE_TRANSFER_FINAL_REVIEW.md`) |
| Continued pretraining ≤300M token trên Bami/Cafe/mmBERT: ≤0 ở ABSA, kể cả social (NC) và curriculum | E19, Pilot B, round 10, 13, 18 (v4 test: Q1 −0,5…−1,9) |
| KD từ teacher đã fine-tune: âm (logits one-hot trên train) | round 9, 10 |
| Review/văn nói người dùng **không lấy được từ web crawl** (JS-rendered; nhãn register HPLT sai cho tiếng Việt) | round 12, 16, 17 |
| Backbone thay thế đã thử: mDeBERTa, SEA-LION-300M, XLM-R-base đều ≤ Bami; mmBERT-base ≈ Bami trên ABSA **test** (+0,3), hơn ở Topic | round 15, v4 test |
| Validation phóng đại: delta <1 điểm val gần như luôn mất trên test (aspect-pair +1,19 → +0,34; mmBERT +0,9 → +0,3) | round 11 test, v4 test |
| ~30% slot bất đồng là nhãn tranh chấp → trần đo được thấp; cần ≥3 seed và ≥1,5 điểm val để tin | round 14b |

CafeBERT (XLM-R-large 560M + ~4–5B token CPT tiếng Việt) là mốc: Restaurant test 77,1 (2 seed), ViSFD 85,5, Topic 80,8.

## 2. Các hướng còn lại — đánh giá

### A. Phương pháp phía task trên nền mạnh nhất (CafeBERT) — *khả thi, biên nhỏ*
- Aspect-pair input: +0,34 test/2 seed nhưng cần 12 lượt encoder/review (`docs/CAFEBERT_ASPECT_PAIR_TEST_RESULT.md`). Composite augmentation: +1,16 ± 0,18 val/3 seed **trên BamiBERT** (E21) nhưng **−1,02 val trên CafeBERT** (`docs/CAFEBERT_COMPOSITE_PILOT_RESULT.md`, 1 seed) — không chuyển sang nền lớn. Chưa thử: huấn luyện chung Restaurant+Hotel (cùng schema VLSP, gấp đôi dữ liệu in-domain).
- Còn lại thực chất chỉ aspect-pair + joint Rest/Hotel trên Cafe, ≥3 seed. Chi phí ~10–15 GPU-giờ. Kỳ vọng test +0,3–0,8. Xác suất ~40%. Novelty thấp, và aspect-pair đắt ở inference.

### B. Tổng hợp dữ liệu ABSA bằng LLM (data-centric) — *khả thi, chưa thử, novelty trung bình cho tiếng Việt*
- Literature 2025: [LACA (ACL 2025)](https://aclanthology.org/2025.acl-long.41/) — LLM sinh dữ liệu ABSA cross-lingual; [Balanced augmentation for ABSA](https://arxiv.org/abs/2507.09485) — LLM sinh + đánh giá chất lượng (DPO) cải thiện encoder trên 4 benchmark; [semantic-preserved augmentation với reliability-aware fine-tuning](https://www.sciencedirect.com/science/article/pii/S0031320326005947). Điểm chung: gain rõ nhất khi dữ liệu ít và lớp hiếm — đúng chỗ ta yếu (rare label pairs, 33–47% miss ở round 5).
- Cách làm trong tài nguyên hiện có: chạy Qwen3-14B/32B (Apache-2.0) local trên L40S, sinh review nhà hàng/khách sạn/điện thoại **có nhãn aspect–polarity theo đúng schema**, ưu tiên cặp nhãn hiếm và câu nhiều aspect; lọc bằng self-consistency + classifier hiện có; trộn với train thật (tỷ lệ cố định), giữ validation/test nguyên. Không dùng LLM ở inference.
- Chi phí: sinh 50–100K câu ~vài giờ; fine-tune như A. Kỳ vọng val +1–2, test +0,5–1,5. Xác suất ~50%. Rủi ro: lệch phân phối, "hallucinated" nhãn; phải gate bằng validation thật và audit mẫu. Novelty: đầu tiên cho UIT-ABSA/ViSFD với gating chất lượng; ghép với composite (E22) thành một đóng góp data-centric hoàn chỉnh.

### C. CPT quy mô lớn (10–60B token) — *khả thi về dữ liệu, đắt, rủi ro, novelty cao*
- Tiền lệ trực tiếp: [moBERTo](https://arxiv.org/abs/2606.22722) — ModernBERT-base (chỉ tiếng Anh) + **60B token** tiếng Bồ (12B unique × 5 epoch) → tốt nhất PLUE-PT; [TabiBERT](https://arxiv.org/abs/2512.23065) cho tiếng Thổ (train từ đầu 1T token — chế độ khác, chỉ tham khảo kiến trúc). CafeBERT: ~4–5B token trên XLM-R-large. Liều ta thử (250M) bằng 0,4–5% các mốc này — nên kết luận "CPT không giúp" chỉ đúng ở liều nhỏ.
- Dữ liệu: HPLT đủ 12 shard không lấy mẫu (v2 chỉ dùng 5 shard × 1/8) → ~20–50B unique sau lọc; FineWeb2-HQ thêm ~10B. Đủ cho 12B × 5 epoch kiểu moBERTo.
- Chi phí: 2 ngày dữ liệu + 1–2 tuần trên 3 GPU (mmBERT-base 62K tok/s/GPU). Gate liều–đáp ứng 1B → 5B → 20B; dừng ở 5B nếu ≤ O. Xác suất thắng Cafe ~30–40%. Novelty cao (encoder tiếng Việt họ ModernBERT đầu tiên, phát hành được nếu không dùng NC).
- Lưu ý: nền XLM-R-large (560M) là lựa chọn thay thế có bằng chứng CPT hiệu quả (chính CafeBERT), nhưng chậm ~3× và "mới" ít hơn.

### D. Encoder từ decoder mạnh tiếng Việt (Qwen3-0.6B → bidirectional + MLM) — *chưa thử, rẻ để probe, novelty cao*
- [ICLR 2026, "Should we still pretrain encoders with MLM?"](https://arxiv.org/abs/2507.00994): 38 model 210M–1B, 15K run: MLM tốt nhất cho representation, nhưng **CLM rồi MLM** tốt hơn MLM thuần ở cùng ngân sách, và "tiếp tục MLM từ model CLM có sẵn" là chiến lược hiệu quả. [LLM2Vec](https://arxiv.org/abs/2404.05961): bật attention hai chiều + MNTP biến decoder thành encoder; đã có [Qwen-encoder-0.5B](https://huggingface.co/knowledgator/Qwen-encoder-0.5B) làm theo cách này.
- Vì sao hợp với ta: Qwen3-0.6B (0,6B tổng, ≤600M) đã thấy tiếng Việt trong 36T token đa ngữ — nhiều hơn mọi encoder tiếng Việt hiện có; tokenizer 151K. Cảnh báo: các báo cáo chuyển decoder→encoder chủ yếu đo embedding/retrieval; ở phân loại token/câu, encoder gốc cỡ nhỏ (DeBERTa/XLM-R) thường vẫn ngang hoặc hơn — không có bằng chứng sẵn cho ABSA tiếng Việt, phải probe.
- **Bổ sung 2026-09-20 (research kỹ cho chặng 0):** [Gemma Encoder (Google 2025)](https://arxiv.org/abs/2503.02656) — fine-tune hai chiều trực tiếp, không pretrain thêm, GLUE +4,9 so với một chiều; [BidirLM (04/2026)](https://arxiv.org/abs/2604.02045) — adapt chính Qwen3-0.6B bằng MNTP 30B token (có tiếng Việt qua FineWeb2-HQ), checkpoint `BidirLM/BidirLM-0.6B-Base` Apache-2.0 công khai, hơn EuroBERT-610m >1 điểm phân loại; phản chứng [Ettin/Seq vs Seq (ICLR 2026)](https://arxiv.org/abs/2507.11412) — decoder adapt vẫn thua encoder gốc *cùng dữ liệu*. Plan chặng 0: `docs/QWEN3_ENCODER_PROBE_PLAN.md`.
- Probe (~8–10 GPU-giờ): fine-tune Qwen3-0.6B với attention hai chiều + head E14 trên Restaurant/ViSFD/Topic, recipe E14. Nếu ≥ mmBERT (75,6 val Restaurant) chưa cần CPT → nền mới; rồi mới CPT MLM 5–10B token tiếng Việt (kết hợp C). Xác suất probe đạt ~30%; nếu đạt, đường này vừa có tính mới vừa có đà dữ liệu.

### E. Đã đóng, không mở lại
Cơ chế trên encoder; CPT liều nhỏ; task-KD; register/review từ crawl; ensemble (user loại); mDeBERTa/SEA-LION/LFM/EuroBERT-210m như base; mean-pooling (chi tiết recipe, không phải hướng).

### F. Vệ sinh bắt buộc cho mọi hướng
≥3 seed cho quyết định; test mở một lần bằng seed mới, cấu hình đăng ký trước; báo cả val lẫn test; delta val <1,5 coi là chưa có tín hiệu; ghi rõ nhãn tranh chấp ~30% giới hạn trần.

## 3. Xếp hạng theo "khả năng thắng cả hai × chi phí × tính mới"

| # | Hướng | Chi phí | P(thắng Cafe test) | Novelty | Khuyến nghị |
|---|---|---|---|---|---|
| 1 | **B (+ A)** trên Cafe: LLM-synthetic có gate chất lượng + joint Rest/Hotel (+ aspect-pair nếu chấp nhận 12 pass), 3 seed | 3–4 ngày, ~20 GPU-h | ~45% (biên 0,5–1,5) | trung bình (data-centric, tiếng Việt) | **làm trước** — kết quả dương khả dĩ nhất; composite đã âm trên Cafe nên B phải gate riêng |
| 2 | **D probe** Qwen3-0.6B bidirectional | 1 giờ | — (probe) | cao nếu đạt | chạy song song ngày 1 |
| 3 | **C** CPT 12B×5 epoch (mmBERT-base hoặc Qwen3-0.6B nếu D đạt), gate 1B/5B/20B | 2–3 tuần | 30–40% | cao | chỉ nếu chấp nhận rủi ro; gate sớm giới hạn mất mát ở 1–2 ngày |
| 4 | **B** trên mmBERT-base (307M) | như 1 | ~15% thắng Cafe; ~60% thắng Bami rõ | trung bình | phụ, cho câu chuyện "base-size" |

Kết luận: không còn hướng "cơ chế mới rẻ"; còn **hai** hướng có bằng chứng ngoài project — dữ liệu tổng hợp bằng LLM (rẻ, xác suất khá) và CPT quy mô lớn/encoder-từ-decoder (đắt, novelty cao, rủi ro cao). Cách dùng thời gian hợp lý: 1 → 2 → (3 nếu còn thời gian).

## Nguồn
- [Should We Still Pretrain Encoders with MLM? (ICLR 2026)](https://arxiv.org/abs/2507.00994) · [LLM2Vec](https://arxiv.org/abs/2404.05961) · [Qwen-encoder-0.5B](https://huggingface.co/knowledgator/Qwen-encoder-0.5B) · [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- [LACA (ACL 2025)](https://aclanthology.org/2025.acl-long.41/) · [Balanced Training Data Augmentation for ABSA](https://arxiv.org/abs/2507.09485) · [Semantic-preserved augmentation, reliability-aware fine-tuning](https://www.sciencedirect.com/science/article/pii/S0031320326005947)
- [moBERTo (ModernBERT → Portuguese, 60B token)](https://arxiv.org/abs/2606.22722) · [TabiBERT (Turkish ModernBERT)](https://arxiv.org/abs/2512.23065) · [BamiBERT](https://arxiv.org/abs/2607.02259)
- Local: `docs/ENCODER_100_600M_DEEP_RESEARCH_20260918.md`, `docs/NEW_DIRECTIONS_VI_ENCODER.md`, `docs/MMBERT_VI_CPT_PILOT_V4_RESULT.md`, `docs/CAFEBERT_ASPECT_PAIR_TEST_RESULT.md`, `docs/CAFEBERT_COMPOSITE_PILOT_RESULT.md`, `docs/E21_MULTISEED_D1.md`, `docs/CAFEBERT_BAMIBERT_STEP0_BLIND_READING.md`.
