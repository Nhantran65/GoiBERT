# Ba hướng mới cho encoder tiếng Việt (vòng 6, 2026-09-14)

Quy trình: đọc lại E1–E22 → xác định đòn bẩy thật → deep research gap encoder tiếng Việt (BamiBERT, PhoBERT, ViSoBERT, ViDeBERTa, mmBERT/ModernBERT, HuTieuBERT/ViWordFormer, LLM2Vec) → kiểm nhanh một giả thuyết bằng thí nghiệm 12 phút → đối chiếu novelty. Mục tiêu: 1–3 ý tưởng có novelty, khả thi trên 4×L40S, hứa hẹn vượt BamiBERT trên bộ benchmark của nó.

## 0. Đòn bẩy thật rút từ 22 thí nghiệm

| Đòn bẩy | Bằng chứng | Kích thước |
|---|---|---|
| Pretraining/backbone | student 26M → teacher 110M: +3 ABSA; DAPT domain review: Topic +2,6 (110M) | lớn nhất |
| Dữ liệu tổ hợp nhãn (ghép câu) | Restaurant +1,16 ± 0,18 (3 seed), Hotel +0,4, ViSFD 0 | trung bình, phụ thuộc mật độ câu đa aspect sẵn có |
| Recipe fine-tune (LLRD, NC) | +1–2,5 ở 26M; ≈ 0 ở 110M | chỉ model nhỏ |
| Cơ chế đọc/attention (12 nhánh E9–E20) | ≤ ±1, bằng nhiễu; 5 giả thuyết bị bác bỏ có đối chứng | không |
| Từ hiếm (E12) | rare-unit burden +6/+13 pp ViSFD/Restaurant | tín hiệu vocab/coverage |

## 1. Bối cảnh cạnh tranh (Table 2 của BamiBERT, test)

| | BamiBERT | ViSoBERT | PhoBERT | XLM-R |
|---|---:|---:|---:|---:|
| VSFC Sent acc/F1 | 93,86 / 83,41 | 93,15 / 81,49 | **94,10 / 83,27** | 93,56 / 82,20 |
| VSFC Topic acc/F1 | 89,34 / 79,90 | 88,80 / 79,86 | 89,24 / 79,80 | 89,18 / 79,56 |
| ViSpamReviews acc/F1 | 90,76 / 78,20 | **90,99 / 79,06** | 89,83 / 76,18 | 90,16 / 76,55 |
| ViSFD det/cls | **89,14 / 84,24** | 88,63 / 83,55 | 86,03 / 78,76 | 82,73 / 65,81 |
| Hotel det/cls | 79,99 / 72,65 | 79,41 / **74,24** | 79,16 / 73,73 | 77,70 / 71,23 |
| Restaurant det/cls | **88,01 / 74,89** | 86,86 / 73,87 | 86,53 / 73,52 | 82,18 / 71,58 |

BamiBERT: BERT-base, BPE 20k của PhoGPT, 129 GB × 20 epoch, RoBERTa MLM, không ablation. ViSoBERT (97M, pretrain trên social media) thắng ở Hotel cls và ViSpam ⇒ **domain dữ liệu pretrain quyết định** ở nhóm task review/social. Hiện chưa có encoder tiếng Việt nào theo kiến trúc/recipe hiện đại (ModernBERT/NeoBERT: RoPE, GeGLU, unpadding, context dài, 3T tokens); các ngôn ngữ khác đã có (moBERTo-PT 2026, TabiBERT-TR, llm-jp-modernbert-JA, French/Latvian ModernBERT). Đây là gap tài nguyên rõ.

**Phép kiểm 12 phút** (`scripts/research_checks/probe_mmbert_finetune.py`, `artifacts/research_diagnostics/mmbert_probe/`): mmBERT-base (jhu-clsp, ModernBERT đa ngữ 3T tokens, 22L×768, 110M non-embedding, vocab 256k, tokenizer tách đúng 1 âm tiết/token) fine-tune CE thường, LR 5e-5, 20 epoch, **không có bất kỳ pretraining tiếng Việt riêng nào**:

| Validation pair F1 | mmBERT-base (CE thường) | BamiBERT E14 (CE) | goibert-b-dapt E19 (NC+LLRD) |
|---|---:|---:|---:|
| Restaurant | **74,38** | 74,74 | 74,51 |
| Hotel | **73,66** | 73,24 | 73,63 |

⇒ Một encoder hiện đại đa ngữ "nguyên bản" đã ngang BamiBERT trên ABSA tiếng Việt. Cộng thêm continued pretraining tiếng Việt + domain review + recipe của project, xác suất vượt là cao.

## Ý tưởng 1 — ViModernBERT: encoder tiếng Việt hiện đại bằng continued pretraining có ủ domain

**Phương pháp.** Continued pretraining mmBERT-base (giữ kiến trúc ModernBERT, context 8k, vocab Gemma-2) trên 5–10B token tiếng Việt theo lịch **annealed domain learning**: giai đoạn 1 web tổng quát (FineWeb2-vi/CC100-vi, mask 30 → 15%), giai đoạn 2 trộn tăng dần review/social (Shopee/Tiki/Foody/Facebook comment công khai, text train của các benchmark, mask 5–15%), giai đoạn 3 decay chỉ review/social — mô phỏng lịch "annealed language learning" của mmBERT nhưng theo **domain** thay vì ngôn ngữ. Sau đó fine-tune bằng recipe project (LLRD/NC cho model nhỏ; ghép câu nhãn hợp cho Restaurant/Hotel).

**Vì sao tin thắng BamiBERT.** (a) mmBERT nguyên bản đã ngang; (b) DAPT +2,6 Topic ở 110M; (c) ViSoBERT thắng BamiBERT đúng ở task review/social nhờ domain; (d) kiến trúc mới nhanh 2–3× và context 8k (BamiBERT 2k) là điểm cộng tài nguyên. Kỳ vọng: vượt ở ViSFD/ABSA/Spam (+0,5–1,5), ngang ở VSFC.

**Novelty.** Tài nguyên: encoder tiếng Việt đầu tiên theo họ ModernBERT (chưa thấy trên arXiv/HF đến 09/2026). Phương pháp: lịch ủ domain cho CPT đơn ngữ (tiền lệ: annealed language learning của mmBERT; DAPT của Gururangan 2020; các CPT đơn ngữ khác dùng một giai đoạn). Mức novelty phương pháp thấp–trung bình; sức nặng nằm ở SOTA + bộ chẩn đoán của project. Tiền lệ phải trích: mmBERT, ModernBERT, moBERTo, TabiBERT, llm-jp-modernbert, DAPT, ViSoBERT, BamiBERT.

**Khả thi.** 4×L40S: mmBERT-base ~110M non-embedding, MLM head chỉ tính trên token bị mask; ước ~100k token/s tổng ⇒ 5B token ≈ 14 h; 10B ≈ 28 h. Dữ liệu: FineWeb2 vi (hàng chục GB, tải HF), CC100-vi; review/social: các bộ công khai trên HF + text train benchmark (không dùng validation/test). Cần cài flash-attn cho unpadding (tùy chọn).

**Kiểm nhanh (1 ngày).** CPT 1B token (≈3 h) rồi chạy lại đúng probe 12 phút: nếu Restaurant/Hotel tăng ≥ +1 so mmBERT gốc → chạy đủ 5–10B và cả 8 benchmark; nếu không tăng → xem lại data/LR trước khi đầu tư.

## Ý tưởng 2 — Tokenizer tách thanh điệu (tone-factorized) + pretraining khử nhiễu dấu

**Phương pháp.** Âm tiết tiếng Việt = khung chữ (skeleton) + dấu (6 thanh + các dấu chữ ă/â/ê/ô/ơ/ư/đ). Chuẩn hóa NFD, tách mỗi token BPE thành (skeleton token, mã mẫu dấu); embedding đầu vào = E_skeleton[token] + E_diacritic[mẫu dấu]. Bảng skeleton chia sẻ thống kê giữa các biến thể thanh điệu (ngon/ngón/ngọn), giảm sparsity đúng chỗ E12 chỉ ra (rare-unit). Objective CPT thêm: **diacritic-denoising** — với p% âm tiết, xóa dấu ở đầu vào nhưng vẫn dự đoán token đầy đủ (mô phỏng văn bản người dùng không dấu), cộng MLM thường. Có thể ghép vào Ý tưởng 1 (vocab transfer: khởi tạo E_skeleton từ trung bình embedding các biến thể có dấu của mmBERT).

**Vì sao đáng thử.** Văn bản social/review thường thiếu dấu; benchmark social (ViSpamReviews, và các bộ ViSoBERT dùng: ViHSD, VSMEC, ViOCD…) sẽ hưởng lợi rõ; trên 5 task hiện tại, tỉ lệ câu có đơn vị thiếu dấu chỉ 2–8% (E12) ⇒ gain trên các task này nhỏ (≤ +0,5), gain chính ở robustness và social tasks. Đây là ý tưởng có novelty kiến trúc/objective đặc thù tiếng Việt cao nhất trong ba.

**Novelty.** Tương tự ChineseBERT (glyph + pinyin embedding, 2021) nhưng cho hệ chữ Latin có dấu/thanh điệu; chưa tìm thấy encoder tiếng Việt nào tách thanh điệu ở embedding hoặc dùng denoising theo dấu khi pretrain (các công trình diacritic restoration là task riêng; HuTieuBERT/ViWordFormer/ACL-2026 "Morphology hides in plain sight" đi hướng ranh giới từ/morpheme, không phải thanh điệu). Cần audit thêm trước claim.

**Kiểm nhanh (2 h, không train).** (a) Đo độ suy giảm của E19/mmBERT khi xóa dấu toàn bộ validation (robustness gap); (b) đếm tỉ lệ token hiếm trong train có "song sinh" cùng skeleton xuất hiện ≥100 lần (mức chia sẻ thống kê thu được). Nếu gap lớn và tỉ lệ song sinh cao → chạy ablation CPT 500M token có/không tách dấu.

## Ý tưởng 3 — Tự huấn luyện cân bằng theo tổ hợp nhãn trên corpus review không nhãn

**Phương pháp.** Kết luận E22: chỉ dữ liệu tổ hợp mới sửa lỗi đa aspect, nhưng ghép câu nhân tạo có trần (không giúp ViSFD, Hotel yếu). Thay bằng câu **tự nhiên**: (1) gán nhãn giả cho corpus review không nhãn lớn bằng hệ thống tốt nhất (ensemble E19/d1 3 seed); (2) khai thác câu có tổ hợp nhãn hiếm/chưa gặp với độ tin cậy cao (đồng thuận ensemble + margin); (3) lấy mẫu cân bằng theo **tổ hợp nhãn** (không phải theo nhãn) rồi trộn vào fine-tune; lặp 1–2 vòng. Kết hợp với ghép câu cho phần tổ hợp không tìm được.

**Vì sao đáng thử.** Nhắm đúng nhóm lỗi tồn tại ở mọi quy mô (cặp hiếm missed 33–48%, trái polarity 15–19%); câu tự nhiên tránh lệch phong cách của composite (đã thấy trong probe: composite làm giảm envelope trên câu tự nhiên).

**Novelty.** Self-training/pseudo-labeling và class-balanced self-training (CReST 2021) có tiền lệ; biến thể cân bằng theo *tổ hợp nhãn* cho multi-label ABSA và việc kết hợp với chẩn đoán tổ hợp là phần mới nhưng nhỏ. Xếp sau hai ý tưởng trên; hữu ích như thành phần của hệ thống SOTA.

**Kiểm nhanh (1 ngày).** Cần corpus review không nhãn (Shopee/Foody/Tiki công khai, ~1M câu); gán nhãn bằng d1 Restaurant/Hotel; đo số câu tổ hợp hiếm thu được; fine-tune một vòng, so cặp hiếm/trái polarity và F1 3 seed.

## Khuyến nghị thứ tự

1. **Ý tưởng 1** (đầu tư chính, xác suất SOTA cao, 1–2 ngày GPU): bắt đầu bằng kiểm nhanh CPT 1B token.
2. **Ý tưởng 2** chạy kiểm nhanh không-train ngay (2 h) để quyết định có ghép vào Ý tưởng 1 làm đóng góp phương pháp hay không.
3. **Ý tưởng 3** sau khi có encoder mới, dùng như lớp dữ liệu cho ABSA.

Bài báo mục tiêu: "ViModernBERT: encoder tiếng Việt hiện đại với ủ domain và tách thanh điệu" — SOTA trên 8 benchmark BamiBERT + phân tích cơ chế từ E1–E22 (phần phân tích đã có sẵn, là điểm khác biệt so với các bài "CPT ModernBERT sang ngôn ngữ X").
