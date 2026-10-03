# Tóm tắt phiên làm việc 2026-09-20 → 2026-09-21 — từ "học thêm tiếng Việt" đến "cắt vocab + student nhỏ"

Tài liệu này tóm tắt toàn bộ chuỗi quyết định, thí nghiệm và kết luận trong phiên, kèm đường dẫn tới plan/prompt/kết quả để tra cứu. Mục tiêu dự án không đổi: **encoder tiếng Việt mới trong dải 100–600M, vượt BamiBERT (103M) và CafeBERT (560M) trên 5 benchmark local (VSFC Sentiment/Topic, ViSFD, UIT-ABSA Restaurant/Hotel), có tính mới**; gần cuối phiên mở thêm nhánh **model nhỏ ≤100M**.

## 0. Bối cảnh trước phiên (đã có sẵn)

- 18 vòng thí nghiệm trước: cơ chế gắn thêm vào encoder, KD từ teacher fine-tune, CPT liều nhỏ, lấy review từ web crawl — đều đóng. Tổng hợp: `docs/EXPERIMENTS_SUMMARY_STANDALONE.md`, `docs/ENCODER_100_600M_DEEP_RESEARCH_20260918.md`, `docs/CAFEBERT_BAMIBERT_STEP0_BLIND_READING.md` (≈30 % bất đồng là nhãn tranh chấp).
- mmBERT-base baseline 5 task qua gate (`docs/BASE_ENCODER_BASELINE_RESULT.md`); pilot CPT HPLT v1–v4 trên mmBERT (`docs/MMBERT_VI_CPT_PILOT_V4_RESULT.md`, kể cả test mở): 3 arm × 250M token đều không vượt mmBERT gốc ở ViSFD/Topic; chỉ C_inf +0,6 Restaurant test.

## 1. Deep research: còn hướng nào khả thi? — `docs/NEXT_DIRECTIONS_DEEP_RESEARCH_20260920.md`

Xếp hạng 4 hướng còn lại: (B) dữ liệu ABSA sinh bằng LLM; (D) encoder từ decoder Qwen3-0.6B (probe rẻ); (C) CPT quy mô lớn 12B×5 kiểu moBERTo; (A) phương pháp phía task trên Cafe (composite đã −1,02 trên Cafe: `docs/CAFEBERT_COMPOSITE_PILOT_RESULT.md`; aspect-pair +0,34 test: `docs/CAFEBERT_ASPECT_PAIR_TEST_RESULT.md`). Người dùng chọn thử **D** trước.

## 2. Chặng 0 — Qwen3-0.6B đọc hai chiều (probe)

- Research + tự kiểm của root: transformers 4.49 không có Qwen3 → venv riêng `.venv-qwen-probe` (torch 2.6.0+cu124, transformers 4.57.6); **bẫy**: Qwen3 hai chiều cần *cả* `is_causal=False` *và* mask 4D — mask 2D có padding âm thầm giữ một chiều; tokenizer Qwen 1,42 token/từ (= XLM-R), 0 % vượt 256; phát hiện checkpoint công khai **BidirLM-0.6B-Base** (Qwen3-0.6B + MNTP 30B token, Apache-2.0, nhánh `transformers-v4`).
- Plan/prompt: `docs/QWEN3_ENCODER_PROBE_PLAN.md`, `docs/QWEN3_ENCODER_PROBE_PROMPT.md`. Kết quả: `docs/QWEN3_ENCODER_PROBE_RESULT.md` (root đã đối chiếu artifacts).

| Arm | Restaurant | ViSFD | Topic |
|---|---:|---:|---:|
| C — Qwen3 một chiều | 74,16 | 84,22 | 79,00 |
| B0 — bật hai chiều, không luyện | 71,77 | 83,70 | 79,19 |
| **B2 — BidirLM (đã MNTP 30B)** | **75,72** | **84,85** | **80,74** |

Kết luận: đòn bẩy là **pha MNTP** (B2 − B0 = +3,95), chỉ đổi mặt nạ thì hại (−2,4); B2 > Bami cả 3, ≈ mmBERT, < Cafe 2,0/0,6/1,0; LR tốt nhất 1e-5.

## 3. Chặng 1 — MNTP tiếng Việt 250M trên BidirLM

- Plan/prompt: `docs/BIDIRLM_VI_MNTP_STAGE1_PLAN.md`, `docs/BIDIRLM_VI_MNTP_STAGE1_PROMPT.md`. Kết quả: `docs/BIDIRLM_VI_MNTP_STAGE1_RESULT.md`.
- Dữ liệu: pool general v3 (CC0) tokenize lại bằng tokenizer Qwen (1,18B token), 250M train; MNTP 4,5 GPU-giờ; held-out loss 1,74 → 0,83 (**model học tiếng Việt thật**). Bẫy mới: `attention_mask=None` + `position_ids` → transformers coi là packed và ép causal; phải truyền mask 2D toàn 1.
- P_vi: Restaurant 76,15 / ViSFD 85,44 / Topic 80,69 so O 75,72 / 84,85 / 80,74 → cổng **yếu**; root kiểm đường cong: trung bình epoch 11–30 của P_vi và O gần trùng (75,0 vs 74,9) → +0,43 là chọn đỉnh trên nhiễu. **Lần thứ ba** CPT 250M ≈ 0 ở ABSA (Bami, mmBERT, BidirLM).

## 4. XLM-R-large gốc — Cafe hơn nhờ to hay nhờ CPT? — `docs/XLMR_LARGE_BASELINE_PLAN.md` / `_PROMPT.md` / `_RESULT.md`

| | Restaurant (mean 2 seed) | ViSFD | Topic |
|---|---:|---:|---:|
| XLM-R-large gốc (2e-5) | **77,01** (77,81 / 76,20) | **85,48** | 80,61 |
| CafeBERT | 77,06 (77,72 / 76,40) | 85,47 | 81,74 |

Kết luận: **CafeBERT = XLM-R-large gốc** trên Restaurant/ViSFD (CPT 4–5B token chỉ +1,1 ở Topic). Hệ quả: (1) đóng hẳn hướng "dạy thêm tiếng Việt" ở mọi liều, không dùng LUMI cho việc này; (2) encoder xây từ decoder (B2/P_vi) thua encoder gốc cùng cỡ ~2 điểm Restaurant (khớp phản chứng Ettin); (3) "thắng Cafe" ≡ "thắng XLM-R-large fine-tune"; dư địa còn lại nằm ở **dữ liệu có nhãn** (gộp Restaurant+Hotel, dữ liệu sinh bằng LLM, aspect-pair).

Giải thích nguyên nhân (trao đổi trong phiên): nút thắt không phải kiến thức tiếng Việt — các nền đã thấy tiếng Việt gấp hàng chục–trăm lần liều thêm; dữ liệu CPT là web/báo chí, không phải review; lỗi còn lại là quyết định theo quy ước nhãn (cặp hiếm, neutral, cảm xúc ngầm), chỉ học từ dữ liệu có nhãn và bị chặn bởi nhiễu nhãn; fine-tune 30 epoch ghi đè thay đổi nhỏ của pretraining; validation 771 review nhiễu ±1. Tổng quát hóa có điều kiện: đúng cho các task phân loại ngắn của ta; chưa kiểm cho NER/POS/QA; model tiếng Việt nhỏ (Bami 103M ≈ mmBERT 307M) vẫn có giá trị *hiệu quả*.

## 5. Rẽ sang model nhỏ: distill + cắt vocab

- Trao đổi: distill xuống ≤100M **không giữ nguyên** điểm 560M (mất ~1–2 điểm; 26M mất ~5 — GoiTurn-S cũ 69,1 test), nhưng 3 lần KD trước làm sai cách (KD lúc fine-tune với teacher thuộc lòng train; khác tokenizer). Quantization không đổi số tham số, chỉ là bước triển khai cuối. **Cắt vocab** (bỏ ~85 % token không dùng cho tiếng Việt) là bước miễn phí: đo 40K bài web chỉ dùng 39.794/250.002 token XLM-R, 30K phổ biến nhất phủ 99,96 %.
- Bảng tổng hợp benchmark + mục tiêu student (trong phiên): mục tiêu ≤100M, ≥ Bami cả 5 task, Restaurant ≥ 75,5 mean 3 seed, ≈ mmBERT-base.
- Research: Ushio et al. EMNLP 2023 (vocabulary trimming, ≤40 % vocab giữ nguyên điểm); MiniLMv2 (student công khai `nreimers/mMiniLMv2-{L12,L6}-H384-distilled-from-XLMR-Large`, XNLI 72,9/69,3); Sajjad et al. 2020 (bỏ lớp trên cùng).
- Plan/prompt: `docs/VOCAB_TRIM_STUDENT_PLAN.md`, `docs/VOCAB_TRIM_STUDENT_PROMPT.md` (bước 1 cắt vocab + bước 2 đo sàn; bước 3 distill là plan riêng).

### Trạng thái lúc 09:04 UTC 2026-09-21 (đang chạy)

| Đã xong | Kết quả |
|---|---|
| Đếm + tập giữ | K = 45.000 token, phủ pool 99,998 %, benchmark 100 % |
| Export 4 model cắt vocab | XLM-R-large 560M → **348,9M**; XLM-R-base 278M → **120,0M**; mMiniLMv2-L12 → 38,8M; L6 → 28,1M; unk 0,0045 %; tokenization khớp 100 % |
| **Chứng minh không mất điểm** | checkpoint XLM-R-large đã fine-tune, sau cắt: **77,8147 = 77,8147**, max\|Δ logits\| = 0 |
| Đợt 1 Restaurant | B12 (XLM-R-base 12 lớp, 120M) **73,54** ≈ round 15 (73,61) · B8 (8 lớp, 84M) 71,09 · M12 (39M) 61,96 · M6 (28M) 62,93 |
| Đợt 2 (đang chạy) | M12 ViSFD 80,04 / Topic 79,68 · M6 78,57 / 78,78 · B8 ~79,7 / ~79,0 · B12 chưa |

Đọc sơ bộ: cắt vocab thành công trọn vẹn; student MiniLM 28–39M sụp ở ABSA (−12 so Bami); B8 84M chưa distill = 71,1, cách mục tiêu 75,5 tới 4,4 điểm → nhiều khả năng đổi mục tiêu sang **~100–120M (12 lớp × 768, vocab 25–30K) distill MiniLMv2 từ XLM-R-large**. Chờ đợt 2–3 và `docs/VOCAB_TRIM_STUDENT_RESULT.md` để chốt.

## 6. Bảng điểm tổng hợp (validation, seed 42, recipe E14, best LR)

| Model | Tham số | Sentiment | Topic | ViSFD | Restaurant | Hotel |
|---|---:|---:|---:|---:|---:|---:|
| GoiTurn-S (cũ) | 26M | 80,6* | 78,6* | 79,8* | 69,1* | 71,8* |
| BamiBERT | 103M | 86,33 | 79,53 | 84,38 | 74,74 | 73,24 |
| XLM-R-base | 278M (→120M cắt vocab) | 86,34 | 81,89 | 83,82 | 73,61 | 73,05 |
| mmBERT-base | 307M | 86,70 | 81,18 | 83,64 | 75,63 | 73,70 |
| CafeBERT | 560M | 87,69 | 81,74 | 85,47 | 77,72 (2 seed 77,06) | 73,34 |
| XLM-R-large gốc | 560M (→349M cắt vocab) | — | 80,61 | 85,48 | 77,81 (2 seed 77,01) | — |
| B2 (BidirLM) | 596M | — | 80,74 | 84,85 | 75,72 | — |
| P_vi (B2 + 250M MNTP VN) | 596M | — | 80,69 | 85,44 | 76,15 | — |

\* test 5 seed. Test đã mở: Bami 83,03/78,18/84,50/74,15/73,94 · Cafe 82,36/80,77/85,48/77,12/75,17 · mmBERT —/80,81/84,38/74,48/—.

## 7. Hạ tầng và quy ước

- **Git:** repo `git@github.com:Nhantran65/GoiBERT.git`, commit đầu `1bb54d4` tác giả Kiet Tran (key `id_ed25519`), **không** dòng Co-Authored-By. `.gitignore` loại venv, weights, corpus, predictions, export HF; dataset bên thứ ba giữ local theo `data/.gitignore` (gói `tmp/goibert_benchmark_data.tgz` 6,4 MB để scp). Kết quả từ chặng 0 trở đi **chưa commit** (40 file thay đổi).
- **LUMI:** dự định cho việc dài (đã có hướng dẫn kiểm internet/ROCm/offline HF); sau kết luận §4, không còn nhu cầu CPT liều lớn — LUMI để dành cho loạt seed/distill nếu cần.
- **Venv:** `.venv-e0-cuda` (transformers 4.49; XLM-R, mmBERT), `.venv-qwen-probe` (4.57.6; Qwen3, BidirLM).
- **Quy tắc:** một job/GPU, GPU 0–2, không dừng job người khác; test chỉ mở một lần với cấu hình đăng ký trước; ≥3 seed cho quyết định; mọi trả lời kết thúc bằng "Tóm tắt dễ hiểu".
- **Bẫy kỹ thuật đã ghi:** Qwen3 hai chiều cần mask 4D; `attention_mask=None`+`position_ids` → causal; sandbox `pgrep` không thấy tiến trình ngoài → dùng tmux và kiểm `nvidia-smi` trước khi launch lại; BidirLM custom code chậm 1,7× (mask T×T mỗi lớp) → micro 16.

## 8. Việc tiếp theo (chờ quyết định)

1. Review `docs/VOCAB_TRIM_STUDENT_RESULT.md` khi xong; chốt cỡ student (84M vs 100–120M) và plan bước 3 (MiniLMv2 từ XLM-R-large cắt vocab, khởi tạo từ XLM-R-base, KD cross-fit + composite/LLRD ở fine-tune, ≥3 seed).
2. Song song (nhánh 100–600M): ghép phương pháp phía dữ liệu có nhãn lên XLM-R-large (gộp Restaurant+Hotel → LLM-synthetic có gate → aspect-pair), 3 seed, test một lần.
3. Commit/push kết quả chặng 0–2 khi người dùng đồng ý.
