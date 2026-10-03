# Chặng 1 — MNTP tiếng Việt 250M token trên BidirLM-0.6B-Base, screen 3 task (bắt buộc, local)

**Trạng thái:** plan bàn giao, chưa chạy. Ngày chốt 2026-09-20. Kế thừa chặng 0 (`docs/QWEN3_ENCODER_PROBE_PLAN.md` + `_RESULT.md`): arm **B2 = BidirLM-0.6B-Base** (Qwen3-0.6B-Base + MNTP 30B token đa ngữ) đạt Restaurant 75,72 / ViSFD 84,85 / Topic 80,74 (validation, seed 42) — hơn BamiBERT, ngang mmBERT, kém CafeBERT 2,0/0,6/1,0. Chặng 0 cũng cho thấy **đòn bẩy là pha MNTP** (B2 − B0 = +3,95), còn tiếng Việt chỉ là ~1/20 dữ liệu MNTP của họ. **Một câu hỏi duy nhất:** dạy tiếp MNTP bằng 250M token tiếng Việt sạch có đẩy B2 lên gần/qua CafeBERT không? Một arm, một cổng, ~10 GPU-giờ, ~8 giờ wall trên 3 GPU. Bản 1B, 5 task × 2 seed, LUMI: **chưa** thuộc nhiệm vụ này.

## 1. Arm và tham chiếu

| Arm | Là gì | Trạng thái |
|---|---|---|
| **O** | BidirLM-0.6B-Base @ `29794508bbb4e378b4789eebff42ae4c2d47d524` (nhánh transformers-v4, Apache-2.0), fine-tune như chặng 0 | đã có: Restaurant 75,72 (1e-5) / 75,13 (2e-5) / 73,89 (5e-5), seed 43 75,72; ViSFD 84,85; Topic 80,74 (`artifacts/qwen3_encoder_probe/B2/`). **Cần thêm 1 run: Restaurant LR 5e-6 seed 42** để cùng grid với P_vi |
| **P_vi** | O + MNTP **250M token** tiếng Việt (pool general v3, CC0), rồi fine-tune y hệt O | chạy trong chặng này |

Tham chiếu validation seed 42: BamiBERT 74,74 / 84,38 / 79,53 · mmBERT 75,63 / 83,64 / 81,18 · CafeBERT 77,72 / 85,47 / 81,74.

## 2. Dữ liệu (CPU, ~1 giờ) — `data/bidirlm_vi_mntp/`

- Nguồn: **general pool v3** (bins 7–9 sau lọc v3, cap host 2 %, 1,132B token Gemma, spam ~1 %, **đã decontam** train/dev/test 5 benchmark ở stage 1 v1; CC0). Lấy danh sách docs qua `scripts/mmbert_vi_cpt_pilot/v3/audit_v3.build_general_v3` (read-only) và đọc text như `scripts/mmbert_vi_cpt_pilot/v4/build_arms_v4.py`. **Không** Wikipedia, **không** ViSoBERT/NC, không text benchmark.
- Tokenize bằng **tokenizer BidirLM** (cùng vocab Qwen3; `pad = 151643`, `mask_token = <|mask|> = 151663`, `len = 151669`); mỗi doc kết thúc bằng EOS `151643`; nối và cắt thành chuỗi **512 token**, không BOS trong dữ liệu (BOS `151644` do model/loader prepend lúc train), không pad. Tokenize **toàn bộ pool** (1,13B) một lần, lưu `tokens.npy` (uint32) + `order` xáo trộn seed 42; **250M token đầu theo order** = tập train chặng này (≈488K chuỗi); 2.000 chuỗi held-out tách trước, loại khỏi train. Ghi `manifest.json`: số doc, token, hash tokens/order, tỷ lệ token/từ (kỳ vọng ~1,4), license.

## 3. MNTP 250M (1 GPU, ~4–5 giờ) — `artifacts/bidirlm_vi_mntp/mntp/`

**Objective (đúng cách BidirLM đã luyện, đã đọc `modeling_bidirlm.py`):** input = `[BOS 151644] + 512 token` trong đó 20 % token bị chọn; token bị chọn thay bằng `<|mask|>` 151663 (80 %) / token ngẫu nhiên (10 %) / giữ nguyên (10 %); attention **hai chiều**; loss CE tại **hidden của vị trí j−1 dự đoán token gốc ở vị trí j** cho các j bị chọn (= CLM-shift chuẩn với labels −100 ngoài vị trí chọn) — tương đương `BidirLMForMaskedLM` (hidden đã dịch một vị trí + `ForMaskedLMLoss` không shift). lm_head tied với embedding.

**Đường chạy nhanh (chính):** nạp weights BidirLM vào `Qwen3ForCausalLM` chuẩn (khóa BidirLM `embed_tokens/layers.*/norm` → thêm tiền tố `model.`; `lm_head` tied; `load_state_dict(strict=True)`), đặt `is_causal=False` trên mọi module attention, batch **không padding** nên `attention_mask=None` (chặng 0 đã đo: unpadded + is_causal False → hai chiều, Δ = 8,49); chỉ tính lm_head tại các vị trí j−1 cần (gather) để tránh logits 152K × 513 × batch. **Kiểm tương đương bắt buộc trước khi train:** cùng một batch, logits tại vị trí chọn của đường nhanh và của `BidirLMForMaskedLM` (custom code) khớp `max|Δ| ≤ 1e-2` fp32 (và loss khớp ≤ 1e-3). Không khớp → dùng custom code `BidirLMForMaskedLM` cho MNTP (chậm ~1,7×), ghi rõ.

**Siêu tham số:** 250M non-pad token, 1 epoch; global batch **64 × 512** (micro 16 × accum 4; tăng micro nếu VRAM cho phép) ≈ 7,6K bước; AdamW peak **2e-5**, betas (0,9, 0,98), wd 0,01, clip 1,0; **WSD**: warmup 5 % → hằng → decay tuyến tính 70–100 %; bf16 autocast + master fp32; seed 42; RNG mask theo (step, micro). Held-out 2.000 chuỗi, mask cố định seed 1234: MNTP loss + accuracy tại 0 % (= mốc BidirLM gốc trên tiếng Việt) / 50 % / 100 %. Profile 100 bước rồi khởi tạo lại; ghi throughput token/s, VRAM, GPU-giờ.

**Export:** checkpoint cuối ngân sách lưu **định dạng BidirLM** (khóa không tiền tố `model.`, kèm `config.json`, `configuration_bidirlm.py`, `modeling_bidirlm.py`, tokenizer từ snapshot pinned) tại `artifacts/bidirlm_vi_mntp/mntp/hf_bidirlm/` để `finetune.py` nạp bằng **đúng custom code như arm B2** → so sánh O/P_vi không có confound đường tính. Kiểm sau export: nạp lại bằng `AutoModel(trust_remote_code)`, đếm tham số 596 049 920, hướng hai chiều trên batch có padding (như preflight chặng 0), và hidden của model export ≡ model train trên một batch (≤ 1e-3).

## 4. Fine-tune screen (3 GPU, ~2 giờ wall) — `artifacts/bidirlm_vi_mntp/ft/`

Recipe **y hệt chặng 0 arm B2** (`scripts/qwen3_encoder_probe/finetune.py`, thêm registry `P_vi` trỏ vào export và `O` = BidirLM gốc): custom code, mean pooling, micro 16 × accum 2, head E14, `attention_dropout 0,1`, 30 epoch, maxlen 256, bf16 autocast + master fp32, AdamW wd 0,01, warmup 10 %, clip 1, best epoch theo validation, reload/prediction/ID checks. **Không test, không Hotel/Sentiment.**

| Run | Arm | Task | LR | Seed | Ghi chú |
|---|---|---|---|---|---|
| 1 | O | Restaurant | **5e-6** | 42 | chạy ngay trong lúc MNTP (GPU rảnh) |
| 2–4 | P_vi | Restaurant | 5e-6, 1e-5, 2e-5 | 42 | grid dịch xuống vì chặng 0 cho 1e-5 tốt nhất, 5e-5 tệ nhất |
| 5–6 | P_vi | ViSFD, Topic | LR tốt nhất Restaurant của P_vi | 42 | so với O ở LR 1e-5 (đã có) |

Ước tính với custom code: Restaurant ~47 phút, ViSFD ~53, Topic ~79 → ~4,7 GPU-giờ.

## 5. Cổng (validation, seed 42; O = tốt nhất trong grid {5e-6, 1e-5, 2e-5} của O)

- **Đạt:** P_vi Restaurant (LR tốt nhất) ≥ **76,5** *và* ViSFD ≥ O − 0,5 *và* Topic ≥ O − 0,5 → root mở bước tiếp (1B token; 5 task × 2 seed; có thể trên LUMI).
- **Yếu:** Restaurant trong [O + 0,2, 76,5) → chưa phân biệt được; root quyết định (thường: thêm seed 43 cho P_vi Restaurant trước khi chi thêm).
- **Không đạt:** Restaurant < O + 0,2 hoặc ViSFD/Topic tụt > 1,0 → dừng hướng "MNTP tiếng Việt liều nhỏ"; ghi held-out loss để biết model có học tiếng Việt thêm không (tách "không học được" khỏi "học nhưng không giúp task").
- Chẩn đoán kèm: P_vi − O từng task; held-out MNTP loss 0/50/100 %; LR tốt nhất của P_vi so với O.

## 6. Artifacts, báo cáo, điểm dừng

Source `scripts/bidirlm_vi_mntp/` (`prepare_data.py`, `mntp_train.py`, `export_bidirlm.py`, `preflight.py`, `run_ft.sh` — launcher copy từ `scripts/qwen3_encoder_probe/run_grid.sh`: một job/GPU, VRAM trống ≥ 20 GB, không stack lên GPU có job của pilot, pgrep neo `^\S*python scripts/bidirlm_vi_mntp/`, không dừng job người khác; GPU theo chỉ dẫn root, mặc định 0–2); dữ liệu `data/bidirlm_vi_mntp/`; artifacts `artifacts/bidirlm_vi_mntp/{data_manifest.json, preflight.json, mntp/, ft/<arm>/<task>/lr_<lr>/seed_<s>/, summary.json}`; báo cáo `docs/BIDIRLM_VI_MNTP_STAGE1_RESULT.md`: dữ liệu (counts, hash, token/từ), kiểm tương đương đường nhanh, siêu tham số, loss/held-out 0/50/100 %, throughput/VRAM/GPU-giờ, export checks, bảng O/P_vi × task × LR (val F1, best epoch, phút) kèm Bami/mmBERT/Cafe, delta P_vi − O, quyết định cổng §5, license (CC0 + Apache-2.0, không NC), giới hạn (một seed, validation-only, một liều). Không sửa chặng 0, baseline, raw gold/split; **không test, không 1B, không LUMI**. Blocker không giữ được protocol (pool không đọc được; kiểm tương đương và custom code đều lỗi; VRAM không đủ micro 16; export không nạp lại được) → báo cụ thể, dừng trước khi đổi phương pháp.
