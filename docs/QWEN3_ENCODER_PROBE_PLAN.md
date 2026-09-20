# Qwen3-0.6B → encoder — chặng 0: probe "decoder đọc hai chiều" trên Restaurant / ViSFD / Topic

**Trạng thái:** plan bàn giao, chưa chạy. Ngày chốt 2026-09-20. Root đã research và **tự kiểm trước** các điểm kỹ thuật (mục 3); executor triển khai theo đặc tả. Mục tiêu: trong ~8–10 GPU-giờ (~3–4 giờ wall trên 3 GPU) trả lời *"một decoder 0,6B đã đọc 36T token đa ngữ, khi được đổi sang đọc hai chiều, có bằng/hơn BamiBERT–mmBERT–CafeBERT trên validation không?"*. Gate **mềm** theo yêu cầu người dùng: probe là bước xem có đáng theo, không phải bước chứng minh.

## 1. Câu hỏi và cơ sở

| Câu hỏi | Arm trả lời | Cơ sở |
|---|---|---|
| Q1: đổi mặt nạ một chiều → hai chiều **ngay lúc fine-tune** có lợi? | B0 − C | [Gemma Encoder (Google 2025)](https://arxiv.org/abs/2503.02656): fine-tune Gemma-2B hai chiều, không pretrain thêm, GLUE 89,4 vs 84,5 một chiều; dropout 10 % giúp |
| Q2: pha **MNTP** (che token, dự đoán từ vị trí trước) có cần không? | B2 − B0, B1 − B0 | [LLM2Vec](https://arxiv.org/abs/2404.05961): 1.000 bước MNTP đủ để quen hai chiều; bật hai chiều **không** luyện làm Llama-2 tụt. [BidirLM (04/2026)](https://arxiv.org/abs/2604.02045): adapt **chính Qwen3-0.6B** bằng MNTP 30B token → XNLI +2,7 so với hai chiều không MNTP; BidirLM-0.6B hơn EuroBERT-610m >1 điểm phân loại; BidirLM-270M ≈ mmBERT-base |
| Q3: nền này có **đáng** thay mmBERT/Bami làm điểm xuất phát cho CPT tiếng Việt? | arm tốt nhất vs tham chiếu | [ICLR 2026 2507.00994](https://arxiv.org/abs/2507.00994): CLM rồi MLM tốt hơn MLM thuần cùng ngân sách; tiếp tục MLM từ checkpoint CLM có sẵn hiệu quả. **Phản chứng** [Seq vs Seq / Ettin (ICLR 2026)](https://arxiv.org/abs/2507.11412): decoder adapt bằng 50B token MNTP vẫn thua encoder gốc **cùng dữ liệu** (MNLI: encoder 150M 89,2 vs decoder 400M 88,2). Cược của ta: encoder tiếng Việt hiện có *không* cùng dữ liệu — Qwen3 thấy nhiều tiếng Việt hơn hẳn |

Tham chiếu validation seed 42 (LR chọn theo grid E14): BamiBERT Restaurant **74,74** / ViSFD **84,38** / Topic **79,53**; mmBERT-base 75,63 / 83,64 / 81,18; CafeBERT (2e-5) 77,72 / 85,47 / 81,74 (`artifacts/base_encoder_baseline/summary.json`).

## 2. Model, license, kích thước (đã kiểm)

| Arm | Checkpoint | Pin | License | Ghi chú |
|---|---|---|---|---|
| C, B0, B1 | `Qwen/Qwen3-0.6B-Base` | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` | Apache-2.0 | **596,0M tổng / 440,5M không-embedding** (embedding 151.936×1024 = 155,6M, tied); 28 lớp, hidden 1024, GQA 16/8, head 128. `Qwen3Model.from_pretrained` load sạch (không missing/unexpected) |
| B2 | `BidirLM/BidirLM-0.6B-Base` | nhánh `transformers-v4` @ `29794508bbb4e378b4789eebff42ae4c2d47d524` | Apache-2.0 | = Qwen3-0.6B-Base + MNTP 30B token (FineWeb-Edu, FineWeb2-HQ 20 ngôn ngữ **có tiếng Việt** 59 GB, FineMath, Stack v2), merge 50/50 với gốc; custom code (`trust_remote_code`), cùng 596,0M/440,5M; tự prepend `<|im_start|>` và trả hidden dịch một vị trí (thiết kế MNTP của họ) |

Gọi đúng tên trong báo cáo: **Qwen3-0.6B-Base 596M tổng (440M không-embedding)** — trong trần 600M, ngang CafeBERT (560M) về kích thước.

## 3. Điểm kỹ thuật root đã kiểm (executor tái kiểm trong preflight, không cần khám phá lại)

- **Môi trường:** `.venv-e0-cuda` (transformers 4.49.0) **không có Qwen3**. Tạo venv riêng `.venv-qwen-probe` (uv, Python 3.12): `torch==2.6.0` (cu124, cùng wheel project), `transformers==4.57.6` (bản 4.x cuối; đúng yêu cầu nhánh transformers-v4 của BidirLM: `>=4.57.6,<5`), `numpy 2.5.x`, `tokenizers 0.22.x`, `safetensors`, `huggingface-hub`. Chạy script với `PYTHONPATH=src`. Root đã kiểm `goibert.train`, `goibert_e14.modeling`, `scripts/base_encoder_baseline/common.py` import được trong tổ hợp này, và cả hai model chạy được trên L40S. Không cài `-e .` (pyproject ghim transformers 4.49). Không đụng `.venv-e0-cuda`.
- **Hai chiều cho Qwen3 (B0/B1) — công thức đã kiểm trên model thật:** (a) `mod.is_causal = False` cho mọi module có thuộc tính đó; **và** (b) truyền **mask 4D bool** `(B,1,L,L)`: `key_ok = am.bool()[:, None, None, :].expand(B,1,L,L); mask = key_ok | eye(L)` (True = được attend; hàng pad tự attend chính nó để không có hàng toàn False), vì `create_causal_mask` trả mask 4D nguyên trạng. **Bẫy đã xác nhận:** chỉ đặt `is_causal=False` mà vẫn truyền mask 2D thì batch **không padding** thành hai chiều nhưng batch **có padding** vẫn một chiều → mọi kiểm tra hướng phải chạy trên batch có padding. Bất biến padding: chuỗi ngắn pad tới L cho hidden trùng chuỗi không pad (sai số 0 ở fp32).
- **Tokenizer Qwen3 trên 3 tập train:** 1,42 token/từ Restaurant (XLM-R 1,41, Gemma 1,42), ViSFD 1,41, Topic 1,14; **0 % vượt 256** ở cả ba (ViSFD 0,03 % — bằng Gemma) → giữ `MAX_LENGTH=256`. Qwen3 **không tự thêm BOS/EOS** (`add_special_tokens=True` cho `[58766, 139548]`), `pad = <|endoftext|> 151643`, padding_side right, `len(tokenizer)=151669 < vocab 151936`.
- **Thêm EOS:** `tok._tokenizer.post_processor = TemplateProcessing(single="$A <|endoftext|>", pair="$A <|endoftext|> $B <|endoftext|>", special_tokens=[("<|endoftext|>", 151643)])` → mọi chuỗi kết thúc bằng EOS, truncation vẫn ≤256 và giữ EOS (đã kiểm). Áp cho C/B0/B1; B2 dùng tokenizer BidirLM nguyên trạng (custom code tự prepend BOS).
- **Thông lượng L40S (bf16 autocast, master fp32, AdamW, batch 32):** Qwen3 hai chiều seq 64: 170 ms/step, 12,6 GB; seq 128: 279 ms, 17,7 GB. BidirLM (đường varlen của họ) micro 16 × seq 64: 263 ms / 32 mẫu, 13 GB; micro 32 × seq 128 lên 40 GB → **B2 dùng micro-batch 16, accum 2**. Ước Restaurant ~25 phút/run, ViSFD ~35, Topic ~30.

## 4. Arm và cấu hình chung

| Arm | Nền | Attention | Pooling | Khác |
|---|---|---|---|---|
| **C** (đối chứng) | Qwen3-0.6B-Base | một chiều (gốc, mask 2D) | token cuối = EOS (`h[b, len_b−1]`, padding right) | — |
| **B0** | Qwen3-0.6B-Base | hai chiều (mục 3) | mean trên token thật (kể cả EOS) | — |
| **B1** (tùy chọn, chạy cuối) | Qwen3-0.6B-Base + **MNTP tiếng Việt ngắn** (mục 5) | hai chiều | mean | không NC |
| **B2** | BidirLM-0.6B-Base | hai chiều (custom code) | mean trên `last_hidden_state` với attention_mask gốc | micro 16 |

Chung cho cả bốn arm (recipe E14 như `scripts/base_encoder_baseline/finetune.py`): head `ClassificationHead` E14 (input-dropout 0,1 → dense 1024 → tanh → dropout 0,1 → logits; đưa `pooled.unsqueeze(1)` vào head để giữ nguyên code `[:, 0]`), CE, 30 epoch, eff. batch 32, maxlen 256, bf16 autocast + master fp32, AdamW wd 0,01 betas (0,9, 0,999), warmup 10 % tuyến tính, clip 1,0, seed 42, best epoch theo validation. **`attention_dropout = 0.1`** trong config của cả bốn arm (Gemma Encoder: dropout 10 % là tối ưu; Qwen3 mặc định 0) — ghi rõ là khác E14 gốc. LR grid **{1e-5, 2e-5, 5e-5}** trên Restaurant cho mỗi arm (đúng protocol baseline); ViSFD và Topic chạy ở LR tốt nhất Restaurant của arm đó. Nếu 5e-5 phân kỳ (NaN/loss tăng) ghi nhận, không thêm LR.

## 5. MNTP tiếng Việt ngắn cho B1 (LLM2Vec-style, ~1 GPU-giờ, không NC)

- Dữ liệu: **general pool v3** (bins 7–9 sau lọc, cap host 2 %, 1,132B token, đã decontam 5 benchmark; CC0) — lấy docs qua `scripts/mmbert_vi_cpt_pilot/v3/audit_v3.build_general_v3` (read-only) và đọc text như `scripts/mmbert_vi_cpt_pilot/v4/build_arms_v4.py`. Lấy mẫu seed 42 đủ **~35M token Qwen** train + 2.000 chuỗi held-out; tokenize bằng tokenizer Qwen3, đóng gói 512 tách bằng EOS. Không dùng ViSoBERT/NC.
- Mask token: thêm `<mask>` (id 151669, không cần resize embedding). Mask 20 % (LLM2Vec), thay 80/10/10. Loss = CE tại **vị trí i−1 dự đoán token i bị che** (= CLM shift chuẩn với labels −100 ngoài vị trí che), lm_head tied; chỉ tính logits tại vị trí cần (gather) để tránh 32×512×152K.
- Hai chiều bằng đúng công thức mục 3 trên `Qwen3ForCausalLM.model`. **2.000 bước**, batch 32×512, AdamW 2e-5, warmup 100, decay tuyến tính, wd 0,01, clip 1, bf16 autocast, seed 42. Ghi loss/accuracy masked trên held-out ở bước 0/1000/2000. Export `model.model` làm encoder B1.

## 6. Preflight (bắt buộc trước fine-tune, ghi `artifacts/qwen3_encoder_probe/preflight.json`)

1. Manifest: repo/commit/sha256 file weights cho Qwen3 và BidirLM; phiên bản torch/transformers; đếm tham số (kỳ vọng 596,0M / 440,5M cho cả hai); loading info sạch.
2. **Kiểm hướng trên batch có padding** (2 câu tiếng Việt, câu 1 dài hơn): đổi token áp cuối câu 1 → hidden token đầu câu 1 đổi: **C = False, B0 = True, B2 = True**; bất biến padding (sai số ≤1e-4 fp32) cho B0/B2; ghi cả kết quả "bẫy mask 2D" (is_causal False + mask 2D có padding → False).
3. Tokenizer: mọi chuỗi C/B0/B1 kết thúc bằng EOS; tỷ lệ >256 ở train ba task; head hash ban đầu.
4. Smoke 1 epoch (`--smoke`) mỗi arm trên Restaurant: loss hữu hạn, reload khớp.

## 7. Lịch chạy (GPU theo chỉ dẫn root, mặc định 0–2; một job/GPU; không dừng job người khác)

| Đợt | Run | Số |
|---|---|---|
| 1 | Restaurant × {1e-5, 2e-5, 5e-5} × {C, B0, B2} | 9 |
| 2 | ViSFD, Topic × {C, B0, B2} ở LR tốt nhất Restaurant từng arm | 6 |
| 3 | Restaurant seed 43, arm tốt nhất, LR tốt nhất (sàn nhiễu) | 1 |
| 4 (tùy chọn) | MNTP B1 (1 GPU-giờ) → B1 Restaurant × 3 LR → ViSFD, Topic | 5 |

Đợt 1–3 bắt buộc (~7 GPU-giờ). Đợt 4 chỉ chạy khi đợt 1–3 xong và báo cáo bản chính đã ghi; ghi B1 thành mục riêng. Launcher kiểu `scripts/base_encoder_baseline/run_grid.sh` (chọn GPU theo VRAM trống ≥ 20 GB và không có job probe trên GPU đó; pgrep neo `^\S*python scripts/qwen3_encoder_probe/`).

## 8. Gate (mềm; validation, seed 42; Restaurant = pair micro F1)

Lấy **arm hai chiều tốt nhất** (B0 hoặc B2; B1 nếu có) ở LR tốt nhất:
- **Tín hiệu mạnh:** Restaurant ≥ **75,6** (≥ mmBERT) *hoặc* bất kỳ task nào ≥ CafeBERT → đề xuất chặng 1 ngay (MNTP tiếng Việt 250M–1B token trên nền tốt hơn trong {Qwen3-Base, BidirLM-Base}, 5 task × 2 seed).
- **Đi tiếp (pass):** Restaurant ≥ **74,0** *và* ViSFD ≥ **83,0** *và* Topic ≥ **79,0** (≈ Bami −0,7 / −1,4 / −0,5) → chặng 1 với quy mô như trên; probe không luyện thích nghi nên là cận dưới.
- **Dừng:** mọi arm hai chiều < 73,0 Restaurant ở mọi LR, hoặc không ổn định ở mọi LR → đóng hướng, ghi lý do.
- Chẩn đoán bắt buộc (không phải gate): B0 − C (Q1), B2 − B0 (Q2, MNTP quy mô lớn), B1 − B0 (MNTP tiếng Việt nhỏ), seed 43 − seed 42. |Δ| < 1 coi là chưa phân biệt.

## 9. Artifacts, báo cáo, điểm dừng

Source `scripts/qwen3_encoder_probe/` (`common.py` registry + bidir mask + pooling; `finetune.py` copy từ baseline; `mntp_train.py`; `preflight.py`; `run_grid.sh`; `summarize.py`); artifacts `artifacts/qwen3_encoder_probe/{preflight.json, manifest.json, <arm>/<task>/lr_<lr>/seed_<s>/…, mntp/, summary.json}`; báo cáo `docs/QWEN3_ENCODER_PROBE_RESULT.md`: môi trường/pin/sha, tham số, preflight (hướng, padding, EOS, độ dài), bảng arm × task × LR (val F1, best epoch, phút) kèm Bami/mmBERT/Cafe tham chiếu, delta Q1/Q2/Q3, seed 43, đường cong, throughput/GPU-giờ, checks reload/prediction/ID, license, quyết định gate §8, giới hạn (một seed, validation-only, không test, chưa CPT tiếng Việt). Không sửa baseline/pilot cũ, raw gold/split; **không test, không Hotel/Sentiment, không thêm seed/LR ngoài §7**. Gặp blocker không giữ được protocol (venv không dựng được với pin trên; preflight hướng sai; VRAM không đủ 3 job), báo cụ thể và dừng trước khi đổi phương pháp.
