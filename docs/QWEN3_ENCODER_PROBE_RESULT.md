# Qwen3-0.6B-Base → encoder — kết quả chặng 0 (probe đọc hai chiều) trên Restaurant / ViSFD / Topic

**Trạng thái:** đã chạy xong đợt 1–3 (15 run bắt buộc + 1 run seed 43), validation-only, seed 42/43.
Ngày chạy 2026-09-20. Recipe E14 nguyên vẹn; artifacts `artifacts/qwen3_encoder_probe/`.
Đây là probe "decoder 0,6B đọc hai chiều có bằng/hơn BamiBERT–mmBERT–CafeBERT trên validation không?".

**Kết luận ngắn:** **đi tiếp (tín hiệu mạnh, nhưng sát biên)**. Arm **B2 = BidirLM-0.6B-Base** (Qwen3-0.6B-Base + MNTP 30B token, đã đổi sang hai chiều) đạt **Restaurant 75,72 / ViSFD 84,85 / Topic 80,74**, hơn BamiBERT cả ba task (+0,98 / +0,47 / +1,21), hơn mmBERT ở Restaurant và ViSFD (+0,09 / +1,21) nhưng kém mmBERT ở Topic (−0,44) và kém CafeBERT cả ba task (−2,00 / −0,62 / −1,00). Đổi mặt nạ sang hai chiều **một mình không đủ**: B0 (Qwen3 gốc fine-tune hai chiều) *kém* chính nó một chiều (B0 − C = **−2,39** Restaurant); phần thắng nằm ở pha MNTP quy mô lớn (B2 − B0 = **+3,95**).

## 1. Môi trường, pin, sha256, license

| Thành phần | Giá trị |
|---|---|
| venv | `.venv-qwen-probe` (uv, Python 3.12.3), chạy bằng `PYTHONPATH=src`, **không** cài `-e .`; `.venv-e0-cuda` không bị đụng |
| torch | `2.6.0+cu124` (CUDA 12.4) |
| transformers | `4.57.6` (bản 4.x cuối; đúng yêu cầu `>=4.57.6,<5` của BidirLM) |
| tokenizers / numpy / safetensors / huggingface-hub | `0.22.2` / `2.5.3` / `0.8.0` / `0.36.2` |
| GPU | NVIDIA L40S 46 GB (dùng GPU 0–2; GPU 3 không đụng) |

| Arm | Repo @ commit | weights file | sha256 weights | License |
|---|---|---|---|---|
| C, B0 (B1) | `Qwen/Qwen3-0.6B-Base @ da87bfb608c14b7cf20ba1ce41287e8de496c0cd` | `model.safetensors` | `cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba` | Apache-2.0 |
| B2 | `BidirLM/BidirLM-0.6B-Base @ 29794508bbb4e378b4789eebff42ae4c2d47d524` (nhánh `transformers-v4`) | `model.safetensors` | `f17cb2979f399eb4df938a0a93230add0f564648138c01c96f3d4b84c765c1a5` | Apache-2.0 |

Cả hai đều **không NC**. Manifest đầy đủ (mọi file trong snapshot + sha256): `artifacts/qwen3_encoder_probe/manifest.json`.
BidirLM được nạp bằng `trust_remote_code=True`; `modeling_bidirlm.py` đã được đọc trước khi dùng: nó tự prepend token `151644` (`<|im_start|>`), pack varlen theo `attention_mask` (bỏ mọi vị trí pad), trả `last_hidden_state` đã dịch về đúng vị trí gốc, attention luôn hai chiều (`causal=False`) và dropout lấy từ `module.attention_dropout`.

Gọi tên đúng: **Qwen3-0.6B-Base 596 M tổng (440 M không-embedding)**; BidirLM-0.6B-Base cùng kích thước.

## 2. Tham số và định nghĩa arm

Kiểm bằng mã (`preflight.json → parameters`), cả hai checkpoint: `596 049 920` tham số tổng và `440 467 456` tham số không-embedding (embedding 151.936×1024 = 155,6 M, tied).

| Arm | Nền | Attention | Mask truyền vào encoder | Pooling | Micro-batch |
|---|---|---|---|---|---|
| **C** (đối chứng) | Qwen3-0.6B-Base | một chiều (gốc) | 2D padding mask | token cuối = EOS (`h[b, len_b−1]`) | 32 |
| **B0** | Qwen3-0.6B-Base | hai chiều | **4D bool (B,1,L,L)**, `True` = được attend, `key_ok \| eye(L)` + `is_causal=False` trên 28 module | mean trên token thật (kể cả EOS) | 32 |
| **B2** | BidirLM-0.6B-Base | hai chiều (custom code, pack varlen) | 2D padding mask (code tự pack) | mean trên `last_hidden_state` với mask gốc | 16 (accum 2) |
| B1 | Qwen3 + MNTP tiếng Việt ngắn | hai chiều | như B0 | mean | 32 — **chưa chạy** (đợt 4 tùy chọn, xem §9) |

Chung: head E14 `ClassificationHead` (input-dropout 0,1 → dense 1024 → tanh → dropout 0,1 → logits), đưa `pooled.unsqueeze(1)` để giữ nguyên hợp đồng `[:, 0]`; CE; 30 epoch; eff. batch 32; maxlen 256 (thêm EOS bằng post-processor cho C/B0/B1; B2 dùng tokenizer nguyên trạng); bf16 autocast + **master fp32**; AdamW wd 0,01 betas (0,9/0,999) eps 1e-8; warmup 10 % tuyến tính; clip 1,0; best epoch theo validation F1; seed 42 (seed 43 cho B2); **`attention_dropout = 0,1`** trên cả bốn arm (khác E14 gốc: Qwen3/BidirLM mặc định 0; đã kiểm bằng thực nghiệm dropout có tác dụng — xem §3).

## 3. Preflight (`artifacts/qwen3_encoder_probe/preflight.json`) — 47/47 check PASS

| Nhóm | Kết quả đo |
|---|---|
| Tham số + loading info | cả C/B0/B2: 596 049 920 / 440 467 456, `missing/unexpected/mismatched = 0`; dtype tham số `torch.float32` |
| **Hướng trên batch có padding** (câu 1 dài hơn câu 2; đổi token cuối câu 1 → hidden token đầu câu 1) | **C = 0,0 (một chiều)** · **B0 = 38,53 (hai chiều)** · **B2 = 2,83 (hai chiều)**; chiều xuôi (đổi token đầu → token sau) đều > 0 (C 25,78 / B0 37,29 / B2 37,80) |
| Bẫy `is_causal=False` + mask 2D | batch **có padding vẫn một chiều** (Δ = 0,0, kể cả khi đã tắt `is_causal` trên 28 module); batch **không padding** mới thành hai chiều (Δ = 8,49) → **bắt buộc dùng mask 4D**, đúng như plan §3 |
| Bất biến padding | **nội dung ở ô pad không rò vào token thật: 0,0 tuyệt đối cho cả C/B0/B2** (thay mọi ô pad bằng id rác rồi so lại). So sánh "pad tới L" với "không pad" (khác hình học batch) cho 2,40e-4 (B0) / 6,99e-5 (B2) — đây là **nhiễu thứ tự cộng của kernel**, không phải rò mask: arm **C nguyên bản còn lớn hơn (6,98e-4)**, và cùng batch-width khác batch-size đã lệch 8,3e-5; đã kiểm TF32 off và `enable_cudnn_sdp(False)` không đổi số. Vì vậy gate 1e-4 của plan §6 được áp cho **nội dung mask** (= 0), còn số liệu hình học ghi lại kèm ngưỡng cứng 1e-3 |
| Tokenizer | Qwen3+EOS: `len = 151 669`, pad = 151 643, right padding, `TemplateProcessing`; **mọi chuỗi train của cả 3 task kết thúc bằng EOS** (7 028 + 11 426 + 7 786 = 26 240 chuỗi); tỷ lệ > 256 = 0 % (max 128 / 177 / 265 token trước cắt; ViSFD truncation 0,026 % — khớp plan) |
| attention dropout 0,1 | config + **28 module** mỗi arm; kiểm chức năng: ở `train()` hai lượt liên tiếp lệch 29–69 (dropout hoạt động), đặt về 0,0 thì lệch **đúng 0** |
| head hash ban đầu | cùng seed cho cả ba arm: `8dc0af1b174bd9ac6d660cee4ea5f7bf5d95f9e5ee6349e3065273cb8a0a7f8b` |
| Smoke 1 epoch/arm (Restaurant) | C/B0/B2: loss hữu hạn, `reload_matches_best = True`, peak VRAM 15,0 / 15,0 / 12,5 GB |

## 4. Kết quả fine-tune (val F1 %, seed 42, validation-only)

### 4.1 Restaurant — đợt 1 (9 run, grid LR đầy đủ)

Ô = **val F1 / best epoch / phút-run**.

| Arm | LR 1e-5 | LR 2e-5 | LR 5e-5 | LR chọn |
|---|---|---|---|---|
| C | **74,16** / 23 / 26,1 | 73,55 / 18 / 26,4 | 72,02 / 22 / 27,5 | 1e-5 |
| B0 | 71,05 / 24 / 27,8 | **71,77** / 22 / 25,6 | 70,11 / 17 / 26,0 | 2e-5 |
| B2 | **75,72** / 26 / 47,1 | 75,13 / 12 / 45,6 | 73,89 / 16 / 46,2 | 1e-5 |

Không có run nào phân kỳ ở 5e-5 (không NaN; train loss giảm đều tới 0,0005–0,0008; chỉ Topic có 3 dao động tăng rất nhỏ ≤ 1e-3 sau epoch 5 ở cả ba arm) → không thêm LR.

### 4.2 ViSFD và Topic — đợt 2 (LR tốt nhất Restaurant của từng arm)

| Arm | LR | ViSFD | Topic |
|---|---|---|---|
| C | 1e-5 | 84,22 / 23 / 48,8 | 79,00 / 3 / 42,4 |
| B0 | 2e-5 | 83,70 / 27 / 43,6 | 79,19 / 25 / 41,7 |
| B2 | 1e-5 | **84,85** / 27 / 53,2 | **80,74** / 12 / 78,7 |

### 4.3 So với tham chiếu (validation, seed 42, cùng recipe E14)

| Task | C (một chiều) | B0 (hai chiều, không MNTP) | **B2 (BidirLM)** | BamiBERT | mmBERT-base | CafeBERT |
|---|---|---|---|---|---|---|
| Restaurant | 74,16 | 71,77 | **75,72** | 74,74 | 75,63 | 77,72 |
| ViSFD | 84,22 | 83,70 | **84,85** | 84,38 | 83,64 | 85,47 |
| Topic | 79,00 | 79,19 | **80,74** | 79,53 | 81,18 | 81,74 |

B2 − tham chiếu: Bami **+0,98 / +0,47 / +1,21**; mmBERT **+0,09 / +1,21 / −0,44**; CafeBERT **−2,00 / −0,62 / −1,00**.
B2 là arm hai chiều tốt nhất ở **cả ba** task; thứ tự B2 > C > B0 ở Restaurant và ViSFD, B2 > B0 > C ở Topic.

## 5. Chẩn đoán (plan §8, |Δ| < 1 coi là chưa phân biệt)

| Delta | Restaurant | ViSFD | Topic | Đọc |
|---|---|---|---|---|
| **B0 − C** (Q1: đổi mặt nạ ngay lúc fine-tune) | **−2,39** | −0,52 | +0,19 | đổi hai chiều **một mình** không giúp; ở Restaurant còn hại rõ |
| **B2 − B0** (Q2: MNTP quy mô lớn) | **+3,95** | +1,15 | +1,55 | pha MNTP là phần quyết định |
| B2 − C | +1,56 | +0,63 | +1,74 | nền đã MNTP hai chiều > đọc một chiều |
| (B1 − B0) | — | — | — | đợt 4 không chạy (§9) |
| seed 43 − seed 42 (B2, Restaurant, 1e-5) | **0,00** (75,72 vs 75,72; best epoch 20 vs 26) | — | — | sàn nhiễu rất nhỏ trên ô quyết định |

## 6. Đường cong (val F1 % tại LR được chọn)

| Task | Arm | ep1 | ep5 | ep10 | ep15 | ep20 | ep25 | ep30 | best (ep) |
|---|---|---|---|---|---|---|---|---|---|
| Restaurant | C | 0,00 | 70,04 | 73,31 | 72,97 | 73,38 | 72,95 | 73,64 | 74,16 (23) |
| Restaurant | B0 | 29,73 | 66,73 | 69,18 | 69,69 | 70,03 | 71,00 | 71,36 | 71,77 (22) |
| Restaurant | B2 | 40,31 | 71,86 | 72,73 | 73,94 | 75,15 | 75,29 | 74,81 | 75,72 (26) |
| ViSFD | C | 38,80 | 81,79 | 82,96 | 83,44 | 83,96 | 84,03 | 84,17 | 84,22 (23) |
| ViSFD | B0 | 55,30 | 80,65 | 82,05 | 82,88 | 83,45 | 83,65 | 83,61 | 83,70 (27) |
| ViSFD | B2 | 64,76 | 83,62 | 83,95 | 84,35 | 84,53 | 84,57 | 84,65 | 84,85 (27) |
| Topic | C | 53,50 | 78,01 | 77,86 | 78,23 | 77,24 | 77,77 | 77,84 | 79,00 (3) |
| Topic | B0 | 74,98 | 77,44 | 77,83 | 78,51 | 77,14 | 79,19 | 78,65 | 79,19 (25) |
| Topic | B2 | 78,92 | 79,53 | 80,68 | 79,72 | 78,69 | 79,98 | 79,98 | 80,74 (12) |

Hai điểm đáng chú ý: **C khởi đầu chậm hơn hẳn** (epoch 1 = 0,0 ở Restaurant rồi mới bật lên — head mới + đọc một chiều), còn **B2 khởi đầu rất nhanh** (40,3 Restaurant / 78,9 Topic ở epoch 1) nhưng đỉnh đến sớm ở Topic (ep12). Không arm nào có dấu hiệu phân kỳ.

## 7. Throughput, VRAM, GPU-giờ

| Arm | Task | giây/epoch | mẫu/giây | phút/run | peak VRAM |
|---|---|---|---|---|---|
| C | Restaurant | 51,5 | 136 | 26,1 | 19,0 GB |
| B0 | Restaurant | 54,7 | 129 | 27,8 | 19,0 GB |
| B2 | Restaurant | 93,5 | 75 | 47,1 | 13,0 GB |
| C | ViSFD | 96,4 | 81 | 48,8 | 30,0 GB |
| B0 | ViSFD | 86,1 | 90 | 43,6 | 30,0 GB |
| B2 | ViSFD | 105,2 | 74 | 53,2 | 20,2 GB |
| C | Topic | 84,5 | 135 | 42,4 | 23,2 GB |
| B0 | Topic | 82,8 | 138 | 41,7 | 23,2 GB |
| B2 | Topic | 156,8 | 73 | 78,7 | 12,6 GB |

- **Tổng: 16 run hoàn tất = 10,9 GPU-giờ** (restaurant 3×~27 → 3×~47; ViSFD/Topic như bảng), wall **4 h 39 ph** trên 3 GPU (09:19→13:59 UTC), cộng ~1,2 phút preflight trên 1 GPU. Cao hơn ước lượng plan §3 ("~8–10 GPU-giờ / 3–4 h wall"): C/B0 khớp ước lượng Restaurant (~25–28 ph), còn **B2 chậm hơn ước lượng ~1,7×** vì custom code dựng mask `T×T` cho cả batch packed ở **mỗi lớp**, và GPU 0 có job của người dùng khác chạy song song.
- VRAM thấp hơn nhiều so với trần 44 GB (thấp nhất là B2 nhờ pack varlen + micro 16); launcher giữ luật "VRAM trống ≥ 20 GB, một job/GPU, không stack lên GPU đang có job probe".
- Không có run nào vượt VRAM hay phải giảm batch; không dùng GPU 3, không dừng tiến trình của người khác.

## 8. Checks, artifacts

- **16/16 run** (15 bắt buộc + seed 43) có `postrun_checks` đạt hết: reload checkpoint **trùng** best validation F1 (sai số < 1e-9), file `validation_predictions.json` cho lại đúng F1 đó, dự đoán sau reload **giống hệt**, ID **ổn định và đủ** (771 / 1 112 / 1 583 hàng).
- Không chạy test, không chạm Hotel/Sentiment, không sửa baseline/pilot cũ, raw gold hay split.
- Artifacts: `artifacts/qwen3_encoder_probe/{manifest.json, preflight.json, summary.json, logs/, smoke/, <arm>/<task>/lr_<lr>/seed_<seed>/{resolved_config,epoch_curves,best_validation_metrics,validation_predictions,metadata,status}.json + best_checkpoint.pt}`.
- Lệnh tái lập: `PROBE_GPUS="0 1 2" bash scripts/qwen3_encoder_probe/run_grid.sh` (đợt 1→3, tự chọn GPU theo VRAM, tự bỏ qua ô đã xong); preflight: `PYTHONPATH=src .venv-qwen-probe/bin/python scripts/qwen3_encoder_probe/preflight.py --device cuda:0`.

## 9. Quyết định gate (plan §8) và đợt 4

**Gate = TÍN HIỆU MẠNH (sát biên) — đề xuất mở chặng 1.** Arm hai chiều tốt nhất là **B2** ở LR 1e-5: Restaurant **75,72 ≥ 75,6** (≥ mmBERT) ✓ — vượt ngưỡng +0,12 và hơn mmBERT +0,09; đồng thời thoả điều kiện "pass" ở cả ba task (74,0 / 83,0 / 79,0). Điều kiện mạnh thứ hai ("bất kỳ task nào ≥ CafeBERT") **không** đạt: ViSFD 84,85 < 85,47; Topic 80,74 < 81,74.

Đọc kèm: phần thắng của B2 không đến từ việc "đổi mặt nạ" (B0 − C = −2,39) mà từ **MNTP 30B token** (B2 − B0 = +3,95). Vì vậy chặng 1 nên CPT trên **BidirLM-0.6B-Base**, không phải trên Qwen3-0.6B-Base thô; và vì biên rất mỏng (0,09–0,12 điểm) cần 2 seed như plan chặng 1 đã định, không kết luận từ một seed.

**Đợt 4 (B1 — MNTP tiếng Việt ngắn trên general pool v3) KHÔNG chạy.** Lý do: đợt 1–3 đã dùng **10,9 GPU-giờ**, vượt trần ~8–10 GPU-giờ của probe; đợt 4 cần thêm ~1 GPU-giờ MNTP + 3 run Restaurant + 2 run ViSFD/Topic (~4 GPU-giờ nữa) → ~15 GPU-giờ, vượt ngân sách đã duyệt. Theo plan §7 đợt 4 là *tùy chọn*, nên dừng ở đây theo đúng prompt. Phần đã sẵn: arm B1 có trong registry (`ARMS["B1"]` với mask 4D + mean pooling) và `finetune.py` chạy được ngay khi có encoder export ở `artifacts/qwen3_encoder_probe/mntp/encoder`; **script MNTP (`mntp_train.py`) chưa được viết**. Hệ quả: câu hỏi "MNTP tiếng Việt nhỏ có thay được MNTP đa ngữ 30B không" **để ngỏ**.

## 10. Giới hạn

- **Một seed** (42) cho 15 run bắt buộc; chỉ ô quyết định B2/Restaurant/1e-5 có seed 43, cho Δ = 0,00 — sàn nhiễu *của ô đó* rất nhỏ nhưng không suy ra cho các ô khác.
- **Validation-only**, chưa chạm test; số tham chiếu Bami/mmBERT/Cafe lấy từ `artifacts/base_encoder_baseline/summary.json` cùng recipe E14/seed 42/validation.
- **Chưa CPT tiếng Việt**: B2 chỉ là bản MNTP đa ngữ 30B token; probe không luyện thích nghi tiếng Việt nên được coi là **cận dưới**.
- **Confound ở Q1**: arm C đọc token cuối (EOS) còn B0/B2 mean-pooling, nên `B0 − C` trộn "đổi hướng attention" với "đổi pooling". Thiết kế này do plan §4 chốt; muốn tách bạch cần thêm arm (ví dụ Qwen3 một chiều + mean-pooling, hoặc Qwen3 hai chiều + last-token) — ngoài ngân sách probe.
- `attention_dropout = 0,1` là khác biệt có chủ đích so với E14 gốc (0,0) theo plan §4 (Gemma Encoder), nên so sánh với tham chiếu là so sánh hai recipe khác nhau đúng một hyperparameter.
- B2 dùng micro-batch 16 (accum 2) và đường attention varlen của tác giả (mask `T×T` mỗi lớp) — cùng effective batch 32 nhưng khác hình học bước so với C/B0 (micro 32, accum 1); số epoch/optimizer steps là như nhau.
- SDPA backward dùng kernel memory-efficient (torch báo "non-deterministic algorithm") nên một lượt chạy không bitwise-deterministic giữa các lần chạy lại; mức ảnh hưởng quan sát được qua seed 43 − seed 42 là 0,00 trên ô quyết định.
