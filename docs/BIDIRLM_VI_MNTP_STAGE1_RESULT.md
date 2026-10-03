# Chặng 1 — MNTP tiếng Việt 250M token trên BidirLM-0.6B-Base: kết quả và cổng

**Trạng thái:** đã chạy xong toàn bộ (dữ liệu → MNTP → export → screen 3 task), validation-only, seed 42.
Ngày chạy 2026-09-20/21. Artifacts `artifacts/bidirlm_vi_mntp/`, dữ liệu `data/bidirlm_vi_mntp/`,
source `scripts/bidirlm_vi_mntp/`. Không chạy test, không 1B, không LUMI, không thêm seed/LR/task.

**Tiếp nối 2026-09-21:** Sentiment/Hotel và test 5 task được chạy sau đó theo yêu cầu người dùng; xem [báo cáo 5 task và test](BIDIRLM_VI_MNTP_5TASK_TEST_RESULT.md). Các câu "không test" trong tài liệu này chỉ mô tả phạm vi chặng 1 ban đầu.

**Kết luận ngắn — cổng §5 = YẾU (chưa phân biệt được).** Arm **P_vi** (BidirLM-0.6B-Base + MNTP
250M token tiếng Việt) đạt **Restaurant 76,15 / ViSFD 85,44 / Topic 80,69** (val F1, seed 42, LR 1e-5),
tức **P_vi − O = +0,43 / +0,58 / −0,04** so với chính nó không MNTP (O = 75,72 / 84,85 / 80,74).
Ngưỡng "đạt" của cổng là Restaurant ≥ **76,5**: ta đạt **76,15**, rơi vào dải **yếu [O+0,2; 76,5)**;
ViSFD và Topic không task nào tụt quá 1,0 nên không rơi vào "không đạt". Nói cách khác: liều MNTP tiếng
Việt 250M token **tăng ở Restaurant và ViSFD, gần như không đổi ở Topic; chưa đủ lớn để tách khỏi nhiễu** — MNTP rõ ràng có học
tiếng Việt (held-out loss 1,739 → 0,825; acc 0,653 → 0,805), nhưng lợi ích trên task còn nhỏ so với mục tiêu
76,5. P_vi hiện **hơn BamiBERT cả ba task**, **hơn mmBERT ở Restaurant và ViSFD**, **ngang CafeBERT ở ViSFD
(−0,03)** nhưng vẫn kém CafeBERT 1,57 điểm ở Restaurant và 1,05 ở Topic.

## 1. Môi trường, pin

| Thành phần | Giá trị |
|---|---|
| venv | `.venv-qwen-probe` (Python 3.12.3, `PYTHONPATH=src`, tái dùng từ chặng 0; không cài `-e .`) |
| torch / transformers | `2.6.0+cu124` / `4.57.6` (tokenizers 0.22.2, numpy 2.5.3) |
| Model | `BidirLM/BidirLM-0.6B-Base @ 29794508bbb4e378b4789eebff42ae4c2d47d524` (nhánh `transformers-v4`, Apache-2.0), weights `model.safetensors` sha256 `f17cb2979f399eb4df938a0a93230add0f564648138c01c96f3d4b84c765c1a5` |
| Tokenizer | BidirLM/Qwen2 (từ snapshot pinned): `len 151 669`, `pad 151 643`, `mask <|mask|> 151 663`, `eos 151 643`, right padding, `ByteLevel` post-processor |
| GPU | NVIDIA L40S 46 GB, GPU 0–2 (một job/GPU; GPU 3 không đụng) |

Gọi tên đúng: **BidirLM-0.6B-Base = Qwen3-0.6B-Base 596M tổng (440M không-embedding) + MNTP**.
`artifacts/bidirlm_vi_mntp/summary.json` ghi rõ ô nào tái dùng từ chặng 0 (`stage0_reuse`).

**Cách gọi tên dùng trong báo cáo** (để "BidirLM" không còn là hộp đen):

- **O** = *Qwen-encoder 0,6B đọc hai chiều*: encoder khởi từ **Qwen3-0.6B-Base** (decoder Qwen, 596M tổng /
  440M không-embedding, Apache-2.0), được tác giả đổi attention một chiều → **hai chiều** rồi dạy **MNTP 30B
  token đa ngữ** (FineWeb-Edu, FineWeb2-HQ 20 ngôn ngữ có tiếng Việt, FineMath, Stack v2) và merge 50/50 với bản
  gốc; checkpoint đó được phát hành dưới tên **`BidirLM/BidirLM-0.6B-Base`**. Đây là "Qwen encoder hai chiều"
  mà ta so.
- **P_vi** = **O + dạy thêm MNTP 250M token tiếng Việt** sạch (general pool v3, CC0), rồi export lại — mọi thứ
  khác (kiến trúc, tokenizer, recipe fine-tune E14, head) giữ nguyên, nên hiệu số P_vi − O đo đúng phần đóng góp
  của liều MNTP tiếng Việt.

## 2. Dữ liệu (`data/bidirlm_vi_mntp/`, `artifacts/bidirlm_vi_mntp/data_manifest.json`)

Nguồn: **general pool v3** lấy read-only qua `audit_v3.build_general_v3` (bins 7–9 sau lọc v3, cap host 2 %,
đã loại informal bucket và đã decontam 5 benchmark từ v1). **Không Wikipedia, không ViSoBERT/NC, không text
benchmark; license CC0.**

| Chỉ số | Giá trị |
|---|---|
| Docs tokenize lại | 1 226 854 (toàn bộ pool) |
| Token pool (Gemma, theo audit v3) | 1 132 387 854 |
| Token Qwen/BidirLM (đếm được) | **1 182 830 405** (tokens/word **1,309**; 903 461 082 từ) |
| Chuỗi 512 (mỗi doc + EOS 151643, nối, cắt; không BOS, không pad) | **2 310 215** |
| `order` xáo trộn seed 42 | permutation đủ 2 310 215 (đã kiểm: unique = số chuỗi) |
| Train chặng này | **488 281 chuỗi ≈ 250,0M token** (250 000 000 token = 488 281 × 512); thực dùng 488 256 chuỗi = 249 987 072 token (7 629 bước × 64) |
| Held-out | **2 000 chuỗi** = `order[488 281 : 490 281]`, tách **trước** khi train, đã kiểm **giao = 0** với train |
| sha256 | tokens.npy `08be4eea…1174`, order.npy `3d4283a5…4e26`, heldout `946d187c…0a44` |

Thời gian tokenize toàn pool: **41,7 phút** (CPU). Log: `artifacts/bidirlm_vi_mntp/logs/prepare_data.log`.

## 3. Preflight MNTP (`artifacts/bidirlm_vi_mntp/preflight.json`) — PASS

Trước khi train, đường nhanh (nạp weights BidirLM vào **Qwen3ForCausalLM chuẩn**) được kiểm với custom code:

| Check | Kết quả |
|---|---|
| Nạp weights strict (`model.*` + `lm_head` tied) | missing 0 / unexpected 0; `lm_head.weight is embed_tokens.weight` = True; **596 049 920** tham số |
| `is_causal=False` | đặt trên **28** module attention |
| Batch không pad (512 token, `attention_mask` toàn 1) | hai chiều: đổi token cuối câu → hidden token đầu đổi **1,604** |
| **Bẫy `attention_mask=None`** | **vẫn một chiều: Δ = 0,0** — với `position_ids` khác None và mask None, transformers 4.57.6 coi batch là *packed sequence* và **dựng mask nhân quả** (`allow_is_causal_skip=False`), bất kể `is_causal=False` trên mọi module (cùng họ bẫy với chặng 0, khác nguyên nhân) |
| **Tương đương với `BidirLMForMaskedLM`** (cùng batch đã mask, fp32) | **logits tại vị trí chọn: max‖Δ‖ = 2,57e-4** (ngưỡng 1e-2) · **loss: Δ = 3,6e-7** (ngưỡng 1e-3) → dùng được đường nhanh, không phải rơi về custom code |

Ghi chú kỹ thuật: custom `BidirLMModel` **bắt buộc** có `attention_mask` (nó pack theo `cu_seqlens`), nên batch
tương đương truyền mask 2D toàn 1 cho custom code và mask toàn 1 cho đường nhanh; hai đường đều đọc hai chiều.

## 4. MNTP 250M token — siêu tham số, đường loss, held-out

| Siêu tham số | Giá trị |
|---|---|
| Objective | MNTP: input `[151644] + 512 token`; chọn **20 %** vị trí; thay **80 % `<|mask|>` 151663 / 10 % token ngẫu nhiên / 10 % giữ** |
| Loss | CE tại hidden **j−1** dự đoán token gốc **j** (gather rồi `lm_head`, không tính full 513×151 936 logits); `lm_head` tied |
| Batch | global **64 × 512** (micro 16 × accum 4) = 32 768 token/bước → **7 629 bước**, 1 epoch = **250,0M token** |
| Optimizer | AdamW peak **2e-5**, betas (0,9, 0,98), eps 1e-8, wd 0,01, clip 1,0 |
| Schedule | WSD: warmup 5 % tuyến tính → hằng → decay tuyến tính về 0 trong 70–100 % |
| Precision / seed | bf16 autocast + **master fp32**; seed 42; RNG mask theo `default_rng([seed, step, micro])` |
| Profile | 100 bước đầu rồi **khởi tạo lại** model trước khi train thật |

**Đường loss (train, trung bình trong mỗi khối):**

| Bước | 0–2000 | 2000–4000 | 4000–6000 | 6000–7629 |
|---|---|---|---|---|
| loss đầu khối | 1,7392 | 1,0544 | 0,8658 | 0,8117 |
| loss cuối khối (TB 50 bước) | 0,9959 | 0,9233 | 0,8547 | 0,8240 |

**Held-out 2 000 chuỗi (mask cố định seed 1234, 204 265 token bị mask):**

| Mốc | 0 % (BidirLM gốc) | 50 % (bước 3 814) | 100 % (bước 7 629) |
|---|---|---|---|
| MNTP loss | 1,7386 | 0,9244 | **0,8246** |
| MNTP accuracy | 0,6530 | 0,7864 | **0,8053** |

→ Model **học tiếng Việt thật** (−0,914 loss, +15,2 điểm accuracy trên held-out), tức "không giúp task" (nếu có)
**không** phải vì "không học được".

## 5. Throughput, VRAM, GPU-giờ

| Hạng mục | Đo được |
|---|---|
| MNTP (đường nhanh, micro 16 × 512, bidirectional) | **2,1295 s/bước**, **15 387 token/s**, batch 32 768 token → 1,54e4 token/s |
| Peak VRAM MNTP | **32,9 GB** (micro 16; micro 32 không vừa 44 GB nên giữ micro 16 như plan) |
| Thời gian MNTP | 16 233 s = **4,509 GPU-giờ** (4,59 h wall) |
| Token toàn bộ chặng 1 | MNTP 4,51 + preflight/export ~0,01 + FT: O Restaurant 5e-6 0,76 + P_vi 5 run 4,73 → **≈ 10,0 GPU-giờ** |
| Wall toàn chặng | ≈ **7,5–8 h** (data 42 ph → MNTP 4,6 h (chạy song song run O 5e-6) → export → screen 2 đợt: 48,5 ph + 82,6 ph) |
| Peak VRAM FT (custom code, micro 16) | 13,0–13,1 GB (Restaurant/Topic), 20,2 GB (ViSFD) |

## 6. Export (`artifacts/bidirlm_vi_mntp/mntp/hf_bidirlm/`) — PASS

Checkpoint cuối (bước 7 629) được lưu lại **đúng định dạng BidirLM** để P_vi đi **đúng đường custom code như O**
(không có confound đường tính): `model.safetensors` (khóa có tiền tố `model.` như checkpoint gốc), `config.json`,
`configuration_bidirlm.py`, `modeling_bidirlm.py`, và tokenizer copy từ snapshot pinned.

| Check | Kết quả |
|---|---|
| Nạp lại `AutoModel(trust_remote_code=True)` | `BidirLMModel`, missing 0 / unexpected 0, **596 049 920** tham số |
| Hai chiều trên batch **có padding** | Δ = **5,2195** (đổi token cuối hàng dài → hidden vị trí 0 của hàng đó đổi) |
| Hidden export ≡ model train (1 batch) | max‖Δ‖ = **6,28e-4** (ngưỡng 1e-3) |
| sha256 | `model.safetensors d2cc7fe0…a2bf`; config `3c9792c7…7ad7`; `modeling_bidirlm.py b12c7ec7…f257`; tokenizer.json `d0fd37c4…4cad8` (đầy đủ trong `mntp/export_report.json`) |

## 7. Screen 3 task (val F1 %, seed 42, validation-only, recipe E14 y hệt chặng 0)

Ô = **val F1 / best epoch / phút-run**. LR grid **{5e-6, 1e-5, 2e-5}** (dịch xuống vì chặng 0 cho 1e-5 tốt nhất).

**Restaurant (grid đầy đủ):**

| Arm | 5e-6 | 1e-5 | 2e-5 | LR chọn |
|---|---|---|---|---|
| O (BidirLM gốc) | 74,11 / 29 / 45,7 | **75,72** / 26 / 47,1 | 75,13 / 12 / 45,6 | 1e-5 |
| **P_vi** (+250M VN MNTP) | 75,77 / 20 / 48,5 | **76,15** / 13 / 46,2 | 75,84 / 26 / 44,8 | 1e-5 |

**ViSFD và Topic (ở LR tốt nhất Restaurant của từng arm = 1e-5):**

| Arm | ViSFD | Topic |
|---|---|---|
| O | 84,85 / 27 / 53,2 | 80,74 / 12 / 78,7 |
| **P_vi** | **85,44** / 27 / 61,9 | 80,69 / 22 / 82,6 |

**So với tham chiếu (validation, seed 42, cùng recipe E14):**

| Task | O | **P_vi** | P_vi − O | BamiBERT | mmBERT | CafeBERT | P_vi vs Bami / mmBERT / Cafe |
|---|---|---|---|---|---|---|---|
| Restaurant | 75,72 | **76,15** | **+0,43** | 74,74 | 75,63 | 77,72 | +1,41 / +0,52 / −1,57 |
| ViSFD | 84,85 | **85,44** | **+0,58** | 84,38 | 83,64 | 85,47 | +1,06 / +1,80 / −0,03 |
| Topic | 80,74 | **80,69** | **−0,04** | 79,53 | 81,18 | 81,74 | +1,16 / −0,49 / −1,05 |

P_vi cao hơn O ở **Restaurant và ViSFD**, còn Topic thấp hơn 0,04 điểm; bảng đầy đủ + `stage0_reuse` trong
`artifacts/bidirlm_vi_mntp/summary.json`. Tất cả **10/10 ô** (5 P_vi + 5 O) có `postrun_checks` đạt:
reload trùng best F1, file prediction cho lại đúng F1, dự đoán sau reload giống hệt, ID ổn định và đủ
(7028 / 7786 / 11426 hàng train; 771 / 1112 / 1583 hàng validation).

## 8. Quyết định cổng (plan §5)

- **Đạt** (Restaurant ≥ 76,5 **và** ViSFD ≥ O−0,5 **và** Topic ≥ O−0,5): **không đạt** — Restaurant 76,15 < 76,5.
- **Yếu** (Restaurant ∈ [O+0,2; 76,5) = [75,92; 76,5)): **ĐÚNG** — 76,15 nằm trong dải; ViSFD +0,58 và Topic −0,04
  đều trong dung sai ±1,0.
- **Không đạt** (Restaurant < 75,92 hoặc ViSFD/Topic tụt > 1,0): không.

**→ Cổng = YẾU / chưa phân biệt được.** Chẩn đoán kèm (không phải cổng): P_vi − O = **+0,43 / +0,58 / −0,04**;
held-out MNTP loss **1,7386 → 0,8246**; LR tốt nhất của P_vi **trùng O (1e-5)**, và toàn bộ grid P_vi
(75,77 / 76,15 / 75,84) đều **cao hơn** grid O tương ứng (74,11 / 75,72 / 75,13) ở cùng LR — dấu dương nhất quán
nhưng biên độ nhỏ (0,4–1,7 điểm, trung bình ~+0,4 ở Restaurant).

Khuyến nghị bước tiếp (theo plan §5, root quyết): trước khi chi tiếp, **chạy seed 43 cho P_vi Restaurant ở LR 1e-5**
để đo sàn nhiễu của chính ô quyết định (chặng 0 đo được seed 43 − seed 42 = 0,00 trên ô B2/Restaurant, nên nếu
sàn nhiễu nhỏ tương tự thì +0,43 là thật nhưng vẫn dưới ngưỡng 76,5); nếu muốn cải thiện thật, liều 250M có thể
chưa đủ — nên cân nhắc 1B token (ngoài phạm vi chặng này) hoặc trộn thêm dữ liệu review/ngắn đặc thù task ABSA.

## 9. License

| Thành phần | License |
|---|---|
| Dữ liệu MNTP (general pool v3) | **CC0** (không NC, không Wikipedia, không text benchmark) |
| BidirLM-0.6B-Base (O) | **Apache-2.0** |
| P_vi (export) | Apache-2.0 + CC0 data (không NC) |
| Recipe/đánh giá | E14/E1 của repo, không đổi |

## 10. Giới hạn

- **Một seed** (42) cho toàn bộ screen; **một liều** MNTP (250M token, 1 epoch). Chưa có seed 43 cho P_vi, chưa
  quét liều 1B, chưa học rate khác 2e-5 cho MNTP.
- **Validation-only**, chưa chạm test; tham chiếu Bami/mmBERT/Cafe lấy từ `artifacts/base_encoder_baseline/`.
- **Cổng "yếu" là kết luận chưa phân biệt**, không phải bằng chứng tác dụng bằng 0: cả 3 task đều dương/không đổi
  và held-out cho thấy model học thêm tiếng Việt rõ rệt.
- P_vi được fine-tune ở **LR 1e-5** giống O; grid P_vi ở 5e-6/2e-5 cũng đều dương so với O, nên kết luận không
  phụ thuộc việc chọn LR.
- MNTP dùng **đường nhanh Qwen3ForCausalLM** thay vì custom code (đã kiểm tương đương: logits Δ 2,6e-4, loss Δ 3,6e-7);
  custom code chậm hơn ~1,7× nên không dùng. Batch MNTP **không pad** (độ dài cố định 512) — bẫy `attention_mask=None`
  đã ghi ở §3.
- Tokenizer dùng cờ mặc định của snapshot pinned (không bật `fix_mistral_regex`), giống chặng 0 để hai chặng so sánh được;
  đây là một chi tiết cần giữ nguyên nếu tái lập.
- SDPA backward dùng kernel memory-efficient (torch cảnh báo non-deterministic) nên không bitwise-deterministic.

## 11. Artifacts và lệnh tái lập

```
scripts/bidirlm_vi_mntp/{prepare_data.py,mntp_train.py,export_bidirlm.py,preflight.py,common.py,finetune.py,summarize.py,run_ft.sh}
data/bidirlm_vi_mntp/{tokens.npy,order.npy,heldout_indices.npy}
artifacts/bidirlm_vi_mntp/{data_manifest.json,preflight.json,summary.json,logs/}
artifacts/bidirlm_vi_mntp/mntp/{final_state.pt,ckpt_step_*.pt,mntp_report.json,heldout.json,export_report.json,hf_bidirlm/}
artifacts/bidirlm_vi_mntp/ft/{O,P_vi}/<task>/lr_<lr>/seed_42/{resolved_config,epoch_curves,best_validation_metrics,validation_predictions,metadata,status}.json + best_checkpoint.pt
```

```bash
# 1) dữ liệu (CPU, ~42 ph)
HF_HUB_OFFLINE=1 PYTHONPATH=src .venv-qwen-probe/bin/python scripts/bidirlm_vi_mntp/prepare_data.py
# 2) preflight tương đương đường nhanh
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 PYTHONPATH=src .venv-qwen-probe/bin/python scripts/bidirlm_vi_mntp/preflight.py --device cuda:0
# 3) MNTP 250M token (1 GPU, ~4,5 h)
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 PYTHONPATH=src .venv-qwen-probe/bin/python scripts/bidirlm_vi_mntp/mntp_train.py --device cuda:0
# 4) export + kiểm tra
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 PYTHONPATH=src .venv-qwen-probe/bin/python scripts/bidirlm_vi_mntp/export_bidirlm.py --device cuda:0
# 5) screen (đợt 1: P_vi Restaurant x 3 LR; đợt 2: ViSFD+Topic ở LR tốt nhất) — một job/GPU, GPU 0-2
FT_WAVES=1,2 bash scripts/bidirlm_vi_mntp/run_ft.sh
# (run O Restaurant LR 5e-6 chạy trong lúc MNTP: FT_WAVES=O bash scripts/bidirlm_vi_mntp/run_ft.sh)
```
