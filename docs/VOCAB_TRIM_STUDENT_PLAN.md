# Student nhỏ từ XLM-R-large — bước 1 (cắt vocab, không mất điểm) + bước 2 (đo sàn bằng student có sẵn)

**Trạng thái:** plan bàn giao, chưa chạy. Ngày chốt 2026-09-21. Chi phí ~4 GPU-giờ + vài giờ CPU, ~4 giờ wall. Validation-only, không test (chỉ dùng **text** test để đo tỷ lệ `<unk>`, không dùng nhãn, không chọn gì theo test).

## 0. Bối cảnh và mục tiêu

XLM-R-large gốc (560M, MIT) = CafeBERT trên benchmark (Restaurant 77,0 / ViSFD 85,5 / Topic 80,6; `docs/XLMR_LARGE_BASELINE_RESULT.md`); CPT tiếng Việt ở mọi liều đã thử ≈ 0. Hướng còn lại về phía **model nhỏ**: student ≤100M vượt BamiBERT (103M: 74,74 / 84,38 / 79,53 / Hotel 73,24 / Sentiment 86,33 val) và tiệm cận mmBERT-base (307M: 75,63 / 83,64 / 81,18). Model đa ngữ mang embedding 250K token (XLM-R-large 256M = 46 % model; XLM-R-base 192M = 69 %), trong khi tiếng Việt dùng ~16 % bảng (đo: 40K bài FineWeb2-vi → 39.794 token khác nhau; 20K token phổ biến nhất phủ 99,81 %, 30K phủ 99,96 %). Kế hoạch tổng: **cắt vocab (bước 1) → đo sàn bằng student công khai (bước 2) → tự distill MiniLMv2 (bước 3, plan riêng nếu bước 2 cho tín hiệu)**.

Cơ sở: [Ushio et al., EMNLP 2023 Findings — Vocabulary Trimming](https://aclanthology.org/2023.findings-emnlp.981/) (XLM-R/mT5: cắt còn ≤40 % vocab giữ nguyên điểm; post-FT VT bền hơn pre-FT; thư viện [lm-vocab-trimmer](https://github.com/asahi417/lm-vocab-trimmer), MIT); [MiniLMv2, Findings ACL 2021](https://aclanthology.org/2021.findings-acl.188/) (student từ XLM-R-large: L12×H384 XNLI 72,9; L6×H384 69,3; XLM-R-base 76,2); [Sajjad et al. 2020 — dropping layers](https://arxiv.org/abs/2004.03844) (bỏ lớp **trên cùng** giữ điểm tốt nhất; bỏ 1/3 lớp mất ~1–2 điểm).

## 1. Bước 1 — Cắt vocab XLM-R cho tiếng Việt (CPU + 1 lượt eval, ~3 giờ)

### 1.1 Đếm tần suất (tokenizer XLM-R, `sentencepiece.bpe.model` chung của 4 model §2.1 — kiểm sha256 trùng nhau; nếu khác, cắt riêng từng model)
- Nguồn đếm: **text pool general v3** (reuse loader `scripts/bidirlm_vi_mntp/prepare_data.py` → `C2.part_texts`; lấy mẫu đều ≥ 300M token XLM-R, seed 42), **train + dev** của 5 benchmark, `data/raw/social/*` (ViCTSD, vihsd, uit-vsmec, vispam; text-only), Wikipedia v2 filtered. **Không** dùng text test để chọn.
- Tập giữ **K** = token đặc biệt {`<s>`,`<pad>`,`</s>`,`<unk>`,`<mask>`} ∪ {token có count ≥ 3 trong pool} ∪ {mọi token xuất hiện trong train/dev benchmark + social} ∪ {mọi piece 1 ký tự trong vocab thuộc bảng chữ tiếng Việt/Latin, chữ số, dấu câu, emoji có mặt}. Kỳ vọng |K| ≈ 30–45K; nếu > 45K, giữ top-45K theo tần suất pool nhưng **không bao giờ bỏ** token benchmark/social/đặc biệt. Ghi |K|, độ phủ trên pool (kỳ vọng ≥ 99,99 % lần xuất hiện) và trên train/dev.

### 1.2 Hai cách áp dụng, dùng cả hai
- **(A) Ánh xạ id** — tokenizer **giữ nguyên**; `new_id = map[old_id]`, id ngoài K → `<unk>` (3); embedding lấy đúng các hàng K theo thứ tự id cũ (token đặc biệt giữ id 0–3, `<mask>` cuối). Tokenization **giống hệt** → dùng để **chứng minh không mất điểm**.
- **(B) Tokenizer cắt** — `tokenizer.json` (model Unigram): lọc `model.vocab` theo K giữ nguyên thứ tự và score, token đặc biệt/`<mask>` đúng vị trí, `unk_id` = 3; cắt tương ứng `sentencepiece.bpe.model` (proto) nếu làm được, hoặc ghi rõ chỉ ship fast tokenizer. Đây là bản để **phát hành/fine-tune**. Đo **độ khớp tokenization** giữa (B) và gốc trên validation 5 task: % chuỗi có id (sau ánh xạ) giống hệt (kỳ vọng ≥ 99,5 %), liệt kê 20 ví dụ khác nhau.
- **Tỷ lệ `<unk>`** với (A) và (B): trên validation và **text** test của 5 task (đọc text, không đọc nhãn): % token và % câu bị ảnh hưởng (kỳ vọng < 0,05 % token). Ghi riêng, không dùng để chọn K.

### 1.3 Chứng minh không mất điểm (không train)
- Lấy checkpoint đã fine-tune `artifacts/xlmr_large_baseline/xlmr_large/absa_restaurant/lr_2e-05/seed_42/best_checkpoint.pt` (val 77,81), cắt embedding theo (A), đánh giá lại Restaurant validation với cùng loader/metric: **F1 phải bằng 77,81** (ghi max|Δ logits| — kỳ vọng ≤ 1e-5 bf16/1e-6 fp32) — đây là bằng chứng "cắt vocab = 0 mất mát". Làm thêm với (B): F1 kỳ vọng trong ±0,1 (chỉ khác ở câu re-tokenize).
- Xuất thư mục HF đã cắt (tokenizer B + `config.vocab_size=|K|` + weights cắt embedding, LM head/bias cắt tương ứng, tie giữ nguyên): `models/trimmed/{xlmr_large_vi, xlmr_base_vi, mminilmv2_l12_vi, mminilmv2_l6_vi}` (ngoài git). Ghi tham số trước/sau từng model (kỳ vọng XLM-R-large 560M → ~340M; XLM-R-base 278M → ~110M; mMiniLMv2-L12 117M → ~34M; L6 107M → ~24M).

## 2. Bước 2 — Đo sàn: student có sẵn + XLM-R-base bỏ lớp (E14, 3 task, ~3–4 GPU-giờ)

### 2.1 Ứng viên (đều dùng vocab đã cắt (B))
| Mã | Model | Pin | Kích thước sau cắt | Ý nghĩa |
|---|---|---|---|---|
| M12 | `nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large` | `d828558d1a570cbbb5e62a8dbf85c8f18bf7982a` | ~34M (21M lớp + 13M emb) | student công khai từ đúng teacher ta định dùng |
| M6 | `nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large` | `160deb78aca30f63754e512a93337ce8013a32ca` | ~24M | sàn cho cỡ 20–30M |
| B8 | XLM-R-base @ `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`, **giữ 8 lớp dưới cùng** (bỏ 4 lớp trên, Sajjad et al.), chưa distill | — | ~84M (57M + 27M) | khởi tạo dự kiến cho bước 3; sàn "chưa distill" |
| B12 | XLM-R-base cắt vocab, đủ 12 lớp | — | ~112M | kiểm cắt vocab pre-FT (kỳ vọng ≈ round 15: 73,61 ở 5e-5) |

License: XLM-R MIT; mMiniLMv2 theo UniLM/MIT (kiểm card, ghi vào báo cáo).

### 2.2 Recipe và run
E14 như `scripts/base_encoder_baseline/finetune.py`: CLS → dropout 0,1 → dense(hidden của model: 384 hoặc 768) → tanh → dropout 0,1 → logits; CE; 30 epoch; eff. batch 32; maxlen 256; bf16/SDPA; AdamW wd 0,01; warmup 10 %; clip 1; best epoch theo validation; reload/prediction/ID checks. LR grid: B8/B12 **{1e-5, 2e-5, 5e-5}**; M12/M6 **{2e-5, 5e-5, 1e-4}** (model hẹp cần LR cao hơn). Seed 42.

| Đợt | Run |
|---|---|
| 1 | Restaurant × 3 LR × {M12, M6, B8, B12} = 12 |
| 2 | ViSFD, Topic × 4 model ở LR tốt nhất Restaurant = 8 |
| 3 | Restaurant seed 43 cho M12 và B8 ở LR tốt nhất = 2 |

Ước tính: M6 ~4 phút/run, M12 ~6, B8 ~7, B12 ~10 → ≤ 4 GPU-giờ. Launcher như `run_grid.sh` (một job/GPU, VRAM trống ≥ 20 GB, không stack, pgrep neo `^\S*python scripts/vocab_trim_student/`, GPU theo chỉ dẫn root, mặc định 0–2, không dừng job người khác).

### 2.3 Đọc kết quả (validation, seed 42; tham chiếu Bami 74,74 / 84,38 / 79,53 · mmBERT 75,63 / 83,64 / 81,18 · XLM-R-large 77,81 / 85,48 / 80,61)
- **B12 ≈ 73,6 ± 0,5** → cắt vocab pre-FT không mất điểm trên model chưa fine-tune (kiểm bổ sung cho §1.3).
- **B8 ≥ 74,0 Restaurant** → bước 3 (MiniLMv2 từ XLM-R-large, student khởi tạo B8, ~84M) có cơ sở đạt ≥ 75,5; B8 < 72 → cân nhắc giữ 10 lớp hoặc khởi tạo khác.
- **M12 ≥ 73,0** → mục tiêu ~34M là thực tế; M6 chỉ mô tả.
- Bất kỳ ứng viên nào ≥ Bami ở ≥ 2/3 task → ghi đậm, là kết quả tự thân ("student ≤100M ≥ Bami 103M" mà chưa distill riêng).
- |Δ| < 1,0 coi là chưa phân biệt; không claim significance.

## 3. Artifacts, báo cáo, điểm dừng
Source `scripts/vocab_trim_student/` (`count_vocab.py`, `trim.py` (A+B), `verify_equivalence.py`, `finetune.py` copy từ baseline với registry 4 model + tham số `--keep-layers`, `run_grid.sh`, `summarize.py`); artifacts `artifacts/vocab_trim_student/{vocab_counts.json, keep_set.json, trim_report.json (unk, khớp tokenization, tham số trước/sau, F1 tương đương), <model>/<task>/lr_<lr>/seed_<s>/…, summary.json}`; model cắt tại `models/trimmed/` (ngoài git). Báo cáo `docs/VOCAB_TRIM_STUDENT_RESULT.md`: |K|/độ phủ/unk/khớp tokenization, bằng chứng §1.3 (F1 bằng nhau), bảng tham số, bảng 4 model × task × LR (val F1, best epoch, phút) kèm tham chiếu, seed 43, đọc theo §2.3, license, giới hạn (1–2 seed, validation-only, chưa distill). **Không test (ngoài đo unk trên text), không Hotel/Sentiment, không distill, không sửa artifacts cũ.** Blocker (không tải được checkpoint đúng pin; §1.3 không bằng nhau; tokenizer cắt không load được) → báo cụ thể, dừng.
