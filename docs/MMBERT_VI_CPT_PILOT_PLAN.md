# mmBERT-base + Vietnamese CPT — pilot 250M token với đối chứng register/curriculum

**Trạng thái:** plan bàn giao, chưa chạy. Screening seed 42 (Restaurant thêm seed 43) trên validation; dừng tại gate. Ngày chốt 2026-09-19.

## 1. Câu hỏi và bối cảnh

Baseline `docs/BASE_ENCODER_BASELINE_RESULT.md`: mmBERT-base (`jhu-clsp/mmBERT-base @ c5955035`, **307M tổng / 110M không-embedding**) với recipe E14 đạt validation Restaurant 75,63 / Topic 81,18 / ViSFD 83,64 / Hotel 73,70 / Sentiment 86,70 — ≥ BamiBERT−0,5 trên 4/5 task, còn kém CafeBERT ~2 điểm Restaurant/ViSFD. Bước 0 (`docs/CAFEBERT_BAMIBERT_STEP0_BLIND_READING.md`) kết luận không có lỗi cụ thể để nhắm bằng cơ chế; khoảng cách còn lại phải đóng bằng pretraining tiếng Việt.

Lịch sử project: plain MLM trên web tiếng Việt ở 30–300M token **không giúp hoặc làm giảm ABSA** (E19, Pilot B, round 13) và giúp Topic. Vì vậy pilot này hỏi ba câu, tách biệt được bằng bốn arm cùng compute:

1. **Q1 (Vietnamese MLM):** mmBERT học thêm MLM tiếng Việt chất lượng cao có tăng downstream không? (P_web − O)
2. **Q2 (dữ liệu register ý kiến/review):** thêm 20% văn bản register opinion/review/thảo luận vào cùng ngân sách có giúp ABSA không? (P_mix − P_web)
3. **Q3 (curriculum/anneal):** cùng dữ liệu và số lần xuất hiện, dồn phần register ý kiến vào chặng decay có hơn trộn đều không? (C_mix − P_mix)

Tên gọi thành thật: đây là *register/domain-targeted data selection + annealing có đối chứng cùng corpus* (họ hàng với DAPT/DSIR và lịch anneal của ModernBERT/mmBERT), **không** là cơ chế mới. Giá trị nằm ở bằng chứng trên tiếng Việt và ở model phát hành. Không claim "thu thập web".

## 2. Arm và ngân sách

| Arm | Khởi tạo | Dữ liệu 250M token | Thứ tự |
|---|---|---|---|
| **O** | mmBERT-base gốc | không | — |
| **P_web** | gốc | HPLT bin 9–10 chung 95% + Wikipedia 5% | trộn đều |
| **P_mix** | gốc | HPLT chung 65% + register ý kiến/review/thảo luận 20% + HPLT bin 8 chung 10% + Wikipedia 5% | trộn đều |
| **C_mix** | gốc | **cùng document, cùng số lần xuất hiện** với P_mix | curriculum: chặng 1 (0–40% token) chỉ chung + Wikipedia; chặng 2 (40–70%) 50/50; chặng 3 (70–100%, trùng chặng decay LR) 60% register / 40% chung |

Tổng: **3 lượt CPT 250M** (mỗi lượt ~1,5–2 GPU-giờ trên L40S ước lượng, đo lại khi profile) + **15 fine-tune** (§6). Không sweep LR/mask/mixture/ngân sách. 1B token **chưa** thuộc pilot này; chỉ mở sau gate §6 và quyết định của root.

## 3. Corpus và gate dữ liệu (làm trước, không GPU train)

### Nguồn (pin và ghi manifest)

1. **HPLT 3.0 `vie_Latn`** ([card](https://huggingface.co/datasets/HPLT/HPLT3.0), HF sha `3394d6ba8dae4da834e3b11771daf95028a960b1`, [shard map](https://data.hplt-project.org/three/sorted/vie_Latn.map)): tải `10_1`, `9_1`, `9_2` trước; `8_1–8_4` và `7_1–7_3` chỉ khi cần cho bucket register hoặc bin 8. Ghi SHA256 từng file `.jsonl.zst`. Schema đã kiểm: `text`, `u` (URL), `crawl_id`, `ts`, `lang`/`prob`, `seg_langs`, `cluster_size`, `filter`, `pii` (span), `doc_scores`, `web-register` (xác suất: `MT LY SP ID NA HI IN OP IP` + sub-register gồm `re` = review). License: CC0 cho packaging; **không** sở hữu trang gốc — ghi rõ, không phát hành lại raw.
2. **Wikipedia tiếng Việt**: `wikimedia/wikipedia` config `20231101.vi` trên HF, pin revision; CC BY-SA 4.0 (ghi vào manifest mixture). Chỉ lấy đủ 5% ngân sách.
3. **Holdout MLM (diagnostic, không train):** (a) HPLT held-out theo document (0,5% pool, cap 5K doc); (b) 5K document từ shard FineWeb2 local `data/raw/fineweb2_vie/data/vie_Latn/train/000_00000.parquet` (sha `337df3c3…`) không trùng pool; (c) 2K document register ý kiến held-out. Cả ba decontam như pool.

Không dùng: MADLAD/CulturaX/OSCAR (license/overlap, không cần ở pilot), ViSoBERT (NC), corpus review nhỏ (346K token — có thể để vòng sau làm đuôi anneal, không thuộc pilot này), **text benchmark** (không bao giờ vào CPT).

### Lọc cố định (không chỉnh theo F1)

- Document: `filter == "keep"`; `lang[0] == "vie_Latn"` và `prob[0] ≥ 0,9`; ≥90% segment `vie_Latn`; loại `web-register.MT ≥ 0,5` (dịch máy); loại doc có PII span >5 hoặc >1% ký tự (ghi số); kiểm tiếng Việt theo nội dung (≥2% ký tự đặc trưng hoặc ≥15% từ chức năng, như `scripts/cafe_bami_repdistill_pilot/common.py::is_vietnamese`); loại doc <50 hoặc >20.000 token mmBERT.
- Chuẩn hóa: NFC, collapse whitespace; giữ dấu/cách viết. Không word-segment.
- Dedup: HPLT đã global-dedup; vẫn kiểm exact (casefold) trong pool và chéo Wikipedia; near-dup document bằng shingle 8 từ lấy mẫu (hash mod 16), loại nếu ≥50% shingle đã gặp. Ghi số.
- **Decontam text-only** với train/dev/test của **cả năm benchmark** (reader `scripts/cafe_bami_repdistill_pilot/common.py::benchmark_texts`, không đọc nhãn): exact key hoặc shingle 8 từ; loại **đoạn** chứa overlap, loại cả doc nếu >20% đoạn dính. Ghi số theo split.
- **Register bucket** (từ `web-register`): `opinion_review` = max(OP, re) ≥ 0,5; `discussion` = ID ≥ 0,5 (và không thuộc opinion_review); `general` = còn lại. Bucket "register ý kiến" trong mixture = opinion_review ∪ discussion, lấy từ bin 7–10, ưu tiên bin cao. **Không** dùng benchmark text để chọn document; nếu cần thu hẹp miền, chỉ dùng lexicon từ khóa cố định bốn miền (nhà hàng/khách sạn/điện thoại/học tập) ghi trong source và báo tỷ lệ.

### Audit tay (subagent đọc, ghi nhãn, không sửa dữ liệu)

100 document ngẫu nhiên (seed 42) cho mỗi ô: bin 10 general, bin 9 general, bin 8 general, `opinion_review` (bin 9–10), `opinion_review` (bin 7–8), `discussion`, `MT ≥ 0,5` (để kiểm cờ), Wikipedia. Mỗi doc gán: tiếng Việt thật (y/n), boilerplate/nav chiếm đa số (y/n), dịch máy/văn bản sinh (y/n), người lớn/spam (y/n), và với hai ô opinion: **văn bản ý kiến/review/thảo luận ở ngôi người dùng** (y/n). Gate: opinion_review precision ≥70%; general bucket ≥90% tiếng Việt thật và ≤10% boilerplate; cờ MT ≥70% đúng là dịch máy (nếu không, ghi và vẫn loại theo cờ nhưng nêu giới hạn).

### Gate dữ liệu

- Pool đã lọc/decontam ≥ **1,2B token mmBERT unique** (để chương trình 1B ≤1 pass sau này), trong đó bucket register ý kiến ≥ **240M** (đủ 20% của 1B); **pilot 250M chỉ cần ≥ 50M** register và ≥ 200M general — nếu chỉ đạt mức pilot, vẫn chạy pilot và báo rõ thiếu cho 1B.
- Mọi ô audit đạt ngưỡng trên. Manifest ghi: file/sha/revision, license, counts trước/sau từng bước, phân bố bin/register/token, decontam theo split, holdout IDs hash, mixture thực tế theo token cho từng arm, hash thứ tự batch của từng arm.
- Không đạt → viết báo cáo, dừng, không hạ ngưỡng, không đổi nguồn.

## 4. Preflight model/tokenizer (trước lượt train)

- `ModernBertForMaskedLM.from_pretrained("jhu-clsp/mmBERT-base", revision=c5955035…)`: 0 missing/mismatched key; head MLM pretrained; kiểm tie decoder/embedding; 22L/768/vocab 256.000; tokenizer Gemma 256K, pad id 0, mask id, special IDs; reload sạch. Ghi param: encoder 306,9M, không-embedding 110,3M, tổng.
- Attention: `flash_attention_2` nếu có trong venv, nếu không `sdpa`; ghi rõ và dùng **cùng** implementation cho mọi arm.
- Đóng gói: document → chuỗi 512 token, tách bằng `</s>`/`[SEP]` theo tokenizer, được phép nối nhiều doc trong một chuỗi, không cắt giữa câu nếu tránh được; không pad ngoài chuỗi cuối. Đếm non-pad thật.
- Mask 15%, 80/10/10, không special/pad; RNG mask theo (step, micro) cố định → **P_mix và C_mix cùng mask theo vị trí step** (khác data), P_web cùng seed.
- Smoke 50 step: loss/grad hữu hạn, tokens/s, VRAM; hash batch step 0–3 mỗi arm; hai lần dựng cùng arm cho cùng hash.

## 5. Recipe CPT (giống hệt cho ba arm)

- **250M non-pad token mỗi arm**, global batch 128 chuỗi × 512 (65.536 token/step) → ~3.815 step; sai số ≤1 batch; ≤1 pass trên pool của arm.
- AdamW lr **5e-5** peak, betas (0,9, 0,98), eps 1e-6, weight decay 0,01, clip 1,0; lịch **WSD**: warmup 6% → giữ hằng đến 70% → decay tuyến tính về 0 ở 70–100% (chặng 3 của C_mix trùng chặng decay). BF16 autocast, FP32 master. Seed 42.
- Checkpoint **cuối ngân sách**, không chọn bằng downstream. Log MLM loss train mỗi 20 step; holdout MLM (ba tập §3, mask cố định seed 1234) ở step 0 (=O), 40%, 70%, 100% cho mọi arm.
- Export encoder-only (`ModernBertModel`) + tokenizer; kiểm reload, không chứa head/projection lạ; smoke inference một pass.
- Profile 100 step một arm rồi **khởi tạo lại từ checkpoint gốc** trước full run; không đổi batch/lr giữa các arm. GPU theo chỉ dẫn root lúc launch (mặc định 0–2, một job/GPU), không dừng job người khác.

## 6. Fine-tune và gate

Recipe E14 y hệt baseline (`scripts/base_encoder_baseline/finetune.py` làm mẫu; head CLS E14 hidden 768; CE; 30 epoch; eff. batch 32; maxlen 256; bf16/SDPA; best epoch validation; reload/prediction/ID checks). LR **khóa từ baseline mmBERT**, không quét: Restaurant 5e-5, ViSFD 5e-5, Topic 2e-5. Không NC/KD/LLRD/mean-pooling (head giữ nguyên để so được với O/Bami/Cafe).

| Run | Task | Seed | Arm |
|---|---|---|---|
| 1–6 | Restaurant | 42, 43 | P_web, P_mix, C_mix |
| 7–9 | ViSFD | 42 | P_web, P_mix, C_mix |
| 10–12 | Topic | 42 | P_web, P_mix, C_mix |
| 13–15 | Restaurant, ViSFD, Topic | 43 | **O** (mmBERT gốc, sàn nhiễu; seed 42 reuse baseline) |

Mốc O seed 42 (validation): Restaurant 75,63 / ViSFD 83,64 / Topic 81,18; Cafe 77,72 / 85,47 / 81,74; Bami 74,74 / 84,38 / 79,53.

**Gate 250M** (validation; Restaurant dùng **trung bình ghép cặp 2 seed**, task khác seed 42):

- G1 (Q1): P_web − O ≥ +0,5 ở ≥1/3 task và không ≤ −0,5 ở Restaurant.
- G2 (Q2): P_mix − P_web ≥ +0,5 ở Restaurant.
- G3 (Q3): C_mix − P_mix ≥ +0,5 ở Restaurant.
- An toàn: arm tốt nhất không dưới O quá 0,5 ở ViSFD và Topic.

Quyết định (báo root, không tự chạy tiếp): G2∧G3 → curriculum có đóng góp, đề xuất 1B với C_mix; G2 ∧ ¬G3 → dữ liệu register có ích, thứ tự không → 1B với P_mix, đổi tên đóng góp; ¬G2 ∧ G1 → Vietnamese MLM chung giúp, register không → cân nhắc 1B P_web hoặc dừng; không G nào → dừng cấu hình này. Gate cuối 1B (nếu mở): Restaurant ≥76,5 và hơn plain-MLM cùng mixture ≥0,5; ViSFD không giảm quá 0,3 so O; ≥2/3 task tăng so O. Đây là heuristic, không significance; **không mở test** ở pilot; không cộng macro F1 Topic với micro F1 ABSA.

## 7. Artifacts, correctness và điểm dừng

- Source `scripts/mmbert_vi_cpt_pilot/`; dữ liệu `data/mmbert_vi_cpt_pilot/` (raw shard tải về đặt `data/raw/hplt3_vie/`); runs `artifacts/mmbert_vi_cpt_pilot/{data_audit, cpt/<arm>, ft/<arm>/<task>/seed_<s>}`; báo cáo `docs/MMBERT_VI_CPT_PILOT_RESULT.md` (kể cả khi dừng ở gate dữ liệu/preflight). Reuse `scripts/base_encoder_baseline/{common,finetune}.py` và helper của `scripts/cafe_bami_repdistill_pilot/` bằng adapter/copy có ghi hash; **không sửa** source/gold/artifacts cũ.
- Báo cáo phải có: provenance/hash/license, bảng audit tay, phân bố register/token, mixture thực tế và hash thứ tự từng arm, MLM/holdout losses ba tập, throughput/GPU-giờ, bảng O/P_web/P_mix/C_mix/Cafe/Bami từng task và seed, các delta G1–G3, best epoch, đường cong, checks, quyết định gate và giới hạn (một seed ngoài Restaurant; validation đã dùng chọn LR; audit bởi một người đọc).
- Gọi model đúng: **mmBERT-base 307M tổng (110M không-embedding)**. Dừng đúng gate; không seed thêm, không Hotel/Sentiment, không 1B, không mean-pooling, không thêm nguồn, không test.
