# mmBERT Vietnamese CPT — pilot v3: "tiếng Việt văn nói" — 3 arm P_web / P_inf / C_inf, 250M token

**Trạng thái:** plan bàn giao, chưa chạy. Pilot rút gọn để **kiểm ý tưởng trước khi đầu tư bản đầy đủ**; ~4 GPU-giờ (wall ~3 giờ trên 3 GPU song song). Kế thừa v1 (`docs/MMBERT_VI_CPT_PILOT_PLAN.md`) §4–§5 recipe và v2 (`…_V2_DATA_FIX.md`) pool general; đọc `docs/MMBERT_VI_CPT_PILOT_V2_RESULT.md` để biết vì sao v2 dừng. Ngày chốt 2026-09-19.

## 1. Câu hỏi và vì sao đổi hướng

v2 cho pool general sạch **1,134B token** (spam xoay từ 0–2%, tiếng Việt 100%) nhưng không lấy được "review" từ web crawl (review nạp bằng JavaScript, không có trong HTML); bộ lọc spam S2 lại loại 38% bài văn nói thật (teencode bị coi là spam). Đo trên pool general: chỉ ~11% bài / 16% token có dấu hiệu văn nói, 67% giọng báo chí; từ điển teencode 384 dạng chỉ 32% xuất hiện ≥5 lần.

Giả thuyết được kiểm ở v3 (thay cho "review register"): **cho mmBERT học thêm tiếng Việt văn nói (diễn đàn, hỏi đáp, teencode, ngôi thứ nhất) có cải thiện ABSA/phản hồi sinh viên hơn là chỉ học tiếng Việt báo chí sạch?** Cơ sở: ViSoBERT (97M, 1 GB social) thắng BamiBERT (129 GB) ở Hotel/ViSpam; benchmark của ta gần như toàn văn nói. Tên gọi thành thật: *register-targeted continued pretraining có đối chứng cùng compute*; không claim cơ chế mới.

## 2. Arm và ngân sách pilot

| Arm | Dữ liệu 250M token (1 epoch, lịch WSD trọn vẹn) | Trả lời |
|---|---|---|
| **O** | mmBERT-base gốc (đã có: Restaurant 75,63 / ViSFD 83,64 / Topic 81,18, seed 42) | mốc |
| **P_web** | pool general v2 (bin 7–9 sau lọc, cap host 2%) 95% + Wikipedia lọc 5%, trộn đều | Q1: tiếng Việt sạch có giúp? |
| **P_inf** | cùng ngân sách, **≥30% token văn nói** (bucket §3) + 65% general + 5% Wikipedia, trộn đều | Q2: văn nói có giúp hơn P_web? |
| **C_inf** | **cùng document, cùng số lần xuất hiện** với P_inf (tổng văn nói 30%); curriculum: chặng 1 (0–40% token) văn nói **10%**, chặng 2 (40–70%) **20%**, chặng 3 (70–100%, trùng decay WSD) **66,7%** — kiểm: 0,4·0,10 + 0,3·0,20 + 0,3·0,667 = 0,30 | Q3: dồn văn nói vào chặng decay có hơn trộn đều? |

**3 lượt CPT 250M** (mỗi lượt ~1,1 GPU-giờ, chạy song song 3 GPU) + **13 fine-tune** (§5). Không sweep. Không test. Q3 chỉ diễn giải nếu Q2 đạt.

## 3. Bucket văn nói và sửa bộ lọc (không GPU, làm trước)

### 3.1 Sửa bộ lọc spam để không phạt văn nói
- Gán nhãn thêm **≥300 bài văn nói thật** (từ bucket `discussion`, host UGC, bài có cue văn nói trong general/bị loại ở v2; seed 42) là *không spam*; huấn luyện lại S2 (char 3–5-gram + S1) trên 1.200 nhãn v2 + 300 mới; S1 tham chiếu thêm text văn nói (câu review v3 CC-BY, `discussion` đã audit). Khóa ngưỡng trước khi chấm.
- **Gate tác dụng phụ:** trên 100 bài văn nói thật (đọc tay, có ≥1 teencode/emoji/ngôi thứ nhất thân mật), bộ lọc mới loại **≤15%**; trên 100 bài general bin 9, spam xoay từ vẫn **≤5%**. Không đạt → dùng S1-only cho bucket văn nói và ghi rõ.

### 3.2 Nguồn văn nói (chỉ HPLT + dữ liệu có license đã có local; không tải nguồn web mới)
- **HPLT**: (a) bài bị S2 v2 loại nhưng qua bộ lọc mới và có cue văn nói; (b) `discussion` nới **ID ≥ 0,3**; (c) bài trong general có cue văn nói (≥3 teencode từ `data/cpt/teencode_map.json`, hoặc emoji, hoặc ≥3 "mình/tui/em/tớ" + ≥1 teencode); (d) bin 5–6 (`5_1`, `6_1`, tải nếu chưa có) **chỉ** lấy bài có cue văn nói, qua bộ lọc. Mọi bài qua decontam 5 benchmark như v1.
- **Dữ liệu có license, nhỏ**: v3 review (CC BY, 208K token) và ViLexNorm (kiểm license trong card; nếu research-only → chỉ làm holdout). Không dùng ViSoBERT/NC, không dùng text benchmark.
- Định nghĩa "văn nói" đo được: bài có ≥3 teencode, hoặc ≥1 emoji, hoặc ≥3 đại từ thân mật + ≥1 teencode (cùng regex ghi trong source). Ghi tỷ lệ văn nói theo token cho từng arm.

### 3.3 Gate dữ liệu v3 (100 bài/ô, seed 42, đọc tay, cột nhãn v2 + `informal_voice`)
| Ô | Ngưỡng |
|---|---|
| bucket văn nói (gộp) | tiếng Việt thật ≥90%; **văn nói thật ≥70%**; spam ≤5%; boilerplate ≤15% |
| general bin 9 sau lọc mới | spam ≤5% |
| tác dụng phụ (3.1) | ≤15% |
| Lượng | bucket văn nói **≥75M token unique** (30% × 250M); được lặp tối đa **2 lần** nếu 37,5–75M (ghi rõ); <37,5M → dừng |
| Độ phủ teencode | top-100 dạng teencode phổ biến (theo tần suất trong bucket) mỗi dạng ≥100 lần trong arm P_inf; báo bảng |

Không đạt → báo cáo, dừng; không hạ ngưỡng, không thêm nguồn web.

## 4. Preflight và recipe (như v1 §4–§5, khác chỗ ghi)
- mmBERT-base `@ c5955035`, `ModernBertForMaskedLM` head pretrained/tied, tokenizer Gemma 256K, attention `sdpa` (như v2 preflight), đóng gói 512, mask 15% 80/10/10, RNG mask theo (step, micro) chung ba arm; P_inf và C_inf cùng multiset document (hash ghi manifest), chỉ khác thứ tự.
- **250M non-pad token mỗi arm, 1 epoch**, batch 128×512 (~3.815 step), AdamW 5e-5, betas (0,9, 0,98), wd 0,01, clip 1, **WSD: warmup 6% → hằng → decay tuyến tính 70–100%**, bf16, seed 42. Cùng số step ba arm; hash thứ tự batch ghi manifest.
- Holdout MLM (mask cố định seed 1234) ở 0/50/100%: HPLT held-out, FineWeb2 local, **văn nói held-out** (2K bài, tách trước), v3 review (không train nếu dùng làm holdout — chọn một vai trò, ghi rõ).
- Checkpoint cuối ngân sách; export encoder-only + tokenizer; reload/param checks; **không khởi động CPT trước verdict 3.3**.
- GPU theo chỉ dẫn root lúc launch (mặc định 0–2), một job/GPU, không dừng job người khác. Profile 100 step rồi khởi tạo lại.

## 5. Fine-tune (recipe E14 như baseline, LR khóa, validation-only)
| Run | Task | Seed | Arm |
|---|---|---|---|
| 1–6 | Restaurant | 42, 43 | P_web, P_inf, C_inf |
| 7–9 | ViSFD | 42 | P_web, P_inf, C_inf |
| 10–12 | Topic | 42 | P_web, P_inf, C_inf |
| 13 | Restaurant | 43 | O (sàn nhiễu; O seed 42 reuse baseline) |

LR: Restaurant 5e-5, ViSFD 5e-5, Topic 2e-5; head CLS E14; 30 epoch; reload/prediction/ID checks. Không Hotel/Sentiment, không mean-pooling, không test.

## 6. Gate quyết định (validation; Restaurant = trung bình ghép cặp seed 42/43)
- **Q2 (chính):** P_inf − P_web ≥ **+0,5** Restaurant **và** P_inf ≥ O − 0,5 ở ViSFD và Topic → **có tín hiệu văn nói** → đề xuất bản đầy đủ (3 epoch, 1,13B+, 5 task × 2 seed).
- **Q3 (chỉ khi Q2 đạt):** C_inf − P_inf ≥ +0,5 Restaurant → thứ tự có đóng góp, bản đầy đủ dùng C_inf; |Δ| < 0,5 → chỉ cần dữ liệu, bản đầy đủ dùng P_inf.
- **Q1:** P_web − O ≥ +0,5 ở ≥1/3 task và Restaurant ≥ O − 0,5 → tiếng Việt sạch có ích.
- Q1 đạt, Q2 không → bản đầy đủ chỉ với P_web. Không gì đạt → dừng, không đầu tư thêm.
Heuristic, không significance; báo cả đường cong/best epoch; |Δ| < 0,5 coi là chưa phân biệt được.

## 7. Artifacts và điểm dừng
Source `scripts/mmbert_vi_cpt_pilot/v3/`; dữ liệu `data/mmbert_vi_cpt_pilot/v3/`; artifacts `artifacts/mmbert_vi_cpt_pilot/v3/{spun_filter_v3, informal_bucket, data_audit_v3, DATA_GATE_VERDICT_V3.json, arms, cpt/<arm>, ft/<arm>/<task>/seed_<s>}`; báo cáo `docs/MMBERT_VI_CPT_PILOT_V3_RESULT.md` (kể cả khi dừng ở gate): nguồn/lượng/tỷ lệ văn nói từng arm, bảng audit, độ phủ teencode, losses/holdout bốn tập, throughput/GPU-giờ, bảng O/P_web/P_inf/C_inf (kèm Bami/Cafe), delta Q1/Q2/Q3, best epoch, checks, quyết định. Không sửa v1/v2, raw gold/split, artifacts cũ. Gọi model đúng: **mmBERT-base 307M tổng (110M không-embedding)**. Dừng tại gate; không tự chạy bản đầy đủ.
