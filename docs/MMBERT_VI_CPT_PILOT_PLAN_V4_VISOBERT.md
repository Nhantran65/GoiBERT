# mmBERT Vietnamese CPT — pilot v4: bucket văn nói từ corpus ViSoBERT (research-only), 3 arm × 250M

**Trạng thái:** plan bàn giao, chưa chạy. Thay thế v3 (`docs/MMBERT_VI_CPT_PILOT_PLAN_V3_INFORMAL.md`) sau khi v3 dừng ở gate (`docs/MMBERT_VI_CPT_PILOT_V3_RESULT.md`): ba lần lấy văn nói từ HPLT/web crawl đều thất bại vì văn bản người dùng không nằm trong HTML crawl và cue teencode/char n-gram không tách được văn nói khỏi spam. **Mọi mục không ghi ở đây giữ nguyên v3** (§2 arm, §4 recipe, §5 fine-tune, §6 gate, §7 artifacts). Ngày chốt 2026-09-19.

## 0. Ràng buộc license — đọc trước

Bucket văn nói của v4 lấy từ **corpus ViSoBERT** (`phucdev/ViSoBERT`, revision `74dbc64e92c627c70670dd91344b776c8500a104`, đã có local tại `data/cdbk_dapt_pilot/raw/`, dùng ở round 10): ~15,7M bình luận Facebook/TikTok/YouTube 2020–2022; điều khoản **research purposes only** (card `uitnlp/visobert`) / **CC BY-NC 4.0** (SEACrowd). Hệ quả, ghi vào mọi manifest/báo cáo/tên checkpoint:

- Checkpoint P_inf và C_inf của pilot này là **research-only, không phát hành, không dùng thương mại**; đặt tên thư mục `…_NC_research_only`.
- Mục đích duy nhất: **kiểm giả thuyết** "văn nói người dùng có giúp mmBERT trên ABSA/phản hồi sinh viên không". Nếu Q2 đạt, bước sau là tìm nguồn social **có license** cho bản phát hành (quyết định root); không dùng checkpoint pilot làm nền cho bản đó.
- P_web và O không chứa dữ liệu NC; kết quả Q1 vẫn dùng được cho model phát hành.

## 1. Câu hỏi (như v3)

Q1: P_web − O — MLM tiếng Việt sạch có giúp? Q2: P_inf − P_web — thêm ≥30% văn nói người dùng có giúp hơn? Q3 (chỉ khi Q2 đạt): C_inf − P_inf — dồn văn nói vào chặng decay có hơn trộn đều? Tiền nghiệm trung thực: round 10 (CafeBERT-large + 300M ViSoBERT + KD) không tăng Restaurant; mmBERT + đối chứng cùng compute là phép thử khác, kỳ vọng vừa phải.

## 2. Dữ liệu

| Thành phần | Nguồn | Đã có | Việc cần làm |
|---|---|---|---|
| General | pool general v3 (`artifacts/mmbert_vi_cpt_pilot/v3/general_pool_v3.json`, bin 7–9 sau lọc mới, cap host 2%, 1,132B token, spam 1%) | ✅ đã qua gate | reuse nguyên, không lọc lại |
| Wikipedia | `wiki_filtered.jsonl` v2 (68,9M token, ≥300 từ, bỏ stub) | ✅ | reuse |
| **Văn nói** | corpus ViSoBERT 3 parquet shard pinned (sha trong `data/cdbk_dapt_pilot/manifest.json`) | ✅ raw local | chuẩn hóa NFC, loại rỗng/không chữ, exact dedup casefold, **decontam text-only 5 benchmark** (exact + shingle 8 từ) — **cùng logic D1** `scripts/cdbk_dapt_pilot/prepare_corpus.py` nhưng **tokenize bằng mmBERT (Gemma 256K)**, không dùng mảng XLM-R cũ; tách trước 2.000 bình luận held-out; ghi counts |
| Holdout MLM | HPLT held-out (v2), FineWeb2 local, ViSoBERT held-out 2K, v3 review CC-BY (208K token, chỉ holdout) | — | mask cố định seed 1234, đo 0/50/100% |

Đóng gói ViSoBERT: nhiều bình luận ngắn nối trong một chuỗi 512, tách bằng token EOS, không xuyên chuỗi sang general (mỗi chuỗi thuần một nguồn); đếm non-pad thật.

### Gate dữ liệu v4 (100 mẫu/ô, seed 42, đọc tay)
| Ô | Ngưỡng |
|---|---|
| ViSoBERT sau prep | tiếng Việt thật ≥90%; **văn nói người dùng ≥70%**; spam/quảng cáo/người lớn ≤10% (social có nhiều rác; ghi tỷ lệ, không lọc thêm ngoài quy tắc D1) |
| Lượng | ≥75M token mmBERT unique (30% × 250M) không lặp; ước D1 cho 166M token XLM-R → kỳ vọng đủ |
| Độ phủ teencode | báo bảng top-100 dạng **sau khi loại các mục mơ hồ ≤2 ký tự và các mục đa nghĩa đã ghi ở v3** (`vs, a, s, e, t, d, i, ô, dừa, bia, seo, âu, mí, kon`); không làm điều kiện dừng, chỉ mô tả |
| Decontam | số bình luận bị loại theo split; benchmark test/dev/train text không có trong bucket |

Không đạt → báo cáo, dừng.

## 3. Arm (ngân sách 250M non-pad token, 1 epoch, WSD như v3 §4)
- **P_web**: general 95% + Wikipedia 5%, trộn đều (không NC).
- **P_inf**: ViSoBERT **30%** + general 65% + Wikipedia 5%, trộn đều.
- **C_inf**: cùng multiset với P_inf; chặng 1 (0–40%) ViSoBERT 10%, chặng 2 (40–70%) 20%, chặng 3 (70–100%, trùng decay) 66,7%.
Hash thứ tự và multiset ghi manifest; RNG mask theo (step, micro) chung ba arm. Tỷ lệ văn nói theo token của từng arm ghi thật.

## 4. Fine-tune, gate, artifacts
Như v3 §5–§7: 13 fine-tune (Restaurant seed 42+43 × 3 arm; ViSFD, Topic seed 42 × 3 arm; O seed 43 Restaurant), LR khóa, head CLS E14, validation-only; gate Q1/Q2/Q3 với ngưỡng 0,5 (Restaurant trung bình 2 seed). Source `scripts/mmbert_vi_cpt_pilot/v4/`, dữ liệu `data/mmbert_vi_cpt_pilot/v4/`, artifacts `artifacts/mmbert_vi_cpt_pilot/v4/`, báo cáo `docs/MMBERT_VI_CPT_PILOT_V4_RESULT.md` — thêm mục **License** nêu rõ tình trạng NC của P_inf/C_inf. Không sửa v1–v3, raw gold/split, artifacts cũ. Gọi model đúng: **mmBERT-base 307M tổng (110M không-embedding)**. Dừng tại gate; không bản đầy đủ, không test.
