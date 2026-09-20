# BamiBERT aspect–sentiment CPT — pilot kiểm tính khả thi

**Trạng thái:** plan để bàn giao; chưa huấn luyện theo plan này. Chỉ chạy screening seed 42 trên validation và dừng ở gate. Ngày chốt: 2026-09-18.

## 1. Quyết định nghiên cứu

Mục tiêu dài hạn là một encoder **khoảng 103M**, một lượt encoder cho mỗi văn bản, vượt BamiBERT gốc và CafeBERT (~560M) trên nhiều nhất có thể trong năm benchmark local. Lượt này chỉ hỏi: **trên cùng corpus và cùng số token, che có ưu tiên cụm aspect–ý kiến khi continued pretraining BamiBERT có đem lại tín hiệu downstream hơn MLM ngẫu nhiên không?**

Đây là biến thể nhỏ lấy cảm hứng từ [SKEP (ACL 2020)](https://aclanthology.org/2020.acl-main.374/), không tái hiện đầy đủ các loss word-polarity/pair-prediction của SKEP và không nhận là cơ chế chưa ai làm. Thay đổi nằm ở **phân bố vị trí bị che trong giai đoạn pretraining**. Không thêm module khi inference. Nếu thành công, sản phẩm là checkpoint BamiBERT mới có bằng chứng cải thiện, không phải kiến trúc hoàn toàn mới.

### Vì sao chọn BamiBERT và cần đối chứng gì

| Task | BamiBERT test seed 42 | CafeBERT test seed 42 | Cafe − Bami (điểm F1) |
|---|---:|---:|---:|
| VSFC Sentiment | 83,03 | 82,36 | −0,67 |
| VSFC Topic | 78,18 | 80,77 | +2,59 |
| ViSFD | 84,50 | 85,48 | +0,98 |
| Restaurant | 74,15 | 77,39 | +3,24 |
| Hotel | 73,94 | 75,17 | +1,23 |

Local references: `docs/E14_BAMIBERT_BASELINE_RESULT.md`, `docs/SEALION_300M_BENCHMARK_RESULT.md`. VSFC dùng fixed-label macro F1; ba ABSA dùng end-to-end aspect–polarity pair micro F1. Đây là test một seed, không phải khoảng cách chắc chắn qua nhiều seed. Hai task VSFC cùng nguồn văn bản; không tính thành hai miền độc lập.

GoiBERT-B DAPT cũ = BamiBERT + plain MLM 40K steps trên corpus 1,67GB có nguồn benchmark oversample và lịch sử tiếp xúc validation text không nhãn. Nó tăng Topic validation khoảng +2,64 điểm nhưng các task khác gần như không đổi (`docs/E19_RESULT.md`, `docs/EXPERIMENTS_SUMMARY.md`). E6 plain MLM trên 33K train text làm student 26M giảm toàn bộ năm task (`docs/E6_RESULT.md`). CafeBERT + 300M social MLM không tăng Restaurant (`docs/CDBK_DAPT_PILOT_RESULT.md`). Các kết quả này không loại biến thể hiện tại, nhưng buộc phải có **arm plain MLM đồng corpus**. MLM loss giảm riêng lẻ không đủ để gọi pilot thành công.

## 2. Phạm vi và các arm

| Arm | Khởi tạo | CPT trên corpus đã khóa | Fine-tune |
|---|---|---|---|
| **O** | BamiBERT gốc | Không | Reuse baseline E14 seed 42; không train lại |
| **R** | BamiBERT gốc | Random MLM chuẩn | Restaurant, Hotel, Topic |
| **S** | BamiBERT gốc | Aspect–sentiment selective masking + MLM | Cùng ba task |

CafeBERT seed 42 là **mốc phải vượt**, không là arm phải train. Tổng tối đa sau gate dữ liệu: **2 CPT runs + 6 fine-tune runs**, một seed, không test. Restaurant và Hotel kiểm hai miền ABSA sát corpus; Topic là phép kiểm model không làm mất năng lực ngoài sentiment/ABSA và là khoảng cách lớn còn lại. ViSFD và VSFC Sentiment chỉ mở ở lượt sau nếu pilot qua gate. Một gain trên Restaurant/Hotel chưa chứng minh thắng rộng năm task.

Không KD, composite, NC, LLRD, aspect-pair hoặc thay tokenizer/head trong pilot. Inference của O/R/S đều một pass. Số tham số classifier phải được đếm lại và ghi; kỳ vọng xấp xỉ E14 103M.

## 3. Gate dữ liệu — phải xong trước khi dùng GPU train

### Nguồn ứng viên đã xác định

1. [`phamluan/vn_msa_sar_v3`](https://huggingface.co/datasets/phamluan/vn_msa_sar_v3), **chỉ split train**, chỉ văn bản thực sự tiếng Việt, các nguồn review restaurant/hotel. Card ghi CC BY 4.0. Một dòng là cặp *(câu, aspect)*, nhiều dòng thuộc cùng review; **không đếm 60.406 dòng train thành 60.406 review**. Pin revision, ghi attribution và kiểm nhãn ngôn ngữ bằng mẫu thủ công vì preview có ví dụ text Việt mang tag `en`.
2. `data/raw/fineweb2_vie/data/vie_Latn/train/000_00000.parquet`, nguồn [FineWeb2 vie_Latn](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), card ODC-By 1.0. Chỉ trích xuất đoạn **review/phản hồi ở ngôi người dùng** thuộc restaurant/hotel. Không dùng nguyên shard web chung. Kiểm nhanh 12.000 document tại thời điểm lập plan: keyword restaurant xuất hiện ở 22,9%, hotel 4,9%, nhưng 15 ví dụ nhìn tay chứa nhiều tin tức/quảng cáo/mô tả khách sạn. **Keyword hit không phải bằng chứng đúng miền.** Pin ID/hash shard và bộ lọc trước CPT.

Không dùng corpus ViSoBERT research-only/CC BY-NC, ViHoRec/CAMEO CC BY-NC, hay nguồn không xác định quyền sử dụng như `pqthinh232/vietnamese-restaurant-review-sentiment-dataset`. License dataset không tự giải quyết mọi quyền đối với trang gốc; ghi nguồn URL/TOS và giới hạn đó trong manifest. Lượt pilot không phát hành lại raw text.

### Quy trình lọc cố định trước khi nhìn downstream F1

- Chuẩn hóa HTML unescape, Unicode NFC, khoảng trắng; giữ dấu/cách viết gốc. Tách đoạn dài thành các câu/đoạn tối đa 256 **BamiBERT tokens**, không cắt giữa câu nếu tránh được. Loại trống, boilerplate rõ, spam lặp, văn bản ngoài tiếng Việt.
- Từ FineWeb2 chỉ chọn đoạn có marker restaurant/hotel **và** một biểu thức trải nghiệm/đánh giá ở ngôi người dùng; loại danh mục phòng, mô tả cơ sở lưu trú, tin tức, quảng cáo và trang SEO. Quy tắc là heuristic để lấy candidate, **không được điều chỉnh theo F1**.
- Gộp các câu của cùng `review_id` ở v3 trước khi tính số document, nhưng có thể giữ câu riêng cho MLM. Dedup theo normalized text và near-duplicate/boilerplate trên toàn candidate; split holdout CPT theo document/review ID trước khi pack/batch. Nêu rõ nếu không thể tái dựng review hoàn chỉnh từ v3.
- Decontam candidate với **text của train/validation/test cả năm task**: exact normalized match và shared 8-word shingle như D1 (`scripts/cdbk_dapt_pilot/prepare_corpus.py`). Chỉ đọc text ở split đánh giá để loại trùng, không dùng nhãn/test predictions. Ghi số bị loại theo nguồn/split. Không đưa văn bản benchmark train vào CPT: đây là phép thử chuyển tri thức từ corpus ngoài, không lặp train labels dưới tên pretraining.
- Lấy ngẫu nhiên cố định **100 đoạn/source × domain** sau lọc để chấm tay `firsthand opinion/feedback` hay `other`. Một nhóm chỉ vào corpus nếu ≥70% là firsthand opinion/feedback. Không dùng tỷ lệ từ khóa thay bước này.
- Sau lọc **và tách holdout CPT theo document**, phần train phải còn **≥4M BamiBERT non-special, non-pad tokens duy nhất**, trong đó Restaurant và Hotel mỗi miền ≥0,5M. Như vậy ngân sách 20M token ở §5 không vượt 5 passes. Đây là **ngưỡng tối thiểu cho pilot**, không phải khẳng định đủ cho model cuối. Nếu thiếu dữ liệu/chất lượng/quyền dùng, **dừng và báo gate fail, không thay bằng social/web chung hoặc tự hạ ngưỡng**.

Artifact gate: `data/bami_as_cpt_pilot/corpus_manifest.json` gồm revision/hash nguồn, giấy phép/attribution, counts trước/sau, token/domain, manual audit (chỉ ID + nhãn audit trong bản báo cáo), overlap removals, holdout IDs/hashes. Corpus R và S phải là **cùng ID, cùng thứ tự**.

## 4. Tạo tín hiệu để selective masking

Chỉ dùng **phần train CPT của v3 sau decontam và tách holdout**, không dùng nhãn của năm benchmark. Đây là weak supervision ngoài benchmark để tạo danh sách cụm từ; không đưa nhãn v3 trực tiếp vào loss CPT. Dùng `aspect` và dấu của `sentiment` để khai thác từ/cụm 1–3 âm tiết có tương quan với (a) nhóm aspect và (b) polarity; loại câu có cả nhãn dương và âm khi khai thác opinion để giảm nhập nhằng. Chuẩn hóa tần suất theo **câu duy nhất**, không theo số dòng `(sentence, aspect)`; `min_doc_freq=10`, bỏ stopwords/từ xuất hiện ở đa số câu, giới hạn top 50/nhóm aspect và top 200/dấu polarity bằng log-odds có làm trơn. Đóng băng lexicon trước CPT; lưu thuật toán, thống kê, hashes và 50 mẫu kiểm tay mỗi loại. Không map thẳng nhãn 9 aspect của v3 sang 12/34 slot benchmark.

Tokenizer **BamiBERT gốc** phải trả character offsets cho cụm đã khớp; che toàn bộ BPE tokens nằm trong cụm. Ưu tiên cặp một cụm aspect và một cụm opinion trong **cùng câu**, cách nhau tối đa 12 BamiBERT tokens; chỉ ghép trong ranh giới câu, không qua document. Nếu không có cặp, ưu tiên opinion. Không suy polarity của cặp từ khoảng cách này: đây chỉ là heuristic để chọn vị trí cần khôi phục.

S dùng **cùng tổng 15% vị trí token được chọn mỗi batch** như R (sai khác do làm tròn ≤1 token/sequence), cùng 80/10/10 mask/random/unchanged. Trên 50% sequence, tối đa 2/3 vị trí mask được dành cho cặp/cụm như trên rồi lấp phần còn lại bằng random; 50% sequence còn lại dùng random MLM hoàn toàn. Nếu không có hit hoặc cụm quá dài thì fallback random. R dùng toàn bộ vị trí random. Loss của cả hai là **MLM CE trên đúng vị trí đã chọn**, không thêm head/loss ở inference. Đây là bản **SKEP-inspired masking**, không gọi là full SKEP.

Preflight bắt buộc: kiểm 0 special/pad bị chọn; không cross-document masks; span→BPE offsets đúng trên mẫu tiếng Việt có dấu/không dấu; mask count/corruption 80/10/10; gradients finite; coverage selective ≥15% sequence có hit và ≥3% có cặp sau lọc. Nếu coverage thấp hơn, dừng trước CPT vì S sẽ gần R và không kiểm được giả thuyết. Báo thực tế tỷ lệ target mask, pair mask, fallback random theo từng domain. Nếu S tốt hơn R, pilot **chưa cô lập** lợi ích semantic khỏi whole-span/cặp masking; đối chứng random-span tương ứng thuộc vòng xác nhận sau, không thêm ở pilot.

## 5. CPT đối chứng khớp ngân sách

- `models/bamibert-base-original` đúng hash E14 `c8c8abe5…`; `RobertaForMaskedLM` với **MLM head pretrained**, tied decoder/embedding, tokenizer gốc. Kiểm missing/unexpected/mismatched keys, 12L/768/12 heads/vocab 20.481, mask/special IDs, pointer tie, reload. Không khởi tạo encoder/head mới nếu load sạch. Checkpoint R và S đều bắt đầu từ **cùng checkpoint gốc**, không nối tiếp nhau.
- Cùng corpus train/holdout, batch ID order, seed 42, optimizer, LR schedule, maxlen 256, BF16/SDPA, AdamW `lr=2e-5`, betas `(0.9,0.98)`, eps `1e-6`, weight decay `0.01`, warmup 6%, clip 1. **20M non-pad input tokens mỗi arm, không quá 5 passes**. Cùng số optimizer updates; nếu không đạt đồng thời hai điều kiện thì dừng ở gate dữ liệu. Dùng dynamic padding và đếm non-pad thực, không tính pad làm data budget.
- Profile 100 steps/smoke để đo memory, throughput và ETA; dùng GPU còn VRAM, không dừng job khác. Cố định thiết lập sau profile rồi chạy đúng hai arms. Không sweep LR/token budget/corpus filter.
- Export checkpoint **cuối ngân sách cố định**, không chọn bằng downstream validation. Đánh giá holdout MLM với **cùng random masks đã khóa** cho O/R/S, chỉ là diagnostic. Loss của S trên target masks và R trên random masks không so trực tiếp để tuyên bố S tốt hơn. Lưu SHA256 source/config/model, token count, passes, wall time, mask statistics và mọi warning.

## 6. Fine-tune và gate kết quả

Từ encoder R và S, bỏ MLM head; gắn **E14 CLS head mới**. Giữ loader/split/evaluator E14, seed 42, effective batch 32, maxlen 256, BF16/SDPA, AdamW/CE/warmup/clip như E14, 30 epochs; **Restaurant LR 5e-5, Hotel LR 5e-5, Topic LR 1e-5** (LR đã chọn từ E14). Không quét LR. R/S cùng head-init và review/data order cho từng task; xác nhận bằng hashes/smoke. O là E14 baseline cùng task/seed, không rerun. Dùng best validation epoch theo metric gốc, reload checkpoint và recompute từ predictions.

Validation references (%):

| Task | O: Bami E14 | CafeBERT | Metric |
|---|---:|---:|---|
| Restaurant | 74,7418 | 77,7197 | Pair micro F1 |
| Hotel | 73,2362 | 73,3438 | Pair micro F1 |
| Topic | 79,5328 | 81,7446 | Fixed-label macro F1 |

**Gate đầu tư** (heuristic, không phải significance): S phải hơn `max(O,R)` ít nhất **+0,5 điểm validation ở ≥2/3 task**, trong đó ≥1 task ABSA; task còn lại không thấp hơn O quá 0,5. Ngoài điểm best, báo đường cong/độ ổn định epoch. Nếu không đạt: dừng và kết luận *chưa có tín hiệu ở budget/corpus/recipe này*, không nói mọi aspect-aware CPT đều sai. Nếu đạt: **dừng tại gate**, báo root để quyết định vòng sau (seed 43 với O/R/S ghép cặp, ViSFD/Sentiment, random-span control, cuối cùng test đã khóa và latency). Không tự chạy mở rộng.

Kết quả phải có bảng O/R/S/Cafe, `S−R`, `S−O`, `S−Cafe`, best epoch, CPT tokens/runtime, FT runtime, mask coverage, data audit và giới hạn diễn giải. Không gọi validation gain là thắng SOTA, không trộn với paper F1 chưa xác minh evaluator. Test hiện đã được nhìn nhiều vòng trong project; **không mở test ở pilot này**.

## 7. Nơi ghi và điểm dừng

- Source mới: `scripts/bami_as_cpt_pilot/`; dữ liệu/manifest mới: `data/bami_as_cpt_pilot/`; runs/checks mới: `artifacts/bami_as_cpt_pilot/`.
- Báo cáo: `docs/BAMIBERT_ASPECT_SENTIMENT_CPT_PILOT_RESULT.md`, kể cả khi dừng ở gate dữ liệu.
- Giữ nguyên raw gold/split, source và artifacts E1–E22, Pilot B, CDBK. Có thể **reuse** loader/evaluator/E14 head và DAPT helpers, nhưng không sửa code cũ; adapter/source mới phải ghi hashes.
- Subagent thực thi tự làm trong scope, không tạo thêm subagent. Dừng đúng gate; không tự mở test, seed khác, CPT dài hơn, crawl thêm nguồn, hoặc chuyển backbone.
