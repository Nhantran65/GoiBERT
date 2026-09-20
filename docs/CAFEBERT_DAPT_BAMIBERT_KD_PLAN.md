# Lượt D1 — CafeBERT-DAPT → BamiBERT: pilot Restaurant

Cập nhật 2026-09-17 theo quyết định mới của user. Thay thế thứ tự/gate của bản pilot trước và bản M0–M8 trong archive. Hiện chỉ cập nhật kế hoạch; chưa launch lượt D1.

## 1. Những gì đã có và câu hỏi tiếp theo

Milestone 1 hoàn tất, validation Restaurant seed42:

| Arm | Recipe | Pair F1 |
|---|---|---:|
| T | CafeBERT gốc fine-tune CE | 77,7197 |
| B | BamiBERT gốc fine-tune CE | 74,7418 |
| K | BamiBERT gốc, 0.5 CE + 0.5 KD từ T, temperature2 | 73,9440 |

Nguồn: `docs/CDBK_PILOT_RESULT.md`, `artifacts/cdbk_pilot/summary.json`; checkpoints ở `artifacts/cdbk_pilot/runs/{T,B,K}_absa_restaurant_seed42/`. Không rerun T/B/K.

KD trên train từ teacher chưa DAPT giảm0,80pp; đây không phải phép thử đầy đủ DAPT→KD. Pipeline student26M trước đây có cả DAPT và general distillation; D1 chỉ kiểm **DAPT teacher + task-logit KD**, không tái lập general distillation.

**Câu hỏi D1:** Giữ nguyên student/recipe K, thay teacher bằng CafeBERT đã DAPT có giúp student vượt BamiBERT baseline không?

**Thay đổi quyết định:** không yêu cầu K cũ pass mới được thử DAPT. Cũng không yêu cầu F1 teacher DAPT tăng mới cho chạy student; chất lượng dự đoán teacher và khả năng truyền kiến thức cần đo riêng.

## 2. Scope gọn

- Một task `absa_restaurant`, seed42, validation-only.
- Tối đa **1 DAPT run300M processed tokens + 1 teacher FT + 1 student FT**, cùng một lượt tạo cache train logits.
- Không E8/NC/LLRD, composite, sweep LR/alpha/temperature, seed mới, test hoặc task khác. Không mở thêm web-CPT control trong pilot này.
- Agent chính tự làm. **Không tạo subagent trừ khi user yêu cầu rõ**, theo AGENTS.md mới nhất.
- Tái dùng code, chỉ sửa/adapt tối thiểu trong thư mục mới. Giữ nguyên artifacts/source lượt M1.

## 3. Bước A — Chọn corpus vừa đủ và chạy DAPT

### Corpus

Chưa có corpus review/social đã xác minh cho D1. File `data/cpt/fineweb2_vie_xlmr_512.npy` hiện là **web tổng quát**, không tự đổi tên thành domain review.

- Ưu tiên một nguồn review nhà hàng/ẩm thực tiếng Việt có sẵn; nếu dùng social rộng phải ghi rõ domain mismatch. Kiểm nội dung bằng khoảng50mẫu và điều kiện sử dụng/source revision trước khi chọn. Đây là quyết định chuẩn bị dữ liệu trong scope, không phải mở dự án crawl lớn.
- Nếu không tìm được nguồn phù hợp tải được, báo thiếu dữ liệu; không âm thầm thay bằng web hoặc benchmark lặp nhiều lần.
- Ghi nguồn, số documents/token sau lọc, tỷ lệ review/social, số passes. Giữ văn bản tự nhiên (dấu, emoji, teencode); NFC, bỏ empty/boilerplate và trùng/near-duplicate rõ ràng.
- Loại overlap downstream train/val/test khỏi corpus DAPT mới. Dùng text-only decontamination đối với test, chỉ log số/hash, không đọc nhãn, chạy model hoặc phân tích test. Ghi rõ ngoại lệ truy cập test text cho loại overlap.
- Split document/duplicate-cluster trước packing, holdout1% (cap1M tokens); teacher fine-tune chỉ dùng train downstream sau đó.
- Pilot300M yêu cầu ít nhất100M training token positions sau lọc, tối đa3passes. Đây là floor thực dụng chống lặp quá mức, không định luật. Nếu thiếu, báo số thực trước training, không tự tăng repetition.

### DAPT recipe cố định

- Khởi tạo **CafeBERT pretrained gốc**, cùng revision backbone của T: `uitnlp/CafeBERT@af76fcf2a04096b2b54b348a3e4eb48253c93c5d`. Không dùng checkpoint T đã fine-tune classifier, không dùng Pilot B factorized embedding.
- Tokenizer như T: `FacebookAI/xlm-roberta-large@c23d21b0620b635a76227c604d44e43a9f0ee389`; kiểm sentencepiece/hash. Giữ nguyên vocab/architecture.
- Load pretrained MLM head và kiểm missing/unexpected keys/tied weights; không âm thầm random-init MLM head nếu source không khớp.
- Standard MLM15%, 80/10/10 mask/random/unchanged, không mask special/pad. Không diacritic dropout hay teencode augmentation.
- Seed42, seq512, global batch256, BF16, LR5e-5, AdamW betas(.9,.98), eps1e-6, wd.01 (bias/norm no decay), warmup6% rồi linear decay, clip1.
- Dùng GPUs thực sự rảnh; điều chỉnh accumulation giữ global batch256. Không đụng jobs người khác. Profile ngắn trước run; không xây controller mới.
- Train300M processed tokens (sai số≤1global batch), ghi token positions thực, masks, passes, runtime. MLM holdout loss đầu/cuối, fixed masking để so sánh. Chọn **checkpoint cuối budget** làm encoder D, không chọn theo downstream.
- Tái dùng `scripts/pilot_b/cpt_train.py` làm tham khảo, adapter input/output riêng; chú ý script gốc hardcode corpus/map. Plain MLM không cần khối tách dấu/map. Kiểm sparse MLM nếu tái dùng; save/load export đúng.

## 4. Bước B — Fine-tune teacher TD trên Restaurant

- Khởi tạo encoder từ D cuối300M, **task head mới** theo đúng code/head/RNG init của T. Không mang MLM head vào classifier.
- Giữ recipe T thực tế: seed42, CE, LR2e-5,30epochs, maxlen256, effective batch32, micro32 nếu vừa như M1, BF16/SDPA, AdamW(.9,.999),eps1e-8,wd.01,warmup10%+linear decay,clip1.
- Head CLS→dropout.1→dense1024→tanh→dropout.1→48logits. Chọn best epoch validation pairF1; save checkpoint và predictions.
- Báo TD−T, nhưng **vẫn sang bước C nếu run hợp lệ**, kể cả TD ngang/thấp hơn T. Chỉ dừng khi lỗi dữ liệu/numerics/checkpoint chưa giải quyết được; không dùng quality gate teacher để bỏ câu hỏi khả năng dạy.

## 5. Bước C — Cache TD và train student KD_D

- Cache **raw train logits** TD best checkpoint, FP32, `.eval()`/inference-only, teacher tokenizer riêng, IDs/text hash của7028train rows. Không cache validation/test làm targets.
- Student khởi tạo lại từ `models/bamibert-base-original`, **không từ K/B đã fine-tune**, không phải `models/goibert-b-dapt`.
- Giữ **đúng recipe K**: seed42, LR5e-5,30epochs, effbatch32/micro32, maxlen256, same head-init/data order, optimizer/schedule như K. Head768; không thêm NC/LLRD.
- Loss `0.5*CE(clean,gold) + 0.5*T²*KL(teacher||student)`, T2. FP32 probability math, teacher detached. E2 helper đã nhânT², không nhân lại.
- Restaurant12slots×4classes:0absent,1positive,2negative,3neutral. Softmax riêng từngslot, absent vẫn tham gia CE/KD. Primary metric pair microF1; lưu F1 đầy đủ[0,1], delta_pp=100×difference.
- Student/teacher tokenize cùng raw text bằng tokenizer riêng; join cache theo stable ID, không token ID. Không normalize thêm text ở một phía.
- Chọn best epoch validation; không chọn theo độ giống teacher trên train.

## 6. Kiểm tra tối thiểu và output

Đọc trước:

- `AGENTS.md`, `docs/CDBK_PILOT_RESULT.md`.
- `scripts/cdbk_pilot/{common,finetune,cache_teacher_logits}.py` và resolved configs M1; đây là reference chính để giữ recipe, không dựa riêng con số làm tròn trong chat.
- `src/goibert_e2/losses.py`, `cache.py`; `scripts/pilot_b/cpt_train.py`.

Chỉ cần kiểm: source/checkpoint đúng; loss hữu hạn và gradients đúng; cache ID/text hash/schema; vài hàng cache khớp teacher; best checkpoint reload và F1 recompute từ predictions khớp. Không rerun toàn bộ audit M1 hoặc viết framework mới.

Mỗi arm lưu config/source hashes, curves/log, best checkpoint (DAPT là cuối budget), validation predictions/metrics, status/runtime. Ghi rõ checkpoint model-only không resume được; không giả vờ resume bằng load weights. Nếu infra fail, giữ attempt cũ, báo trạng thái trước fresh restart ngoài run đã hoàn tất.

Đường dẫn mới:

- Code adapters: `scripts/cdbk_dapt_pilot/`.
- Corpus/manifest: `data/cdbk_dapt_pilot/`.
- Artifacts: `artifacts/cdbk_dapt_pilot/{dapt,TD,KD_D,teacher_cache}/`.
- Báo cáo: `docs/CDBK_DAPT_PILOT_RESULT.md`.

Báo cáo cuối bảng **T/B/K/TD/KD_D**; các delta TD−T, KD_D−K, **KD_D−B**; best epochs, runtime/cost, corpus counts/passes. Thêm teacher train confidence T2 tách present/absent từ cache như M1; đây là mô tả, không chứng minh nguyên nhân.

## 7. Cách đọc kết quả và điểm dừng

- **KD_D−B≥0,5pp:** tín hiệu đáng xác nhận thêm seed trong lượt sau.
- **KD_D−B từ0,2 đến<0,5pp:** tín hiệu yếu, báo chưa rõ; không tự mở sweep.
- **KD_D−B<0,2pp:** chưa có tín hiệu đạt mục tiêu; nếu KD_D>K nhưng vẫn dưới B, gọi đúng là bớt hại so KD cũ, chưa thắng baseline.
- Teacher TD tăng mà student không tăng: DAPT có thể giúp teacher nhưng recipe logit KD chưa truyền được lợi ích.
- Teacher TD không tăng nhưng student tăng: báo quan sát, cần xác nhận; không suy F1 teacher là predictor đủ cho khả năng dạy.

Đây đều là heuristic một seed, không significance. Không có matched web-CPT control nên không claim domain là nguyên nhân duy nhất; không suy mọi general distillation thất bại nếu D1 fail.

**Kết thúc D1 sau báo cáo, không tự chạy seed43/E8/composite/general distillation/test hoặc tăng token.** Lượt hiện tại chỉ cập nhật plan/prompt; prompt handoff là yêu cầu thực thi khi user dùng cho phiên tiếp theo.
