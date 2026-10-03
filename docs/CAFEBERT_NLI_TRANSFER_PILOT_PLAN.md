# CafeBERT NLI transfer — pilot v1

Ngày 2026-09-22. Người dùng cho phép lập plan và giao đúng một subagent gpt-5.6-luna, xhigh thực thi coding/training; root nghiên cứu, audit dữ liệu và review kết quả. Không delegation tiếp. Trạng thái: Gate D FAIL (97 questionable+unclear trong mẫu 300, vượt ngưỡng 30); không chạy các stage sau.

## 1. Câu hỏi và phạm vi

Liệu học quan hệ premise–hypothesis có nhãn giúp cùng một encoder CafeBERT chuyển giao tốt hơn tới năm task local so với Cafe gốc và MLM trên cùng văn bản? Không đổi architecture, không cắt vocab, không KD, synthetic hoặc thêm corpus. Pilot seed42, validation-only, không mở/tải test. Thành công pilot là tín hiệu đầu tư, không phải thắng năm task hoặc SOTA.

Nguồn bằng chứng: https://aclanthology.org/2020.acl-main.467/ và https://aclanthology.org/2020.aacl-main.56/ cho thấy intermediate training có thể giúp nhưng không đồng đều; https://arxiv.org/abs/2406.17716 là nguồn ViANLI. Lịch sử local CTI đã cải thiện probe mà không tăng downstream, vì vậy hai gate phải tách biệt.

## 2. Gate D — nguồn và chất lượng, root quyết định

- Chỉ tải uitnlp/ViANLI train/dev, pin commit API trả về trước tải, ghi SHA256 raw và README. Không tải test. Card MIT/CC BY-NC-SA bất nhất: người dùng chấp nhận nghiên cứu theo điều kiện phi thương mại, không phát hành model/dataset.
- Kiểm schema uid/premise/hypothesis/label; label chính xác entailment/neutral/contradiction. Đếm null, duplicate ID, exact duplicate pair và conflicting gold. Không tự sửa hoặc lọc nhãn theo prediction.
- Chuẩn hóa dùng NFC, casefold, gom whitespace cho fingerprints; giữ nguyên text train. Root đo sơ bộ train8012/dev1000; dev866/888 unique premise đã có ở train, exact pair overlap0. Đếm lại từ revision pinned.
- Chỉ dùng train chính thức để tạo fit/select/audit. Dev chính thức chỉ kiểm overlap, không dùng chọn checkpoint/score chính. Group connected components theo premise exact hoặc Jaccard word-5-shingle >=0.8 (premise ngắn <5 từ dùng exact). Nối thêm các nhóm có hypothesis exact trùng để hạn chế rò. Component ID = lexicographically smallest normalized premise; bucket=int(SHA256(id)[:8],16)%10:0..7fit,8select,9audit. Không split theo nhãn hoặc model output. Không claim article-held-out vì không có article ID xác nhận.
- Gate coverage: fit>=4000 rows, select/audit mỗi split>=400 và mỗi class>=80; audit>=50 premise groups có cả entailment và contradiction. Nếu không đạt dừng báo root, không tự đổi split.
- Decontam với text train/dev của năm benchmark bằng exact normalized text và word-5-shingle Jaccard>=0.8 cho từng premise/hypothesis; loại toàn component trước count. Không đọc benchmark test. Ghi không có kiểm decontam test, không claim test-uncontaminated.
- Chuẩn bị 300 mẫu train chính thức, random.Random(42),100/class, IDs/text/label; root đọc mù prediction và ghi supported/questionable/unclear kèm lý do. Không dùng các ghi chú này làm gold huấn luyện. >30 questionable+unclear hoặc mapping/systematic provenance issue => FAIL. Đây là ngưỡng vận hành và audit LLM sơ bộ, không phải chứng nhận người gán nhãn. Root phải viết quyết định trước GPU.

## 3. Gate C — triển khai và đối chứng

Reuse .venv-e0-cuda, scripts/sealion_benchmark/common.py và audited E1 metrics/optimizer/head; không sửa source/artifacts cũ. Cafe uitnlp/CafeBERT@af76fcf2a04096b2b54b348a3e4eb48253c93c5d; tokenizer FacebookAI/xlm-roberta-large@c23d21b0620b635a76227c604d44e43a9f0ee389. MasterFP32/BF16 autocast/SDPA.

Arms O=no intermediate; M=MLM; N=NLI; H=hypothesis-only NLI diagnostic (không downstream). N/H use CLS->linear1024->tanh->dropout0.1->linear3, CE; same head init. Label map entailment0,neutral1,contradiction2. H dùng empty premise + hypothesis theo pair tokenizer, không retain premise text. M dùng MLM head có sẵn từ đúng Cafe checkpoint; strict report missing/unexpected, dừng nếu MLM head pretrained không có, không tự random-init.

Intermediate: seed42, maxlen256 pair tokenization longest_first, batch effective32 (micro8 accumulation4), LR1e-5, AdamW wd0.01, warmup10%, linear decay, gradclip1, dropout0.1, 7epochs, checkpoint/eval mỗi epoch. N/M cùng fit IDs/order/pair truncation, số optimizer updates và token exposure trước mask; M dynamic15% non-special mask80/10/10, independent mask RNG. H shorter inputs: không gọi compute matched. Record processed nonpad tokens, updates, truncation rates per side, runtime/peak memory. Tail accumulation rescale đúng theo thực tế.

Checkpoint N/H select by select macroF1 earliest tie; M select by fixed-mask select MLM loss earliest tie. Audit only once after selection; không dùng downstream chọn intermediate epoch. N export encoder-only, bỏ NLI head; M bỏ MLM head. O dùng same builder.

Preflight: mapping fixture; padding and special masks; N/M ID/order equality; finite/nonzero encoder and head gradients; distinct RNG head initialization vs shuffle; reload parity; encode export parameter count; no labels outside fit; audit split disjointness. 100step profile estimate toàn queue. Dừng nếu tổng dự kiến vượt13 GPUh (gồm diagnostics/intermediate/downstream, không wall); root không tự mở ngân sách. GPUs0/1 mặc định theo quyền gần nhất; cho phép sharing nếu đủ peak measured+2GiB headroom, không cần GPU idle; không dừng job người khác. OOM retry micro4 accum8 một lần, effective32 không đổi, thư mục recovery riêng; không giảm seq length/epoch để cứu budget.

## 4. Gate M — kiểm quan hệ, rồi downstream nếu đạt

Report N/H audit macroF1, accuracy, perclass recall; majority-class fit predictor accuracy; exact all-correct rate nhóm premise có >=2labels và entailment/contradiction pair both-correct. Paired bootstrap10000 resamples seed123 theo premise component cho delta accuracy N-H. N inference empty-premise ablation chỉ là input dependence diagnostic, không gọi gold hợp lệ cho input sửa hoặc chứng minh causal neuron.

Đi tiếp downstream khi N audit accuracy >= H+5pp và >= majority+5pp, recall từng class>=0.25, và pair both-correct N>H. CI báo minh bạch, không dùng threshold như significance proof. Nếu fail dừng cấu hình7epochs này, không kết luận NLI nói chung vô ích. Root review gate trước downstream.

Downstream: O/M/N ×5 task (vsfc_sentiment,vsfc_topic,visfd,absa_restaurant,absa_hotel)=15runs seed42 mới. Same E14 Cafe head, LR2e-5,30epochs,maxlen256,effective32,micro16accum2,wd0.01,warmup10%,clip1,CE,BF16/SDPA. Cùng head init/task train order across arms; không NLI head/LLRD/augmentation; cùng kernel. Best val epoch earliest tie. MacroF1 VSFC, pairmicroF1 ABSA/ViSFD theo audited scored slots. Không lấy historical Cafe score thay O mới; không bỏ task yếu. Reload prediction/ID checks và recompute saved-prediction F1 độc lập.

Gate đầu tư: N hơn cả O và M >=0.5pp ở >=3/5task, bao phủ>=2/4datasets; không task nào dưới O hoặc M >0.5pp. Đây là gate screen heuristic, không phải mục tiêu cuối; chỉ1seed. Nếu fail dừng, nếu pass đề xuất plan >=3seeds riêng, không tự chạy. Báo cả delta từng task, mean top3epochs và last10epochs (diagnostic không chọn checkpoint lại).

## 5. Artifacts, bàn giao và stop

Code scripts/cafe_nli_transfer/; data data/cafe_nli_transfer/; artifacts artifacts/cafe_nli_transfer/; report docs/CAFEBERT_NLI_TRANSFER_PILOT_RESULT.md. Manifest provenance/hash, split map, audit sample, root gate decisions, preflight, run configs/curves/checkpoints/predictions, GPU hours và machine-readable summary. Không commit/push/release. Report FAIL ở gate nào cũng là kết quả hoàn tất hợp lệ, không chạy các stage sau để lấp ô.

Agent gửi root sau D preparation, sau C preflight và M, và cuối queue. Root tự audit/review scientific conclusions. Dùng wait/write_stdin nếu tool yêu cầu; không polling shell tiến trình; nếu tool tự thông báo kết quả thì không sleep/poll. Chỉ một execution subagent xuyên suốt.
