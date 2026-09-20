# Kiểm khả năng học cách kết hợp reader

## Câu hỏi và giới hạn

Frozen C4 với max-presence mất precision; điều đó chưa chứng minh logits của các reader không chứa thông tin có thể khai thác. Kiểm tra một combiner nhỏ có học được lợi ích riêng của clause readers hay không. Đây là diagnostic dùng phương pháp đã biết, không candidate novelty, không fine-tune encoder, không benchmark hoặc test result.

Không fit combiner trên E8 train logits rồi kết luận từ việc nó chỉ chọn full-reader: E8 đã fit train rất cao, khiến tình huống lỗi full-reader không đại diện validation. Thay vào đó dùng 3-fold cross-fitting trên official validation cho mục đích exploratory. Encoder đã được chọn checkpoint bằng validation và validation đã dùng trong research: cross-fitting này **không biến nó thành một held-out benchmark độc lập**. Không dùng kết quả để chọn test threshold hoặc báo thắng BamiBERT.

## Dữ liệu và ba arm

Student E8 Restaurant/Hotel seed42, existing FP32 per-reader caches đã audit. Arms:

1. Full-only: dùng một reader full duy nhất.
2. Clause: full + clause readers hiện có.
3. Random masks: cùng số reader và cùng số content-token positions của mỗi clause reader, nhưng lấy random subset content positions không hoàn lại cho mỗi reader; giữ BOS/EOS. Dùng seed cố định tạo từ task+rowID+readerIndex, không Python hash tùy session. Không sửa cache cũ. Full reader cùng numerical path như original C4. Random là control thêm views, không giả vờ là mệnh đề ngữ pháp hợp lệ.

Random inference trên validation only; frozen E8, BF16 outer/manual attention FP32 như reader_operating_point. Nếu provenance/mask/counts không khớp, sửa trước fit. Không train mới backbone hoặc load test.

## Feature và combiner khóa trước

Mỗi aspect slot là một bài toán riêng4classes. Input features của từng row:

- full slot logits4;
- mean/max/std của per-reader slot logits:4 mỗi loại;
- fraction readers có argmax bằng class0/1/2/3:4;
- log(reader_count):1.

Tổng21features. Với full-only, mean/max=full, std=0, reader_count=1. Công thức cùng nhau ở ba arm; không thêm handcrafted lexical features sau nhìn kết quả.

Dùng sklearn StandardScaler fit trong training fold; multinomial LogisticRegression C=1, max_iter=1000, random_state42, không class weighting. Fold KFold(n_splits=3,shuffle=True,random_state42), cùng IDs cho ba arm. Nếu training fold chỉ có1class tại aspect, dùng constant predictor cho aspect đó và báo; thiếu một số classes thì align probability columns đúng numericlabels0..3. Không grid C, không chọn epoch/threshold/hyperparameter bằng held-out fold. Argmax probabilities là decode duy nhất.

Đây là probe đơn giản, không khẳng định failure của nó loại mọi learned readout. Logit summaries cũng có thể mất thông tin về từng reader; báo rõ giới hạn này.

## Báo cáo

Lưu fold assignment, feature config/hash, original and OOF predictions, probabilities, fold fit class supports, convergence warnings và numeric failures. Tính pairF1/detection P/R, missed/spurious/wrong-polarity, cùng subgroup gold-neutral missed (không gọi toàn bộ neutral là factual) và gold-positive/negative missed. Report pooled OOF metrics và từngfold, không suy3fold thành3seeds independent.

So clause vs full-only cho information utility; clause vs random cho linguistic partition utility. Gain chỉ trên một task hoặc mộtfold không được gọi là hướng rộng. Không tự mở rộng training/model/test theo kết quả; root tự review.

## Execution

Cùng Luna xhigh hiện có, không thêm agent. Tổng tối đa0.25GPU-hour cho random inference; fits CPU, tối đa10phútCPU cho toàn bộ probes. Script scripts/research_checks/reader_learnability.py; output artifacts/research_diagnostics/reader_learnability/. No changes to E8/E17 or existing diagnostic scripts/caches. Kiểm feature/class mapping bằng fixture nhỏ trước chạy hai tasks.
