# GoiTurn-S được tạo như thế nào?

## Quy trình bản Stage P

1. Corpus tự dựng: báo cáo ghi 1.67 GB / 9,586,054 dòng. Kết hợp phần CC100-vi và text VSFC, ViSFD, VLSP2018 Restaurant, PhoATIS; domain text lặp 20 lần.
2. BamiBERT được continued pretraining bằng MLM, mask 30%, 40,000 steps, tạo `goibert_b_dapt` làm teacher. Đây là BamiBERT derivative cùng kiến trúc, không phải student.
3. Tạo student RoBERTa cổ điển: 10 layers, hidden 384, 6 heads, FFN 1536; vocabulary 20,481; max_position_embeddings 2050; độ dài train cũ 256.
4. Khởi tạo word embedding bằng PCA 768→384 từ teacher được chọn, chuẩn hóa và rescale; Transformer layers khởi tạo ngẫu nhiên. Không phải cắt bớt hai layer của BamiBERT.
5. General distillation 30,000 steps trên corpus: teacher cố định; student khớp hidden states qua projection và attention trung bình theo head. Kết quả là backbone `goiturn_s_gd` cần kế thừa.
6. Mỗi dataset có checkpoint task riêng: thêm classification/ABSA head và task distillation từ teacher đã fine-tune. Loss gồm CE nhãn thật, KL đầu ra teacher (T=2), hidden MSE và attention MSE. Code dùng KL(student || teacher); phải ghi đúng nếu port sang pipeline mới.
7. Biến thể `_nc` thêm noise-consistency giữa văn bản sạch và bản nhiễu. Đây là biến thể training, không phải tên kiến trúc mới.

## Checkpoint cần lấy

Mục tiêu là encoder Hugging Face ở `experiments/goibert-family-v1/results/pretrained/goiturn_s_gd/`, không phải checkpoint 3 nhãn FINISHED/UNFINISHED/WAIT và không phải checkpoint sentiment/ABSA đã fine-tune.

Đã nhận GD-S ở `models/goiturn-s-gd/`: manifest ghi completed_steps=30000, teacher DAPT, teacher_fallback_used=false. Teacher tại `models/goibert-b-dapt/`, manifest ghi 40,000 steps. Đây là metadata checkpoint đã nhận, chưa phải reproduce training.

## Giới hạn của lịch sử

- `MODEL_SPECS.md` cũ mô tả Stage A trước khi DAPT/GD hoàn tất. Đọc cùng REPORT.md và summary mới hơn; không coi tất cả nhận định trong spec cũ là hiện trạng.
- `build_corpus.py` dùng train+validation text của dataset. Encoder thừa hưởng việc tiếp xúc validation không nhãn; không thể đảo ngược bằng cách fine-tune lại. Ghi rõ provenance này; muốn đối chứng train-only cần tạo lại backbone thích hợp.
- Báo cáo nói corpus/held-out được kiểm soát, nhưng gói này không có dữ liệu hoặc hashes gốc để audit độc lập contamination. Cần kiểm tra trước khi tuyên bố benchmark.
- Điểm benchmark lịch sử bao gồm bước task training, không phải điểm của encoder sau bước 5 dùng trực tiếp.

Code: `reference/ai-foundary/experiments/goibert-family-v1/src/`.
