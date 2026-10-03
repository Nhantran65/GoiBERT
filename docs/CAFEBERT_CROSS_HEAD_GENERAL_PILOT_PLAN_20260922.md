# CafeBERT: pilot kiến trúc attention dùng chung cho năm task

**Trạng thái 2026-09-22: screening ba task hoàn tất, gate KHÔNG ĐẠT.** [Kết quả](CAFEBERT_CROSS_HEAD_SCREEN_RESULT.md). Không mở hai task/seed tiếp theo hoặc test. Nội dung dưới đây là đặc tả đã dùng trước khi xem validation.

## Giả thuyết và giới hạn bằng chứng

Trong attention nhiều head của CafeBERT, mỗi head tự tính score trước softmax. Cho các head trao đổi score trước và trọng số sau softmax có thể kết hợp các kiểu truy hồi đã pretrained. Đây là [Talking-Heads Attention](https://arxiv.org/html/2003.02436), đã có từ 2020. Bài báo cho BERT được **pretrain với operator đó** và cải thiện MNLI/SQuAD; chưa chứng minh gắn operator vào checkpoint CafeBERT pretrained rồi fine-tune sẽ giúp. E11/E15 của project không kiểm đúng phép trộn *giữa các head bên trong block*. [Gap review](ENCODER_REPLACEMENT_GAP_RESEARCH.md), [lịch sử E16](E16_FFN_TEST_RESULT.md).

Pilot này có thể bác bỏ khả năng **retrofit** operator trên Cafe. Kể cả khi dương trên năm task local, đó mới là bằng chứng áp dụng/transfer; chưa được gọi là một phương pháp kiến trúc hoàn toàn mới hay general-purpose Vietnamese encoder đã được kiểm chứng ngoài benchmark.

## Cấu hình khóa trước điểm validation

- Nền `uitnlp/CafeBERT` pinned đúng checkpoint, tokenizer và recipe E14 đã dùng trong năm baseline. Giữ Q/K/V/Wo và các tầng FFN pretrained.
- Thay attention ở **bốn block trên cùng** của 24 block. Với 16 head, thêm một ma trận16×16 trên logits trước softmax và một ma trận16×16 sau softmax ở mỗi block, khởi tạo identity. Tổng **2.048 tham số** thêm cho bốn block; đếm lại khi export. Không quét vị trí block sau khi thấy điểm.
- Kiểm eval logit parity tại init với baseline trên câu có/không padding, gradient hữu hạn của hai ma trận, attention mask không rò và không NaN. Trộn logits hữu hạn trước khi áp `−inf` mask; dùng cùng precision/attention implementation cho mọi arm. Nếu phải bỏ fused SDPA để trộn head, baseline đối chứng cũng đi qua đường attention đó. Đo throughput/VRAM trước full training.
- Ba arm: **MHA nguyên bản**, **diagonal-only** (chỉ scale từng head), **full cross-head**. Diagonal và full có số tham số khác nhau; báo đúng, không dùng tham số chết để giả vờ parameter-match. Cả ba dùng cùng Cafe init, LR2e-5/seed42, 30epoch, batch hiệu dụng32, maxlen256, cùng task head E14 mới khởi tạo. Không thêm MLM, synthetic, KD, LLRD, augmentation hoặc multitask trong pilot.

## Cổng theo miền rồi theo đủ năm task

1. **Preflight** không nhìn validation để chọn hyperparameter: init parity, masks, backward, VRAM/throughput. Lỗi correctness hoặc chi phí vượt mức chấp nhận được → dừng trước GPU grid.
2. **Screen ba nguồn dữ liệu khác nhau:** Restaurant, ViSFD, VSFC Topic; 3 arm ×3 task ×1 seed. Đây là lựa chọn trước khi có điểm. Sentiment dùng cùng văn bản với Topic; Hotel gần Restaurant, nên ba task đầu bao phủ ba nguồn chính, nhưng **chưa đủ để claim thắng năm task**.
3. Chỉ mở Sentiment/Hotel và seeds43/44 nếu full mixing hơn **cả** MHA và diagonal trên ít nhất hai nguồn với biên validation đủ lớn (heuristic ≥0,5 điểm), nguồn còn lại không giảm >0,5. Nếu chỉ hơn MHA nhưng ngang diagonal, không có bằng chứng cần giao tiếp giữa các head. Nếu gain chỉ ở Restaurant, gọi đúng là task-specific và dừng mục tiêu general.
4. Gate xác nhận cần lợi ích trên **ít nhất bốn trong năm task**, không giảm task còn lại quá0,3 điểm, và paired gain qua≥3seed. Đây là quy tắc đầu tư, không phải kiểm định ý nghĩa; final theo protocol cần 5seed, recipe khóa trước test. Không chọn arm khác nhau cho từng task rồi gọi một encoder chung.

## Nếu có tín hiệu tốt

Một wrapper Cafe với hai ma trận identity chưa phải pretrained checkpoint mới: tại init nó tính cùng hàm gốc. Nếu fine-tune **riêng** từng task, ta chỉ chứng minh một *kiến trúc adaptation dùng chung*. Để tạo một backbone có weights cải thiện tổng quát trước fine-tune, phải mở thí nghiệm riêng: cho operator học trên dữ liệu không nhãn rồi đánh giá năm task, với control Cafe plain continued MLM cùng corpus/tokens/steps. Các lượt CPT trước không cho gain ABSA, nên không mặc định bước này sẽ giúp; chỉ đầu tư nếu gate fine-tune có tín hiệu rõ.

Ngoài năm task, cần ít nhất một nhóm NLI/NER hoặc holdout review độc lập trước khi mô tả là encoder tiếng Việt tốt hơn tổng quát. Test local đã được quan sát trong nghiên cứu trước và UIT-ABSA có giới hạn lineage/evaluator. [Benchmark protocol](BENCHMARK_PROTOCOL.md).

## Ước tính công sức và GPU trước khi triển khai

Metadata CafeBERT hiện có cho 30 epoch/seed/LR2e-5: Restaurant **23,33 phút**, ViSFD **24,66**, Topic **30,79**, Hotel **19,70**, Sentiment **31,44**. Peak allocated tương ứng **10,8–14,3 GiB** với đường SDPA cũ. Đây là số đo cũ, không phải benchmark tốc độ cho attention đã thay.

| Giai đoạn | Số run | GPU-giờ tại tốc độ SDPA cũ | Dự toán sau thay attention* |
|---|---:|---:|---:|
| Preflight correctness + profile | Không tính là run 30 epoch | — | ~0,3–1 GPU-giờ; ~1–2 ngày công triển khai/QA |
| Screening 3 task ×3 arm ×seed42 | 9 | **3,94** | **~5–9 GPU-giờ**; ~2–4 giờ wall nếu 3–4 GPU dùng được |
| Mở thêm Sentiment/Hotel seed42 nếu pass | 6 | **2,56** | **~3–6 GPU-giờ** |
| Xác nhận đủ 5 task ×3 arm ×3 seed (tính cả screening) | 45 | **19,49** | **~24–40 GPU-giờ** |
| Final đủ 5 task ×3 arm ×5 seed | 75 | **32,48** | **~40–70 GPU-giờ**, cộng thời gian audit/test đã khóa |

\* Khoảng dự toán giả định operator mới làm mỗi run chậm khoảng **1,2–2,0 lần**; chưa đo. Việc thay bốn block có thể mất fused SDPA và phải giữ baseline cùng đường attention để so công bằng. Preflight phải đo thời gian/peak VRAM trên 100–200 bước và một lượt validation, rồi **cập nhật dự toán hoặc dừng** nếu overhead vượt mức chấp nhận. Không dùng tổng tham số thêm 2.048 để suy runtime nhỏ.

Trên bốn L40S, phép chia tổng GPU-giờ cho bốn chỉ là cận dưới của wall time. Contention với job khác, task dài ngắn khác nhau và lỗi CUDA gần đây khi chạy chồng tải có thể kéo dài hoặc buộc recovery. Nếu chỉ hai GPU ổn định/trống VRAM, screening có thể thành ~3–6 giờ wall. Không dừng job người khác để đạt ETA.

## Diễn giải kết quả âm

Nếu full mixing không hơn diagonal/MHA trong pilot đúng và đủ budget, dừng operator này. Không tự nối thêm dynamic heads, đổi layer, tăng số head hoặc pretrain tiếp chỉ để cứu một screen âm. Kết quả âm không phủ định mọi thay đổi kiến trúc, nhưng là lý do hạ ưu tiên hướng này so với nghiên cứu dữ liệu có nhãn hoặc model nhỏ.
