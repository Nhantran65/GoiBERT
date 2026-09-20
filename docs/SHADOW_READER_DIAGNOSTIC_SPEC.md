# Đối chứng reader trên token states giữ cố định

## Câu hỏi

C4 hiện recompute toàn encoder cho từng reader. CLS bị mask có thể làm thay đổi content states các layer sau vì content tokens đọc CLS. Cần tách đường phản hồi này khỏi việc reader chỉ nhìn một vùng. Đây là diagnostic của giả thuyết scope, không phải claim operator mới hoặc quyết định train C4.

## Phạm vi

Sau CPU decoder feasibility, cùng execution agent Luna xhigh triển khai student E8 seed42, validation Restaurant và Hotel. Không test, không training, không sửa script/artifact cũ. Dùng GPU rảnh, giới hạn một GPU-giờ cho extraction; báo tiến độ nếu chạm giới hạn, không mở rộng model/task. Giữ chính xác tokenizer, batching, FP32 manual attention trong outer BF16 và clause masks của reader diagnostic đã xác minh.

## Hai đường tính

1. Full model chạy như baseline, lưu hidden input của mỗi encoder block (sau block trước), content K/V và output CLS từng layer. Đây là states contextualized bởi lần đọc toàn câu. Không detach bằng cách sửa model nguồn; inference-only hooks/helper riêng.
2. Shadow reader bắt đầu bằng đúng embedding CLS gốc. Ở mỗi layer, dùng Q của shadow CLS; vị trí key/value0 là K/V từ chính shadow state, còn mọi content/special-token positions khác dùng K/V từ full pass cùng layer. Mask chỉ query row reader theo clause mask như C4. Sau weighted attention, chạy attention.output/residual/LayerNorm và FFN/output/residual/LayerNorm gốc trên shadow state. Shadow reader không được ghi trở lại content stack. Vị trí/tensor weights giữ nguyên. Sau layer cuối, chạy head gốc trên shadow CLS.

Full shadow reader không mask phải tái tạo full CLS trajectory. Kiểm từng layer và logits trong FP32 đầu tiên, rồi actual BF16 path; lưu max error và argmax mismatches. BF16 riêng biệt GEMM có thể lệch do shape, nên baseline chính để so C4 phải dùng cùng shadow implementation với full mask. Không dùng sai khác numerical như gain.

## Kiểm tra bắt buộc trước toàn tập

- Tiny fixture kiểm không feedback vào content states và full-mask parity.
- Hai reader masks khác nhau không làm thay đổi cached content K/V.
- Kiểm mask không cho đọc padding; BOS/EOS đúng prototype; không thêm token mới hoặc đổi positions.
- Trước extraction toàn tập, chạy16 validation rows để xác nhận code/output/cache path, không tune theo chất lượng của16rows.

## So sánh và dự đoán

Ba policy: full-shadow, original recomputed C4 từ cache đã có, shadow clause readers. Lưu per-reader FP32 logits, IDs/gold/textSHA và masks. Chọn reader và quét global margin theo đúng reader diagnostic, giữ cstar độc lập với delta0.

Giả thuyết định hướng: giữ full contextual content states có thể giảm correct→wrong-polarity transitions gây bởi clause masking. Không mặc định phải tăng F1 hoặc giữ mọi rescue. Đo số lỗi sửa và lỗi phát sinh, missed/spurious/wrong-polarity, PR comparisons tại cùng vùng recall; root đọc các row36/69 Restaurant đã phát hiện, nhưng metric chính vẫn toàn bộ validation.

Gate quyết định nghiên cứu, không significance claim: nếu cả hai dataset không cải thiện PR/F1 so original C4 và các lỗi scope mục tiêu không giảm, không ưu tiên một learned shared-stack reader dựa trên giả thuyết phản hồi này. Nếu chỉ sửa một vài câu nhưng không có lợi ích tổng thể, giữ kết luận giới hạn đó. Nếu có tín hiệu, root rà prior art và chọn cơ chế học; không tự launch training.

## Artifacts và ownership

scripts/research_checks/shadow_reader.py; artifacts/research_diagnostics/shadow_reader/ gồm parity, cache, report và actual runtime/VRAM. Root tự review nghiên cứu; agent chỉ triển khai phép đo. Operator này có quan hệ với multi-reader/masked pooled-token literature; không gọi shared-stack efficiency là novelty chưa kiểm.
