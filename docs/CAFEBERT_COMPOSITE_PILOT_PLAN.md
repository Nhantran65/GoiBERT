# CafeBERT + composite — pilot một lượt, inference một pass

User yêu cầu pilot hướng khác sau khi aspect-pair không phù hợp production. Lựa chọn: thử augmentation đã có tín hiệu E21/E22 trên backbone CafeBERT mạnh hơn; không đổi kiến trúc hoặc cách inference.

- Restaurant, seed42, một run CE. Khởi tạo CafeBERT pretrained gốc, head48logits như baseline T; LR2e-5,30epochs,effbatch/micro32,maxlen256,BF16/SDPA,AdamWwd.01,warmup10%linear,clip1. Tái dùng frozen runner CDBK qua adapter riêng, không sửa lịch sử.
- Reuse đúng2343câu ghép uniform của E22 seed42, tạo từ7028trainrows: mỗi nguồn có đúng1aspect khác nhau, ghép nguyên câu bằng dấu cách, nhãn hợp theo numeric slots. Kiểm lại từng sourceID/text/gold trước train. Tập train mới9371rows; validation771rows giữ nguyên, không loadtest.
- Tokenizer CafeBERT kiểm lại: không composite nào bịtruncation, max90tokens. Không dùng boundary/tokenIDs của Bami tokenizer để giả định tương đương; pilot chỉ cần text/label, không CTI hoặc token alignment loss.
- Inference vẫn là một review→một encoderpass→12slot predictions. Không KD/NC/LLRD, aspectpair, head mới hoặc augmentation khi dự đoán.
- Có33%thêmsteps vì cùng30epochs trên tập lớn hơn:293steps/epoch,8790total,warmup879 so baseline6600. Đây là practicalscreen, chưa cô lập dữ liệu khỏi extra training; nếu tăng sẽ cần xác nhận/đối chứng sau.
- BaselineT val77.7197 có sẵn, không rerun. Gain≥0.5pp là tín hiệu để xác nhận;0.2–0.5chưa rõ;<0.2không có tín hiệu. Không significance, không tự mở thêmseed/LR/test.
- Chạy tiny smoke trước fullrun. GPU theo freeVRAM, một run mộtGPU, không dừng jobs khác. Chỉ một fullrun.
- Code `scripts/cafebert_composite/run.py`; artifacts `artifacts/cafebert_composite/`; report `docs/CAFEBERT_COMPOSITE_PILOT_RESULT.md`.

Giới hạn: ghép nhãn là giả định augmentation, không bảo đảm câu ghép luôn tự nhiên. Kết quả E22 ở103M không bảo đảm gain trênCafe. Nếu fail, không lặp vô hạn các tỷ lệ ghép trong scope này.
