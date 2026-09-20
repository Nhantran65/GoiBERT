# Prompt lượt D1 — CafeBERT-DAPT → BamiBERT

Hãy tự thực hiện pilot dưới đây tại `/home/kiettqa/GoiBERT`. **Không tạo subagent trừ khi tôi yêu cầu rõ.** Đọc AGENTS.md và `docs/CAFEBERT_DAPT_BAMIBERT_KD_PLAN.md` bản D1 trước khi làm. Bản này thay gate cũ “K phải pass mới DAPT”.

## Context

Milestone1 đã hoàn tất, Restaurant seed42 validation: T CafeBERT77,7197; B BamiBERT74,7418; K BamiBERT KD từT73,9440. Correctness đã kiểm. Đọc `docs/CDBK_PILOT_RESULT.md`, reuse artifacts `artifacts/cdbk_pilot/`; **không rerun T/B/K**.

Câu hỏi mới: DAPT teacher có giúp cùng recipe student KD vượt BamiBERT không? Pipeline cũ26M có general distillation; lần này chỉ kiểm task-logit KD, không tái lập pipeline đó.

## Chỉ làm đúng một lượt nhỏ

1. Chọn một corpus review/social tiếng Việt phù hợp và tải được; kiểm nguồn/nội dung, lọc trùng/overlap benchmark, ghi token count/passes. Không giả gọi FineWeb web tổng quát là domain review. Nếu thiếu corpus phù hợp theo plan, báo blocker, không tự thay bằng benchmark lặp nhiều lần.
2. **Một DAPT run300M processed tokens** từ CafeBERT pretrained gốc, standard MLM15%,seq512,LR5e-5,global batch256,seed42. Recipe đầy đủ trong plan. Không dùng checkpoint T classifier để khởi tạo DAPT.
3. Fine-tune teacher **TD** trên Restaurant đúng recipe T cũ: CE,LR2e-5,30epochs,seed42,effbatch32,maxlen256. Lưu best validation checkpoint thật.
4. **Dù TD không tăng F1, vẫn chạy student nếu run hợp lệ.** Cache TD logits trên train, rồi train **KD_D** từ BamiBERT pretrained gốc bằng đúng recipe K cũ: LR5e-5,30epochs,seed42,loss0.5CE+0.5KD,T2. Không khởi tạo từ B/K đã fine-tune.
5. Báo bảng T/B/K/TD/KD_D, delta TD−T,KD_D−K và quan trọng nhất **KD_D−B**. Student phải vượt74,7418 mới là vượt baseline; chỉ vượt73,9440 chưa đủ.

## Giới hạn

Tối đa1DAPT+2fine-tune runs và một cache mới; Restaurant/seed42/validation-only. Không E8,NC,LLRD,composite,grid,seed khác,test,task khác hoặc tăng token. Xong báo kết quả và dừng, không tự chạy xác nhận.

Giữ nguyên source/artifacts M1. Tái dùng code `scripts/cdbk_pilot/` và Pilot B qua adapter mới ở `scripts/cdbk_dapt_pilot/`; artifacts `artifacts/cdbk_dapt_pilot/`; báo cáo `docs/CDBK_DAPT_PILOT_RESULT.md`. Không xây framework mới.

Kiểm tối thiểu: đúng model/source/MLM head; Restaurant12×4 softmax từngslot; KD=T²KL(teacher||student) chỉnhânT²mộtlần; cache join stable IDs dù tokenizer khác; reload checkpoint/F1 từpredictions khớp. Không sửa gold/split, không kill GPU jobs khác. Decontamination test chỉtext/hash để loạioverlap theo plan, không đọcnhãn/inference.

Báo tiến độ ngắn tại các mốc corpus xong, DAPT xong, TD xong, KD_D xong. Tự tiến hành các bước đã nêu, không hỏi lại xác nhận từng job; chỉ báo blocker nếu không thể thực hiện đúng scope.
