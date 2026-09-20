# E17 — 12 lượt hoàn tất, Hotel bị gián đoạn

## Cập nhật kiểm tra 13/09/2026, khoảng05:56UTC

Root xác minh host không còn controller E17 hoặc3PIDHotel820459/975600/975603. Heartbeat cuối của filecontroller là12/09/2026 19:51:50UTC; trạng thái `running` trongfiles làstale, khôngbằngchứngjobcònlive. Chưa cólogxácnhận nguyênnhân dừng.

12/15trials đủ30epochs: baseline/full_context/cavity cho Sentiment,Topic,ViSFD,Restaurant. Root tínhlạiF1từsavedpredictions, kiểmIDs/goldkhớp3arms; core/runnerhashkhớpapproval. Bảng ở `artifacts/e17/root_interim_readout.json`.

| Task | Baseline valF1 | Full-context | Cavity | Cavity−baseline pp |
|---|---:|---:|---:|---:|
| Sentiment |85.3217|85.3217|85.3217|0.0000|
| Topic |80.1766|80.2348|80.1766|0.0000|
| ViSFD |82.7638|82.7086|82.7565|−0.0073|
| Restaurant |71.4697|71.2699|71.3260|−0.1437|

Hotel chưa hoàn tất: baseline16/30epochs,full_context2/30,cavity2/30. Không dùng điểmpartial để so finalquality. Execution agent mới `/root/e17_recovery` explicitLuna/xhigh đãđượcgiaokiểmtra/phục hồi đúng3Hotel, giữ12completed vàtoànbộpartialprovenance; chưacoi phục hồi đãlaunch khi chưa cóprocess/telemetryxácnhận.

Gatecải thiệnrộng đãkhôngthểđạt từ4taskhoàntất (khôngtasknàocavity>baseline); đây làquyếtđịnhscreening, khôngsignificance hoặc bácbỏmọicavityparameterization. Khôngtest/multiseed/grid. Nhánh cóupdate nhưngcorrectionnorm/logitnorm tạibestepoch trênfixedtrainprobe nhỏ: Sent~.0054%,Topic~.0075%,Vi~.396%,Rest~.0861%; khôngquyđổi thành phầntrămthôngtin hoặcnguyênnhâncausal.

## Snapshot khởi chạy lịch sử

Pilot đãlaunch vàroot xácnhận cả3worker đầu qua epoch1/globalstep358. Scope15trials:3arms×5task,studentGD,seed42,30epochs,micro16accum2,E8NC+LLRD. Chưa test hoặcmultiseed.

Snapshot khởi chạy: controller2418485; VSFCsentiment baseline2420246/GPU2,full_context2420253/GPU1,cavity2420259/GPU0. Queue12trials còn lại. CácPID làsnapshot, phải kiểm lạihostprocess/command trước khi kếtluậnlive/stopped.

- Plan: `docs/E17_PLAN.md`.
- Review/approval: `docs/E17_IMPLEMENTATION_REVIEW.md`, `artifacts/e17/root_approval.json`.
- Live state: `artifacts/e17/controller_state.json`.
- Trial outputs: `artifacts/e17/<arm>/<task>/seed_42/`.
- Logs: `artifacts/e17/logs/`.
- Execution agent: `/root/cavity_algebra_check`, explicit gpt-5.6-luna/xhigh, tiếp tụcmonitor15trials vàbàn giao validationreadout.

Không sửa source/checkpoints trong lúcchạy; helper tổnghợp sauđó phải riêng nếu cần. Hoàn thànhtraining phải kiểm savedpredictions IDs/gold/F1, báo cả5task vàcavity−baseline/cavity−full_context. Root sẽquyết định bước tiếp theo từ kếtquả, khôngchọnarchitecture bằngtest.
