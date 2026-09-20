# E16 — Test xác nhận baseline và FFN-input gate

User đã duyệt test hai arms baseline/ffn_input_gate seed42 trên đủ5tasks, sau khi attention-update gate thất bại. Đây là authorization mới; giữ nguyên manifestvalidation và flags lịch sử, ghi manifesttest riêng chọn đúng10checkpoint từ15checkpointđãkhóa. Không testattention_update_gate, khôngchọnlạiLR/epoch bằngtest.

Execution dùng lại agent e16_implement Luna xhigh, khônghelpers. Rootreviewresults. Khôngtraining mới trong bước1. Inference đúngconstructorE16,sc/sxcalibrationbuffers,FP32gate/fusion vàBF16SDPA,micro16. StrictcheckpointSHA+state vàIDs/gold/order/count, metricrecompute. Rawcodesgiữnguyên,semanticoverlayABSA1pos2neg3neutral; ViSFDOTHERSdetectiononly.

Primarycomparison là baselineE16vsgateFFN trongcùngpipeline. Báo đủ5task: VSFCaccuracy/macroF1,ABSAdetection/pairmicroF1,delta_pp. Khônggộp settingtốt từnhiềuarms. E14/paper chỉhistoricalreferencekhácprotocol.

Deliver testmanifest riêng,accesslog,pertrialpredictions/metrics/logits,summaryCSV/JSON,docs/E16_FFN_TEST_RESULT.md. Oldsource/checkpoints/validationselection khôngđổi.

Bước2 cóđiềukiện: rootđọctestvector đểxemTopic/Hotelcòngain vàRestaurantgiảmbao nhiêu; chỉ nếucótínhiệurộng mới giao chạyseed43cảhaiarms trên5task. Khôngtựmởseed43trướcrootreview,khôngđổi gate/LR/layer. Cácngưỡngcũ làscreeningheuristic; báođộlớn/chiềutừngtask chứkhôngcoi testmộtseedlàstabilityproof. Userkhôngcầnxácnhậnlạitrongbước đãduyệt.
