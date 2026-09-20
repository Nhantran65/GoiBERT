# E16 — Review test baseline và FFN-input gate

Hoàn tất test10checkpoint: baseline/ffn_input_gate ×5task,seed42. Checkpoint chọn từvalidation vàkhóa trước test theo authorizationmới. Khôngtestattention-update gate. Khôngchọnlạiepoch/LR bằngtest.

| Task | BaselineE16 testF1 | FFN-input gate testF1 | Gate−baseline (pp) |
|---|---:|---:|---:|
| Sentiment | 81.9159 | 81.1482 | −0.7677 |
| Topic | 78.9652 | 79.4299 | +0.4647 |
| ViSFD | 84.6019 | 84.2190 | −0.3829 |
| Restaurant | 74.4509 | 74.4230 | −0.0279 |
| Hotel | 74.2146 | 73.8172 | −0.3974 |

VSFC dùngfixed-labelmacroF1; ABSA dùngend-to-endpairmicroF1,ViSFDOTHERS loại khỏipair. F1display100×fraction; delta theođiểmF1. Baseline vàgate cùngmicro16/BF16SDPA/FP32targetFFNfusion,calibrationđãlưu,rawlabels vàevaluator.

## Kết luận của root

Gate chỉ tăngTopic, khônggiữ được lợi íchHotel thấy ởvalidation. Sentiment,ViSFD vàHotel đều giảm rõ hơn cácchênhlệchrất nhỏ trướcđó; Restaurant gầnngang. Tínhiệuvalidation4/5tasktăng khôngchuyển thành cải thiệnrộng trêntest ởseed42.

**Không mởseed43. Dừng nhánh FFN-input gate trong protocolnày**, giữbaseline làmreference. Đây làquyếtđịnh ngân sách từscreening, khôngchứngminh mọiFFNgate đều vôích hay khácbiệt cósignificance. Khôngthayđổi cấu hình hoặc chọncheckpointkhác đểcứuđiểmtest.

Root đã tínhlại cả10primaryF1 từsavedpredictions độc lập với summary; kiểmtracount/uniqueIDs vàID/goldalignment giữa2arms trên5task. Counts3166/3166/2224/1938/2030. Nguồn: `artifacts/e16/test_summary_ffn_baseline.json`, `test_authorization_ffn_baseline.json`, `test_access_log_ffn_baseline.json`, pertrial `test_predictions_ffn_baseline.json` vàlogits. Khôngtraining/seedmới sautest.
