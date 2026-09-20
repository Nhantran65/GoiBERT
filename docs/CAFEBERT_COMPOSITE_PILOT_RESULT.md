# CafeBERT + composite — pilot Restaurant

Một run mới, seed42, validation-only; không test hoặc training thêm. Inference giữ nguyên encoder/head48logits, một pass mỗi review.

| Arm | Best epoch | Validation pair microF1 | Delta | Runtime |
|---|---:|---:|---:|---:|
| CafeBERT CE reference |21|77.7197|—|23.3min lịch sử|
| CafeBERT CE + uniform composite |19|76.7045|-1.0152pp|25.5min|

## Cách làm

7028reviewtrain +2343composites của E22uniformseed42 =9371rows. Cùng checkpoint CafeBERT pretrained, head và lossCE như reference;LR2e-5,30epochs,effbatch32,maxlen256,BF16SDPA. KhôngKD/NC/LLRD. Mọi sourceID/text/unionlabel composite được kiểm, max90tokens theoXLM-R, khôngtruncation. Validation giữ nguyên771reviews.

## Kiểm tra

Smoke/reload/F1checks passed trong metadata. Report writer đếm lại pairF1 từ savedpredictions độc lập và đối chiếu ID/gold với reference; khớp1e-12.30epochs hoàn tất. Best checkpoint chọn validation, không đổi bằngtest.

## Diễn giải

Chưa có tín hiệu đạt mục tiêu; dừng cấu hình này.

Có33%thêmoptimizersteps (8790 vs6600) do trainset lớn hơn ở cùng30epochs; chưa cô lập gain doaugmentation với extra training. Inference model không tăng sốpasses hoặcparams. Runtime trênGPUkhác điều kiệnchia sẻ khôngphải benchmarklatency. Chỉmộtseed/mộttask; khôngclaimSOTA hoặc thắngrộng.

Artifacts: artifacts/cafebert_composite/summary.json và run_seed42/. Scope kết thúc, không tự thêmseed/LR/test.
