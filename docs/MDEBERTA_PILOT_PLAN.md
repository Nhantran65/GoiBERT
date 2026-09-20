# mDeBERTa-v3-base — pilot 5 task

2026-09-18. User ủy quyền tạo execution subagent Luna xhigh và dùng toàn bộ GPU0–3. Scope mới thay đề xuất2task×2LR trong deep research: **5task×1LR×1seed =5runs**. Không mở sweep/test trong vòng này.

## Cấu hình

- Model `microsoft/mdeberta-v3-base`, revision `a0484667b22365f84929a935b5e50a51f71f159d`; tokenizer gốc, raw text, không word segmentation của ViDeBERTa.
- Tasks: VSFC Sentiment/Topic, ViSFD, UIT-ABSA Restaurant/Hotel.
- Seed42, LR2e-5,30epochs, effectivebatch32, maxlen256.
- CE; AdamW wd.01,betas(.9,.999),eps1e-8; warmup10% rồi linear decay; gradient clip1. FP32 masterweights+BF16autocast. Native eager attention nếu backbone không hỗ trợSDPA, ghi đúng implementation thực tế.
- Native DeBERTa context pooler/classifier semantics; output VSFC categorical, ABSA softmax riêngslot. Không ép head SEA-LION/CafeBERT vào model rồi gọi native.
- Microbatch32 nếu vừa; nếu16×accum2 thì chuẩn hóa batch cuối theo số samples thực. Ghi actualconfig.
- Không DAPT/KD/NC/LLRD/composite, không baseline retrain.

## Dữ liệu và checks nhỏ

Giữ frozenloader/evaluator/splits/gold. ABSA0absent1positive2negative3neutral; ViSFDwidths(4,4,4,4,4,2,4,4,4,4,4),OTHERSindex5 thamgiaCE nhưng loại pairF1. VSFC macroF1, ABSA end-to-endpairmicroF1.

Pin model/tokenizer/source hashes; tinyGPUpreflight5schemas/finitegrad; giữ best validationcheckpoint, predictions IDs/gold, curves/runtime; reload và recomputeF1 khớp. Không xây framework/testsuite lớn hoặc thay source lịch sử.

## Mốc validation để so sánh

| Task | Bami E14 | Cafe CE |
|---|---:|---:|
| Sentiment | 86.3252 | 87.6924 |
| Topic | 79.5328 | 81.7446 |
| ViSFD | 84.3834 | 85.4665 |
| Restaurant | 74.7418 | 77.7197 |
| Hotel | 73.2362 | 73.3438 |

Bami đã tune3LR, Cafe cốđịnh2e-5, mDeBERTa pilot1LR: báo rõ ngân sách. Đây là screen mộtseed, không significance/best-attainable/SOTAclaim. Không trộn với test.

## Vận hành và deliverables

- Execution agent `gpt-5.6-luna`,reasoning `xhigh`,rootreview.
- GPU0–3 được cho phép, giữ nguyên jobs khác. Ban đầu1worker/GPU; workerthứ5 có thể chia sẻGPU đủheadroom sauprofile. Không tăng tổng5runs để lấpGPU.
- Log/status/process bền vững; chỉ1owner/trial; kiểmPID/state trước restart. Không serializePopen/filehandle trong JSONcontroller.
- Source `scripts/mdeberta_pilot/`, artifacts `artifacts/mdeberta_pilot/`, report `docs/MDEBERTA_PILOT_RESULT.md`.
- Báo root actualPID/GPUmapping ngay saulaunch; tiếp tục theo dõi tới đủ5runs, report rồi dừng. Không tự thêmLR/seed/test sau khi xem kết quả.
