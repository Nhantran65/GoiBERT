# Trạng thái vòng thiết kế và kiểm chứng độc lập

## Cập nhật vòng 4 (2026-09-13)

Root tự chạy các probe inference/prompt-only trên student E8 đóng băng (GPU1 rồi GPU0 theo yêu cầu người dùng; không chạm job khác; validation-only; không test). Tài liệu: [SLOT_READER_ROUTING_PROPOSAL.md](SLOT_READER_ROUTING_PROPOSAL.md). Ba bằng chứng mới:

1. Locus khác nhau giữa student và BamiBERT trên composite: student lỗi ở CLS trộn (fork theo đoạn phục hồi ≈ đơn lẻ), BamiBERT lỗi ở token states (fork vẫn báo B present 30%).
2. Reader probe (K CLS slot + ~8k tham số, encoder/head đóng băng): tách composite khi train trên composite, nhưng PR envelope trên validation thật thấp hơn E8 ở mọi cấu hình có composite; không composite thì không tách. Cổng cạnh tranh CGSA chỉ +2 điểm trong nhiễu.
3. Trần đọc-theo-đoạn trên câu tự nhiên với đối chứng random-mask: oracle split liên tục 79,68/78,75 nhưng oracle random-mask 77,70/77,23 (Restaurant/Hotel) — phần thuộc cấu trúc đoạn chỉ +2,0/+1,5. Composite battery phóng đại headroom; các fix ở lớp đọc có trần thấp với student hiện tại.

Quyết định: hạ cấp 4A; 4B đã screening (E18, 6 trial) và **không đạt gate** — đóng toàn bộ hướng readout/segment với bằng chứng: trần thấp + cơ chế đạt mà quality không đổi + control ngẫu nhiên ngang. Các đòn bẩy lúc fine-tune (E2–E18) đều ≤ ±1 điểm ≈ nhiễu seed; khoảng cách 2–3 điểm ABSA so BamiBERT nằm ở backbone. Hướng tiếp theo: xem [NEXT_DIRECTION_AFTER_E18.md](NEXT_DIRECTION_AFTER_E18.md).


Mục tiêu đang theo đuổi: thiết kế và tự phản biện liên tục cho đến khi có một ứng viên mới có đóng góp cụ thể, không yêu cầu người dùng nhắc lại từng bước. Root tự nghiên cứu; chỉ một execution agent Luna xhigh hiện thực các phép kiểm tensor tổng hợp theo spec. Không training hoặc chạy dataset mới.


## Cập nhật 2026-09-14: E19, E20 (vòng 5)

- **E19** (goibert-b-dapt 103M + NC + LLRD, 5 task): không đạt gate — Sent −0,2, Topic −0,4, ViSFD +0,7, Restaurant −0,1, Hotel +0,4 so E2 teacher CE ([E19_RESULT.md](E19_RESULT.md)).
- **Chẩn đoán vòng 5** ([SCALE_PERSISTENT_LABEL_SET_BIAS.md](SCALE_PERSISTENT_LABEL_SET_BIAS.md)): cặp nhãn hiếm missed 33–47% và trái polarity wrong 15–18% ở cả 26M và 110M; lỗi cặp hiếm là thay thế bằng bạn đồng hành quen; D2: đổi cách đọc trên token states đóng băng không sửa.
- **5A CTI** ([CTI_PROPOSAL.md](CTI_PROPOSAL.md)) screening **E20** ([E20_RESULT.md](E20_RESULT.md)): nhiễm fork giảm 16,5 → 2,0 (Hotel) nhưng lỗi cặp hiếm/trái polarity và F1 không đổi so đối chứng ghép câu d1 → **bác bỏ**. d1 (ghép câu ưu tiên cặp hiếm) cho Restaurant 75,46 (+0,95, một seed), Hotel −0,3.
- **Kiểm prior quyết định** ([DECISION_LEVEL_DEBIAS_CHECK.md](DECISION_LEVEL_DEBIAS_CHECK.md)): trừ lift đồng xuất hiện (kể cả oracle) không cứu cặp hiếm → bác bỏ giả thuyết prior ở lớp quyết định.
- **E21 multi-seed** ([E21_MULTISEED_D1.md](E21_MULTISEED_D1.md)): d1 (ghép câu nhãn hợp, ưu tiên cặp hiếm) trên nền E19, Restaurant 3 seed: 75,26 ± 0,23 vs 74,10 ± 0,40, chênh cặp +1,16 ± 0,18 — tín hiệu dương ổn định đầu tiên ở quy mô 110M; trái polarity missed/wrong giảm ở cả 3 seed. Hotel −0,3 (1 seed), ViSFD chưa chạy.
- **E22** ([E22_RESULT.md](E22_RESULT.md)): ghép câu nhãn hợp 3 seed: Restaurant +1,16 ± 0,18; Hotel +0,40 ± 0,72 (cặp hiếm missed giảm 3/3 seed); ViSFD −0,13. Ablation: chọn cặp đều (d1_uniform 75,74) ≥ ưu tiên cặp hiếm (75,26) → phần "mới" không tạo gain; gain là augmentation ghép câu thuần (tiền lệ C3DA/Chai).
- Kết luận: ba cơ chế đã được "sửa" có đối chứng (đọc theo đoạn E18, đọc theo token D2, nhiễm token E20) mà lỗi cấu trúc không đổi; phần còn lại nằm ở thống kê nhãn/ontology và quy tắc quyết định. Hướng tiếp: xác nhận d1 multi-seed; thử debias theo prior đồng xuất hiện ở lớp quyết định (không train encoder); adjudication mẫu cặp hiếm Hotel.

## Các vòng trước

| Vòng | Ý tưởng | Quyết định |
|---|---|---|
| 1A | Persistent revisable pair state | Loại broad novelty; Relational/Edge Transformer là tiền lệ trực tiếp |
| 1B | Query-conditioned ordered operators | Chưa đạt novelty; matrix composition, multiplicative RNN và attention-gated scan đã có |
| 2 | FFN finite-response feedback vào attention | Loại bản hiện tại: sensitivity prior art, kernel limit, proxy bỏ LN/W2 |
| 3 | Recipient-excluded normalized keys | Giữ sau review đại số/prior art; novelty đề xuất ở operator cụ thể và factorized evaluation, không ở nguyên lý non-backtracking. E17 screening: không tác động (eta ≈ 0) |
| 4A | Slot readers trong encoder (K CLS bảo toàn hàm, cổng cạnh tranh CGSA) + định tuyến từ composite ghép | Đã đối chiếu novelty và đo trên encoder đóng băng: tách composite (both-correct 9→32%) nhưng không vượt PR envelope validation thật; hạ cấp, không screening |
| 4B | Huấn luyện đọc-được-theo-đoạn: split ẩn liên tục chọn bằng fork CLS của chính model, CE trên fork theo đoạn, inference CLS thường | **Screening E18 không đạt gate** ([E18_RESULT.md](E18_RESULT.md)): cơ chế đạt (gap đoạn Restaurant 0,65→1,74) nhưng F1 không tăng, control random-mask ngang ứng viên. Đóng hướng đoạn/reader |

## Bằng chứng mới của vòng 3

- Đã viết forward, initialization, bias policy, mask policy, scope partial-path invariance và controls trong [proposal](RECIPIENT_EXCLUDED_KEYS_PROPOSAL.md).
- Đã đối chiếu NBA-GNN, line-graph SLU, SAOBP, contextual keys, attribution qua LN và implicit self-correction. Có tiền lệ cho từng nhóm nguyên lý; chưa xác định nguồn trùng full operator/algorithm. Không coi search absence là chứng nhận novelty tuyệt đối.
- Root đã chứng minh trên giấy partial-path exclusion và construction score rank vượt dot-product cùng head width. Những tính chất này không đồng nghĩa model thắng benchmark.
- [Kiểm tensor đầu](../artifacts/research_checks/cavity_key_algebra/REPORT.md): FP64 forward12cases maxabs5.33e-15, backward maxabs1.42e-14; permutation/invariance/no-LN controls pass; ranking witness phân biệt với mask diagonal. Root đọc code và raw JSON, sửa một lỗi broadcasting và cách đặt reference denominator qua execution agent trước khi chốt kết quả.
- Cùng phép kiểm phát hiện FP32 scaled-cancellation maxabs2.79648; raw failure được giữ. Guarded evaluation được đặc tả để fallback tính trực tiếp theo chunk ở pairs có nguy cơ cancellation. Đây là sửa đường tính của cùng operator, không đổi giả thuyết để tìm gain.

## Kết quả chốt

Execution agent đã hoàn tất [guarded evaluation](CAVITY_KEY_GUARDED_SPEC.md), không tải model/data. Root đã review code, rawresults, sửa một lỗi guard-term qua agent và xác minh sourceSHA. Ca scaled-cancellation maxabs giảm2.79648→1.90735e-6; sixFP32cases pass criterion đã chốt, FP64forward/backward pass3cases bổ sung. Nonfinite overflow recovery chưa verified; không training.

[Review cuối](RECIPIENT_EXCLUDED_KEYS_REVIEW.md) chốt giữ một ứng viên có forward, property, thuật toán và prior-art distinction đủ cụ thể. Giai đoạn tìm ý tưởng được hoàn tất ở phạm vi này; mục tiêu thắng BamiBERT chưa đạt và chưa có kết quả model mới. Không chờ agent nghiên cứu ngang hàng hoặc người dùng nhắc bước1/bước2 trong vòng này.

## Điều không được diễn giải quá mức

Không gọi cả encoder là non-backtracking: chỉ key của preview có loại contribution, ở interface values với routing/input đã cố định. Không gọi tensor checks là benchmark. Không nhận rank cao là bottleneck đã đo trên GoiBERT. Không gọi six-parameter interpolation là innovation chính. Nguồn line-graph SLU mới chỉ truy cập indexed publisher content, chưa fullpaper; giới hạn đó vẫn được công bố.
