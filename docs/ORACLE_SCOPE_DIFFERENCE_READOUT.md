# Kiểm readout theo đóng góp của aspect

## Đề xuất và phép đo

Một ứng viên được xét là đọc polarity bằng delta z_c(x)-z_c(x_without_target_scope), c thuộc positive/negative/neutral. Trên fixture hai mệnh đề có scope biết trước, x_without_target_scope là standalone của mệnh đề kia. Root dùng logits đã lưu; không inference/training mới.

Phép đo dùng oracle scope và oracle presence: chỉ chấm hai targets được annotate và luôn chọn một polarity. Vì vậy comparator phải là full reader cũng argmax trên ba polarity, không phải argmax bốn lớp có absent. Nếu bỏ điều kiện này, tăng recall do oracle presence bị hiểu nhầm thành lợi ích cơ chế.

| Model/task | Full với oracle presence: joint polarity accuracy | Delta với cùng oracle presence |
|---|---:|---:|
|Student Restaurant|87.50%|87.50%|
|Student Hotel|62.50%|81.25%|
|Student ViSFD|84.375%|90.625%|
|Bami Restaurant|100%|100%|
|Bami Hotel|75%|87.50%|
|Bami ViSFD|93.75%|93.75%|

Mỗi hàng chỉ32variants từ2template pairs. Không coi đây là natural validation improvement hoặc một benchmark result. Source deletion cũng thay đổi độ dài/vị trí; không giả định đây là intervention chỉ vào một biến latent.

## Lỗi mục tiêu Restaurant vẫn chưa sửa

Khi FOOD negative/SERVICE positive, full reader đọc SERVICE negative ở cả4variants. Delta đọcneutral ở hai bản dấu chấm, negative ở hai bản 'và'; không variant nào được sửa thànhpositive. Như vậy ứng viên chưa giải quyết failure mode đã dùng để đề xuất, dù có tín hiệu ở các templates domain khác.

Phép trừ logits thuộc dòng counterfactual effects/attribution đã biết. Chưa có claimchart chứng minh novelty của một architecture hoàn chỉnh; không launch training chỉ từ các con số oracle này.

## Audit

Artifact chính: artifacts/research_diagnostics/controlled_aspect_interventions/root_oracle_scope_logit_difference.json. Root sửa một lỗi trong lượt tính đầu: ViSFD có slot OTHERS hai lớp nên offsets sau index5 không bằng4*aspect. Bản lỗi được giữ ở file hậu tố `_superseded_index_error.json`, không dùng để diễn giải. Comparator oracle-presence đồng điều kiện được bổ sung cùng lượt sửa.

Kết luận: giữ kết quả như diagnostic có giới hạn, chưa chọn đây là ý tưởng mới khả thi. Trước mọi bước sau phải so cả source-only oracle, và phải định nghĩa cách suy scope/presence trên văn bản thật.

## Đối chứng source-only đã hoàn tất

Cho source-only cùng oracle presence và đọc riêng mỗi mệnh đề đúng target, joint polarity accuracy của student đạt100% trên cả ba domains. Vì vậy delta không hơn source-only ở bất kỳ domain nào của student; các gain so full context chưa chứng minh cần phép trừ. Bami source-only đạt100/75/100% ở Restaurant/Hotel/ViSFD; delta chỉ hơn source-only ở Hotel (87.5%).

Đọc standalone BamiHotel xác định một lỗi gốc: 'Phòng ngủ không sạch sẽ' vẫn được đọcpositive (logits positive2.515625, negative2.078125). Không coi lỗi này là do ghép aspect. Delta có thể sửa một số biến thể context nhưng chưa cho một rule tổng quát.

Artifact `root_source_only_oracle_control.json`. Quyết định: không ưu tiên xây readout delta cho student từ probe này. Không tiếp tục thêm biến thể phép trừ trước khi có một trường hợp mà context thật sự cần thiết và source-only không đủ.
