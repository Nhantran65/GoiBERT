# Audit độ tin cậy và hướng nghiên cứu tiếp theo

## Kết luận của root

Đã xác nhận nhiều lỗi ở tên nhãn, diagnostic và diễn giải thống kê. Những lỗi này làm suy yếu các nhận định về cơ chế, nhưng chưa có bằng chứng chúng làm thay đổi bảng F1 numeric E8/E10/E11. Không có lý do tự động hủy checkpoint hay retrain toàn bộ.

Hướng tiếp theo nên tập trung vào **gắn bằng chứng với đúng đối tượng/category**, bắt đầu bằng kiểm chứng trên model sẵn có, thay vì tiếp tục thêm nhánh ký tự hoặc thay công thức attention trên cả năm task ngay. Một nhánh quan hệ trong encoder là ứng viên sau diagnostic, chưa phải kiến trúc đã được bằng chứng lựa chọn.

## Issue ledger

| Vấn đề | Bằng chứng | Phạm vi ảnh hưởng | Quyết định |
|---|---|---|---|
| Tên polarity UIT-ABSA sai | `src/goibert/data.py:56–57` ghi1negative2neutral3positive; raw examples và nguồn preprocessing dùng1positive2negative3neutral | Metadata và diễn giải theo tên nhãn; diagnostic dùng tên→ID | Giữ integer codes; dùng mapping đính chính, không hoán vị nhãn/checkpoint |
| E11 synthetic ABSA dùng sai expected IDs | `scripts/build_e11_diagnostic.py:40–49` | Correctness và margins trên synthetic Restaurant/Hotel | Không dùng bản cũ làm bằng chứng semantic failure |
| Hai cặp Hotel diagnostic hỏi sai aspect | “phòng sạch sẽ” gán FACILITIES#CLEANLINESS, thay vì ROOMS#CLEANLINESS | Hai cặp sai slot; saved selected-slot predictions không đủ khôi phục đúng slot | Loại khỏi kết luận; nếu cần sẽ đánh giá lại đúng slot trong lượt riêng |
| Conditional flip rate E11 sai công thức | `scripts/evaluate_e11_diagnostics.py`: changed đếm mọi disagreement, chia base_correct | Có thể sai cả mẫu số/phạm vi; không phải độ bền trước biến đổi câu | Tính lại từ observations; tách model disagreement và pair robustness |
| Nhãn hiển thị ViSFD ở E12 cũng sai | `build_absa_slot_rows`: fallback nonOTHERS dùng map negative/neutral/positive | Bảng tên polarity theo slot | Sửa lớp report E12, giữ numeric metrics |
| Chia zero-gold thành0 che false positives | `safe_rate(events,0)=0`; ViSFD có18validation rows zero scored-gold,15spurious events trên12rows | Row burden và standardized strata zero-gold | Để burden undefined; báo FP/row riêng, recalc positive-support coverage |
| Điều chỉnh E12 chưa tách rarity/fragmentation | VSFC strata length+class; ABSA length+gold-positive-bin | Không đủ gọi đây là hiệu ứng riêng của rarity hay đã kiểm soát aspect/polarity | Hạ mức diễn giải; nếu follow-up thì joint-control thật |
| Flag mẫu VSFC gây hiểu nhầm | error là sentiment OR topic, gold/pred mặc định sentiment | Ba trong15mẫu topic-only nhìn như “đúng nhưng bị đánh sai” | Hiển thị riêng hai outcome, không đổi metric |
| Provenance dữ liệu chưa đầy đủ | Mirror local README có counts không khớp bảntrain/dev; nguồn VLSP tham chiếu không chứng minh byte lineage của UITmirror | Claim so paperBamiBERT và diễn giải rawannotation | Giữ caveat; không gọi mọi mismatchgold là lỗi model |
| Audit trước có dẫn chứng dòng Test.csv | `E12_LABEL_MAPPING_AUDIT.md` ban đầu dẫn Hotel Test.csv:3 | Không có test inference mới, nhưng không được gọi đó là không đọc test | Ghi rõ access history; audit mở rộng chỉ train/val/source |

Nguồn xác nhận mapping: [preprocessing VLSP](https://github.com/ds4v/absa-vlsp-2018#ii-the-vlsp-2018-aspect-based-sentiment-analysis-dataset) và [định nghĩa task VLSP](https://vlsp.org.vn/vlsp2018/eval/sa). Nguồn thứ nhất ghi integer mapping, nguồn thứ hai giải thích ontology/polarity; không suy từ chúng rằng bảnmirror hiện tại đã được xác minh toàn bộ lineage.

## Những gì vẫn dùng được

Loader đọc integer CSV không remap theo tên. CE học integer labels, argmax trả integer, evaluator so integer và presence>0. Nếu một permutation chỉ thay tên của ba polarity và không đổi absent, accuracy/exact pair F1 không đổi. Vì vậy lỗi tên vừa xác nhận không giải thích được các mức tăng/giảm numeric trong E10/E11.

Điều này không đồng nghĩa mọi pipeline đã được chứng minh hoàn hảo. Cần kiểm tra riêng đường KD/caches và bất kỳ thao tác có điều kiện theo tên polarity. Báo cáo audit của agent bổ sung phạm vi rà soát cụ thể; không lấy lập luận permutation để miễn audit.

E12 raw missed/spurious/wrong-code counts vẫn hữu ích. Nhưng burden chia số gold không phải F1; không gọi +6pp burden thành −6pp F1. Với zero-gold, tổngF1 vẫn có thể tính FP đúng trong khi một bảng slice riêng lại bỏ mất FP. Đây là hai đường tính khác nhau.

## Phát hiện ngữ nghĩa còn đáng tin

Review60mẫu ở [E12_SEMANTIC_REVIEW.md](E12_SEMANTIC_REVIEW.md) chỉ ra ba loại cần tách:

1. **Target/category attribution:** câu về giáo viên bỏ buổi thực hành bị gán topic chương trình; phụ kiện sạc bị gán pin máy; khen quản lý bị gán quán nói chung.
2. **Đánh giá ngầm và aggregation:** mong cô tiếp tục dạy được đọc thành chê; nhận xét chung nhẹ bị bỏ trong câu nhiều aspect; ý kiến trái chiều trong cùng aspect cần convention nhãn.
3. **Gold/convention chưa rõ:** một số câu Hotel có tên/mã polarity khó khớp nghĩa; một số câu VSFC thiếu antecedent hoặc có topic gold khó suy từ câu riêng.

Các mẫu đúng với emoji/viết tắt, và mẫu phân biệt đúng camera tốt/pin kém, là phản bằng chứng cho claim tổng quát “model không xử lý được noise/phủ định/multi-aspect”. Không đủ căn cứ mở tiếp character branch.

## Hướng mới: kiểm chứng binding trước, rồi mới thiết kế relation branch

**Giả thuyết:** lỗi có thể đến từ cạnh tranh giữa các category/đối tượng cùng xuất hiện hoặc từ khóa trùng domain, thay vì độ phân mảnh đầu vào. Cơ chế đáng cân nhắc là trao đổi thông tin có hướng giữa các token/span biểu diễn đối tượng và phần đánh giá, dùng chung cho các task. Đây là đề xuất có điều kiện, không kết luận bottleneck.

Tiền lệ đã có: [R-GAT](https://aclanthology.org/2020.acl-main.295/) dùng quan hệ cú pháp cho kết nối aspect–opinion; [Syntax-Aware Opinion Role Labeling](https://aclanthology.org/2020.acl-main.297/) nghiên cứu vai trò opinion. Vì vậy, chỉ thêm graph/dependency parser không phải novelty. Các công trình aspect-term có target được cung cấp cũng không chuyển nguyên xi sang task category của project, nơi không có gold span target ở inference.

Đề nghị bước kế tiếp là diagnostic nhỏ, không training mới:

- Khoảng24–40cặp, cân bằng các domain; một nhóm đổi đối tượng/quan hệ, một nhóm paraphrase giữ quan hệ. Không dùng nguyên60mẫu làm bộ đánh giá độc lập; đó là nguồn tạo giả thuyết, cần thêm mẫu được chọn có quy tắc.
- Ghi expected theo code đã đính chính và slot cụ thể. Với Topic, đổi phủ định không mặc định đổi nhãn; với ABSA, thêm aspect không mặc định giữ vectorgold.
- Đánh giá E8 và checkpoint E11signed/convex đã có trên cùngcặp để xem lợi ích E11 có trùng lỗi binding không. Không train và không đọc test. Chỉ sau root duyệt từng nhãn kỳ vọng mới inference.
- Báo pair success (cảhai phíađúng), sensitivity đúngkỳvọng, falseflip trên label-preserving pairs, phân biệt với candidate-vsbaseline disagreement. Lưu full logits để có thể sửa cách tổng hợp mà không phải inference lại.
- Có các control chỉ chứa một đối tượng và mẫu modelđãđúng. Chỉ ưu tiên relation branch nếu lỗi binding lặp ở nhiều domain, không bị gold ambiguity giải thích, và có khoảng trống rõ so với cách thêm attention E11.

Nếu gate đạt, thiết kế encoder branch phải không dùng goldaspect/target ở inference; có đối chứng sameparameter không truyền quan hệ và control đảo cạnh. Giữ E8 recipe, bắt đầu student. Vị trí/layer/supervision của branch chỉ chốt sau diagnostic. Nếu chỉ cải thiện ABSA, báo task-specific, không gọi thắngrộng.

Nếu không đạt, dừng chuỗi block nhỏ theo phỏng đoán. Lựa chọn kế tiếp là nghiên cứu đối chứng kiến trúc ở quy mô lớn hơn hoặc phạm vi task khác, nhưng hiện chưa có bằng chứng đủ để chọn một block cụ thể. Không nên tự động tăng compute chỉ vì đã thử nhiều lần thất bại.

## Trạng thái và bảo toàn kết quả

Root đã xác nhận các lỗi code/định nghĩa trên và hoàn tất review60mẫu. Agent E12 tiếp tục sửa lớp reporting không frozen, tính lại phần sửa được từ saved observations và lập bản audit chi tiết; mọi kết quả hiệu chỉnh phải có provenance và không ghi đè âm thầm lịch sử.

Không sửa rawgold, không đổi checkpoint, không train mới. Nguồn local chính: `src/goibert/{data,train,metrics}.py`, E11diagnosticbuilder/evaluator, E12script/CSV, E8savedpredictions. Các quyết định kiến trúc mới là đề xuất, chưa launch.
