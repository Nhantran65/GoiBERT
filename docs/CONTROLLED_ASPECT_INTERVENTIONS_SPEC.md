# Phép can thiệp polarity có đối chứng

## Mục tiêu

Tách lỗi không nhận ra polarity của một mệnh đề khỏi lỗi do mệnh đề khác làm nhiễu. Fixture do root tự viết ở CONTROLLED_ASPECT_INTERVENTIONS.json; không lấy từ test và không dùng để train hoặc chọn hyperparameter. Đây là kiểm tra nhỏ trên câu đơn giản, không đại diện validation tự nhiên và không phải đóng góp novelty.

## Tạo input

Với mỗi cặp A/B có aspect khác nhau:

1. Bốn câu standalone: Apositive, Anegative, Bpositive, Bnegative.
2. Full factorial: polarity A/B mỗi bên hai trạng thái, thứ tự AB/BA, connector '. ' hoặc ' và ':16inputs.
3. Apositive/Anegative/Bpositive/Bnegative ghép filler bằng '. ', thử hai thứ tự:8inputs. Filler chỉ dùng để đo target đã khai báo, không gán nhãn absent cho mọi category khác.

Tổng28inputs/cặp,168inputs toàn bộ. Lưu text, target slot indices, expected labels, factor metadata, token counts, SHA và immutable IDs. Không lowercase hoặc chỉnh dấu câu tùy kết quả. Giữ mô tả fixture là annotation của root, không official dataset gold. Connector/capitalization giữ theo file; casing có thể là nuisance cần báo, không sửa sau nhìn predictions.

## Model và protocol

Cùng agent Luna xhigh đã dùng: frozen E8 student và E14 BamiBERT checkpoints đã audit, ba ABSA tasks, mỗi checkpoint chỉ chạy56inputs đúng domain. Full baseline model, không C4/shadow hoặc operator mới. Dùng numerical path của reader full baseline (BF16 outer/manual FP32 attention), local weights/tokenizers. GPU tối đa0.25giờ một GPU rảnh; không training/test/multiseed. Snapshot sha checkpoint và code.

## Báo cáo

- Per-input raw logits và predictions cho mọi slot, metrics chỉ trên target annotations.
- Standalone target accuracy; ability to switch positive/negative correctly within từng target template.
- Composite joint accuracy của hai annotated targets, missed và wrong polarity.
- Giữ nguyên A, đổi B: số A predictions đổi và số từ đúng thành sai; báo riêng theo A polarity, order, connector. Đồng thời kiểm B có phản ứng đúng với thay đổi của chính nó.
- Lặp hoán đổi A/B: hai targets đều được chấm, không chỉ rescue A.
- Filler-vs-standalone target changes và target errors. Không coi filler length-matched hoàn hảo; báo token counts.
- Báo denominators theo templates và inputs. Các biến thể cùng template không độc lập; không p-value hay claim population causal effect.

Nếu standalone đã sai, không coi composite failure là bằng chứng interference. Báo cả unconditional và conditional-on-correct-standalone; không xóa trường hợp sai khỏi raw artifacts.

Nếu hầu hết câu đơn giản đều đúng, kết luận chỉ là probe không tái tạo được failure trên natural data; không tuyên bố đã giải quyết compositional generalization. Nếu có paired interference rõ, root chọn đúng vị trí/biểu diễn để kiểm tiếp và mới đề xuất can thiệp.

## Artifacts

scripts/research_checks/controlled_aspect_interventions.py; artifacts/research_diagnostics/controlled_aspect_interventions/ gồm input manifest, predictions, summary và FACTUAL_REPORT. Agent chỉ triển khai và báo số liệu, không chọn hướng nghiên cứu.
