# Tiếp tục trên server

Cập nhật 2026-09-08: đã tải datasets (xem DATA_STATUS.md), E0 smoke PASS và E1 đã khởi chạy qua subagent. Để tiếp tục công việc hiện tại, đọc EXECUTION_STATUS.md, E1_EXECUTION_LOG.md và E0_E1_EXECUTION_PLAN.md trước. Các bước đóng gói bên dưới là handoff gốc, không có nghĩa cần tải lại hoặc setup lại môi trường đã kiểm chứng.

## 1. Kiểm tra gói

Giải nén `goibert-server-2026-09-07.zip` để nhận project trong folder `goibert/`, gồm code, docs, hai model và bootstrap project. Gói loại môi trường ảo, `.git`, cache tải Hugging Face và ZIP cũ. Chạy `sha256sum -c MANIFEST.sha256` từ root project sau giải nén.

## 2. Tài sản phải lấy riêng

| Tài sản | Tình trạng khi đóng gói | Hành động |
|---|---|---|
| GD-S backbone | Đã có `models/goiturn-s-gd/` | Verify checksum, smoke load trên server |
| DAPT teacher | Đã có `models/goibert-b-dapt/` | Dùng encoder cho general KD; fine-tune head trước task KD |
| Per-task teachers | Không có checkpoint local | Lấy hoặc train từ teacher; không bắt buộc benchmark BamiBERT test |
| Raw datasets/corpus | Người dùng đang xin thêm; chưa có | Lấy đúng revision/split; hash và audit |
| Family per-run JSON/predictions | Không thấy trong checkout local | Lấy từ storage gốc để audit chi tiết |
| BamiBERT tokenizer/config | Đã kèm | Reference metadata; weights chưa kèm |

Báo cáo lịch sử chỉ vị trí LUMI `/scratch/project_462001285/goiturn/`; README chỉ các suffix `experiments/goibert-family-v1/results/pretrained/goiturn_s_gd/` và `goibert_b_dapt/`. Phải tìm/xác nhận trên storage thực tế, chưa có truy cập remote trong đợt đóng gói này. Báo cáo cảnh báo scratch có thể purge khoảng Dec 2026; chưa xác minh chính sách hiện tại. Báo cáo arch-scaling cũ còn chỉ `/projappl/project_462001285/experiments/goiturn-arch-scaling/results/` cho JSON lịch sử.

Đã nhận GD-S đúng config 10L/384H/6heads/1536FFN. Manifest ghi 30,000 steps, teacher DAPT, không fallback; teacher ghi 40,000 MLM steps. Xem MODEL_STATUS.md. Đường dẫn LUMI trong metadata là lịch sử; dùng đường dẫn `models/` ở root project mới. Code reference chưa tự nhận layout model mới.

Corpus cần xin: `mlm_mix.txt` (~1.67 GB) và `stats.json`, dự kiến `data/corpus_v1/`. Raw datasets cần text, labels, official splits: VSFC sentiment/topic, ViSFD, UIT-ABSA Restaurant/Hotel. Không thay Restaurant bằng VLSP2018. Có thể smoke model và dựng evaluator ngay. Supervised baseline cần raw datasets, không cần corpus; continued general distillation cần corpus.

## 3. Môi trường

Trên server: kiểm tra `nvidia-smi`, driver, Python, RAM/disk. Tạo venv mới; không copy `.venv` local. Chọn PyTorch CUDA phù hợp driver. Reference cũ dùng transformers 4.49.0, pandas 2.2.3, scikit-learn 1.6.1, safetensors 0.5.2; đây là thông tin lịch sử, chưa phải lock đã kiểm chứng trên L40S. Lưu `pip freeze`, torch/CUDA/driver, GPU model và commit sau smoke.

Giữ layout `reference/ai-foundary` nếu dùng code cũ vì scripts suy ra repo root tương đối. Đọc README family để hiểu flags. Không chạy setup_and_run.sh/LUMI jobs như entrypoint GoiBERT mới. Chúng là workflow cũ và có default launch không phù hợp phạm vi mới.

## 4. Các bước triển khai tiếp

1. Đọc BENCHMARK_PROTOCOL và E0.
2. Nhận artifacts, verify config 10L/384H/6heads/1536FFN và checkpoint teacher identity.
3. Port baseline supervised-only vào pipeline mới; code family hiện tổ chức quanh baseline teacher + task distillation, không mặc định là E1 runner.
4. Bổ sung loader VSFC-topic/UIT-ABSA Restaurant/Hotel và evaluator tương ứng.
5. Smoke load/tokenize/forward/backward/save/load, kiểm thử metrics, rồi chạy E1.

## Prompt handoff cho cộng sự trên server

“Đọc README.md và docs/ của GoiBERT. Mục tiêu là encoder chung vượt BamiBERT paper trên VSFC sentiment/topic, ViSFD và UIT-ABSA Restaurant/Hotel. Bắt đầu từ GD-S backbone, cho phép tăng tham số, 4 L40S. Không có yêu cầu turn detection/TwoHearts và không bắt buộc rerun BamiBERT benchmark. Xác minh artifacts và metric trước, thực hiện E0 rồi E1. Phân biệt rõ reported historical results với new measured results; không chạy full training trước smoke và không sử dụng test để chọn phương án.”
