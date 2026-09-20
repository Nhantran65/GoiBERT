# Data & Checkpoint Provenance

Branch này chứa code + kết quả của GoiTurn architecture-scaling experiment. Các file lớn và
dataset bên thứ ba KHÔNG commit vào git (giới hạn 100MB của GitHub + license bên thứ ba).
Tải lại như sau để dựng đúng môi trường (đường dẫn tương đối từ gốc branch).

## BamiBERT checkpoint (bắt buộc cho baseline/distill)

```bash
wget -P models/bamibert/ https://huggingface.co/Qualcomm-AI-Research/BamiBERT/resolve/main/model.safetensors
# MD5 phải là: b5f369196f27fdcf4c3e38bca27a2be1  (412MB, revision 57bc1340, 2026-07-03)
```

Config + tokenizer đã có sẵn trong `models/bamibert/`. License: BSD-3-Clause-Clear +
Qualcomm Responsible AI License (không redistribute weights trong repo này).

## Datasets (vào `data/raw/`)

```bash
mkdir -p data/raw && cd data/raw
wget https://huggingface.co/datasets/visolex/VLSP2018-ABSA-Restaurant/resolve/main/VLSP2018-ABSA-Restaurant.csv
wget https://huggingface.co/datasets/visolex/ViSFD/resolve/main/ViSFD.csv
wget https://huggingface.co/datasets/ura-hcmut/UIT-VSFC/resolve/main/UIT-VSFC_train.csv
wget https://huggingface.co/datasets/ura-hcmut/UIT-VSFC/resolve/main/UIT-VSFC_valid.csv
wget https://huggingface.co/datasets/ura-hcmut/UIT-VSFC/resolve/main/UIT-VSFC_test.csv
```

License lưu ý: UIT-VSFC là CC-BY-NC-SA-4.0 (mirror ura-hcmut); ViSFD/VLSP2018 qua mirror
visolex. Chỉ dùng nghiên cứu nội bộ, không redistribute trong repo.

## Chạy

Xem `experiments/goiturn-arch-scaling/README.md` và `REPORT.md`. Trên LUMI: `lumi_job.sh`
(sbatch) + `setup_and_run.sh`. Kết quả đo ngày 2026-08-30: `experiments/goiturn-arch-scaling/results-lumi/summary.md`.
