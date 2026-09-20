# E0 environment and checkpoint report

Ngày kiểm tra: 2026-09-08. Đây là kiểm tra môi trường và checkpoint, không phải kết quả fine-tune.

## Kết luận

E0 environment/checkpoint smoke **PASS** trên một NVIDIA L40S (GPU1). Student và teacher tải được từ local checkpoint, tokenizer tương thích, forward hữu hạn, head mới có backward và optimizer update thật, save/load round-trip khôi phục logits trong sai số số học, và source weights không bị thay đổi.

Máy có đúng **4× NVIDIA L40S** và driver `580.126.20`. Tại thời điểm inventory, GPU0 có 4,175 MiB đang dùng, GPU1 có 24 MiB, GPU2 có 29,178 MiB (`VLLM::EngineCore` chiếm khoảng 29 GiB), GPU3 có 132 MiB. Smoke chạy với `CUDA_VISIBLE_DEVICES=1`; không dừng hoặc chiếm process của GPU0/GPU2/GPU3. GPU1 đã trở về 24 MiB sau khi smoke kết thúc.

## Môi trường đã đo

- Python: `/usr/bin/python3` 3.12.3, venv `.venv-e0-cuda`.
- PyTorch: `2.6.0+cu124`; CUDA runtime 12.4; cuDNN 9.1; `torch.cuda.device_count() = 4`.
- Transformers 4.49.0, safetensors 0.5.2, tokenizers 0.21.0, accelerate 1.4.0.
- NumPy 2.2.3, scikit-learn 1.6.1, pandas 2.2.3.
- Driver/GPU: NVIDIA L40S, driver 580.126.20, 46,068 MiB mỗi GPU.
- Precision smoke/profile: FP32 và BF16 autocast đã được smoke. BF16 có thể chọn cho E1 sau khi runner lặp lại smoke train theo từng task.

Freeze đầy đủ của venv CUDA nằm tại [pip-freeze-cuda.txt](../artifacts/e0/pip-freeze-cuda.txt); overlay test có pytest tại [pip-freeze-cuda-dev.txt](../artifacts/e0/pip-freeze-cuda-dev.txt). Lock cài đặt là [e0-cuda.txt](../requirements/e0-cuda.txt), test overlay là [e0-dev.txt](../requirements/e0-dev.txt), local CPU fallback là [e0-cpu.txt](../requirements/e0-cpu.txt). `pyproject.toml` đã đổi Python range thành `>=3.12,<3.14` vì interpreter đã kiểm chứng trên máy là 3.12.3.

## Checkpoint

| Checkpoint | Kiến trúc đã load | Encoder parameters | SHA256 `model.safetensors` |
|---|---:|---:|---|
| `goiturn-s-gd` | 10 layers / hidden 384 / 6 heads / FFN 1536 | 26,397,696 | `d5473da3c7ad54835b48e8887043502659f8925096e58f26c97521352068ccca` |
| `goibert-b-dapt` | 12 layers / hidden 768 / 12 heads / FFN 3072 | 102,360,576 | `a639c941dca7574b416ca2ff5d9899ff4e97a247f87902a95b8d34bcb4282a5f` |

Teacher weight file có 102,973,185 tham số lưu trong toàn bộ checkpoint vì còn chứa MLM head; số encoder được load trong smoke là 102,360,576. Student weight file chỉ chứa encoder và có đúng 26,397,696 tham số theo metadata. Các số encoder trong bảng tương ứng với `AutoModel(..., add_pooling_layer=False)` mà pipeline sẽ dùng.

## Smoke evidence

JSON đầy đủ: [environment.json](../artifacts/e0/environment.json) và [smoke_models_gpu1.json](../artifacts/e0/smoke_models_gpu1.json).

Bản smoke có BF16 backward/optimizer là [smoke_models_gpu1_bf16.json](../artifacts/e0/smoke_models_gpu1_bf16.json). CPU fallback cũng PASS trong [smoke_models_cpu.json](../artifacts/e0/smoke_models_cpu.json), nhưng không dùng CPU để suy ra throughput GPU.

- Student output: `(2, 14, 384)`, teacher output: `(2, 14, 768)`; cả hai finite.
- Hai tokenizer có cùng vocabulary 20,481 mục.
- Loss head 3 lớp: `1.055635`; loss và gradients finite; gradient L2 `5.96567`.
- Optimizer cập nhật encoder với max delta `0.00100249` và head với max delta `0.00100051`.
- Encoder save/load hidden max absolute diff `1.91e-6`; head logits max absolute diff `2.09e-7`.
- Hash weights trước/sau smoke giống nhau.

Profile synthetic batch 32, FP32, một warmup và hai bước đo:

| Max length | Peak allocated | Peak reserved | Sequences/s | Tokens/s |
|---:|---:|---:|---:|---:|
| 256 | 988 MiB | 1,180 MiB | 2,321.6 | 594,336.9 |
| 512 | 1,133 MiB | 1,469 MiB | 1,023.2 | 523,869.8 |

Đây là forward-only synthetic profile của student trên một L40S; throughput thực tế của E1 sẽ thấp hơn vì backward, optimizer, dynamic padding và data loading. Không dùng profile này để chọn test result.

BF16 autocast cũng **PASS** với batch 32 và AdamW thật: loss/gradient finite ở cả hai độ dài. Full train-step profile đạt peak allocated 2,592 MiB ở length 256 và 3,789 MiB ở length 512; peak reserved tương ứng 2,724 MiB và 4,166 MiB. Throughput là 1,181.2 sequences/s (256) và 785.1 sequences/s (512), tính trên hai bước đo sau một warmup. BF16 có thể được chọn cho E1 theo protocol, với điều kiện runner giữ loss/gradient finite trong smoke train của từng task.

## Điều kiện chuyển sang E1

E0 hardware/checkpoint gate đã đạt. Runner E1 vẫn phải hoàn tất smoke train/validation nhỏ, kiểm tra loader/evaluator và artifact contract trước khi chạy grid 30 epochs. GPU2 hiện chỉ còn khoảng 16 GiB trống do workload khác; không giả định mọi GPU đều có cùng memory khả dụng. Inventory/process evidence có trong `environment.json`, bao gồm GPU UUID để phân biệt process đang chạy.

Không có claim nào ở đây tái tạo lịch sử training hoặc chứng minh chất lượng benchmark; đây chỉ là bằng chứng runtime, checkpoint integrity và tài nguyên hiện tại.

Các lệnh tái chạy trên máy này:

```bash
.venv-e0-cuda/bin/python scripts/check_environment.py --output artifacts/e0/environment.json
CUDA_VISIBLE_DEVICES=1 .venv-e0-cuda/bin/python scripts/smoke_models.py --device cuda --profile --bf16-profile --profile-steps 2 --profile-warmup 1 --output artifacts/e0/smoke_models_gpu1_bf16.json
```
