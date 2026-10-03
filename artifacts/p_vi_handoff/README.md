# P_vi base — handoff package

This is the **shared P_vi encoder before supervised fine-tuning**. It starts from
`BidirLM/BidirLM-0.6B-Base` at commit
`29794508bbb4e378b4789eebff42ae4c2d47d524`, then receives MNTP training on
249,987,072 Vietnamese tokens. The model has 596,049,920 parameters. It is **not**
one of the five task-specific classifiers; downstream users must add a task head
and fine-tune for their own data.

## Contents

| Directory | Weights file | Use |
|---|---:|---|
| `fp32/` | 2,384,234,968 bytes | Canonical model for downstream fine-tuning; also works on CPU for inference. |
| `bf16/` | 1,192,135,120 bytes | Lower-memory NVIDIA GPU inference candidate. |
| `gpu_int8/` | 753,145,844 bytes | bitsandbytes INT8 GPU **inference** candidate; do not use as the default fine-tuning source. |

Each directory contains its own config, tokenizer, and pinned BidirLM modeling
code. `manifest.json` records SHA-256 hashes for every variant file.

## Load the model

Tested with Python 3.12.3, PyTorch 2.6.0+cu124, Transformers 4.57.6, and
Safetensors 0.8.0. INT8 also requires bitsandbytes 0.50.2 and Accelerate
1.15.0. The supplied custom modeling code is loaded with `trust_remote_code`.

```python
import torch
from transformers import AutoModel, AutoTokenizer

path = "fp32"  # or an absolute path to this subdirectory
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
model = AutoModel.from_pretrained(
    path, trust_remote_code=True, local_files_only=True,
    attn_implementation="sdpa", dtype=torch.float32,
).eval()  # .to("cuda") for GPU; leave on CPU for CPU inference

batch = tokenizer(["Ví dụ tiếng Việt."], padding=True, truncation=True,
                  max_length=256, return_tensors="pt")
with torch.inference_mode():
    hidden = model(**batch, return_dict=True).last_hidden_state
```

For BF16 GPU, use `bf16/`, `dtype=torch.bfloat16`, and move model/batch to CUDA.
For INT8 GPU, use `gpu_int8/` with `device_map={"": 0}`; its config loads the
saved bitsandbytes quantization automatically. Always pass the tokenizer's
`attention_mask`, including when a batch has padding. Keep the tokenizer as
shipped: changing its regex fix option would change tokenization relative to
the training/evaluation runs.

## Validation and limits

- `validation_report.json`: FP32 CPU loads and produces finite hidden states;
  BF16 GPU reloads with all 596,049,920 parameters, reads bidirectionally,
  and has pooled-vector cosine at least 0.99992 relative to FP32 GPU on six
  short Vietnamese inputs.
- `gpu_int8_validation.json`: 196 linear modules converted; two independent
  loads of the saved artifact produced identical hidden states; pooled-vector
  cosine at least 0.99849 relative to BF16 on the same inputs. Bidirectional
  attention and padding isolation were also checked.
- These are **encoder smoke and numerical checks**, not downstream F1
  validation of the BF16/INT8 base variants. The five published benchmark
  scores belong to five separately fine-tuned FP32 checkpoints. Select a
  compressed variant for a downstream application only after measuring that
  application's validation quality and latency.
- `gpu_latency_probe.json` is a short shared-GPU model-only timing sample;
  contention was substantial, so its latency numbers are not a production
  speed guarantee. GPU INT8 reduces stored weights and VRAM but is not
  necessarily faster for short sequences.
- Exploratory CPU dynamic INT8 changed pooled vectors much more on small
  probes (minimum cosine about 0.95). A CPU INT8 variant is therefore not
  included. The `fp32/` variant is the validated CPU fallback.

The upstream BidirLM checkpoint is Apache-2.0; the extra Vietnamese MNTP
corpus used here is CC0. Check the licensing of any separate supervised
dataset before distributing a model fine-tuned on it.
