Bản review ban đầu dưới đây là lịch sử. Audit sau chạy đã PASS: [E3_REVIEW_FINAL.md](E3_REVIEW_FINAL.md).

# E3 independent review

Status: implementation review pending. No E3 training or inference was run by
this reviewer.

## Frozen E1 contract checked

The current E1 recipe uses a fresh `models/goiturn-s-gd` encoder and fresh
`CLS -> dropout(0.1) -> dense(384) -> tanh -> dropout(0.1) -> logits` head.
For ABSA it computes the ordinary per-slot cross-entropy over the complete
model label space, including class `0` (absent), with the fixed slot schema
from `goibert.data`. It uses AdamW (weight decay 0.01 except bias/LayerNorm),
betas `(0.9, 0.999)`, epsilon `1e-8`, linear warmup/decay with 10% warmup,
gradient clipping at 1.0, effective batch 32, deterministic seed/shuffle,
and validation pair micro-F1 for ABSA checkpoint selection.

The two E3 trials must preserve those settings at LR `1e-4`, 30 epochs,
microbatch 32, max length 256, BF16/SDPA, and `include_test=False`.

## Loss review criteria

The auxiliary term must be exactly:

`CE_E1 + 0.5 * mean(CE(logits[gold-present, 1:4], gold-label - 1))`.

The mean pools all eligible cells in the batch. It must exclude all absent
cells and the binary ViSFD `OTHERS` slot, keep the native label ordering for
ViSFD and Restaurant, and return a differentiable zero when a batch has no
eligible cells. The original E1 CE must remain unchanged.

Before full training I will check the isolated loss and runner against these
conditions, the exact model/optimizer/scheduler/loader contract, and the
test-split boundary. Afterward I will independently recompute validation
metrics from saved predictions and verify provenance/source hashes.
