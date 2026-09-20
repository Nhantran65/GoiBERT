# E2 loss and noise contract

This document freezes the loss/noise interface used by the E2 runner.  It is
deliberately separate from `src/goibert/` so the E1 runner and its source hash
remain unchanged.

## Tensor conventions

The task runner passes logits in one of these forms:

| Task shape | Logits | Labels |
| --- | --- | --- |
| VSFC sentiment/topic | `[B, C]` | `[B]` |
| Uniform ABSA | `[B, S, 4]` | `[B, S]` |
| Packed ABSA | `[B, sum(widths)]` | `[B, S]` |

`slot_class_counts` is the canonical schema.  With the current E1 loader's
aspect order, ViSFD must pass
`(4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4)`: ten polarity slots and a binary
`OTHERS` slot at index 5.  A different tuple such as `(4,) * 10 + (2,)` is
valid only if the caller has also reordered the aspect slots and labels.  If
padded `[B, S, max_width]` logits are used, columns beyond a slot's declared
width are sliced before any probability operation.  They are invalid classes
and never contribute to CE, KD, or NC.

Labels are in model space.  For ViSFD, the data adapter has already converted
raw `OTHERS=4` to model `OTHERS=1`; raw labels must not be passed to CE.
Absent (`0`) is a real target and is included in CE.  Only `ignore_index=-100`
or an explicitly false `valid_slot_mask` excludes a target.

## Task KD (E2a)

```python
logits_kd_loss(
    student_logits,
    teacher_logits,
    temperature=2.0,
    slot_class_counts=counts,
)
```

The implementation computes logits-only

```text
KD = T² · mean KL(softmax(teacher/T) || softmax(student/T))
```

The teacher distribution is detached and should be evaluated under
`torch.inference_mode()` by the runner.  All logits/probability arithmetic is
performed in FP32 even when the surrounding model uses BF16/FP16 autocast.
No hidden-state or attention tensors are read.  For the hard-label component:

```python
task_kd_loss = (1 - alpha) * CE(student_logits, labels) + alpha * KD
```

`alpha` defaults to `0.5`; the runner records the resolved value and the
individual `ce`, `kd`, and `total` values from `task_kd_components`.
`logit_distillation_loss`, `logit_kd_loss`, and `kd_loss` are compatibility
aliases for `logits_kd_loss` and therefore already include `T²`; callers must
not multiply their return value by `T²` again.

## Noise consistency (E2b)

```python
noise_consistency_loss(clean_logits, noisy_logits, temperature=1.0)
```

NC is a separate symmetric divergence:

```text
NC = 0.5 · KL(clean || noisy) + 0.5 · KL(noisy || clean)
```

It uses `T=1` by default and has no `T²` factor.  Both forward passes remain
attached to the graph: gradients flow into clean and noisy logits.  The
E2b total is:

```text
CE(clean_logits, labels) + lambda_nc · NC
```

The clean CE is the only hard-label term.  The noisy view has no second CE.

## Combined E2c

```python
combined_kd_noise_loss(
    clean_logits,
    noisy_logits,
    teacher_logits,
    labels,
    alpha=0.5,
    noise_weight=lambda_nc,
    kd_temperature=2.0,
    nc_temperature=1.0,
)
```

It computes:

```text
(1 - alpha) · CE(clean) + alpha · KD(clean, teacher)
    + lambda_nc · NC(clean, noisy)
```

`combined_kd_noise_components` exposes all four terms for logging.  The
teacher is used only for logits KD; hidden/attention losses are out of scope.

## Deterministic text noise

```python
noisy_view(text, seed=seed, epoch=epoch, stable_id=example_id)
```

The record-local RNG seed is derived with SHA-256 from the integer seed,
epoch, and stable ID.  Python's process-global `hash()` and global
`random` state are never used.  The mutually exclusive operation probabilities
are:

| Operation | Probability |
| --- | ---: |
| unchanged | 50% |
| remove Vietnamese accents (`đ` → `d`) | 25% |
| tiny explicit word-boundary teen-code map | 20% |
| one internal-letter typo | 5% |

The map excludes negation words, and word matching is Unicode word-boundary
and case-aware.  The typo operation edits only an internal letter of a
non-negating alphabetic token; digits, punctuation, and whitespace are copied
verbatim.  Noise never truncates, strips, deletes a digit/punctuation mark, or
turns an empty input into a fabricated token.  `noisy_batch` returns the noisy
strings plus operation counts and the *actual* changed-row rate for runner
logs.  The actual rate can be below 50% of rows chosen for an edit when a row
has no eligible teen-code word or typo candidate.

## Focused fixtures

`tests/test_e2_losses_noise.py` covers:

* KL direction and the `T²` multiplier;
* padded invalid-class masking;
* absent CE and ViSFD ten-by-four-plus-binary slot shape;
* NC symmetry, `T=1`, and gradients to both views;
* deterministic `(seed, epoch, stable_id)` streams, independent IDs, global
  RNG preservation, raw-text preservation, operation/change-rate logging, and
  negation/boundary behavior.
