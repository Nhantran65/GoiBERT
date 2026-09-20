# E15 — Centered pair-value residual on original BamiBERT

Protocol: `E15-BamiBERT-original-pair-value-residual-v1`; revision `e15-20260912-r1`; seed `42`.

E15 transferred the E13 pair-value residual to the pinned original BamiBERT checkpoint. The branch was inserted at block index 7 of the 12-layer, 768-wide, 12-head encoder. Both variants used rank 64, chunk size 64, exact GELU, mean routing over the original 12 attention maps, FP32 routing scores/softmax under BF16 autocast, no routing dropout, and query-padding delta zero. The centered branch used `GELU(q+k)-GELU(q)-GELU(k)`; the control used `GELU(q+k)`. Each variant added exactly 147,456 parameters.

## Preflight and provenance

All preflight checks passed: official source and weight identity, 12×768×12 architecture, five task output shapes, exact E14 common-state and t0-logit parity with P initialized to zero, FP32 routing, centered null/linear identities, chunked forward/gradient parity, P gradient at step 1, U/W gradients after a P step, uniform AdamW coverage, and strict checkpoint branch/head roundtrip. The actual CUDA profile used BF16 model768 backward at batch32 on GPUs 0–2 and batch16 with accumulation 2 on GPU3.

The backbone is `models/bamibert-base-original`, from `Qualcomm-AI-Research/BamiBERT` at HF commit `57bc1340debbe4e348ec549047a763caebe4a977`; `model.safetensors` SHA256 is `c8c8abe5914f2543bd6a65019b685dacf25ad39da5f01b680b082876e1947627`. The source runner hash frozen before dispatch is `878fd05cf5557876f7927cd6457333e5e932bc2389f95f045d59d17cf0b72473`. E1–E14 source and artifacts were left unchanged. Numeric gold labels are unchanged; report names use ABSA `0=absent, 1=positive, 2=negative, 3=neutral`; ViSFD OTHERS is detection-only and pair-masked.

## Validation results

Scores are fractions in `[0,1]`. Delta columns are percentage points (`100 × difference`) against the validation-selected E14 local baseline and the same-task uncentered control.

| Task | E14 | Centered | Uncentered | Centered − E14 (pp) | Centered − control (pp) | Centered best epoch | Control best epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| vsfc_sentiment | 0.863252 | 0.848146 | 0.854288 | -1.511 | -0.614 | 29 | 14 |
| vsfc_topic | 0.795328 | 0.800794 | 0.797093 | +0.547 | +0.370 | 5 | 5 |
| visfd | 0.843834 | 0.838806 | 0.840872 | -0.503 | -0.207 | 25 | 30 |
| absa_restaurant | 0.747418 | 0.737535 | 0.737682 | -0.988 | -0.015 | 17 | 30 |
| absa_hotel | 0.732362 | 0.723861 | 0.731349 | -0.850 | -0.749 | 30 | 22 |

The predeclared screening gate did not pass. Centered improved over E14 on 1/5 tasks, reached at least +0.3pp on 1/5, dropped by more than 0.3pp on 4/5, and was better than the uncentered control on 1/5. This is a one-seed validation screen, not a significance test. Because the broad gate failed, E15 test evaluation was not authorized and no test split was loaded.

## Runtime and artifacts

The ten validation trials completed with the fixed E14-selected rates: sentiment `2e-5`, topic `1e-5`, and ViSFD/Restaurant/Hotel `5e-5`. Trial elapsed time, peak memory, correction probes, checkpoint hashes, predictions, and optimizer metadata are stored per trial under `artifacts/e15/trials/`. Summary and audit files are `artifacts/e15/validation_summary.json`, `artifacts/e15/validation_summary.csv`, `artifacts/e15/locked_checkpoints.json`, `artifacts/e15/preflight.json`, `artifacts/e15/runtime_stats.json`, `artifacts/e15/source_checkpoint_manifest.json`, and `artifacts/e15/.orchestrator/controller_state.json`.

The result does not support a broad E15 investment signal or a centering-specific improvement on original BamiBERT under this recipe. No new architecture, grid, seed, or follow-up run was opened.
