# E2 CPU diagnostic archive

The first E2 test attempt ran while the sandbox process could not see the host NVIDIA driver. It completed all 15 checkpoints with the wrapper's explicitly requested CPU BF16 fallback, but those outputs are retained as diagnostic evidence only and are excluded from the final E2 readout.

The final readout was regenerated after the same 15 validation-locked checkpoints were evaluated on host CUDA GPUs 1–3 with BF16/SDPA. The original CPU per-checkpoint files are preserved in `original/` and in `e2_cpu_test_outputs.tar`; the aggregate CPU files are `test_readout_cpu.json`, `test_access_log_cpu.json`, `test_comparison_cpu.csv`, and `E2_TEST_EVALUATION_cpu.md`.

No checkpoint, branch, hyperparameter, or selection manifest was changed between the diagnostic and the final CUDA evaluation.
