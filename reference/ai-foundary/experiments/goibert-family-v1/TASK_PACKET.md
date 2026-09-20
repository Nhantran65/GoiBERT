# TASK PACKET — GoiBERT family v1, Stage A (VSFC quick loop)

GOAL: Implement `experiments/goibert-family-v1/` — a 3-model family (GoiTurn-XS, GoiTurn-S,
GoiBERT-B) derived from BamiBERT via orthodox task distillation, with a fair same-protocol
comparison harness, first on UIT-VSFC.

BUSINESS REASON: Team goal — our strongest model must be within 5% relative of BamiBERT on
shared public datasets. This proves the GoiBERT family is real, not vaporware, using
defensible methods (fine-tuning + TinyBERT-style task distillation), no tricks.

SUCCESS CRITERIA:
1. `python src/run_family.py --dataset vsfc --smoke` completes on CPU (tiny caps).
2. Full mode trains, per dataset: (a) BamiBERT baseline fine-tune over lr grid
   {1e-5, 2e-5, 5e-5}; (b) GoiBERT-B = identical to baseline (BamiBERT init, fine-tune) —
   report it as goibert_b; (c) GoiTurn-S distilled from the best baseline checkpoint over lr
   grid {5e-5, 1e-4, 2e-4}; (d) GoiTurn-XS same distillation, own grid.
3. Each run writes JSON (metrics + config + exact param counts); a final
   `results/summary_<dataset>.md` table shows Accuracy, macro F1, P/R, params, and
   **relative gap % vs baseline** for every model.
4. Best checkpoints saved under `results/checkpoints/`.

FILES (paths relative to repo root D:\01_Workspaces\TwoHearts\ml-models):
- CREATE `experiments/goibert-family-v1/src/run_family.py` (orchestrator)
- CREATE `experiments/goibert-family-v1/src/distill.py` (distillation trainer)
- CREATE `experiments/goibert-family-v1/configs/family.json` (three model configs)
- COPY from `experiments/goiturn-arch-scaling/src/`: `modeling.py`, `data.py`, `train.py`
  into `experiments/goibert-family-v1/src/` and adapt the copies. NEVER modify anything
  under goiturn-arch-scaling/ or goiturn-synthetic-v0/.
- CREATE `experiments/goibert-family-v1/README.md` (run commands, 30 lines max)

MODEL CONFIGS (exact, classic RoBERTa block same as BamiBERT — vocab 20,481, BamiBERT
tokenizer at `models/bamibert/`, max_position_embeddings 2050, absolute positions, GELU):
- `goiturn_xs`: 6 layers, hidden 256, 4 heads, FFN 1024
- `goiturn_s`: 10 layers, hidden 384, 6 heads, FFN 1536  (identical to arch-scaling
  variant_b — reuse its config)
- `goibert_b`: BamiBERT architecture, initialized FROM `models/bamibert/` weights
Print exact parameter counts (sum of p.numel()) at build time and store them in every JSON.

EXISTING PATTERNS TO REUSE:
- `experiments/goiturn-arch-scaling/src/train.py` already has: VSFC/ViSFD/VLSP loaders via
  data.py, dynamic-padding collate, AdamW, best-epoch-by-val-F1 selection, seed handling,
  ROCm guards (GOITURN_AMP / GOITURN_ATTN env vars — KEEP THEM), logit distillation
  (CE*0.5 + T^2*KL*0.5, T=2) and PCA embedding init
  (`experiments/goiturn-arch-scaling/results/pca/bamibert_pca_384.pt`; also 512 exists).
  For XS at hidden 256 compute a fresh PCA projection of BamiBERT word embeddings to 256
  the same way the 384 one was made (read how in the arch-scaling code).
- VSFC metric convention: follow arch-scaling exactly (accuracy + macro F1 + per-class P/R)
  so numbers are comparable with the paper's 93.86 acc / 83.41 F1.

DISTILLATION SPEC (orthodox TinyBERT-style task distillation — implement in distill.py):
- Teacher: the best baseline (BamiBERT fine-tuned) checkpoint for that dataset, frozen,
  eval mode, forward with `output_hidden_states=True, output_attentions=True`.
- Student losses (weights as constants, easy to edit):
  L = 0.5*CE(hard labels) + 0.5*T^2*KL(student/T || teacher/T), T=2
    + 0.1 * mean over mapped layers of MSE(W_i @ student_hidden_i, teacher_hidden_map(i))
      where W_i is a learned Linear(H_student -> 768) per mapped layer (trained jointly),
      layer map: student layer i (1-based) -> teacher layer round(i * 12 / L_student);
      also map embeddings layer 0 -> 0. Use [all tokens] with attention-mask averaging.
    + 0.1 * mean over mapped layers of MSE(student_attn_i, teacher_attn_map(i)) over valid
      positions (attention probabilities; detach teacher).
- Init: student word embeddings from PCA projection of BamiBERT embeddings (as arch-scaling
  did); other weights default init.
- Students train 15 epochs; baseline/goibert_b 10 epochs (paper-style grids above); batch 32,
  max_length 256, seed 42; best epoch by val macro F1.
- Report every (model, lr) run in JSON; summary table shows the best-by-val run per model
  evaluated once on test. NO test-set selection (that would be cheating): pick by val, report
  test of that pick.

ORCHESTRATOR (`run_family.py`):
- `--dataset {vsfc,vlsp_restaurant,visfd}` (phoatis intent may come in Stage B — do not
  build it now), `--stage {baseline,students,all}`, `--smoke`, `--seed 42`.
- Order: baseline grid -> pick best by val -> save checkpoint + record which lr won ->
  goibert_b IS that run (alias in summary, honestly labeled "BamiBERT init, fine-tuned") ->
  students distill from it over their grid.
- Summary row format: model | params | lr | acc | macro F1 | gap% F1 vs baseline
  (gap% = (base - ours) / base * 100, negative = we beat it).
- Every JSON includes torch/transformers versions and device name (copy the pattern from
  goiturn-synthetic-v0/src/train_turn.py result dict).

CONSTRAINTS:
- Write ONLY inside experiments/goibert-family-v1/ (plus nothing else).
- stdlib + torch + transformers + sklearn only. Match existing code style (plain argparse
  scripts).
- Do not touch models/bamibert/, other experiments, tracker/plan files, LUMI scripts.

NON-GOALS: No PhoATIS task yet. No modern block (RoPE/GeGLU/RMSNorm) — classic block only.
No MLM pretraining. No ONNX/quantization. No turn-detection data. No factorized embeddings.

DONE WHEN:
- `python -m py_compile` passes on all new/copied files.
- `python src/run_family.py --dataset vsfc --smoke` runs end-to-end IF torch is importable;
  torch is NOT importable on this Windows host, so instead: verify with py_compile plus a
  `--dry-run` flag that builds configs, counts params arithmetically (no torch), prints the
  planned run matrix, and exits. State clearly which checks ran.
- README documents: smoke command (Docker), LUMI full command per stage.

VALIDATE (run and report):
- `python -m py_compile experiments/goibert-family-v1/src/*.py`
- `python experiments/goibert-family-v1/src/run_family.py --dataset vsfc --dry-run`

STOP RULES:
- Missing prerequisite -> implement what you can, report what is blocked. Do not fake.
- Ambiguous requirement -> stop and report the gap instead of guessing.
- Only a hack remains -> stop and report the blocker.
- Unsure a loss implementation is solid -> add a toy self-test (e.g. distill a 2-layer
  student on random data for 20 steps, assert losses decrease and shapes match) before done.

ROLLBACK: additive only under experiments/goibert-family-v1/; delete the folder to undo.
